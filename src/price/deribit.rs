use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info, warn};

use super::hyperliquid::{FeedHealth, HealthTracker, reconnect_delay};
use super::types::{PriceSource, PriceTick};

pub const DEFAULT_DERIBIT_WS_URL: &str = "wss://www.deribit.com/ws/api/v2";

/// Normalizes a ticker name, Hyperliquid coin (`xyz:NVDA`), Deribit index name (`nvda_usdc`),
/// or Deribit channel (`deribit_price_index.nvda_usdc`) to the lowercase Deribit index name
/// `<sym>_usdc` (e.g. `"NVDA"` -> `"nvda_usdc"`).
pub fn deribit_index_name(symbol_or_coin: &str) -> String {
    let trimmed = symbol_or_coin.trim();
    let stripped = trimmed
        .strip_prefix("deribit_price_index.")
        .unwrap_or(trimmed);
    let base = stripped.split(':').next_back().unwrap_or(stripped);
    let lower = base.to_ascii_lowercase();
    if lower.ends_with("_usdc") {
        lower
    } else {
        format!("{lower}_usdc")
    }
}

/// Returns the Deribit public index subscription channel (`deribit_price_index.<sym>_usdc`).
pub fn deribit_index_channel(symbol_or_coin: &str) -> String {
    format!("deribit_price_index.{}", deribit_index_name(symbol_or_coin))
}

/// Returns the optional Deribit perpetual ticker channel (`ticker.<SYM>_USDC-PERPETUAL.100ms`).
pub fn deribit_ticker_channel(symbol_or_coin: &str) -> String {
    let sym = deribit_ticker_symbol(symbol_or_coin);
    format!("ticker.{sym}_USDC-PERPETUAL.100ms")
}

/// Extracts the uppercase base ticker symbol (`"NVDA"`, `"TSLA"`) from a ticker name,
/// Hyperliquid coin (`"xyz:NVDA"`), Deribit index name (`"nvda_usdc"`), or Deribit channel.
pub fn deribit_ticker_symbol(symbol_or_coin: &str) -> String {
    let idx = deribit_index_name(symbol_or_coin);
    let sym = idx.strip_suffix("_usdc").unwrap_or(&idx);
    let sym = sym.strip_suffix("-perpetual").unwrap_or(sym);
    sym.to_ascii_uppercase()
}

/// Extracts the lowercase Deribit index name (`"nvda_usdc"`) from a Deribit ticker channel
/// (`"ticker.NVDA_USDC-PERPETUAL.100ms"`) or instrument name (`"NVDA_USDC-PERPETUAL"`).
pub fn deribit_ticker_index_name(channel_or_instrument: &str) -> Option<String> {
    let trimmed = channel_or_instrument.trim();
    if trimmed.is_empty() {
        return None;
    }
    let body = trimmed.strip_prefix("ticker.").unwrap_or(trimmed);
    let instrument = body.split('.').next().unwrap_or(body);
    let lower = instrument.to_ascii_lowercase();
    let idx = lower.strip_suffix("-perpetual").unwrap_or(&lower);
    if idx.is_empty() {
        None
    } else {
        Some(idx.to_string())
    }
}

/// Minimum interval (ms) between carried unchanged samples when liveness is proven.
pub const DERIBIT_CARRY_INTERVAL_MS: u64 = 1_000;

/// Single normalized socket event consumed by `DeribitSampler` (shared by live feed & replay).
#[derive(Debug, Clone, PartialEq)]
pub enum DeribitSampleEvent {
    /// `deribit_price_index.<sym>_usdc` print (the sole authoritative settlement price source).
    IndexPrint {
        index_name: String,
        price: f64,
        ts_ms: u64,
    },
    /// `ticker.<SYM>_USDC-PERPETUAL.100ms` update: updates perp book mid/mark and proves
    /// socket liveness only; never used as an index settlement price.
    Ticker {
        index_name: String,
        mid_px: Option<f64>,
        mark_price: Option<f64>,
        ts_ms: u64,
    },
    /// Any other proven-alive socket frame (`public/test` response with `usOut`, heartbeat,
    /// `trades.*`, etc.) that proves liveness without changing index prices.
    Liveness { ts_ms: u64 },
}

impl DeribitSampleEvent {
    pub fn ts_ms(&self) -> u64 {
        match self {
            Self::IndexPrint { ts_ms, .. }
            | Self::Ticker { ts_ms, .. }
            | Self::Liveness { ts_ms } => *ts_ms,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CoinSampleState {
    last_emitted_cents: Option<i64>,
    last_emitted_price: Option<f64>,
    last_emitted_ts_ms: Option<u64>,
    latest_mid_px: Option<f64>,
    latest_mid_ts_ms: Option<u64>,
    latest_mark_px: Option<f64>,
    latest_mark_ts_ms: Option<u64>,
}

impl CoinSampleState {
    fn fresh_book(&self, now_ms: u64, max_age_ms: u64) -> (Option<f64>, Option<f64>) {
        let mid = match (self.latest_mid_px, self.latest_mid_ts_ms) {
            (Some(v), Some(ts)) if now_ms.saturating_sub(ts) <= max_age_ms => Some(v),
            _ => None,
        };
        let mark = match (self.latest_mark_px, self.latest_mark_ts_ms) {
            (Some(v), Some(ts)) if now_ms.saturating_sub(ts) <= max_age_ms => Some(v),
            _ => None,
        };
        (mid, mark)
    }
}

/// Shared Deribit sample-emission rule used identically by both the live WebSocket stream
/// and `replay::read_drb_jsonl_reader`:
/// - (a) Emits immediately ONLY on a `deribit_price_index.*` print whose cent value differs
///   from the last emitted value. `ticker.*` messages update perp book mid/mark and prove
///   liveness only; their `index_price` is never used as a settlement price.
/// - (b) Emits a carried unchanged sample (of the last printed index value) whenever `>= 1 s`
///   has elapsed since the coin's last emitted sample AND socket liveness is proven by an
///   incoming event on the same socket.
/// - Attaches the latest perp book mid `(best_bid + best_ask) / 2` in `PriceTick.mid_px` and
///   perp `mark_price` in `PriceTick.oracle_px` when fresh (`<= book_max_age_ms`, else `None`).
#[derive(Debug, Clone)]
pub struct DeribitSampler {
    coins: Vec<String>,
    states: HashMap<String, CoinSampleState>,
    book_max_age_ms: u64,
}

impl DeribitSampler {
    pub fn new<I, S>(coins: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut ordered = Vec::new();
        let mut states = HashMap::new();
        for coin in coins {
            let name = deribit_index_name(&coin.into());
            if !states.contains_key(&name) {
                ordered.push(name.clone());
                states.insert(name, CoinSampleState::default());
            }
        }
        Self {
            coins: ordered,
            states,
            book_max_age_ms: crate::v2::DERIBIT_BOOK_MAX_AGE_MS,
        }
    }

    /// Clears cached prices on WebSocket disconnect so reconnect never emits stale carried values
    /// before a fresh post-reconnect index/ticker reading arrives.
    pub fn reset_on_disconnect(&mut self) {
        for state in self.states.values_mut() {
            *state = CoinSampleState::default();
        }
    }

    /// Applies the shared 1b/1c sample rule to one socket event and returns all samples to emit.
    pub fn on_event(&mut self, event: &DeribitSampleEvent) -> Vec<(String, PriceTick)> {
        let mut emitted = Vec::new();
        let event_ts_ms = event.ts_ms();

        // Step 1: Update ticker metadata and handle Rule (a) immediate index-value change.
        let mut emitted_coin: Option<&str> = None;
        match event {
            DeribitSampleEvent::IndexPrint {
                index_name,
                price,
                ts_ms,
            } => {
                if price.is_finite() && *price > 0.0 {
                    if let Some(state) = self.states.get_mut(index_name) {
                        let cents = (*price * 100.0).round() as i64;
                        if state.last_emitted_cents != Some(cents) {
                            let emit_ts = state
                                .last_emitted_ts_ms
                                .map_or(*ts_ms, |prev| (*ts_ms).max(prev));
                            state.last_emitted_cents = Some(cents);
                            state.last_emitted_price = Some(*price);
                            state.last_emitted_ts_ms = Some(emit_ts);
                            let (mid_px, oracle_px) =
                                state.fresh_book(emit_ts, self.book_max_age_ms);
                            emitted.push((
                                index_name.clone(),
                                PriceTick {
                                    ts_ms: emit_ts,
                                    price: *price,
                                    mid_px,
                                    oracle_px,
                                },
                            ));
                            emitted_coin = Some(index_name.as_str());
                        }
                    }
                }
            }
            DeribitSampleEvent::Ticker {
                index_name,
                mid_px,
                mark_price,
                ts_ms,
            } => {
                if let Some(state) = self.states.get_mut(index_name) {
                    if let Some(mid) = mid_px.filter(|v| v.is_finite() && *v > 0.0) {
                        state.latest_mid_px = Some(mid);
                        state.latest_mid_ts_ms = Some(*ts_ms);
                    }
                    if let Some(mark) = mark_price.filter(|v| v.is_finite() && *v > 0.0) {
                        state.latest_mark_px = Some(mark);
                        state.latest_mark_ts_ms = Some(*ts_ms);
                    }
                }
            }
            DeribitSampleEvent::Liveness { .. } => {}
        }

        // Step 2: Rule (b) — for any tracked coin whose last emitted sample is >= 1 s old,
        // emit a carried unchanged sample proven alive by this socket event.
        for coin in &self.coins {
            if emitted_coin == Some(coin.as_str()) {
                continue;
            }
            if let Some(state) = self.states.get_mut(coin) {
                if let (Some(price), Some(last_ts)) =
                    (state.last_emitted_price, state.last_emitted_ts_ms)
                {
                    if event_ts_ms.saturating_sub(last_ts) >= DERIBIT_CARRY_INTERVAL_MS {
                        state.last_emitted_ts_ms = Some(event_ts_ms);
                        let (mid_px, oracle_px) =
                            state.fresh_book(event_ts_ms, self.book_max_age_ms);
                        emitted.push((
                            coin.clone(),
                            PriceTick {
                                ts_ms: event_ts_ms,
                                price,
                                mid_px,
                                oracle_px,
                            },
                        ));
                    }
                }
            }
        }

        emitted
    }
}

#[derive(Debug, Clone)]
pub struct DeribitConfig {
    pub ws_url: String,
    /// Lowercase Deribit index names (`"nvda_usdc"`, `"tsla_usdc"`, ...), matching sender keys.
    pub coins: Vec<String>,
    /// Subscribe to `ticker.<SYM>_USDC-PERPETUAL.100ms` for index updates, perp mid, and mark.
    pub subscribe_ticker: bool,
    pub stale_after: Duration,
    pub health_interval: Duration,
    /// Interval between `public/test` liveness probes on the socket (default 1 s).
    pub liveness_interval: Duration,
    pub heartbeat_interval: Duration,
    pub reconnect_base_delay: Duration,
    pub reconnect_max_delay: Duration,
}

impl Default for DeribitConfig {
    fn default() -> Self {
        Self {
            ws_url: DEFAULT_DERIBIT_WS_URL.to_string(),
            coins: vec!["nvda_usdc".to_string(), "tsla_usdc".to_string()],
            subscribe_ticker: true,
            stale_after: PriceSource::Deribit.timing().stale_after,
            health_interval: Duration::from_millis(500),
            liveness_interval: Duration::from_secs(1),
            heartbeat_interval: Duration::from_secs(30),
            reconnect_base_delay: Duration::from_secs(1),
            reconnect_max_delay: Duration::from_secs(30),
        }
    }
}

#[derive(Debug)]
pub struct DeribitStreamHandle {
    pub health_rx: watch::Receiver<HashMap<String, FeedHealth>>,
    pub prev_day_px: Arc<RwLock<HashMap<String, f64>>>,
    pub join_handle: JoinHandle<()>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeribitPriceIndex {
    pub index_name: String,
    pub price: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeribitTicker {
    pub channel: String,
    pub instrument_name: Option<String>,
    pub mark_price: Option<f64>,
    pub index_price: Option<f64>,
    pub best_bid_price: Option<f64>,
    pub best_ask_price: Option<f64>,
    pub last_price: Option<f64>,
    /// Perp 24h price change in percent (`stats.price_change`), used to derive `prev_day_px`.
    pub price_change_pct: Option<f64>,
    pub timestamp: u64,
}

impl DeribitTicker {
    /// Returns the perp order-book midpoint `(best_bid + best_ask) / 2` when both sides exist.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid_price, self.best_ask_price) {
            (Some(bid), Some(ask)) if bid.is_finite() && ask.is_finite() && bid > 0.0 && ask > 0.0 => {
                Some(0.5 * (bid + ask))
            }
            _ => None,
        }
    }

    /// Derives the 24h reference price `last_price / (1 + stats.price_change / 100)`.
    /// Returns `None` when `last_price` or `price_change_pct` is missing, non-finite,
    /// `last_price <= 0`, or `price_change_pct <= -100`.
    pub fn derived_prev_day_px(&self) -> Option<f64> {
        let last = self.last_price?;
        let change_pct = self.price_change_pct?;
        if !last.is_finite() || last <= 0.0 || !change_pct.is_finite() || change_pct <= -100.0 {
            return None;
        }
        let denom = 1.0 + change_pct / 100.0;
        if !denom.is_finite() || denom <= 0.0 {
            return None;
        }
        let prev = last / denom;
        if prev.is_finite() && prev > 0.0 {
            Some(prev)
        } else {
            None
        }
    }

    /// Resolves the lowercase index name (`"nvda_usdc"`) for this ticker frame.
    pub fn index_name(&self) -> Option<String> {
        self.instrument_name
            .as_deref()
            .and_then(deribit_ticker_index_name)
            .or_else(|| deribit_ticker_index_name(&self.channel))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeribitMessage {
    PriceIndex(DeribitPriceIndex),
    Ticker(DeribitTicker),
    Heartbeat { kind: String },
    RpcResult {
        id: Option<u64>,
        result: serde_json::Value,
        /// Server response timestamp derived from Deribit JSON-RPC `usOut` (`usOut / 1000`).
        us_out_ms: Option<u64>,
    },
    RpcError { id: Option<u64>, code: i64, message: String },
}

#[derive(Debug, Deserialize)]
struct JsonRpcEnvelope {
    id: Option<u64>,
    method: Option<String>,
    params: Option<serde_json::Value>,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
    #[serde(rename = "usOut")]
    us_out: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

#[derive(Debug, Deserialize)]
struct SubscriptionParams {
    channel: String,
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct HeartbeatParams {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct RawPriceIndexData {
    index_name: String,
    price: f64,
    timestamp: u64,
}

#[derive(Debug, Deserialize)]
struct RawTickerStats {
    price_change: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawTickerData {
    instrument_name: Option<String>,
    mark_price: Option<f64>,
    index_price: Option<f64>,
    best_bid_price: Option<f64>,
    best_ask_price: Option<f64>,
    last_price: Option<f64>,
    stats: Option<RawTickerStats>,
    timestamp: u64,
}

fn positive_finite(val: Option<f64>) -> Option<f64> {
    val.filter(|v| v.is_finite() && *v > 0.0)
}

pub(crate) fn parse_deribit_index_value(data: serde_json::Value) -> Option<DeribitPriceIndex> {
    let raw: RawPriceIndexData = serde_json::from_value(data).ok()?;
    if !raw.price.is_finite() || raw.price <= 0.0 {
        return None;
    }
    Some(DeribitPriceIndex {
        index_name: raw.index_name.to_ascii_lowercase(),
        price: raw.price,
        timestamp: raw.timestamp,
    })
}

pub(crate) fn parse_deribit_ticker_value(
    channel: &str,
    data: serde_json::Value,
) -> Option<DeribitTicker> {
    let raw: RawTickerData = serde_json::from_value(data).ok()?;
    let price_change_pct = raw
        .stats
        .and_then(|s| s.price_change)
        .filter(|v| v.is_finite());
    Some(DeribitTicker {
        channel: channel.to_string(),
        instrument_name: raw.instrument_name,
        mark_price: positive_finite(raw.mark_price),
        index_price: positive_finite(raw.index_price),
        best_bid_price: positive_finite(raw.best_bid_price),
        best_ask_price: positive_finite(raw.best_ask_price),
        last_price: positive_finite(raw.last_price),
        price_change_pct,
        timestamp: raw.timestamp,
    })
}

/// Parses one JSON-RPC 2.0 frame from Deribit's WebSocket API (`subscription`, `heartbeat`,
/// RPC `result`, or RPC `error`).
pub fn parse_deribit_message(text: &str) -> Option<DeribitMessage> {
    let env: JsonRpcEnvelope = serde_json::from_str(text).ok()?;
    if let Some(err) = env.error {
        return Some(DeribitMessage::RpcError {
            id: env.id,
            code: err.code,
            message: err.message,
        });
    }
    if let Some(result) = env.result {
        return Some(DeribitMessage::RpcResult {
            id: env.id,
            result,
            us_out_ms: env.us_out.filter(|&us| us > 0).map(|us| us / 1_000),
        });
    }
    match env.method.as_deref() {
        Some("heartbeat") => {
            let params: HeartbeatParams = serde_json::from_value(env.params?).ok()?;
            Some(DeribitMessage::Heartbeat { kind: params.kind })
        }
        Some("subscription") => {
            let params: SubscriptionParams = serde_json::from_value(env.params?).ok()?;
            if params.channel.starts_with("deribit_price_index.") {
                parse_deribit_index_value(params.data).map(DeribitMessage::PriceIndex)
            } else if params.channel.starts_with("ticker.") {
                parse_deribit_ticker_value(&params.channel, params.data).map(DeribitMessage::Ticker)
            } else {
                None
            }
        }
        _ => None,
    }
}

struct MonotonicWallClock {
    start_wall_ms: u64,
    start_instant: tokio::time::Instant,
}

impl MonotonicWallClock {
    fn new() -> Self {
        let start_wall_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            start_wall_ms,
            start_instant: tokio::time::Instant::now(),
        }
    }

    fn now_ms(&self) -> u64 {
        self.start_wall_ms
            .saturating_add(self.start_instant.elapsed().as_millis() as u64)
    }
}

struct DeribitStream {
    config: DeribitConfig,
    senders: HashMap<String, mpsc::Sender<PriceTick>>,
    health_tx: watch::Sender<HashMap<String, FeedHealth>>,
    prev_day_px: Arc<RwLock<HashMap<String, f64>>>,
}

enum HandleOutcome {
    Continue,
    SendText(String),
    Stop,
}

impl DeribitStream {
    fn subscribe_channels(&self) -> Vec<String> {
        let mut channels = Vec::with_capacity(
            self.config.coins.len() * if self.config.subscribe_ticker { 2 } else { 1 },
        );
        for coin in &self.config.coins {
            channels.push(deribit_index_channel(coin));
            if self.config.subscribe_ticker {
                channels.push(deribit_ticker_channel(coin));
            }
        }
        channels
    }

    async fn run(self) {
        let clock = MonotonicWallClock::new();
        let mut tracker =
            HealthTracker::new(self.config.coins.iter().cloned(), self.config.stale_after);
        let mut sampler = DeribitSampler::new(self.config.coins.iter().cloned());
        let health_period = self.config.health_interval.max(Duration::from_millis(1));
        let mut health_ticker = tokio::time::interval(health_period);
        health_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut reconnect_attempts = 0u32;
        let mut next_rpc_id = 1u64;

        loop {
            if self.all_senders_closed() {
                return;
            }

            info!(
                ws_url = %self.config.ws_url,
                reconnect_attempts,
                "connecting to Deribit stream"
            );

            let connect_fut = connect_async(&self.config.ws_url);
            tokio::pin!(connect_fut);
            let connect_result = loop {
                tokio::select! {
                    res = &mut connect_fut => break res,
                    _ = health_ticker.tick() => {
                        publish_health(&self.health_tx, tracker.tick(clock.now_ms()));
                    }
                }
            };

            match connect_result {
                Ok((socket, _)) => {
                    info!("Deribit stream connected");
                    reconnect_attempts = 0;
                    sampler.reset_on_disconnect();
                    tracker.set_reconnect_attempts(0);
                    publish_health(&self.health_tx, tracker.snapshot());

                    let (mut write, mut read) = socket.split();

                    let hb_secs = self.config.heartbeat_interval.as_secs().max(10);
                    let hb_req = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": next_rpc_id,
                        "method": "public/set_heartbeat",
                        "params": { "interval": hb_secs }
                    })
                    .to_string();
                    next_rpc_id = next_rpc_id.wrapping_add(1);

                    let channels = self.subscribe_channels();
                    let sub_req = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": next_rpc_id,
                        "method": "public/subscribe",
                        "params": { "channels": channels }
                    })
                    .to_string();
                    next_rpc_id = next_rpc_id.wrapping_add(1);

                    let mut setup_ok = true;
                    if let Err(error) = write.send(Message::Text(hb_req.into())).await {
                        warn!(?error, "failed to send Deribit set_heartbeat");
                        setup_ok = false;
                    } else if let Err(error) = write.send(Message::Text(sub_req.into())).await {
                        warn!(?error, "failed to send Deribit subscribe");
                        setup_ok = false;
                    }

                    if setup_ok {
                        let liveness_period = self
                            .config
                            .liveness_interval
                            .max(Duration::from_millis(250));
                        let liveness_start = tokio::time::Instant::now() + liveness_period;
                        let mut liveness_ticker =
                            tokio::time::interval_at(liveness_start, liveness_period);
                        liveness_ticker
                            .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

                        loop {
                            tokio::select! {
                                maybe_msg = read.next() => {
                                    match maybe_msg {
                                        Some(Ok(Message::Text(text))) => {
                                            match self.handle_text(&text, clock.now_ms(), &mut tracker, &mut sampler, &mut next_rpc_id).await {
                                                HandleOutcome::Continue => {}
                                                HandleOutcome::SendText(reply) => {
                                                    if let Err(error) = write.send(Message::Text(reply.into())).await {
                                                        warn!(?error, "Deribit heartbeat reply send error");
                                                        break;
                                                    }
                                                }
                                                HandleOutcome::Stop => return,
                                            }
                                        }
                                        Some(Ok(Message::Binary(bytes))) => {
                                            if let Ok(text) = std::str::from_utf8(&bytes) {
                                                match self.handle_text(text, clock.now_ms(), &mut tracker, &mut sampler, &mut next_rpc_id).await {
                                                    HandleOutcome::Continue => {}
                                                    HandleOutcome::SendText(reply) => {
                                                        if let Err(error) = write.send(Message::Text(reply.into())).await {
                                                            warn!(?error, "Deribit heartbeat reply send error");
                                                            break;
                                                        }
                                                    }
                                                    HandleOutcome::Stop => return,
                                                }
                                            }
                                        }
                                        Some(Ok(Message::Ping(payload))) => {
                                            if let Err(error) = write.send(Message::Pong(payload)).await {
                                                warn!(?error, "Deribit pong send error");
                                                break;
                                            }
                                        }
                                        Some(Ok(Message::Pong(_))) | Some(Ok(Message::Frame(_))) => {}
                                        Some(Ok(Message::Close(frame))) => {
                                            info!(?frame, "Deribit stream closed by peer");
                                            break;
                                        }
                                        Some(Err(error)) => {
                                            warn!(?error, "Deribit stream read error");
                                            break;
                                        }
                                        None => {
                                            info!("Deribit stream ended");
                                            break;
                                        }
                                    }
                                }
                                _ = liveness_ticker.tick() => {
                                    let test_req = serde_json::json!({
                                        "jsonrpc": "2.0",
                                        "id": next_rpc_id,
                                        "method": "public/test",
                                        "params": {}
                                    })
                                    .to_string();
                                    next_rpc_id = next_rpc_id.wrapping_add(1);
                                    if let Err(error) = write.send(Message::Text(test_req.into())).await {
                                        warn!(?error, "Deribit public/test send error");
                                        break;
                                    }
                                }
                                _ = health_ticker.tick() => {
                                    publish_health(&self.health_tx, tracker.tick(clock.now_ms()));
                                }
                            }
                        }
                    }
                    sampler.reset_on_disconnect();
                }
                Err(error) => {
                    warn!(?error, "failed to connect to Deribit stream");
                }
            }

            if self.all_senders_closed() {
                return;
            }

            reconnect_attempts = reconnect_attempts.saturating_add(1);
            tracker.set_reconnect_attempts(reconnect_attempts);
            publish_health(&self.health_tx, tracker.tick(clock.now_ms()));

            let delay = reconnect_delay(
                reconnect_attempts,
                self.config.reconnect_base_delay,
                self.config.reconnect_max_delay,
            );
            debug!(
                reconnect_attempts,
                delay_ms = delay.as_millis() as u64,
                "waiting before Deribit reconnect"
            );

            let sleep_fut = tokio::time::sleep(delay);
            tokio::pin!(sleep_fut);
            loop {
                tokio::select! {
                    _ = &mut sleep_fut => break,
                    _ = health_ticker.tick() => {
                        publish_health(&self.health_tx, tracker.tick(clock.now_ms()));
                    }
                }
            }
        }
    }

    fn mark_all_alive(&self, now_ms: u64, tracker: &mut HealthTracker) {
        for coin in &self.config.coins {
            tracker.on_sample(coin, now_ms);
        }
        publish_health(&self.health_tx, tracker.snapshot());
    }

    async fn emit_samples(&self, samples: Vec<(String, PriceTick)>) -> bool {
        for (index_name, tick) in samples {
            if let Some(tx) = self.senders.get(&index_name) {
                if tx.send(tick).await.is_err() {
                    warn!(index = %index_name, "Deribit price receiver dropped");
                    if self.all_senders_closed() {
                        return false;
                    }
                }
            }
        }
        true
    }

    async fn handle_text(
        &self,
        text: &str,
        now_ms: u64,
        tracker: &mut HealthTracker,
        sampler: &mut DeribitSampler,
        next_rpc_id: &mut u64,
    ) -> HandleOutcome {
        let Some(msg) = parse_deribit_message(text) else {
            return HandleOutcome::Continue;
        };

        match msg {
            DeribitMessage::PriceIndex(idx) => {
                self.mark_all_alive(now_ms, tracker);
                let ts_ms = if idx.timestamp > 0 {
                    idx.timestamp
                } else {
                    now_ms
                };
                let event = DeribitSampleEvent::IndexPrint {
                    index_name: idx.index_name,
                    price: idx.price,
                    ts_ms,
                };
                if !self.emit_samples(sampler.on_event(&event)).await {
                    return HandleOutcome::Stop;
                }
                HandleOutcome::Continue
            }
            DeribitMessage::Ticker(ticker) => {
                self.mark_all_alive(now_ms, tracker);
                debug!(
                    channel = %ticker.channel,
                    mark_price = ?ticker.mark_price,
                    index_price = ?ticker.index_price,
                    bid = ?ticker.best_bid_price,
                    ask = ?ticker.best_ask_price,
                    last = ?ticker.last_price,
                    ts = ticker.timestamp,
                    "Deribit ticker update"
                );
                let ts_ms = if ticker.timestamp > 0 {
                    ticker.timestamp
                } else {
                    now_ms
                };
                if let (Some(index_name), Some(prev)) =
                    (ticker.index_name(), ticker.derived_prev_day_px())
                {
                    self.prev_day_px
                        .write()
                        .expect("prev_day_px rwlock poisoned")
                        .insert(index_name, prev);
                }
                let event = match ticker.index_name() {
                    Some(index_name) => DeribitSampleEvent::Ticker {
                        index_name,
                        mid_px: ticker.mid_price(),
                        mark_price: ticker.mark_price,
                        ts_ms,
                    },
                    None => DeribitSampleEvent::Liveness { ts_ms },
                };
                if !self.emit_samples(sampler.on_event(&event)).await {
                    return HandleOutcome::Stop;
                }
                HandleOutcome::Continue
            }
            DeribitMessage::Heartbeat { kind } => {
                self.mark_all_alive(now_ms, tracker);
                let event = DeribitSampleEvent::Liveness { ts_ms: now_ms };
                if !self.emit_samples(sampler.on_event(&event)).await {
                    return HandleOutcome::Stop;
                }
                if kind == "test_request" {
                    let reply = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": *next_rpc_id,
                        "method": "public/test",
                        "params": {}
                    })
                    .to_string();
                    *next_rpc_id = next_rpc_id.wrapping_add(1);
                    HandleOutcome::SendText(reply)
                } else {
                    HandleOutcome::Continue
                }
            }
            DeribitMessage::RpcResult {
                id,
                result,
                us_out_ms,
            } => {
                debug!(?id, %result, ?us_out_ms, "Deribit RPC result");
                self.mark_all_alive(now_ms, tracker);
                let ts_ms = us_out_ms.unwrap_or(now_ms);
                let event = DeribitSampleEvent::Liveness { ts_ms };
                if !self.emit_samples(sampler.on_event(&event)).await {
                    return HandleOutcome::Stop;
                }
                HandleOutcome::Continue
            }
            DeribitMessage::RpcError { id, code, message } => {
                warn!(?id, code, %message, "Deribit RPC error");
                HandleOutcome::Continue
            }
        }
    }

    fn all_senders_closed(&self) -> bool {
        !self.senders.is_empty() && self.senders.values().all(|sender| sender.is_closed())
    }
}

fn publish_health(
    health_tx: &watch::Sender<HashMap<String, FeedHealth>>,
    next: HashMap<String, FeedHealth>,
) {
    health_tx.send_if_modified(|current| {
        if *current == next {
            false
        } else {
            *current = next;
            true
        }
    });
}

/// Spawns a background task subscribing to Deribit `deribit_price_index.<sym>_usdc`
/// (and optionally `ticker.<SYM>_USDC-PERPETUAL.100ms`) and tracking per-coin staleness.
pub fn spawn_deribit_stream(
    config: DeribitConfig,
    senders: HashMap<String, mpsc::Sender<PriceTick>>,
) -> DeribitStreamHandle {
    let initial_health =
        HealthTracker::new(config.coins.iter().cloned(), config.stale_after).snapshot();
    let (health_tx, health_rx) = watch::channel(initial_health);
    let prev_day_px = Arc::new(RwLock::new(HashMap::new()));
    let stream = DeribitStream {
        config,
        senders,
        health_tx,
        prev_day_px: Arc::clone(&prev_day_px),
    };
    let join_handle = tokio::spawn(stream.run());

    DeribitStreamHandle {
        health_rx,
        prev_day_px,
        join_handle,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deribit_channel_and_symbol_mapping_handles_tickers_and_coins() {
        assert_eq!(deribit_index_name("NVDA"), "nvda_usdc");
        assert_eq!(deribit_index_name("xyz:TSLA"), "tsla_usdc");
        assert_eq!(deribit_index_name("nvda_usdc"), "nvda_usdc");
        assert_eq!(
            deribit_index_name("deribit_price_index.nvda_usdc"),
            "nvda_usdc"
        );
        assert_eq!(
            deribit_index_channel("NVDA"),
            "deribit_price_index.nvda_usdc"
        );
        assert_eq!(
            deribit_ticker_channel("NVDA"),
            "ticker.NVDA_USDC-PERPETUAL.100ms"
        );
        assert_eq!(deribit_ticker_symbol("nvda_usdc"), "NVDA");
        assert_eq!(
            deribit_ticker_symbol("deribit_price_index.tsla_usdc"),
            "TSLA"
        );
    }

    #[test]
    fn parses_index_ticker_heartbeat_and_subscription_result_messages() {
        let idx_msg = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"deribit_price_index.nvda_usdc","data":{"index_name":"nvda_usdc","price":146.12,"timestamp":1795450000123}}}"#;
        assert_eq!(
            parse_deribit_message(idx_msg),
            Some(DeribitMessage::PriceIndex(DeribitPriceIndex {
                index_name: "nvda_usdc".to_string(),
                price: 146.12,
                timestamp: 1_795_450_000_123,
            }))
        );

        let ticker_msg = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","mark_price":146.15,"index_price":146.12,"best_bid_price":146.10,"best_ask_price":146.20,"last_price":146.14,"stats":{"price_change":-2.5},"timestamp":1795450000150}}}"#;
        assert_eq!(
            parse_deribit_message(ticker_msg),
            Some(DeribitMessage::Ticker(DeribitTicker {
                channel: "ticker.NVDA_USDC-PERPETUAL.100ms".to_string(),
                instrument_name: Some("NVDA_USDC-PERPETUAL".to_string()),
                mark_price: Some(146.15),
                index_price: Some(146.12),
                best_bid_price: Some(146.10),
                best_ask_price: Some(146.20),
                last_price: Some(146.14),
                price_change_pct: Some(-2.5),
                timestamp: 1_795_450_000_150,
            }))
        );

        let hb_msg = r#"{"jsonrpc":"2.0","method":"heartbeat","params":{"type":"test_request"}}"#;
        assert_eq!(
            parse_deribit_message(hb_msg),
            Some(DeribitMessage::Heartbeat {
                kind: "test_request".to_string(),
            })
        );

        let sub_res = r#"{"jsonrpc":"2.0","id":2,"usOut":1795450001123456,"result":["deribit_price_index.nvda_usdc","deribit_price_index.tsla_usdc"]}"#;
        match parse_deribit_message(sub_res) {
            Some(DeribitMessage::RpcResult {
                id: Some(2),
                result,
                us_out_ms: Some(1_795_450_001_123),
            }) => {
                assert_eq!(result.as_array().map(Vec::len), Some(2));
            }
            other => panic!("unexpected parsed subscription result: {other:?}"),
        }

        let err_msg = r#"{"jsonrpc":"2.0","id":3,"error":{"code":10020,"message":"invalid channel"}}"#;
        assert_eq!(
            parse_deribit_message(err_msg),
            Some(DeribitMessage::RpcError {
                id: Some(3),
                code: 10020,
                message: "invalid channel".to_string(),
            })
        );

        // Trades subscription is ignored cleanly (not an index or ticker).
        let trades_msg = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"trades.NVDA_USDC-PERPETUAL.100ms","data":[{"price":146.14,"amount":10.0}]}}"#;
        assert_eq!(parse_deribit_message(trades_msg), None);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn live_deribit_handler_emits_1hz_carried_samples_ignores_ticker_index_price_and_handles_dead_socket() {
        let (tx, mut rx) = mpsc::channel(16);
        let mut senders = HashMap::new();
        senders.insert("nvda_usdc".to_string(), tx);
        let (health_tx, health_rx) = watch::channel(HashMap::new());
        let config = DeribitConfig {
            coins: vec!["nvda_usdc".to_string()],
            ..DeribitConfig::default()
        };
        assert!(config.subscribe_ticker);
        assert_eq!(config.liveness_interval, Duration::from_secs(1));

        let stream = DeribitStream {
            config,
            senders,
            health_tx,
            prev_day_px: Arc::new(RwLock::new(HashMap::new())),
        };
        let mut tracker = HealthTracker::new(["nvda_usdc"], Duration::from_secs(3));
        let mut sampler = DeribitSampler::new(["nvda_usdc"]);
        let mut rpc_id = 10u64;

        // 1. Initial index print emits immediately with mid_px=None, oracle_px=None.
        let idx_msg = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"deribit_price_index.nvda_usdc","data":{"index_name":"nvda_usdc","price":146.12,"timestamp":1795450000100}}}"#;
        assert!(matches!(
            stream
                .handle_text(idx_msg, 1_795_450_000_120, &mut tracker, &mut sampler, &mut rpc_id)
                .await,
            HandleOutcome::Continue
        ));
        assert_eq!(
            rx.recv().await.expect("initial index tick"),
            PriceTick::new(1_795_450_000_100, 146.12)
        );

        // 2. Same-price ticker within < 1 s updates mid/mark without flooding.
        let quiet_ticker = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","mark_price":146.15,"index_price":146.12,"best_bid_price":146.10,"best_ask_price":146.20,"timestamp":1795450000250}}}"#;
        assert!(matches!(
            stream
                .handle_text(quiet_ticker, 1_795_450_000_260, &mut tracker, &mut sampler, &mut rpc_id)
                .await,
            HandleOutcome::Continue
        ));
        assert!(rx.try_recv().is_err());

        // 3. Quiet index + alive socket (public/test response with usOut every 1 s) -> 1 Hz carried
        //    samples using usOut server time and carrying latest mid_px and oracle_px.
        let test_resp_1 = r#"{"jsonrpc":"2.0","id":10,"usOut":1795450001100500,"result":{"version":"1.2.26"}}"#;
        assert!(matches!(
            stream
                .handle_text(test_resp_1, 1_795_450_001_120, &mut tracker, &mut sampler, &mut rpc_id)
                .await,
            HandleOutcome::Continue
        ));
        let carried_1 = rx.recv().await.expect("1 Hz carried sample #1");
        assert_eq!(carried_1.ts_ms, 1_795_450_001_100);
        assert_eq!(carried_1.price, 146.12);
        assert!((carried_1.mid_px.unwrap() - 146.15).abs() < 1e-12);
        assert_eq!(carried_1.oracle_px, Some(146.15));

        let test_resp_2 = r#"{"jsonrpc":"2.0","id":11,"usOut":1795450002100900,"result":{"version":"1.2.26"}}"#;
        assert!(matches!(
            stream
                .handle_text(test_resp_2, 1_795_450_002_120, &mut tracker, &mut sampler, &mut rpc_id)
                .await,
            HandleOutcome::Continue
        ));
        let carried_2 = rx.recv().await.expect("1 Hz carried sample #2");
        assert_eq!(carried_2.ts_ms, 1_795_450_002_100);
        assert_eq!(carried_2.price, 146.12);
        assert!((carried_2.mid_px.unwrap() - 146.15).abs() < 1e-12);
        assert_eq!(carried_2.oracle_px, Some(146.15));
        assert!(!health_rx.borrow()["nvda_usdc"].stale);

        // 4. Ticker with a DIFFERENT index_price (146.99) within < 1 s must NOT emit a price
        //    change; it only updates mid/mark. When >= 1 s later another ticker with index_price
        //    146.99 arrives, it triggers a carried sample of the LAST PRINTED value (146.12).
        let diverged_ticker_early = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","mark_price":146.18,"index_price":146.99,"best_bid_price":146.14,"best_ask_price":146.20,"timestamp":1795450002300}}}"#;
        assert!(matches!(
            stream
                .handle_text(
                    diverged_ticker_early,
                    1_795_450_002_320,
                    &mut tracker,
                    &mut sampler,
                    &mut rpc_id,
                )
                .await,
            HandleOutcome::Continue
        ));
        assert!(rx.try_recv().is_err(), "ticker index_price must not emit a price change");

        let diverged_ticker_1s = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","mark_price":146.18,"index_price":146.99,"best_bid_price":146.14,"best_ask_price":146.20,"timestamp":1795450003150}}}"#;
        assert!(matches!(
            stream
                .handle_text(
                    diverged_ticker_1s,
                    1_795_450_003_170,
                    &mut tracker,
                    &mut sampler,
                    &mut rpc_id,
                )
                .await,
            HandleOutcome::Continue
        ));
        let carried_3 = rx.recv().await.expect("carried sample of last printed index");
        assert_eq!(carried_3.ts_ms, 1_795_450_003_150);
        assert_eq!(carried_3.price, 146.12);
        assert!((carried_3.mid_px.unwrap() - 146.17).abs() < 1e-12);
        assert_eq!(carried_3.oracle_px, Some(146.18));

        // 5. Real deribit_price_index print with new cent value (146.14) emits immediately;
        //    unchanged repeat 20 ms later is suppressed.
        let new_idx = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"deribit_price_index.nvda_usdc","data":{"index_name":"nvda_usdc","price":146.14,"timestamp":1795450003300}}}"#;
        assert!(matches!(
            stream
                .handle_text(new_idx, 1_795_450_003_320, &mut tracker, &mut sampler, &mut rpc_id)
                .await,
            HandleOutcome::Continue
        ));
        let printed = rx.recv().await.expect("immediate deribit_price_index change");
        assert_eq!(printed.ts_ms, 1_795_450_003_300);
        assert_eq!(printed.price, 146.14);
        assert!((printed.mid_px.unwrap() - 146.17).abs() < 1e-12);
        assert_eq!(printed.oracle_px, Some(146.18));

        let dup_idx = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"deribit_price_index.nvda_usdc","data":{"index_name":"nvda_usdc","price":146.14,"timestamp":1795450003320}}}"#;
        assert!(matches!(
            stream
                .handle_text(dup_idx, 1_795_450_003_330, &mut tracker, &mut sampler, &mut rpc_id)
                .await,
            HandleOutcome::Continue
        ));
        assert!(rx.try_recv().is_err());

        // 6. Dead socket: no incoming frames -> zero carried samples emitted, stale after > 3 s.
        let snap_2s = tracker.tick(1_795_450_005_330);
        assert!(!snap_2s["nvda_usdc"].stale);
        assert!(rx.try_recv().is_err());
        let snap_3s_plus = tracker.tick(1_795_450_006_331);
        assert!(snap_3s_plus["nvda_usdc"].stale);
        assert!(rx.try_recv().is_err());

        // 7. Heartbeat test_request produces a public/test JSON-RPC reply.
        let hb_msg = r#"{"jsonrpc":"2.0","method":"heartbeat","params":{"type":"test_request"}}"#;
        match stream
            .handle_text(hb_msg, 1_795_450_006_400, &mut tracker, &mut sampler, &mut rpc_id)
            .await
        {
            HandleOutcome::SendText(reply) => {
                assert!(reply.contains(r#""method":"public/test""#), "got: {reply}");
                assert_eq!(rpc_id, 11);
            }
            _ => panic!("expected SendText reply for test_request heartbeat"),
        }
    }

    #[test]
    fn deribit_sampler_drops_book_mid_and_mark_when_older_than_book_max_age_ms() {
        let mut sampler = DeribitSampler::new(["nvda_usdc"]);
        let t0 = 1_795_450_000_000_u64;
        let _ = sampler.on_event(&DeribitSampleEvent::IndexPrint {
            index_name: "nvda_usdc".to_string(),
            price: 146.12,
            ts_ms: t0,
        });
        let _ = sampler.on_event(&DeribitSampleEvent::Ticker {
            index_name: "nvda_usdc".to_string(),
            mid_px: Some(146.20),
            mark_price: Some(146.15),
            ts_ms: t0 + 100,
        });

        // At t0 + 2100 ms, ticker age is 2000 ms (<= book_max_age_ms) -> book still attached.
        let fresh = sampler.on_event(&DeribitSampleEvent::Liveness { ts_ms: t0 + 2_100 });
        assert_eq!(fresh.len(), 1);
        assert_eq!(fresh[0].1.mid_px, Some(146.20));
        assert_eq!(fresh[0].1.oracle_px, Some(146.15));

        // At t0 + 3101 ms, ticker age is 3001 ms (> book_max_age_ms) -> mid/oracle cleared to None.
        let stale = sampler.on_event(&DeribitSampleEvent::Liveness { ts_ms: t0 + 3_101 });
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].1.mid_px, None);
        assert_eq!(stale[0].1.oracle_px, None);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn deribit_ticker_derives_prev_day_px_and_ignores_invalid_price_change() {
        let (tx, _rx) = mpsc::channel(8);
        let mut senders = HashMap::new();
        senders.insert("nvda_usdc".to_string(), tx);
        let (health_tx, _health_rx) = watch::channel(HashMap::new());
        let prev_day_px = Arc::new(RwLock::new(HashMap::new()));
        let stream = DeribitStream {
            config: DeribitConfig {
                coins: vec!["nvda_usdc".to_string()],
                ..DeribitConfig::default()
            },
            senders,
            health_tx,
            prev_day_px: Arc::clone(&prev_day_px),
        };
        let mut tracker = HealthTracker::new(["nvda_usdc"], Duration::from_secs(3));
        let mut sampler = DeribitSampler::new(["nvda_usdc"]);
        let mut rpc_id = 1u64;

        // Recorded ticker format: last_price=155.13, stats.price_change=-2.977 -> 155.13 / 0.97023 ≈ 159.89.
        let valid_ticker = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","mark_price":155.15,"index_price":155.12,"best_bid_price":155.10,"best_ask_price":155.16,"last_price":155.13,"stats":{"high":160.10,"low":154.00,"price_change":-2.977,"volume":1234.5},"timestamp":1795450000100}}}"#;
        let _ = stream
            .handle_text(valid_ticker, 1_795_450_000_120, &mut tracker, &mut sampler, &mut rpc_id)
            .await;
        let stored = prev_day_px.read().unwrap().get("nvda_usdc").copied().unwrap();
        let expected = 155.13_f64 / (1.0 - 2.977_f64 / 100.0);
        assert!((stored - expected).abs() < 1e-12);
        assert!((stored - 159.89).abs() < 0.005);

        // Missing stats, missing last_price, or price_change <= -100 must be ignored and keep last valid value.
        for bad_ticker in [
            r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","last_price":155.13,"stats":{"price_change":-100.0},"timestamp":1795450000200}}}"#,
            r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","last_price":155.13,"stats":{"price_change":-120.5},"timestamp":1795450000300}}}"#,
            r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","last_price":155.13,"timestamp":1795450000400}}}"#,
            r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","stats":{"price_change":1.5},"timestamp":1795450000500}}}"#,
        ] {
            let _ = stream
                .handle_text(bad_ticker, 1_795_450_000_600, &mut tracker, &mut sampler, &mut rpc_id)
                .await;
            assert_eq!(
                prev_day_px.read().unwrap().get("nvda_usdc").copied(),
                Some(stored)
            );
        }

        // Direct unit check on invalid DeribitTicker values.
        let mut t = DeribitTicker {
            channel: "ticker.NVDA_USDC-PERPETUAL.100ms".to_string(),
            instrument_name: Some("NVDA_USDC-PERPETUAL".to_string()),
            mark_price: None,
            index_price: None,
            best_bid_price: None,
            best_ask_price: None,
            last_price: Some(155.13),
            price_change_pct: Some(f64::NAN),
            timestamp: 1,
        };
        assert_eq!(t.derived_prev_day_px(), None);
        t.price_change_pct = Some(-100.0);
        assert_eq!(t.derived_prev_day_px(), None);
        t.price_change_pct = Some(0.0);
        assert_eq!(t.derived_prev_day_px(), Some(155.13));
    }
}
