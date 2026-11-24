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
use super::types::PriceTick;

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

#[derive(Debug, Clone)]
pub struct DeribitConfig {
    pub ws_url: String,
    /// Lowercase Deribit index names (`"nvda_usdc"`, `"tsla_usdc"`, ...), matching sender keys.
    pub coins: Vec<String>,
    /// Optional subscription to `ticker.<SYM>_USDC-PERPETUAL.100ms` for display/diagnostics only.
    pub subscribe_ticker: bool,
    pub stale_after: Duration,
    pub health_interval: Duration,
    pub heartbeat_interval: Duration,
    pub reconnect_base_delay: Duration,
    pub reconnect_max_delay: Duration,
}

impl Default for DeribitConfig {
    fn default() -> Self {
        Self {
            ws_url: DEFAULT_DERIBIT_WS_URL.to_string(),
            coins: vec!["nvda_usdc".to_string(), "tsla_usdc".to_string()],
            subscribe_ticker: false,
            stale_after: Duration::from_secs(3),
            health_interval: Duration::from_millis(500),
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
    pub mark_price: Option<f64>,
    pub index_price: Option<f64>,
    pub best_bid_price: Option<f64>,
    pub best_ask_price: Option<f64>,
    pub last_price: Option<f64>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeribitMessage {
    PriceIndex(DeribitPriceIndex),
    Ticker(DeribitTicker),
    Heartbeat { kind: String },
    RpcResult { id: Option<u64>, result: serde_json::Value },
    RpcError { id: Option<u64>, code: i64, message: String },
}

#[derive(Debug, Deserialize)]
struct JsonRpcEnvelope {
    id: Option<u64>,
    method: Option<String>,
    params: Option<serde_json::Value>,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
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
struct RawTickerData {
    mark_price: Option<f64>,
    index_price: Option<f64>,
    best_bid_price: Option<f64>,
    best_ask_price: Option<f64>,
    last_price: Option<f64>,
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
    Some(DeribitTicker {
        channel: channel.to_string(),
        mark_price: positive_finite(raw.mark_price),
        index_price: positive_finite(raw.index_price),
        best_bid_price: positive_finite(raw.best_bid_price),
        best_ask_price: positive_finite(raw.best_ask_price),
        last_price: positive_finite(raw.last_price),
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
        return Some(DeribitMessage::RpcResult { id: env.id, result });
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
    #[allow(dead_code)]
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
                        let ping_period = self.config.heartbeat_interval.max(Duration::from_secs(10));
                        let ping_start = tokio::time::Instant::now() + ping_period;
                        let mut ping_ticker = tokio::time::interval_at(ping_start, ping_period);
                        ping_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

                        loop {
                            tokio::select! {
                                maybe_msg = read.next() => {
                                    match maybe_msg {
                                        Some(Ok(Message::Text(text))) => {
                                            match self.handle_text(&text, clock.now_ms(), &mut tracker, &mut next_rpc_id).await {
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
                                                match self.handle_text(text, clock.now_ms(), &mut tracker, &mut next_rpc_id).await {
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
                                _ = ping_ticker.tick() => {
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

    async fn handle_text(
        &self,
        text: &str,
        now_ms: u64,
        tracker: &mut HealthTracker,
        next_rpc_id: &mut u64,
    ) -> HandleOutcome {
        let Some(msg) = parse_deribit_message(text) else {
            return HandleOutcome::Continue;
        };

        match msg {
            DeribitMessage::PriceIndex(idx) => {
                tracker.on_sample(&idx.index_name, now_ms);
                publish_health(&self.health_tx, tracker.snapshot());

                if let Some(tx) = self.senders.get(&idx.index_name) {
                    let ts_ms = if idx.timestamp > 0 {
                        idx.timestamp
                    } else {
                        now_ms
                    };
                    if tx.send(PriceTick::new(ts_ms, idx.price)).await.is_err() {
                        warn!(index = %idx.index_name, "Deribit price receiver dropped");
                        if self.all_senders_closed() {
                            return HandleOutcome::Stop;
                        }
                    }
                }
                HandleOutcome::Continue
            }
            DeribitMessage::Ticker(ticker) => {
                debug!(
                    channel = %ticker.channel,
                    mark_price = ?ticker.mark_price,
                    index_price = ?ticker.index_price,
                    bid = ?ticker.best_bid_price,
                    ask = ?ticker.best_ask_price,
                    last = ?ticker.last_price,
                    ts = ticker.timestamp,
                    "Deribit diagnostic ticker"
                );
                HandleOutcome::Continue
            }
            DeribitMessage::Heartbeat { kind } => {
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
            DeribitMessage::RpcResult { id, result } => {
                debug!(?id, %result, "Deribit RPC result");
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

        let ticker_msg = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"mark_price":146.15,"index_price":146.12,"best_bid_price":146.10,"best_ask_price":146.20,"last_price":146.14,"timestamp":1795450000150}}}"#;
        assert_eq!(
            parse_deribit_message(ticker_msg),
            Some(DeribitMessage::Ticker(DeribitTicker {
                channel: "ticker.NVDA_USDC-PERPETUAL.100ms".to_string(),
                mark_price: Some(146.15),
                index_price: Some(146.12),
                best_bid_price: Some(146.10),
                best_ask_price: Some(146.20),
                last_price: Some(146.14),
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

        let sub_res = r#"{"jsonrpc":"2.0","id":2,"result":["deribit_price_index.nvda_usdc","deribit_price_index.tsla_usdc"]}"#;
        match parse_deribit_message(sub_res) {
            Some(DeribitMessage::RpcResult { id: Some(2), result }) => {
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
    async fn live_deribit_handler_emits_index_tick_and_replies_to_heartbeat_test_request() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut senders = HashMap::new();
        senders.insert("nvda_usdc".to_string(), tx);
        let (health_tx, _health_rx) = watch::channel(HashMap::new());
        let stream = DeribitStream {
            config: DeribitConfig::default(),
            senders,
            health_tx,
            prev_day_px: Arc::new(RwLock::new(HashMap::new())),
        };
        let mut tracker = HealthTracker::new(["nvda_usdc"], Duration::from_secs(3));
        let mut rpc_id = 10u64;

        let idx_msg = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"deribit_price_index.nvda_usdc","data":{"index_name":"nvda_usdc","price":146.12,"timestamp":1795450000123}}}"#;
        assert!(matches!(
            stream
                .handle_text(idx_msg, 1_795_450_000_200, &mut tracker, &mut rpc_id)
                .await,
            HandleOutcome::Continue
        ));
        let tick = rx.recv().await.expect("expected index PriceTick");
        assert_eq!(tick, PriceTick::new(1_795_450_000_123, 146.12));

        // Diagnostic ticker does not emit a settlement PriceTick.
        let ticker_msg = r#"{"jsonrpc":"2.0","method":"subscription","params":{"channel":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"mark_price":146.15,"index_price":146.12,"timestamp":1795450000250}}}"#;
        assert!(matches!(
            stream
                .handle_text(ticker_msg, 1_795_450_000_260, &mut tracker, &mut rpc_id)
                .await,
            HandleOutcome::Continue
        ));
        assert!(rx.try_recv().is_err());

        // Heartbeat test_request produces a public/test JSON-RPC reply.
        let hb_msg = r#"{"jsonrpc":"2.0","method":"heartbeat","params":{"type":"test_request"}}"#;
        match stream
            .handle_text(hb_msg, 1_795_450_001_000, &mut tracker, &mut rpc_id)
            .await
        {
            HandleOutcome::SendText(reply) => {
                assert!(reply.contains(r#""method":"public/test""#), "got: {reply}");
                assert_eq!(rpc_id, 11);
            }
            _ => panic!("expected SendText reply for test_request heartbeat"),
        }
    }
}
