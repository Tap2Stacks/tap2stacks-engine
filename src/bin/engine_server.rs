use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use st_engine::logging::init_tracing;
use st_engine::price::{
    FeedHealth, HealthTracker, HyperliquidConfig, PriceTick, read_hl_jsonl,
    spawn_hyperliquid_stream, spawn_price_pipeline,
};
use st_engine::protocol::{EngineIn, EngineOut};
use st_engine::quote::{CellKey, QuoteEngine};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, mpsc, watch};
use tokio::time::{Instant, MissedTickBehavior, interval, interval_at};
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::tungstenite::{Bytes, Message, Utf8Bytes};
use tracing::{debug, error, info, warn};

const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8090";
const DEFAULT_TICKERS: &str = "NVDA=xyz:NVDA:3,TSLA=xyz:TSLA:8";
const STALE_AFTER: Duration = Duration::from_secs(3);
const HL_HEALTH_INTERVAL: Duration = Duration::from_millis(500);
const QUOTE_INTERVAL: Duration = Duration::from_millis(100);
const HEALTH_BROADCAST_INTERVAL: Duration = Duration::from_secs(1);
const REPLAY_WRAP_DELAY: Duration = Duration::from_secs(1);
const MAX_COMPLETED_BARS: usize = 600;
const GRID_LOOKBACK_BARS: usize = 180;
const TICK_CHANNEL_CAPACITY: usize = 4096;
const EXPOSURE_CHANNEL_CAPACITY: usize = 256;
const MAX_QUEUED_SAMPLES_PER_CLIENT: usize = 256;

#[derive(Debug, Clone)]
struct TickerSpec {
    ticker: String,
    coin: String,
    band_cents: u32,
}

#[derive(Clone)]
struct MonotonicWallClock {
    start_wall_ms: u64,
    start_instant: Instant,
}

impl MonotonicWallClock {
    fn new() -> Self {
        let start_wall_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            start_wall_ms,
            start_instant: Instant::now(),
        }
    }

    fn now_ms(&self) -> u64 {
        self.start_wall_ms
            .saturating_add(self.start_instant.elapsed().as_millis() as u64)
    }
}

struct MailboxState {
    closed: bool,
    pongs: VecDeque<Bytes>,
    samples: VecDeque<Utf8Bytes>,
    latest_health: HashMap<String, Utf8Bytes>,
    latest_quotes: HashMap<String, Utf8Bytes>,
}

impl Default for MailboxState {
    fn default() -> Self {
        Self {
            closed: false,
            pongs: VecDeque::new(),
            samples: VecDeque::new(),
            latest_health: HashMap::new(),
            latest_quotes: HashMap::new(),
        }
    }
}

struct ClientMailbox {
    state: Mutex<MailboxState>,
    notify: Notify,
}

impl ClientMailbox {
    fn new() -> Self {
        Self {
            state: Mutex::new(MailboxState::default()),
            notify: Notify::new(),
        }
    }

    fn push_out(&self, msg: &EngineOut, payload: Utf8Bytes) {
        let mut state = self.state.lock().expect("client mailbox mutex poisoned");
        if state.closed {
            return;
        }
        match msg {
            EngineOut::Quote(q) => {
                state.latest_quotes.insert(q.ticker.clone(), payload);
            }
            EngineOut::Health { ticker, .. } => {
                state.latest_health.insert(ticker.clone(), payload);
            }
            EngineOut::Sample { .. } => {
                if state.samples.len() >= MAX_QUEUED_SAMPLES_PER_CLIENT {
                    state.samples.pop_front();
                }
                state.samples.push_back(payload);
            }
        }
        drop(state);
        self.notify.notify_one();
    }

    fn push_pong(&self, payload: Bytes) {
        let mut state = self.state.lock().expect("client mailbox mutex poisoned");
        if state.closed {
            return;
        }
        state.pongs.push_back(payload);
        drop(state);
        self.notify.notify_one();
    }

    fn close(&self) {
        let mut state = self.state.lock().expect("client mailbox mutex poisoned");
        state.closed = true;
        drop(state);
        self.notify.notify_one();
    }

    fn drain(&self) -> (bool, Vec<Bytes>, Vec<Utf8Bytes>) {
        let mut state = self.state.lock().expect("client mailbox mutex poisoned");
        let closed = state.closed;
        let pongs: Vec<Bytes> = state.pongs.drain(..).collect();
        let mut frames = Vec::with_capacity(
            state.samples.len() + state.latest_health.len() + state.latest_quotes.len(),
        );
        frames.extend(state.samples.drain(..));
        for (_, frame) in state.latest_health.drain() {
            frames.push(frame);
        }
        for (_, frame) in state.latest_quotes.drain() {
            frames.push(frame);
        }
        (closed, pongs, frames)
    }
}

struct ClientHub {
    next_id: AtomicU64,
    clients: Mutex<HashMap<u64, Arc<ClientMailbox>>>,
}

impl ClientHub {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            clients: Mutex::new(HashMap::new()),
        }
    }

    fn register(self: &Arc<Self>) -> ClientRegistration {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mailbox = Arc::new(ClientMailbox::new());
        self.clients
            .lock()
            .expect("client hub mutex poisoned")
            .insert(id, Arc::clone(&mailbox));
        ClientRegistration {
            id,
            hub: Arc::clone(self),
            mailbox,
        }
    }

    fn publish(&self, msg: &EngineOut) {
        let json = match serde_json::to_string(msg) {
            Ok(json) => json,
            Err(err) => {
                warn!(error = %err, "failed to serialize outbound message");
                return;
            }
        };
        let payload: Utf8Bytes = json.into();
        let mailboxes: Vec<Arc<ClientMailbox>> = {
            let clients = self.clients.lock().expect("client hub mutex poisoned");
            clients.values().cloned().collect()
        };
        for mailbox in mailboxes {
            mailbox.push_out(msg, payload.clone());
        }
    }
}

struct ClientRegistration {
    id: u64,
    hub: Arc<ClientHub>,
    mailbox: Arc<ClientMailbox>,
}

impl Drop for ClientRegistration {
    fn drop(&mut self) {
        self.mailbox.close();
        self.hub
            .clients
            .lock()
            .expect("client hub mutex poisoned")
            .remove(&self.id);
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("engine_server fatal error: {err}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    init_tracing("info,st_engine=info");

    let expected_token = match env::var("ENGINE_TOKEN") {
        Ok(val) if !val.trim().is_empty() => val,
        _ => {
            return Err("ENGINE_TOKEN environment variable is required and must not be empty".into());
        }
    };
    let expected_token: Arc<str> = Arc::from(expected_token);

    let bind_addr = env::var("ENGINE_BIND_ADDR")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_BIND_ADDR.to_string());

    let tickers_raw = env::var("ENGINE_TICKERS")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_TICKERS.to_string());
    let tickers = parse_tickers(&tickers_raw)?;

    let replay_file = env::var("ENGINE_REPLAY_FILE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());

    let hl_ws_url = env::var("HL_WS_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());

    let clock = Arc::new(MonotonicWallClock::new());
    let hub = Arc::new(ClientHub::new());

    let mut tick_senders: HashMap<String, mpsc::Sender<PriceTick>> =
        HashMap::with_capacity(tickers.len());
    let mut tick_receivers: HashMap<String, mpsc::Receiver<PriceTick>> =
        HashMap::with_capacity(tickers.len());
    let mut exposure_senders: HashMap<String, mpsc::Sender<Vec<(CellKey, f64)>>> =
        HashMap::with_capacity(tickers.len());

    for spec in &tickers {
        let (tick_tx, tick_rx) = mpsc::channel(TICK_CHANNEL_CAPACITY);
        tick_senders.insert(spec.coin.clone(), tick_tx);
        tick_receivers.insert(spec.coin.clone(), tick_rx);
    }

    let health_rx = if let Some(ref replay_path) = replay_file {
        info!(replay_path = %replay_path, "starting engine_server in replay mode");
        spawn_replay_stream(
            Path::new(replay_path),
            &tickers,
            tick_senders,
            Arc::clone(&clock),
        )?
    } else {
        let mut hl_config = HyperliquidConfig {
            coins: tickers.iter().map(|t| t.coin.clone()).collect(),
            stale_after: STALE_AFTER,
            health_interval: HL_HEALTH_INTERVAL,
            ..HyperliquidConfig::default()
        };
        if let Some(url) = hl_ws_url {
            hl_config.ws_url = url;
        }
        info!(ws_url = %hl_config.ws_url, "starting engine_server with Hyperliquid stream");
        let handle = spawn_hyperliquid_stream(hl_config, tick_senders);
        handle.health_rx
    };

    for spec in tickers {
        let tick_rx = tick_receivers
            .remove(&spec.coin)
            .expect("missing tick receiver for configured coin");
        let (exposure_tx, exposure_rx) = mpsc::channel(EXPOSURE_CHANNEL_CAPACITY);
        exposure_senders.insert(spec.ticker.clone(), exposure_tx);

        let hub_clone = Arc::clone(&hub);
        let clock_clone = Arc::clone(&clock);
        let health_rx_clone = health_rx.clone();
        tokio::spawn(run_ticker_task(
            spec,
            tick_rx,
            exposure_rx,
            health_rx_clone,
            hub_clone,
            clock_clone,
        ));
    }

    let exposure_senders = Arc::new(exposure_senders);
    let listener = TcpListener::bind(&bind_addr).await?;
    let local_addr = listener.local_addr()?;
    info!(bind_addr = %local_addr, "engine_server listening");

    loop {
        let (stream, remote_addr) = listener.accept().await?;
        let token_clone = Arc::clone(&expected_token);
        let hub_clone = Arc::clone(&hub);
        let exposure_clone = Arc::clone(&exposure_senders);
        tokio::spawn(async move {
            if let Err(err) =
                serve_client(stream, &token_clone, hub_clone, exposure_clone).await
            {
                debug!(%remote_addr, error = %err, "client connection closed");
            }
        });
    }
}

async fn run_ticker_task(
    spec: TickerSpec,
    mut raw_tick_rx: mpsc::Receiver<PriceTick>,
    mut exposure_rx: mpsc::Receiver<Vec<(CellKey, f64)>>,
    mut health_rx: watch::Receiver<HashMap<String, FeedHealth>>,
    hub: Arc<ClientHub>,
    clock: Arc<MonotonicWallClock>,
) {
    let (price_runtime, _price_worker_handle) = spawn_price_pipeline(
        MAX_COMPLETED_BARS,
        GRID_LOOKBACK_BARS,
        TICK_CHANNEL_CAPACITY,
    );
    let mut grid_rx = price_runtime.grid_rx;
    let mut quote_engine = QuoteEngine::new(spec.ticker.clone(), spec.band_cents);

    let stale_after_ms = STALE_AFTER.as_millis() as u64;
    let mut quote_tick = interval_at(Instant::now() + QUOTE_INTERVAL, QUOTE_INTERVAL);
    quote_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut health_tick = interval(HEALTH_BROADCAST_INTERVAL);
    health_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut last_sample_ms: Option<u64> = None;
    let mut last_sent_ts_ms: u64 = 0;
    let mut pending_tick: Option<PriceTick> = None;
    let mut feed_health = health_rx
        .borrow()
        .get(&spec.coin)
        .copied()
        .unwrap_or_default();

    loop {
        tokio::select! {
            maybe_tick = raw_tick_rx.recv() => {
                let Some(tick) = maybe_tick else {
                    break;
                };
                if !tick.price.is_finite() || tick.price <= 0.0 {
                    continue;
                }

                hub.publish(&EngineOut::Sample {
                    ticker: spec.ticker.clone(),
                    ts_ms: tick.ts_ms,
                    price: tick.price,
                });

                let pipeline_ts = tick.ts_ms.max(last_sent_ts_ms);
                last_sent_ts_ms = pipeline_ts;
                last_sample_ms = Some(match last_sample_ms {
                    Some(prev) => prev.max(tick.ts_ms),
                    None => tick.ts_ms,
                });
                let pipeline_tick = PriceTick {
                    ts_ms: pipeline_ts,
                    price: tick.price,
                };
                pending_tick = Some(tick);

                if price_runtime.tick_tx.send(pipeline_tick).await.is_err() {
                    error!(ticker = %spec.ticker, "price pipeline closed unexpectedly");
                    break;
                }
            }
            changed = grid_rx.changed() => {
                if changed.is_err() {
                    break;
                }
                let maybe_snapshot = grid_rx.borrow_and_update().clone();
                if let (Some(tick), Some(snapshot)) = (pending_tick.take(), maybe_snapshot) {
                    quote_engine.on_sample(tick, &snapshot.history);
                }
            }
            maybe_exposure = exposure_rx.recv() => {
                let Some(cells) = maybe_exposure else {
                    break;
                };
                quote_engine.set_exposure(cells);
            }
            changed = health_rx.changed() => {
                if changed.is_err() {
                    break;
                }
                feed_health = health_rx
                    .borrow_and_update()
                    .get(&spec.coin)
                    .copied()
                    .unwrap_or_default();
            }
            _ = quote_tick.tick() => {
                let server_ts_ms = clock.now_ms().max(last_sample_ms.unwrap_or(0));
                let stale = is_ticker_stale(feed_health, last_sample_ms, server_ts_ms, stale_after_ms);
                let mut quote = quote_engine.price(server_ts_ms);
                if stale {
                    quote.cells.clear();
                }
                hub.publish(&EngineOut::Quote(quote));
            }
            _ = health_tick.tick() => {
                let server_ts_ms = clock.now_ms().max(last_sample_ms.unwrap_or(0));
                let stale = is_ticker_stale(feed_health, last_sample_ms, server_ts_ms, stale_after_ms);
                let last_sample_age_ms = last_sample_ms.map(|ts| server_ts_ms.saturating_sub(ts));
                hub.publish(&EngineOut::Health {
                    ticker: spec.ticker.clone(),
                    server_ts_ms,
                    last_sample_age_ms,
                    stale,
                });
            }
        }
    }
}

fn is_ticker_stale(
    feed_health: FeedHealth,
    last_sample_ms: Option<u64>,
    server_ts_ms: u64,
    stale_after_ms: u64,
) -> bool {
    feed_health.stale
        || last_sample_ms.is_some_and(|ts| server_ts_ms.saturating_sub(ts) > stale_after_ms)
}

fn spawn_replay_stream(
    path: &Path,
    tickers: &[TickerSpec],
    senders: HashMap<String, mpsc::Sender<PriceTick>>,
    clock: Arc<MonotonicWallClock>,
) -> Result<watch::Receiver<HashMap<String, FeedHealth>>, Box<dyn Error + Send + Sync>> {
    let coins: Vec<String> = tickers.iter().map(|t| t.coin.clone()).collect();
    let tracker = Arc::new(Mutex::new(HealthTracker::new(coins, STALE_AFTER)));
    let initial_health = tracker
        .lock()
        .expect("replay health mutex poisoned")
        .snapshot();
    let (health_tx, health_rx) = watch::channel(initial_health);

    let mut per_coin_ticks: Vec<(String, Vec<PriceTick>)> = Vec::with_capacity(tickers.len());
    for spec in tickers {
        let ticks = read_hl_jsonl(path, &spec.coin)?;
        if ticks.is_empty() {
            warn!(coin = %spec.coin, path = %path.display(), "no replay ticks found for coin");
        }
        per_coin_ticks.push((spec.coin.clone(), ticks));
    }

    let tracker_tick = Arc::clone(&tracker);
    let health_tx_tick = health_tx.clone();
    let clock_tick = Arc::clone(&clock);
    tokio::spawn(async move {
        let mut ticker = interval(HL_HEALTH_INTERVAL);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            let snapshot = {
                let mut guard = tracker_tick.lock().expect("replay health mutex poisoned");
                guard.tick(clock_tick.now_ms())
            };
            publish_health(&health_tx_tick, snapshot);
        }
    });

    for (coin, ticks) in per_coin_ticks {
        if ticks.is_empty() {
            continue;
        }
        let Some(tx) = senders.get(&coin).cloned() else {
            continue;
        };
        let tracker_coin = Arc::clone(&tracker);
        let health_tx_coin = health_tx.clone();
        let clock_coin = Arc::clone(&clock);

        tokio::spawn(async move {
            loop {
                let mut prev_rx = ticks[0].ts_ms;
                if !emit_replay_tick(
                    &coin,
                    ticks[0].price,
                    &tx,
                    &tracker_coin,
                    &health_tx_coin,
                    &clock_coin,
                )
                .await
                {
                    return;
                }

                for tick in &ticks[1..] {
                    let delay_ms = tick.ts_ms.saturating_sub(prev_rx);
                    prev_rx = tick.ts_ms;
                    if delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                    if !emit_replay_tick(
                        &coin,
                        tick.price,
                        &tx,
                        &tracker_coin,
                        &health_tx_coin,
                        &clock_coin,
                    )
                    .await
                    {
                        return;
                    }
                }

                tokio::time::sleep(REPLAY_WRAP_DELAY).await;
            }
        });
    }

    Ok(health_rx)
}

async fn emit_replay_tick(
    coin: &str,
    price: f64,
    tx: &mpsc::Sender<PriceTick>,
    tracker: &Mutex<HealthTracker>,
    health_tx: &watch::Sender<HashMap<String, FeedHealth>>,
    clock: &MonotonicWallClock,
) -> bool {
    let now_ms = clock.now_ms();
    let snapshot = {
        let mut guard = tracker.lock().expect("replay health mutex poisoned");
        guard.on_sample(coin, now_ms);
        guard.snapshot()
    };
    publish_health(health_tx, snapshot);
    tx.send(PriceTick { ts_ms: now_ms, price }).await.is_ok()
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

async fn serve_client(
    stream: TcpStream,
    expected_token: &str,
    hub: Arc<ClientHub>,
    exposure_senders: Arc<HashMap<String, mpsc::Sender<Vec<(CellKey, f64)>>>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let websocket = accept_hdr_async(stream, |req: &Request, res: Response| {
        let provided = req
            .headers()
            .get("x-engine-token")
            .and_then(|val| val.to_str().ok());
        let authorized = provided.is_some_and(|val| constant_time_eq(val.as_bytes(), expected_token.as_bytes()));
        if authorized {
            Ok(res)
        } else {
            let mut resp = ErrorResponse::new(Some("Unauthorized".to_string()));
            *resp.status_mut() = StatusCode::UNAUTHORIZED;
            Err(resp)
        }
    })
    .await?;

    let registration = hub.register();
    let mailbox_writer = Arc::clone(&registration.mailbox);
    let mailbox_reader = Arc::clone(&registration.mailbox);

    let (mut writer, mut reader) = websocket.split();

    let write_task = async move {
        loop {
            let (closed, pongs, frames) = mailbox_writer.drain();
            for pong in pongs {
                writer.send(Message::Pong(pong)).await?;
            }
            for frame in frames {
                writer.send(Message::Text(frame)).await?;
            }
            if closed {
                break;
            }
            mailbox_writer.notify.notified().await;
        }
        Ok::<(), Box<dyn Error + Send + Sync>>(())
    };

    let read_task = async move {
        while let Some(incoming) = reader.next().await {
            match incoming? {
                Message::Text(text) => {
                    handle_client_text(&text, &exposure_senders).await;
                }
                Message::Ping(payload) => {
                    mailbox_reader.push_pong(payload);
                }
                Message::Close(_) => break,
                Message::Binary(_) | Message::Pong(_) | Message::Frame(_) => {}
            }
        }
        mailbox_reader.close();
        Ok::<(), Box<dyn Error + Send + Sync>>(())
    };

    let result = tokio::select! {
        res = write_task => res,
        res = read_task => res,
    };

    drop(registration);
    result
}

async fn handle_client_text(
    text: &str,
    exposure_senders: &HashMap<String, mpsc::Sender<Vec<(CellKey, f64)>>>,
) {
    let Ok(msg) = serde_json::from_str::<EngineIn>(text) else {
        warn!("ignoring invalid inbound engine message");
        return;
    };
    match msg {
        EngineIn::Exposure { ticker, cells } => {
            let Some(tx) = exposure_senders.get(&ticker) else {
                warn!(%ticker, "ignoring exposure update for unknown ticker");
                return;
            };
            let converted: Vec<(CellKey, f64)> = cells
                .into_iter()
                .map(|c| {
                    (
                        CellKey {
                            window_start_ms: c.window_start_ms,
                            lower_cents: c.lower_cents,
                        },
                        c.liability,
                    )
                })
                .collect();
            if let Err(err) = tx.send(converted).await {
                warn!(%ticker, error = %err, "failed to deliver exposure update");
            }
        }
    }
}

fn parse_tickers(raw: &str) -> Result<Vec<TickerSpec>, Box<dyn Error + Send + Sync>> {
    let mut specs = Vec::new();
    let mut seen_tickers = HashSet::new();
    let mut seen_coins = HashSet::new();

    for entry in raw.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (ticker_part, rest) = trimmed
            .split_once('=')
            .ok_or_else(|| format!("invalid ENGINE_TICKERS entry '{trimmed}': missing '='"))?;
        let ticker = ticker_part.trim();
        let (coin_part, band_part) = rest
            .rsplit_once(':')
            .ok_or_else(|| format!("invalid ENGINE_TICKERS entry '{trimmed}': missing ':'"))?;
        let coin = coin_part.trim();
        let band_cents: u32 = band_part
            .trim()
            .parse()
            .map_err(|_| format!("invalid bandCents in ENGINE_TICKERS entry '{trimmed}'"))?;

        if ticker.is_empty() || coin.is_empty() || band_cents == 0 {
            return Err(format!("invalid ENGINE_TICKERS entry '{trimmed}'").into());
        }
        if !seen_tickers.insert(ticker.to_string()) {
            return Err(format!("duplicate ticker '{ticker}' in ENGINE_TICKERS").into());
        }
        if !seen_coins.insert(coin.to_string()) {
            return Err(format!("duplicate coin '{coin}' in ENGINE_TICKERS").into());
        }

        specs.push(TickerSpec {
            ticker: ticker.to_string(),
            coin: coin.to_string(),
            band_cents,
        });
    }

    if specs.is_empty() {
        return Err("ENGINE_TICKERS must configure at least one ticker".into());
    }
    Ok(specs)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_tickers_with_colon_in_coin() {
        let parsed = parse_tickers(DEFAULT_TICKERS).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].ticker, "NVDA");
        assert_eq!(parsed[0].coin, "xyz:NVDA");
        assert_eq!(parsed[0].band_cents, 3);
        assert_eq!(parsed[1].ticker, "TSLA");
        assert_eq!(parsed[1].coin, "xyz:TSLA");
        assert_eq!(parsed[1].band_cents, 8);
    }

    #[test]
    fn rejects_invalid_ticker_specs() {
        assert!(parse_tickers("").is_err());
        assert!(parse_tickers("NVDA:xyz:NVDA:3").is_err());
        assert!(parse_tickers("NVDA=xyz:NVDA:0").is_err());
        assert!(parse_tickers("NVDA=xyz:NVDA:abc").is_err());
        assert!(parse_tickers("NVDA=xyz:NVDA:3,NVDA=xyz:TSLA:8").is_err());
        assert!(parse_tickers("NVDA=xyz:NVDA:3,NVDA2=xyz:NVDA:8").is_err());
    }

    #[test]
    fn constant_time_eq_checks_exact_match() {
        assert!(constant_time_eq(b"secret-123", b"secret-123"));
        assert!(!constant_time_eq(b"secret-123", b"secret-124"));
        assert!(!constant_time_eq(b"secret-123", b"secret-12"));
    }
}
