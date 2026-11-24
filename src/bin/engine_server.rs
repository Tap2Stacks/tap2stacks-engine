use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use st_engine::grid::{OhlcBar, stock_grid_config_for_ticker};
use st_engine::horizon::{
    DEFAULT_TRIGGER_SIGMA, DEFAULT_WINDOWS, FAR_REFRESH_MS, stagger_phase_ms,
};
use st_engine::logging::init_tracing;
use st_engine::price::{
    FeedHealth, HealthTracker, HyperliquidConfig, PriceTick, fetch_candle_warmup, read_hl_jsonl,
    spawn_hyperliquid_stream, spawn_price_pipeline,
};
use st_engine::protocol::{EngineIn, EngineOut};
use st_engine::quote::{CellKey, QuoteEngine, SampleResult};
use st_engine::v2::PricingModel;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, mpsc, watch};
use tokio::time::{Instant, MissedTickBehavior, interval, interval_at};
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::tungstenite::{Bytes, Message, Utf8Bytes};
use tracing::{debug, error, info, warn};

const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8090";
const DEFAULT_TICKERS: &str = "NVDA=xyz:NVDA:3,TSLA=xyz:TSLA:8,AAPL=xyz:AAPL:8,SP500=xyz:SP500:150";
const DEFAULT_HL_INFO_URL: &str = "https://api.hyperliquid.xyz/info";
const DEFAULT_HL_WARMUP_MINUTES: u64 = 180;
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

    let hl_info_url = env::var("HL_INFO_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_HL_INFO_URL.to_string());
    let warmup_minutes = warmup_minutes_from_env();
    let quote_windows = quote_windows_from_env()?;
    let quote_windows_by_ticker = quote_windows_by_ticker_from_env(&tickers)?;
    let far_trigger_sigma = far_trigger_sigma_from_env();
    let pricing_model = pricing_model_from_env()?;
    let max_multiplier = max_multiplier_from_env()?;

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

    let mut seed_variance_priors: HashMap<String, f64> = HashMap::new();
    let (health_rx, prev_day_px) = if let Some(ref replay_path) = replay_file {
        info!(replay_path = %replay_path, "starting engine_server in replay mode");
        let health_rx = spawn_replay_stream(
            Path::new(replay_path),
            &tickers,
            tick_senders,
            Arc::clone(&clock),
        )?;
        (health_rx, Arc::new(RwLock::new(HashMap::new())))
    } else {
        if warmup_minutes > 0 {
            seed_variance_priors =
                fetch_warmup_priors(&tickers, &hl_info_url, warmup_minutes).await;
        }
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
        (handle.health_rx, handle.prev_day_px)
    };

    let ticker_count = tickers.len();
    for (ticker_index, spec) in tickers.into_iter().enumerate() {
        let tick_rx = tick_receivers
            .remove(&spec.coin)
            .expect("missing tick receiver for configured coin");
        let (exposure_tx, exposure_rx) = mpsc::channel(EXPOSURE_CHANNEL_CAPACITY);
        exposure_senders.insert(spec.ticker.clone(), exposure_tx);

        let hub_clone = Arc::clone(&hub);
        let clock_clone = Arc::clone(&clock);
        let health_rx_clone = health_rx.clone();
        let prev_day_px_clone = Arc::clone(&prev_day_px);
        let seed_variance_prior = seed_variance_priors.get(&spec.coin).copied();
        let ticker_quote_windows =
            resolve_quote_windows(&spec.ticker, quote_windows, &quote_windows_by_ticker);
        let far_phase_ms = stagger_phase_ms(ticker_index, ticker_count, FAR_REFRESH_MS);
        info!(
            ticker = %spec.ticker,
            coin = %spec.coin,
            band_cents = spec.band_cents,
            quote_windows = ticker_quote_windows,
            pricing_model = pricing_model.as_str(),
            max_multiplier = ?max_multiplier,
            far_trigger_sigma,
            far_phase_ms,
            "spawning ticker task"
        );
        tokio::spawn(run_ticker_task(
            spec,
            TickerTaskInput {
                raw_tick_rx: tick_rx,
                exposure_rx,
                health_rx: health_rx_clone,
                prev_day_px: prev_day_px_clone,
                hub: hub_clone,
                clock: clock_clone,
                seed_variance_prior,
                quote_windows: ticker_quote_windows,
                far_trigger_sigma,
                far_phase_ms,
                pricing_model,
                max_multiplier,
            },
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

async fn fetch_warmup_priors(
    tickers: &[TickerSpec],
    info_url: &str,
    warmup_minutes: u64,
) -> HashMap<String, f64> {
    let mut priors = HashMap::new();
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            warn!(error = %err, "hyperliquid warmup client unavailable; using static volatility priors");
            return priors;
        }
    };
    let end_time_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    for spec in tickers {
        match fetch_candle_warmup(&client, info_url, &spec.coin, warmup_minutes, end_time_ms).await
        {
            Ok(estimate) => {
                info!(
                    ticker = %spec.ticker,
                    coin = %spec.coin,
                    candle_count = estimate.candle_count,
                    last_close = estimate.last_close,
                    sigma_per_second = estimate.sigma_per_second,
                    variance_per_second = estimate.variance_per_second,
                    "hyperliquid warmup volatility prior"
                );
                priors.insert(spec.coin.clone(), estimate.variance_per_second);
            }
            Err(err) => {
                warn!(
                    ticker = %spec.ticker,
                    coin = %spec.coin,
                    error = %err,
                    "hyperliquid warmup unavailable; using static volatility prior"
                );
            }
        }
    }
    priors
}

struct TickerTaskInput {
    raw_tick_rx: mpsc::Receiver<PriceTick>,
    exposure_rx: mpsc::Receiver<Vec<(CellKey, f64)>>,
    health_rx: watch::Receiver<HashMap<String, FeedHealth>>,
    prev_day_px: Arc<RwLock<HashMap<String, f64>>>,
    hub: Arc<ClientHub>,
    clock: Arc<MonotonicWallClock>,
    seed_variance_prior: Option<f64>,
    quote_windows: usize,
    far_trigger_sigma: f64,
    far_phase_ms: u64,
    pricing_model: PricingModel,
    max_multiplier: Option<f64>,
}

async fn run_ticker_task(spec: TickerSpec, input: TickerTaskInput) {
    let TickerTaskInput {
        mut raw_tick_rx,
        mut exposure_rx,
        mut health_rx,
        prev_day_px,
        hub,
        clock,
        seed_variance_prior,
        quote_windows,
        far_trigger_sigma,
        far_phase_ms,
        pricing_model,
        max_multiplier,
    } = input;
    let (price_runtime, _price_worker_handle) = spawn_price_pipeline(
        MAX_COMPLETED_BARS,
        GRID_LOOKBACK_BARS,
        TICK_CHANNEL_CAPACITY,
    );
    let mut grid_rx = price_runtime.grid_rx;
    let mut quote_engine = if quote_windows == DEFAULT_WINDOWS
        && pricing_model == PricingModel::V1
        && max_multiplier.is_none()
    {
        QuoteEngine::new(spec.ticker.clone(), spec.band_cents)
    } else {
        let mut cfg = stock_grid_config_for_ticker(&spec.ticker, spec.band_cents, Vec::new());
        cfg.simulation.quote_window_count = quote_windows;
        cfg.simulation.far_trigger_sigma = far_trigger_sigma;
        cfg.simulation.far_phase_ms = far_phase_ms;
        cfg.simulation.pricing_model = pricing_model;
        if let Some(cap) = max_multiplier {
            cfg.pricing.max_multiplier = cap;
        }
        QuoteEngine::with_config(spec.ticker.clone(), spec.band_cents, cfg)
    };
    quote_engine.set_seed_variance_prior(seed_variance_prior);

    let (mc_done_tx, mut mc_done_rx) = mpsc::channel::<Option<SampleResult>>(1);
    let mut mc_busy = false;
    let mut queued_mc: Option<(PriceTick, Vec<OhlcBar>)> = None;

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
                quote_engine.observe_tick(tick);

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
                    mid_px: tick.mid_px,
                    oracle_px: tick.oracle_px,
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
                    if mc_busy {
                        queued_mc = Some((tick, snapshot.history));
                    } else if let Some(job) = quote_engine.prepare_sample_job(tick, snapshot.history) {
                        mc_busy = true;
                        let tx = mc_done_tx.clone();
                        tokio::task::spawn_blocking(move || {
                            let _ = tx.blocking_send(job.run());
                        });
                    }
                }
            }
            maybe_mc = mc_done_rx.recv() => {
                mc_busy = false;
                if let Some(Some(result)) = maybe_mc {
                    quote_engine.apply_sample_result(result);
                }
                if let Some((tick, history)) = queued_mc.take() {
                    if let Some(job) = quote_engine.prepare_sample_job(tick, history) {
                        mc_busy = true;
                        let tx = mc_done_tx.clone();
                        tokio::task::spawn_blocking(move || {
                            let _ = tx.blocking_send(job.run());
                        });
                    }
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
                let latest_prev_day_px = prev_day_px
                    .read()
                    .expect("prev_day_px rwlock poisoned")
                    .get(&spec.coin)
                    .copied();
                hub.publish(&EngineOut::Health {
                    ticker: spec.ticker.clone(),
                    server_ts_ms,
                    last_sample_age_ms,
                    stale,
                    prev_day_px: latest_prev_day_px,
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
                    ticks[0],
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
                        *tick,
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
    tick: PriceTick,
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
    tx.send(PriceTick {
        ts_ms: now_ms,
        price: tick.price,
        mid_px: tick.mid_px,
        oracle_px: tick.oracle_px,
    })
    .await
    .is_ok()
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

fn warmup_minutes_from_env() -> u64 {
    match env::var("HL_WARMUP_MINUTES") {
        Ok(raw) if !raw.trim().is_empty() => match raw.trim().parse::<u64>() {
            Ok(value) => value,
            Err(_) => {
                warn!(value = %raw, "invalid HL_WARMUP_MINUTES; using default");
                DEFAULT_HL_WARMUP_MINUTES
            }
        },
        _ => DEFAULT_HL_WARMUP_MINUTES,
    }
}

fn parse_quote_windows(raw: &str) -> Option<usize> {
    raw.trim().parse::<usize>().ok().filter(|&n| n >= 1)
}

fn quote_windows_from_env() -> Result<usize, Box<dyn Error + Send + Sync>> {
    match env::var("ENGINE_QUOTE_WINDOWS") {
        Ok(raw) if !raw.trim().is_empty() => Ok(parse_quote_windows(&raw).ok_or_else(|| {
            format!("invalid ENGINE_QUOTE_WINDOWS '{raw}': expected integer >= 1")
        })?),
        Ok(_) | Err(env::VarError::NotPresent) => Ok(DEFAULT_WINDOWS),
        Err(env::VarError::NotUnicode(_)) => {
            Err("ENGINE_QUOTE_WINDOWS must be valid UTF-8".into())
        }
    }
}

fn parse_quote_windows_by_ticker(
    raw: &str,
    tickers: &[TickerSpec],
) -> Result<HashMap<String, usize>, Box<dyn Error + Send + Sync>> {
    let mut overrides: HashMap<String, usize> = HashMap::new();
    if raw.trim().is_empty() {
        return Ok(overrides);
    }
    let known: HashSet<&str> = tickers.iter().map(|spec| spec.ticker.as_str()).collect();
    let configured_list = || -> String {
        tickers
            .iter()
            .map(|spec| spec.ticker.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };

    for entry in raw.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            return Err(
                format!("invalid ENGINE_QUOTE_WINDOWS_BY_TICKER '{raw}': empty entry").into(),
            );
        }
        let (ticker_part, windows_part) = trimmed.split_once('=').ok_or_else(|| {
            format!(
                "invalid ENGINE_QUOTE_WINDOWS_BY_TICKER entry '{trimmed}': expected TICKER=WINDOWS"
            )
        })?;
        let ticker = ticker_part.trim();
        let windows_raw = windows_part.trim();
        if ticker.is_empty() {
            return Err(format!(
                "invalid ENGINE_QUOTE_WINDOWS_BY_TICKER entry '{trimmed}': empty ticker"
            )
            .into());
        }
        if !known.contains(ticker) {
            return Err(format!(
                "unknown ticker '{ticker}' in ENGINE_QUOTE_WINDOWS_BY_TICKER (configured tickers: {})",
                configured_list()
            )
            .into());
        }
        let windows = parse_quote_windows(windows_raw).ok_or_else(|| {
            format!(
                "invalid window count '{windows_raw}' for ticker '{ticker}' in ENGINE_QUOTE_WINDOWS_BY_TICKER: expected integer >= 1"
            )
        })?;
        if overrides.insert(ticker.to_string(), windows).is_some() {
            return Err(format!(
                "duplicate ticker '{ticker}' in ENGINE_QUOTE_WINDOWS_BY_TICKER"
            )
            .into());
        }
    }

    Ok(overrides)
}

fn quote_windows_by_ticker_from_env(
    tickers: &[TickerSpec],
) -> Result<HashMap<String, usize>, Box<dyn Error + Send + Sync>> {
    match env::var("ENGINE_QUOTE_WINDOWS_BY_TICKER") {
        Ok(raw) => parse_quote_windows_by_ticker(&raw, tickers),
        Err(env::VarError::NotPresent) => Ok(HashMap::new()),
        Err(env::VarError::NotUnicode(_)) => {
            Err("ENGINE_QUOTE_WINDOWS_BY_TICKER must be valid UTF-8".into())
        }
    }
}

fn resolve_quote_windows(
    ticker: &str,
    default_windows: usize,
    overrides: &HashMap<String, usize>,
) -> usize {
    overrides.get(ticker).copied().unwrap_or(default_windows)
}

fn parse_trigger_sigma(raw: &str) -> Option<f64> {
    raw.trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn far_trigger_sigma_from_env() -> f64 {
    match env::var("ENGINE_FAR_TRIGGER_SIGMA") {
        Ok(raw) if !raw.trim().is_empty() => match parse_trigger_sigma(&raw) {
            Some(value) => value,
            None => {
                warn!(value = %raw, "invalid ENGINE_FAR_TRIGGER_SIGMA; using default");
                DEFAULT_TRIGGER_SIGMA
            }
        },
        _ => DEFAULT_TRIGGER_SIGMA,
    }
}

fn pricing_model_from_env() -> Result<PricingModel, Box<dyn Error + Send + Sync>> {
    match env::var("ENGINE_PRICING_MODEL") {
        Ok(raw) if !raw.trim().is_empty() => Ok(PricingModel::parse(&raw).ok_or_else(|| {
            format!("invalid ENGINE_PRICING_MODEL '{raw}': expected v1, v2, or m4")
        })?),
        Ok(_) | Err(env::VarError::NotPresent) => Ok(PricingModel::V1),
        Err(env::VarError::NotUnicode(_)) => {
            Err("ENGINE_PRICING_MODEL must be valid UTF-8".into())
        }
    }
}

fn parse_max_multiplier(raw: &str) -> Option<f64> {
    raw.trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value >= 1.05)
}

fn max_multiplier_from_env() -> Result<Option<f64>, Box<dyn Error + Send + Sync>> {
    match env::var("ENGINE_MAX_MULTIPLIER") {
        Ok(raw) if !raw.trim().is_empty() => Ok(Some(parse_max_multiplier(&raw).ok_or_else(|| {
            format!("invalid ENGINE_MAX_MULTIPLIER '{raw}': expected number >= 1.05")
        })?)),
        Ok(_) | Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => {
            Err("ENGINE_MAX_MULTIPLIER must be valid UTF-8".into())
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
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[0].ticker, "NVDA");
        assert_eq!(parsed[0].coin, "xyz:NVDA");
        assert_eq!(parsed[0].band_cents, 3);
        assert_eq!(parsed[1].ticker, "TSLA");
        assert_eq!(parsed[1].coin, "xyz:TSLA");
        assert_eq!(parsed[1].band_cents, 8);
        assert_eq!(parsed[2].ticker, "AAPL");
        assert_eq!(parsed[2].coin, "xyz:AAPL");
        assert_eq!(parsed[2].band_cents, 8);
        assert_eq!(parsed[3].ticker, "SP500");
        assert_eq!(parsed[3].coin, "xyz:SP500");
        assert_eq!(parsed[3].band_cents, 150);
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

    #[test]
    fn quote_windows_parser_defaults_to_six_and_accepts_opt_in_24() {
        assert_eq!(DEFAULT_WINDOWS, 6);
        assert_eq!(parse_quote_windows("24"), Some(24));
        assert_eq!(parse_quote_windows(" 6 "), Some(6));
        assert_eq!(parse_quote_windows("0"), None);
        assert_eq!(parse_quote_windows("-1"), None);
        assert_eq!(parse_quote_windows("abc"), None);
    }

    #[test]
    fn quote_windows_by_ticker_parses_overrides_and_falls_back_to_base() {
        let tickers = parse_tickers(DEFAULT_TICKERS).unwrap();
        let empty = parse_quote_windows_by_ticker("   ", &tickers).unwrap();
        assert!(empty.is_empty());
        assert_eq!(resolve_quote_windows("NVDA", DEFAULT_WINDOWS, &empty), 6);

        let single = parse_quote_windows_by_ticker("NVDA=24", &tickers).unwrap();
        assert_eq!(resolve_quote_windows("NVDA", DEFAULT_WINDOWS, &single), 24);
        assert_eq!(resolve_quote_windows("TSLA", DEFAULT_WINDOWS, &single), 6);
        assert_eq!(resolve_quote_windows("AAPL", DEFAULT_WINDOWS, &single), 6);
        assert_eq!(resolve_quote_windows("SP500", DEFAULT_WINDOWS, &single), 6);

        let multi = parse_quote_windows_by_ticker(" NVDA = 24 , TSLA = 6 ", &tickers).unwrap();
        assert_eq!(resolve_quote_windows("NVDA", 12, &multi), 24);
        assert_eq!(resolve_quote_windows("TSLA", 12, &multi), 6);
        assert_eq!(resolve_quote_windows("AAPL", 12, &multi), 12);
    }

    #[test]
    fn quote_windows_by_ticker_rejects_unknown_tickers_and_bad_values() {
        let tickers = parse_tickers(DEFAULT_TICKERS).unwrap();

        let unknown = parse_quote_windows_by_ticker("MSFT=24", &tickers)
            .unwrap_err()
            .to_string();
        assert!(unknown.contains("unknown ticker"), "got: {unknown}");
        assert!(unknown.contains("MSFT"), "got: {unknown}");

        let wrong_case = parse_quote_windows_by_ticker("nvda=24", &tickers)
            .unwrap_err()
            .to_string();
        assert!(wrong_case.contains("unknown ticker"), "got: {wrong_case}");

        let dup = parse_quote_windows_by_ticker("NVDA=24,NVDA=6", &tickers)
            .unwrap_err()
            .to_string();
        assert!(dup.contains("duplicate"), "got: {dup}");

        for bad in [
            "NVDA:24",
            "=24",
            "NVDA=",
            "NVDA=0",
            "NVDA=-1",
            "NVDA=abc",
            "NVDA=24,",
            ",NVDA=24",
            "NVDA=24,,TSLA=6",
        ] {
            assert!(
                parse_quote_windows_by_ticker(bad, &tickers).is_err(),
                "expected error for '{bad}'"
            );
        }
    }

    #[test]
    fn trigger_sigma_parser_defaults_to_two_and_rejects_invalid() {
        assert!((DEFAULT_TRIGGER_SIGMA - 2.0).abs() < 1e-12);
        assert_eq!(parse_trigger_sigma("2.0"), Some(2.0));
        assert_eq!(parse_trigger_sigma(" 1.5 "), Some(1.5));
        assert_eq!(parse_trigger_sigma("0"), None);
        assert_eq!(parse_trigger_sigma("-2"), None);
        assert_eq!(parse_trigger_sigma("NaN"), None);
        assert_eq!(parse_trigger_sigma("inf"), None);
    }

    #[test]
    fn pricing_model_and_max_multiplier_parsers_validate_inputs() {
        assert_eq!(PricingModel::parse("v1"), Some(PricingModel::V1));
        assert_eq!(PricingModel::parse("V2"), Some(PricingModel::V2));
        assert_eq!(PricingModel::parse("m4"), Some(PricingModel::M4));
        assert_eq!(PricingModel::parse("M4"), Some(PricingModel::M4));
        assert_eq!(PricingModel::parse("v3"), None);
        assert_eq!(parse_max_multiplier("25"), Some(25.0));
        assert_eq!(parse_max_multiplier(" 50.0 "), Some(50.0));
        assert_eq!(parse_max_multiplier("100"), Some(100.0));
        assert_eq!(parse_max_multiplier("1.04"), None);
        assert_eq!(parse_max_multiplier("0"), None);
        assert_eq!(parse_max_multiplier("NaN"), None);
    }
}
