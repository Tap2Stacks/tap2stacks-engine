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

use super::types::PriceTick;

const PING_PAYLOAD: &str = r#"{"method":"ping"}"#;

#[derive(Debug, Clone)]
pub struct HyperliquidConfig {
    pub ws_url: String,
    pub coins: Vec<String>,
    pub stale_after: Duration,
    pub health_interval: Duration,
    pub ping_interval: Duration,
    pub reconnect_base_delay: Duration,
    pub reconnect_max_delay: Duration,
}

impl Default for HyperliquidConfig {
    fn default() -> Self {
        Self {
            ws_url: "wss://api.hyperliquid.xyz/ws".to_string(),
            coins: vec!["xyz:NVDA".to_string(), "xyz:TSLA".to_string()],
            stale_after: Duration::from_secs(3),
            health_interval: Duration::from_millis(500),
            ping_interval: Duration::from_secs(30),
            reconnect_base_delay: Duration::from_secs(1),
            reconnect_max_delay: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeedHealth {
    pub last_sample_ms: Option<u64>,
    pub stale: bool,
    pub reconnect_attempts: u32,
}

#[derive(Debug)]
pub struct HyperliquidStreamHandle {
    pub health_rx: watch::Receiver<HashMap<String, FeedHealth>>,
    pub prev_day_px: Arc<RwLock<HashMap<String, f64>>>,
    pub join_handle: JoinHandle<()>,
}

/// Tracks per-coin sample freshness independent of incoming message arrivals.
#[derive(Debug, Clone)]
pub struct HealthTracker {
    stale_after_ms: u64,
    baseline_ms: Option<u64>,
    states: HashMap<String, FeedHealth>,
}

impl HealthTracker {
    pub fn new<I, S>(coins: I, stale_after: Duration) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let states = coins
            .into_iter()
            .map(|coin| (coin.into(), FeedHealth::default()))
            .collect();
        Self {
            stale_after_ms: stale_after.as_millis() as u64,
            baseline_ms: None,
            states,
        }
    }

    pub fn on_sample(&mut self, coin: &str, now_ms: u64) {
        if self.baseline_ms.is_none() {
            self.baseline_ms = Some(now_ms);
        }
        let entry = self.states.entry(coin.to_string()).or_default();
        entry.last_sample_ms = Some(match entry.last_sample_ms {
            Some(prev) => prev.max(now_ms),
            None => now_ms,
        });
        entry.stale = false;
    }

    pub fn set_reconnect_attempts(&mut self, attempts: u32) {
        for health in self.states.values_mut() {
            health.reconnect_attempts = attempts;
        }
    }

    pub fn tick(&mut self, now_ms: u64) -> HashMap<String, FeedHealth> {
        let baseline = *self.baseline_ms.get_or_insert(now_ms);
        for health in self.states.values_mut() {
            let ref_ms = health.last_sample_ms.unwrap_or(baseline);
            health.stale = now_ms.saturating_sub(ref_ms) > self.stale_after_ms;
        }
        self.states.clone()
    }

    pub fn snapshot(&self) -> HashMap<String, FeedHealth> {
        self.states.clone()
    }
}

#[derive(Debug, Deserialize)]
struct WsEnvelope {
    channel: String,
    #[serde(default)]
    data: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssetCtx {
    pub coin: String,
    pub mark_px: f64,
    pub prev_day_px: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ActiveAssetCtxData {
    coin: String,
    ctx: ActiveAssetCtx,
}

#[derive(Debug, Deserialize)]
struct ActiveAssetCtx {
    #[serde(rename = "markPx")]
    mark_px: String,
    #[serde(rename = "prevDayPx")]
    prev_day_px: Option<String>,
}

fn parse_positive_finite_price(raw: &str) -> Option<f64> {
    let price = raw.parse::<f64>().ok()?;
    if price.is_finite() && price > 0.0 {
        Some(price)
    } else {
        None
    }
}

pub(crate) fn parse_active_asset_ctx_full(data: serde_json::Value) -> Option<AssetCtx> {
    let payload: ActiveAssetCtxData = serde_json::from_value(data).ok()?;
    let mark_px = parse_positive_finite_price(&payload.ctx.mark_px)?;
    let prev_day_px = payload
        .ctx
        .prev_day_px
        .as_deref()
        .and_then(parse_positive_finite_price);
    Some(AssetCtx {
        coin: payload.coin,
        mark_px,
        prev_day_px,
    })
}

pub(crate) fn parse_active_asset_ctx_value(data: serde_json::Value) -> Option<(String, f64)> {
    parse_active_asset_ctx_full(data).map(|ctx| (ctx.coin, ctx.mark_px))
}

/// Parses a live Hyperliquid WebSocket JSON frame and returns `AssetCtx`
/// when `channel == "activeAssetCtx"`.
pub fn parse_asset_ctx(text: &str) -> Option<AssetCtx> {
    let envelope: WsEnvelope = serde_json::from_str(text).ok()?;
    if envelope.channel != "activeAssetCtx" {
        return None;
    }
    parse_active_asset_ctx_full(envelope.data)
}

/// Parses a live Hyperliquid WebSocket JSON frame and returns `(coin, mark_px)`
/// when `channel == "activeAssetCtx"`.
pub fn parse_mark_price(text: &str) -> Option<(String, f64)> {
    parse_asset_ctx(text).map(|ctx| (ctx.coin, ctx.mark_px))
}

pub fn reconnect_delay(attempt: u32, base: Duration, max: Duration) -> Duration {
    let factor = 2u32.saturating_pow(attempt.saturating_sub(1));
    let delay_ms = base.as_millis().saturating_mul(factor as u128);
    Duration::from_millis(delay_ms.min(max.as_millis()) as u64)
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

struct HyperliquidStream {
    config: HyperliquidConfig,
    senders: HashMap<String, mpsc::Sender<PriceTick>>,
    health_tx: watch::Sender<HashMap<String, FeedHealth>>,
    prev_day_px: Arc<RwLock<HashMap<String, f64>>>,
}

impl HyperliquidStream {
    async fn run(self) {
        let clock = MonotonicWallClock::new();
        let mut tracker = HealthTracker::new(self.config.coins.iter().cloned(), self.config.stale_after);
        let health_period = self.config.health_interval.max(Duration::from_millis(1));
        let mut health_ticker = tokio::time::interval(health_period);
        health_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut reconnect_attempts = 0u32;

        loop {
            if self.all_senders_closed() {
                return;
            }

            info!(
                ws_url = %self.config.ws_url,
                reconnect_attempts,
                "connecting to Hyperliquid stream"
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
                    info!("Hyperliquid stream connected");
                    reconnect_attempts = 0;
                    tracker.set_reconnect_attempts(0);
                    publish_health(&self.health_tx, tracker.snapshot());

                    let (mut write, mut read) = socket.split();

                    let mut subscribe_ok = true;
                    for coin in &self.config.coins {
                        let sub = serde_json::json!({
                            "method": "subscribe",
                            "subscription": {
                                "type": "activeAssetCtx",
                                "coin": coin,
                            }
                        })
                        .to_string();
                        if let Err(error) = write.send(Message::Text(sub.into())).await {
                            warn!(%coin, ?error, "failed to subscribe on Hyperliquid stream");
                            subscribe_ok = false;
                            break;
                        }
                    }

                    if subscribe_ok {
                        let ping_period = self.config.ping_interval.max(Duration::from_secs(1));
                        let ping_start = tokio::time::Instant::now() + ping_period;
                        let mut ping_ticker = tokio::time::interval_at(ping_start, ping_period);
                        ping_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

                        loop {
                            tokio::select! {
                                maybe_msg = read.next() => {
                                    match maybe_msg {
                                        Some(Ok(Message::Text(text))) => {
                                            if !self.handle_text(&text, clock.now_ms(), &mut tracker).await {
                                                return;
                                            }
                                        }
                                        Some(Ok(Message::Binary(bytes))) => {
                                            if let Ok(text) = std::str::from_utf8(&bytes) {
                                                if !self.handle_text(text, clock.now_ms(), &mut tracker).await {
                                                    return;
                                                }
                                            }
                                        }
                                        Some(Ok(Message::Ping(payload))) => {
                                            if let Err(error) = write.send(Message::Pong(payload)).await {
                                                warn!(?error, "Hyperliquid pong send error");
                                                break;
                                            }
                                        }
                                        Some(Ok(Message::Pong(_))) | Some(Ok(Message::Frame(_))) => {}
                                        Some(Ok(Message::Close(frame))) => {
                                            info!(?frame, "Hyperliquid stream closed by peer");
                                            break;
                                        }
                                        Some(Err(error)) => {
                                            warn!(?error, "Hyperliquid stream read error");
                                            break;
                                        }
                                        None => {
                                            info!("Hyperliquid stream ended");
                                            break;
                                        }
                                    }
                                }
                                _ = ping_ticker.tick() => {
                                    if let Err(error) = write.send(Message::Text(PING_PAYLOAD.into())).await {
                                        warn!(?error, "Hyperliquid ping send error");
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
                    warn!(?error, "failed to connect to Hyperliquid stream");
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
                "waiting before Hyperliquid reconnect"
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
    ) -> bool {
        let Some(ctx) = parse_asset_ctx(text) else {
            return true;
        };

        if let Some(prev_day_px) = ctx.prev_day_px {
            let mut guard = self
                .prev_day_px
                .write()
                .expect("prev_day_px rwlock poisoned");
            guard.insert(ctx.coin.clone(), prev_day_px);
        }

        let coin = ctx.coin;
        let price = ctx.mark_px;

        tracker.on_sample(&coin, now_ms);
        publish_health(&self.health_tx, tracker.snapshot());

        if let Some(tx) = self.senders.get(&coin) {
            if tx.send(PriceTick { ts_ms: now_ms, price }).await.is_err() {
                warn!(%coin, "Hyperliquid price receiver dropped");
                if self.all_senders_closed() {
                    return false;
                }
            }
        }
        true
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

/// Spawns a background task subscribing to Hyperliquid `activeAssetCtx` mark
/// prices for all configured coins and tracking per-coin staleness on a timer.
pub fn spawn_hyperliquid_stream(
    config: HyperliquidConfig,
    senders: HashMap<String, mpsc::Sender<PriceTick>>,
) -> HyperliquidStreamHandle {
    let initial_health = HealthTracker::new(config.coins.iter().cloned(), config.stale_after).snapshot();
    let (health_tx, health_rx) = watch::channel(initial_health);
    let prev_day_px = Arc::new(RwLock::new(HashMap::new()));
    let stream = HyperliquidStream {
        config,
        senders,
        health_tx,
        prev_day_px: Arc::clone(&prev_day_px),
    };
    let join_handle = tokio::spawn(stream.run());

    HyperliquidStreamHandle {
        health_rx,
        prev_day_px,
        join_handle,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn parses_active_asset_ctx_mark_price() {
        let line = r#"{"channel":"activeAssetCtx","data":{"coin":"xyz:NVDA","ctx":{"markPx":"157.53","oraclePx":"157.44"}}}"#;
        assert_eq!(parse_mark_price(line), Some(("xyz:NVDA".into(), 157.53)));
    }

    #[test]
    fn parse_asset_ctx_reads_prev_day_px() {
        let with_prev = r#"{"channel":"activeAssetCtx","data":{"coin":"xyz:NVDA","ctx":{"markPx":"154.68","prevDayPx":"154.03","oraclePx":"154.70"}}}"#;
        let parsed = parse_asset_ctx(with_prev).expect("expected valid AssetCtx");
        assert_eq!(parsed.coin, "xyz:NVDA");
        assert_eq!(parsed.mark_px, 154.68);
        assert_eq!(parsed.prev_day_px, Some(154.03));

        let without_prev = r#"{"channel":"activeAssetCtx","data":{"coin":"xyz:NVDA","ctx":{"markPx":"154.68","oraclePx":"154.70"}}}"#;
        let parsed_no_prev = parse_asset_ctx(without_prev).expect("expected valid AssetCtx");
        assert_eq!(parsed_no_prev.mark_px, 154.68);
        assert_eq!(parsed_no_prev.prev_day_px, None);
    }

    #[test]
    fn ignores_bbo_and_subscription_responses() {
        assert_eq!(
            parse_mark_price(r#"{"channel":"bbo","data":{"coin":"xyz:NVDA"}}"#),
            None
        );
        assert_eq!(
            parse_mark_price(r#"{"channel":"subscriptionResponse","data":{}}"#),
            None
        );
    }

    #[tokio::test(start_paused = true)]
    async fn silent_feed_turns_stale_after_3s() {
        let mut tracker = HealthTracker::new(["xyz:NVDA"], Duration::from_secs(3));
        let start = tokio::time::Instant::now();
        let now_ms = |instant: tokio::time::Instant| -> u64 {
            instant.duration_since(start).as_millis() as u64
        };

        tracker.on_sample("xyz:NVDA", now_ms(tokio::time::Instant::now()));

        tokio::time::advance(Duration::from_millis(2_900)).await;
        let health_at_2900 = tracker.tick(now_ms(tokio::time::Instant::now()));
        assert!(!health_at_2900["xyz:NVDA"].stale);

        tokio::time::advance(Duration::from_millis(600)).await;
        let health_at_3500 = tracker.tick(now_ms(tokio::time::Instant::now()));
        assert!(health_at_3500["xyz:NVDA"].stale);
    }

    #[test]
    fn reconnect_delay_doubles_and_caps_at_30s() {
        let base = Duration::from_secs(1);
        let max = Duration::from_secs(30);
        assert_eq!(reconnect_delay(1, base, max), Duration::from_secs(1));
        assert_eq!(reconnect_delay(2, base, max), Duration::from_secs(2));
        assert_eq!(reconnect_delay(3, base, max), Duration::from_secs(4));
        assert_eq!(reconnect_delay(4, base, max), Duration::from_secs(8));
        assert_eq!(reconnect_delay(5, base, max), Duration::from_secs(16));
        assert_eq!(reconnect_delay(6, base, max), Duration::from_secs(30));
        assert_eq!(reconnect_delay(10, base, max), Duration::from_secs(30));
    }
}
