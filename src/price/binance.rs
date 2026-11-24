use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use serde::Deserialize;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info, warn};

use super::types::{PriceTick, StreamHealth};

#[derive(Debug, Clone)]
pub struct BinanceAggTradeConfig {
    pub ws_url: String,
    pub stale_after: Duration,
    pub reconnect_base_delay: Duration,
    pub reconnect_max_delay: Duration,
}

impl Default for BinanceAggTradeConfig {
    fn default() -> Self {
        Self {
            ws_url: "wss://fstream.binance.com/ws/btcusdt@aggTrade".to_string(),
            stale_after: Duration::from_secs(15),
            reconnect_base_delay: Duration::from_secs(1),
            reconnect_max_delay: Duration::from_secs(30),
        }
    }
}

#[derive(Debug)]
pub struct BinanceStreamHandle {
    pub health_rx: watch::Receiver<BinanceStreamHealth>,
    pub join_handle: JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub struct BinanceAggTradeStream {
    tick_tx: mpsc::Sender<PriceTick>,
    health_tx: watch::Sender<BinanceStreamHealth>,
    config: BinanceAggTradeConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinanceConnectionState {
    Connecting,
    Open,
    Reconnecting,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinanceStreamHealth {
    pub connection_state: BinanceConnectionState,
    pub latest_tick_ts_ms: Option<u64>,
    pub last_message_at_ms: Option<u64>,
    pub reconnect_attempts: u32,
}

impl Default for BinanceStreamHealth {
    fn default() -> Self {
        Self {
            connection_state: BinanceConnectionState::Closed,
            latest_tick_ts_ms: None,
            last_message_at_ms: None,
            reconnect_attempts: 0,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AggTradePayload {
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "T")]
    trade_ts_ms: u64,
}

impl BinanceAggTradeStream {
    pub fn new(
        tick_tx: mpsc::Sender<PriceTick>,
        health_tx: watch::Sender<BinanceStreamHealth>,
        config: BinanceAggTradeConfig,
    ) -> Self {
        Self {
            tick_tx,
            health_tx,
            config,
        }
    }

    pub async fn run(self) {
        let mut reconnect_attempts = 0u32;

        loop {
            info!(
                ws_url = %self.config.ws_url,
                reconnect_attempts,
                "connecting to Binance aggTrade stream"
            );
            self.publish_health(BinanceStreamHealth {
                connection_state: BinanceConnectionState::Connecting,
                reconnect_attempts,
                ..current_health(&self.health_tx)
            });

            match connect_async(&self.config.ws_url).await {
                Ok((socket, _)) => {
                    info!("Binance aggTrade stream connected");
                    reconnect_attempts = 0;
                    self.publish_health(BinanceStreamHealth {
                        connection_state: BinanceConnectionState::Open,
                        reconnect_attempts,
                        ..current_health(&self.health_tx)
                    });

                    let (_, mut read) = socket.split();
                    let mut last_message_at_ms = now_ms();

                    while let Some(message) = read.next().await {
                        match message {
                            Ok(Message::Text(text)) => {
                                if let Some(tick) = parse_tick(&text) {
                                    last_message_at_ms = now_ms();
                                    if self.tick_tx.send(tick).await.is_err() {
                                        warn!("price pipeline receiver dropped; stopping Binance stream task");
                                        return;
                                    }
                                    self.publish_health(BinanceStreamHealth {
                                        connection_state: BinanceConnectionState::Open,
                                        latest_tick_ts_ms: Some(tick.ts_ms),
                                        last_message_at_ms: Some(last_message_at_ms),
                                        reconnect_attempts,
                                    });
                                }
                            }
                            Ok(Message::Binary(bytes)) => {
                                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                                    if let Some(tick) = parse_tick(&text) {
                                        last_message_at_ms = now_ms();
                                        if self.tick_tx.send(tick).await.is_err() {
                                            warn!("price pipeline receiver dropped; stopping Binance stream task");
                                            return;
                                        }
                                        self.publish_health(BinanceStreamHealth {
                                            connection_state: BinanceConnectionState::Open,
                                            latest_tick_ts_ms: Some(tick.ts_ms),
                                            last_message_at_ms: Some(last_message_at_ms),
                                            reconnect_attempts,
                                        });
                                    }
                                }
                            }
                            Ok(Message::Ping(_))
                            | Ok(Message::Pong(_))
                            | Ok(Message::Frame(_)) => {}
                            Ok(Message::Close(frame)) => {
                                info!(?frame, "Binance aggTrade stream closed by peer");
                                break;
                            }
                            Err(error) => {
                                warn!(?error, "Binance aggTrade stream read error");
                                break;
                            }
                        }

                        if now_ms().saturating_sub(last_message_at_ms)
                            > self.config.stale_after.as_millis() as u64
                        {
                            warn!(
                                stale_after_ms = self.config.stale_after.as_millis() as u64,
                                "Binance aggTrade stream stale; reconnecting"
                            );
                            break;
                        }
                    }
                }
                Err(error) => {
                    warn!(?error, "failed to connect to Binance aggTrade stream");
                }
            }

            reconnect_attempts = reconnect_attempts.saturating_add(1);
            self.publish_health(BinanceStreamHealth {
                connection_state: BinanceConnectionState::Reconnecting,
                reconnect_attempts,
                ..current_health(&self.health_tx)
            });

            let delay = reconnect_delay(
                reconnect_attempts,
                self.config.reconnect_base_delay,
                self.config.reconnect_max_delay,
            );
            debug!(
                reconnect_attempts,
                delay_ms = delay.as_millis() as u64,
                "waiting before Binance reconnect"
            );
            tokio::time::sleep(delay).await;
        }
    }

    fn publish_health(&self, health: BinanceStreamHealth) {
        let _ = self.health_tx.send(health);
    }
}

pub fn spawn_binance_aggtrade_stream(
    tick_tx: mpsc::Sender<PriceTick>,
    config: BinanceAggTradeConfig,
) -> BinanceStreamHandle {
    let (health_tx, health_rx) = watch::channel(BinanceStreamHealth::default());
    let stream = BinanceAggTradeStream::new(tick_tx, health_tx, config);
    let join_handle = tokio::spawn(stream.run());

    BinanceStreamHandle {
        health_rx,
        join_handle,
    }
}

fn parse_tick(raw: &str) -> Option<PriceTick> {
    let payload: AggTradePayload = serde_json::from_str(raw).ok()?;
    let price = payload.price.parse::<f64>().ok()?;
    Some(PriceTick::new(payload.trade_ts_ms, price))
}

fn reconnect_delay(attempt: u32, base: Duration, max: Duration) -> Duration {
    let factor = 2u32.saturating_pow(attempt.saturating_sub(1));
    let delay_ms = base.as_millis().saturating_mul(factor as u128);
    Duration::from_millis(delay_ms.min(max.as_millis()) as u64)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn current_health(sender: &watch::Sender<BinanceStreamHealth>) -> BinanceStreamHealth {
    *sender.borrow()
}

impl From<BinanceStreamHealth> for StreamHealth {
    fn from(value: BinanceStreamHealth) -> Self {
        Self {
            has_price: value.latest_tick_ts_ms.is_some(),
            latest_tick_ts_ms: value.latest_tick_ts_ms,
            completed_bar_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_binance_aggtrade_payload() {
        let raw = r#"{"p":"50000.12","T":1710000000123}"#;
        let tick = parse_tick(raw).unwrap();
        assert_eq!(tick.ts_ms, 1_710_000_000_123);
        assert_eq!(tick.price, 50_000.12);
    }

    #[test]
    fn reconnect_delay_is_capped() {
        let base = Duration::from_secs(1);
        let max = Duration::from_secs(30);
        assert_eq!(reconnect_delay(1, base, max), Duration::from_secs(1));
        assert_eq!(reconnect_delay(2, base, max), Duration::from_secs(2));
        assert_eq!(reconnect_delay(10, base, max), Duration::from_secs(30));
    }
}
