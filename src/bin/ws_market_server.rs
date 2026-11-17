use std::env;
use std::error::Error;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use st_engine::app::build_view;
use st_engine::grid::{CellQuote, GridConfig, GridSnapshot};
use st_engine::logging::init_tracing;
use st_engine::price::{
    BinanceAggTradeConfig, BinanceConnectionState, BinanceStreamHealth, PriceSnapshot,
    spawn_binance_aggtrade_stream, spawn_price_pipeline,
};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio::time::{MissedTickBehavior, interval};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize)]
struct OutboundMessage {
    server_ts_ms: u64,
    stream: StreamStatusDto,
    market: MarketViewDto,
}

#[derive(Debug, Clone, Serialize)]
struct StreamStatusDto {
    connection_state: String,
    reconnect_attempts: u32,
    latest_tick_ts_ms: Option<u64>,
    last_message_age_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct MarketViewDto {
    latest_price: Option<f64>,
    latest_price_ts_ms: Option<u64>,
    latest_price_server_ts_ms: Option<u64>,
    completed_bar_count: usize,
    cells: Vec<CellDto>,
}

#[derive(Debug, Clone, Serialize)]
struct CellDto {
    server_ts_ms: u64,
    lower: f64,
    upper: f64,
    window_start_secs: u32,
    window_end_secs: u32,
    remaining_start_secs: u32,
    remaining_end_secs: u32,
    betable: bool,
    multiplier: f64,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    init_tracing("info,st_engine=debug");

    let mut config = GridConfig::default();
    apply_grid_env_overrides(&mut config);
    let lock_offset_secs = config.simulation.lock_offset;

    let max_completed_bars = env_usize("DEMO_MAX_BARS", 600);
    let grid_lookback_bars = env_usize("DEMO_LOOKBACK_BARS", 180);
    let tick_channel_capacity = env_usize("DEMO_TICK_CHANNEL_CAPACITY", 4096);
    let ws_bind_addr = env::var("WS_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let push_interval_ms = env_u64("WS_PUSH_INTERVAL_MS", 250);

    let mut binance_config = BinanceAggTradeConfig::default();
    if let Ok(value) = env::var("BINANCE_WS_URL") {
        if !value.trim().is_empty() {
            binance_config.ws_url = value;
        }
    }

    info!("WS Market Server");
    info!(bind_addr = %ws_bind_addr);
    info!(ws_url = %binance_config.ws_url);
    info!(lookback_bars = grid_lookback_bars);
    info!(max_completed_bars);
    info!(tick_channel_capacity);
    info!(parallel_workers = ?config.simulation.parallel_workers);
    info!(k_safety_ttl_ms = config.pricing.k_safety_ttl_ms);
    info!(push_interval_ms);

    let (price_runtime, _price_task) = spawn_price_pipeline(
        max_completed_bars,
        grid_lookback_bars,
        tick_channel_capacity,
    );
    let binance = spawn_binance_aggtrade_stream(price_runtime.tick_tx.clone(), binance_config);
    let (app_runtime, _app_task) = st_engine::app::spawn_app_pipeline(price_runtime.grid_rx.clone(), config);

    let listener = TcpListener::bind(&ws_bind_addr).await?;
    info!("websocket server listening");

    loop {
        let (stream, remote_addr) = listener.accept().await?;
        info!(%remote_addr, "accepted websocket TCP connection");

        let price_rx = price_runtime.snapshot_rx.clone();
        let grid_rx = app_runtime.grid_rx.clone();
        let health_rx = binance.health_rx.clone();

        tokio::spawn(async move {
            if let Err(error) = serve_client(
                stream,
                remote_addr.to_string(),
                price_rx,
                grid_rx,
                health_rx,
                push_interval_ms,
                lock_offset_secs,
            )
            .await
            {
                warn!(%remote_addr, error = %error, "websocket client task exited with error");
            }
        });
    }
}

async fn serve_client(
    stream: TcpStream,
    remote_addr: String,
    mut price_rx: watch::Receiver<PriceSnapshot>,
    mut grid_rx: watch::Receiver<Option<GridSnapshot>>,
    mut health_rx: watch::Receiver<BinanceStreamHealth>,
    push_interval_ms: u64,
    lock_offset_secs: u32,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let websocket = accept_async(stream).await?;
    let (mut writer, mut reader) = websocket.split();
    let mut latest_price = price_rx.borrow().clone();
    let mut latest_grid = grid_rx.borrow().clone();
    let mut latest_health = *health_rx.borrow();
    let mut ticker = interval(Duration::from_millis(push_interval_ms.max(50)));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    info!(remote_addr, "websocket client connected");

    loop {
        tokio::select! {
            changed = price_rx.changed() => {
                if changed.is_err() {
                    break;
                }
                latest_price = price_rx.borrow().clone();
            }
            changed = grid_rx.changed() => {
                if changed.is_err() {
                    break;
                }
                latest_grid = grid_rx.borrow().clone();
            }
            changed = health_rx.changed() => {
                if changed.is_err() {
                    break;
                }
                latest_health = *health_rx.borrow();
            }
            incoming = reader.next() => {
                match incoming {
                    Some(Ok(Message::Close(frame))) => {
                        info!(remote_addr, ?frame, "websocket client closed");
                        break;
                    }
                    Some(Ok(Message::Ping(payload))) => {
                        writer.send(Message::Pong(payload)).await?;
                    }
                    Some(Ok(Message::Text(text))) => {
                        debug!(remote_addr, payload = %text, "received websocket text message");
                    }
                    Some(Ok(Message::Binary(_))) => {}
                    Some(Ok(Message::Pong(_))) => {}
                    Some(Ok(Message::Frame(_))) => {}
                    Some(Err(error)) => {
                        return Err(Box::new(error));
                    }
                    None => break,
                }
            }
            _ = ticker.tick() => {
                let outbound = build_outbound_message(
                    &latest_price,
                    latest_grid.as_ref(),
                    latest_health,
                    lock_offset_secs,
                );
                let payload = serde_json::to_string(&outbound)?;
                writer.send(Message::Text(payload.into())).await?;
            }
        }
    }

    Ok(())
}

fn build_outbound_message(
    price: &PriceSnapshot,
    grid: Option<&GridSnapshot>,
    health: BinanceStreamHealth,
    lock_offset_secs: u32,
) -> OutboundMessage {
    let view = build_view(price, grid);
    OutboundMessage {
        server_ts_ms: now_ms(),
        stream: StreamStatusDto {
            connection_state: match health.connection_state {
                BinanceConnectionState::Connecting => "connecting",
                BinanceConnectionState::Open => "open",
                BinanceConnectionState::Reconnecting => "reconnecting",
                BinanceConnectionState::Closed => "closed",
            }
            .to_string(),
            reconnect_attempts: health.reconnect_attempts,
            latest_tick_ts_ms: health.latest_tick_ts_ms,
            last_message_age_ms: health
                .last_message_at_ms
                .map(|timestamp| now_ms().saturating_sub(timestamp)),
        },
        market: MarketViewDto {
            latest_price: view.latest_price,
            latest_price_ts_ms: view.latest_price_ts_ms,
            latest_price_server_ts_ms: view.latest_price_server_ts_ms,
            completed_bar_count: view.completed_bar_count,
            cells: view
                .grid
                .as_ref()
                .map(|grid| {
                    grid.cells
                        .iter()
                        .filter(|cell| is_betable_window(cell.window.start_secs, lock_offset_secs))
                        .map(|cell| cell_to_dto(cell, lock_offset_secs))
                        .collect()
                })
                .unwrap_or_default(),
        },
    }
}

fn cell_to_dto(cell: &CellQuote, lock_offset_secs: u32) -> CellDto {
    CellDto {
        server_ts_ms: cell.server_ts_ms,
        lower: cell.band.lower,
        upper: cell.band.upper,
        window_start_secs: cell.window.start_secs,
        window_end_secs: cell.window.end_secs,
        remaining_start_secs: cell.window.start_secs,
        remaining_end_secs: cell.window.end_secs,
        betable: is_betable_window(cell.window.start_secs, lock_offset_secs),
        multiplier: cell.multiplier,
    }
}

fn is_betable_window(window_start_secs: u32, lock_offset_secs: u32) -> bool {
    window_start_secs > lock_offset_secs.max(5)
}

fn apply_grid_env_overrides(config: &mut GridConfig) {
    if let Ok(value) = env::var("GRID_PARALLEL_WORKERS") {
        let trimmed = value.trim();
        config.simulation.parallel_workers = if trimmed.eq_ignore_ascii_case("none") {
            None
        } else {
            trimmed.parse::<usize>().ok()
        };
    }

    if let Ok(value) = env::var("GRID_K_SAFETY_TTL_MS") {
        if let Ok(parsed) = value.trim().parse::<u64>() {
            config.pricing.k_safety_ttl_ms = parsed;
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::is_betable_window;

    #[test]
    fn tradable_windows_start_after_pending_window() {
        assert!(!is_betable_window(5, 1));
        assert!(is_betable_window(10, 1));
        assert!(is_betable_window(10, 5));
        assert!(is_betable_window(15, 1));
    }
}
