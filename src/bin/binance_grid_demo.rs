use std::env;
use std::error::Error;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use st_engine::grid::{GridConfig, GridEngine, GridSnapshot};
use st_engine::logging::init_tracing;
use st_engine::price::{
    BinanceAggTradeConfig, BinanceStreamHealth, GridMarketSnapshot, PriceRuntime,
    spawn_binance_aggtrade_stream, spawn_price_pipeline,
};
use tokio::time::{MissedTickBehavior, interval};
use tracing::{debug, info, warn};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    init_tracing("info,st_engine=debug");

    let mut config = GridConfig::default();
    apply_grid_env_overrides(&mut config);

    let max_completed_bars = env_usize("DEMO_MAX_BARS", 600);
    let grid_lookback_bars = env_usize("DEMO_LOOKBACK_BARS", 180);
    let tick_channel_capacity = env_usize("DEMO_TICK_CHANNEL_CAPACITY", 4096);
    let log_interval_ms = env_u64("DEMO_LOG_INTERVAL_MS", 1_000);
    let log_cells = env_flag("DEMO_LOG_CELLS", false);

    let mut binance_config = BinanceAggTradeConfig::default();
    if let Ok(value) = env::var("BINANCE_WS_URL") {
        if !value.trim().is_empty() {
            binance_config.ws_url = value;
        }
    }

    info!("Binance Futures Grid Demo");
    info!(ws_url = %binance_config.ws_url);
    info!(lookback_bars = grid_lookback_bars);
    info!(max_completed_bars);
    info!(tick_channel_capacity);
    info!(parallel_workers = ?config.simulation.parallel_workers);
    info!(k_safety_ttl_ms = config.pricing.k_safety_ttl_ms);
    info!(log_interval_ms);
    info!(log_cells);

    let (price_runtime, _price_task) = spawn_price_pipeline(
        max_completed_bars,
        grid_lookback_bars,
        tick_channel_capacity,
    );
    let binance = spawn_binance_aggtrade_stream(price_runtime.tick_tx.clone(), binance_config);

    run_demo_loop(
        price_runtime,
        binance.health_rx,
        config,
        log_interval_ms,
        log_cells,
    )
    .await
}

async fn run_demo_loop(
    price_runtime: PriceRuntime,
    mut health_rx: tokio::sync::watch::Receiver<BinanceStreamHealth>,
    config: GridConfig,
    log_interval_ms: u64,
    log_cells: bool,
) -> Result<(), Box<dyn Error>> {
    let mut engine = GridEngine::new(config);
    let mut grid_rx = price_runtime.grid_rx.clone();
    let mut latest_market = grid_rx.borrow().clone();
    let mut latest_health = *health_rx.borrow();
    let mut last_logged_state = None;
    let mut ticker = interval(Duration::from_millis(log_interval_ms.max(100)));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            result = grid_rx.changed() => {
                if result.is_err() {
                    return Ok(());
                }
                latest_market = grid_rx.borrow().clone();
            }
            result = health_rx.changed() => {
                if result.is_err() {
                    return Ok(());
                }
                latest_health = *health_rx.borrow();
                log_health(latest_health, &mut last_logged_state);
            }
            _ = ticker.tick() => {
                log_grid_snapshot(&mut engine, latest_market.as_ref(), latest_health, log_cells);
            }
        }
    }
}

fn log_health(
    health: BinanceStreamHealth,
    last_logged_state: &mut Option<(st_engine::price::BinanceConnectionState, u32)>,
) {
    let state_key = (health.connection_state, health.reconnect_attempts);
    if last_logged_state.as_ref() != Some(&state_key) {
        *last_logged_state = Some(state_key);
        info!(
            state = ?health.connection_state,
            reconnect_attempts = health.reconnect_attempts,
            latest_tick_ts_ms = ?health.latest_tick_ts_ms,
            last_message_age_ms = ?health
                .last_message_at_ms
                .map(|timestamp| now_ms().saturating_sub(timestamp)),
            "stream state update"
        );
    } else {
        debug!(
            state = ?health.connection_state,
            reconnect_attempts = health.reconnect_attempts,
            latest_tick_ts_ms = ?health.latest_tick_ts_ms,
            last_message_age_ms = ?health
                .last_message_at_ms
                .map(|timestamp| now_ms().saturating_sub(timestamp)),
            "stream health tick"
        );
    }
}

fn log_grid_snapshot(
    engine: &mut GridEngine,
    market: Option<&GridMarketSnapshot>,
    health: BinanceStreamHealth,
    log_cells: bool,
) {
    let Some(market) = market else {
        debug!(state = ?health.connection_state, "waiting for market snapshot");
        return;
    };

    let started_at = Instant::now();
    match engine.quote_grid(market.spot, &market.history) {
        Ok(snapshot) => {
            print_snapshot_summary(
                &snapshot,
                market.history.len(),
                health,
                started_at.elapsed(),
                log_cells,
            )
        }
        Err(error) => {
            warn!(
                spot = market.spot,
                bars = market.history.len(),
                compute_ms = started_at.elapsed().as_secs_f64() * 1_000.0,
                error = %error,
                "grid quote failed"
            );
        }
    }
}

fn print_snapshot_summary(
    snapshot: &GridSnapshot,
    history_len: usize,
    health: BinanceStreamHealth,
    compute_time: Duration,
    log_cells: bool,
) {
    let windows_per_row = snapshot
        .cells
        .first()
        .map(|first| {
            snapshot
                .cells
                .iter()
                .take_while(|cell| cell.band.row_index == first.band.row_index)
                .count()
        })
        .unwrap_or(1);
    let nearest_row = snapshot
        .cells
        .chunks(windows_per_row)
        .filter_map(|row_cells| row_cells.first())
        .min_by(|left, right| {
            let left_center = (left.band.lower + left.band.upper) / 2.0;
            let right_center = (right.band.lower + right.band.upper) / 2.0;
            (left_center - snapshot.spot)
                .abs()
                .partial_cmp(&(right_center - snapshot.spot).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|cell| cell.band.row_index)
        .unwrap_or(0);

    let nearest_row_cells = snapshot
        .cells
        .iter()
        .filter(|cell| cell.band.row_index == nearest_row)
        .collect::<Vec<_>>();
    let nearest_p_min = nearest_row_cells
        .iter()
        .map(|cell| cell.hit_probability)
        .fold(f64::INFINITY, f64::min);
    let nearest_p_max = nearest_row_cells
        .iter()
        .map(|cell| cell.hit_probability)
        .fold(f64::NEG_INFINITY, f64::max);
    let nearest_m_min = nearest_row_cells
        .iter()
        .map(|cell| cell.multiplier)
        .fold(f64::INFINITY, f64::min);
    let nearest_m_max = nearest_row_cells
        .iter()
        .map(|cell| cell.multiplier)
        .fold(f64::NEG_INFINITY, f64::max);

    info!(
        state = ?health.connection_state,
        spot = snapshot.spot,
        anchor = snapshot.anchor,
        bars = history_len,
        compute_ms = compute_time.as_secs_f64() * 1_000.0,
        sigma = snapshot.volatility.sigma,
        lambda = snapshot.volatility.jump_intensity,
        jump = snapshot.volatility.jump_flag,
        row = nearest_row,
        nearest_p_min,
        nearest_p_max,
        nearest_m_min,
        nearest_m_max,
        "grid snapshot"
    );

    if log_cells {
        for cell in nearest_row_cells {
            info!(
                window_start = cell.window.start_secs,
                window_end = cell.window.end_secs,
                band_lower = cell.band.lower,
                band_upper = cell.band.upper,
                p = cell.hit_probability,
                multiplier = cell.multiplier,
                "nearest-row cell"
            );
        }
    }
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

fn env_flag(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
