use crate::app::types::AppMarketView;
use crate::grid::{GridConfig, GridEngine, GridSnapshot};
use crate::price::GridMarketSnapshot;
use tokio::sync::watch;
use tokio::task::JoinHandle;

#[derive(Debug)]
pub struct AppRuntime {
    pub grid_rx: watch::Receiver<Option<GridSnapshot>>,
}

pub struct GridWorker {
    market_rx: watch::Receiver<Option<GridMarketSnapshot>>,
    grid_tx: watch::Sender<Option<GridSnapshot>>,
    engine: GridEngine,
}

impl GridWorker {
    pub async fn run(mut self) {
        loop {
            if self.market_rx.changed().await.is_err() {
                return;
            }

            let next_grid = self
                .market_rx
                .borrow()
                .clone()
                .and_then(|market| self.engine.quote_grid(market.spot, &market.history).ok());
            let _ = self.grid_tx.send(next_grid);
        }
    }
}

pub fn spawn_app_pipeline(
    grid_market_rx: watch::Receiver<Option<GridMarketSnapshot>>,
    config: GridConfig,
) -> (AppRuntime, JoinHandle<()>) {
    let (grid_tx, grid_rx) = watch::channel(None);

    let grid_worker = GridWorker {
        market_rx: grid_market_rx,
        grid_tx,
        engine: GridEngine::new(config),
    };

    let handle = tokio::spawn(grid_worker.run());

    (AppRuntime { grid_rx }, handle)
}

pub fn build_view(
    price: &crate::price::PriceSnapshot,
    grid: Option<&GridSnapshot>,
) -> AppMarketView {
    AppMarketView {
        latest_price: price.latest_price(),
        latest_price_ts_ms: price.latest_tick.map(|tick| tick.ts_ms),
        latest_price_server_ts_ms: price.latest_server_ts_ms,
        completed_bar_count: price.completed_bars.len(),
        grid: grid.cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{GridConfig, OhlcBar, VolatilitySnapshot};
    use crate::price::{PriceSnapshot, PriceTick, spawn_price_pipeline};
    use tokio::time::{Duration, timeout};

    #[tokio::test(flavor = "current_thread")]
    async fn app_pipeline_emits_latest_grid_snapshot() {
        let (price_runtime, _price_task) = spawn_price_pipeline(16, 4, 16);
        let (app_runtime, _task) = spawn_app_pipeline(
            price_runtime.grid_rx.clone(),
            GridConfig::default(),
        );

        for (index, price) in [100.0, 100.5, 101.0, 101.5].into_iter().enumerate() {
            price_runtime
                .tick_tx
                .send(PriceTick {
                    ts_ms: 1_000 * (index as u64 + 1),
                    price,
                })
                .await
                .unwrap();
        }

        let mut grid_rx = app_runtime.grid_rx.clone();
        timeout(Duration::from_secs(1), async {
            loop {
                grid_rx.changed().await.unwrap();
                if grid_rx.borrow().is_some() {
                    break;
                }
            }
        })
        .await
        .unwrap();

        let grid = grid_rx.borrow().clone();
        assert!(grid.is_some());
    }

    #[test]
    fn build_view_copies_latest_price_and_grid() {
        let price = PriceSnapshot {
            latest_tick: Some(PriceTick {
                ts_ms: 1_000,
                price: 123.0,
            }),
            latest_server_ts_ms: Some(1_111),
            completed_bars: vec![OhlcBar {
                open: 120.0,
                high: 124.0,
                low: 119.0,
                close: 123.0,
            }],
            current_bar: None,
        };
        let grid = GridSnapshot {
            server_ts_ms: 1_222,
            spot: 123.0,
            anchor: 120.0,
            volatility: VolatilitySnapshot {
                variance: 0.0,
                sigma_raw: 0.0,
                sigma: 0.0,
                jump_intensity: 0.0,
                jump_flag: false,
            },
            cells: Vec::new(),
        };

        let view = build_view(&price, Some(&grid));
        assert_eq!(view.latest_price, Some(123.0));
        assert_eq!(view.latest_price_ts_ms, Some(1_000));
        assert_eq!(view.latest_price_server_ts_ms, Some(1_111));
        assert_eq!(view.completed_bar_count, 1);
        assert_eq!(view.grid, Some(grid));
    }

    #[test]
    fn build_view_handles_missing_grid() {
        let price = PriceSnapshot {
            latest_tick: Some(PriceTick {
                ts_ms: 2_000,
                price: 125.0,
            }),
            latest_server_ts_ms: Some(2_111),
            completed_bars: Vec::new(),
            current_bar: None,
        };

        let view = build_view(&price, None);
        assert_eq!(view.latest_price, Some(125.0));
        assert_eq!(view.latest_price_ts_ms, Some(2_000));
        assert_eq!(view.latest_price_server_ts_ms, Some(2_111));
        assert_eq!(view.completed_bar_count, 0);
        assert!(view.grid.is_none());
    }
}
