use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use super::cache::PriceCache;
use super::types::{GridMarketSnapshot, PriceSnapshot, PriceTick};

#[derive(Debug)]
pub struct PriceRuntime {
    pub tick_tx: mpsc::Sender<PriceTick>,
    pub snapshot_rx: watch::Receiver<PriceSnapshot>,
    pub grid_rx: watch::Receiver<Option<GridMarketSnapshot>>,
}

pub struct PriceWorker {
    tick_rx: mpsc::Receiver<PriceTick>,
    snapshot_tx: watch::Sender<PriceSnapshot>,
    grid_tx: watch::Sender<Option<GridMarketSnapshot>>,
    cache: PriceCache,
    grid_lookback_bars: usize,
}

impl PriceWorker {
    pub async fn run(mut self) {
        while let Some(tick) = self.tick_rx.recv().await {
            if self.cache.apply_tick(tick).is_err() {
                continue;
            }

            let snapshot = self.cache.snapshot();
            let _ = self.snapshot_tx.send(snapshot.clone());
            let _ = self
                .grid_tx
                .send(snapshot.bars_for_grid(self.grid_lookback_bars));
        }
    }
}

pub fn spawn_price_pipeline(
    max_completed_bars: usize,
    grid_lookback_bars: usize,
    tick_channel_capacity: usize,
) -> (PriceRuntime, JoinHandle<()>) {
    let (tick_tx, tick_rx) = mpsc::channel(tick_channel_capacity.max(1));
    let (snapshot_tx, snapshot_rx) = watch::channel(PriceSnapshot::default());
    let (grid_tx, grid_rx) = watch::channel(None);

    let worker = PriceWorker {
        tick_rx,
        snapshot_tx,
        grid_tx,
        cache: PriceCache::new(max_completed_bars),
        grid_lookback_bars: grid_lookback_bars.max(1),
    };
    let join_handle = tokio::spawn(worker.run());

    (
        PriceRuntime {
            tick_tx,
            snapshot_rx,
            grid_rx,
        },
        join_handle,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{Duration, timeout};

    #[tokio::test(flavor = "current_thread")]
    async fn pipeline_emits_grid_snapshots_over_watch_channel() {
        let (runtime, _task) = spawn_price_pipeline(8, 4, 8);
        runtime
            .tick_tx
            .send(PriceTick {
                ts_ms: 1_000,
                price: 100.0,
            })
            .await
            .unwrap();
        runtime
            .tick_tx
            .send(PriceTick {
                ts_ms: 2_000,
                price: 101.0,
            })
            .await
            .unwrap();

        let mut grid_rx = runtime.grid_rx.clone();
        timeout(Duration::from_secs(1), grid_rx.changed())
            .await
            .unwrap()
            .unwrap();
        let snapshot = grid_rx.borrow().clone().unwrap();

        assert_eq!(snapshot.spot, 101.0);
        assert_eq!(snapshot.history.len(), 2);
    }
}
