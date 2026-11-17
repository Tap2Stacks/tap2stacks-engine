use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::grid::OhlcBar;

use super::types::{GridMarketSnapshot, PriceError, PriceSnapshot, PriceTick, StreamHealth};

#[derive(Debug, Clone)]
pub struct PriceCache {
    max_completed_bars: usize,
    latest_tick: Option<PriceTick>,
    completed_bars: VecDeque<(u64, OhlcBar)>,
    current_second: Option<u64>,
    current_bar: Option<OhlcBar>,
}

impl PriceCache {
    pub fn new(max_completed_bars: usize) -> Self {
        Self {
            max_completed_bars: max_completed_bars.max(1),
            latest_tick: None,
            completed_bars: VecDeque::with_capacity(max_completed_bars.max(1)),
            current_second: None,
            current_bar: None,
        }
    }

    pub fn apply_tick(&mut self, tick: PriceTick) -> Result<(), PriceError> {
        if tick.price <= 0.0 {
            return Err(PriceError::NonPositivePrice);
        }
        if let Some(latest) = self.latest_tick {
            if tick.ts_ms < latest.ts_ms {
                return Err(PriceError::OutOfOrderTick);
            }
        }

        let second = tick.ts_ms / 1_000;
        match self.current_second {
            None => {
                self.current_second = Some(second);
                self.current_bar = Some(new_bar(tick.price));
            }
            Some(current_second) if second == current_second => {
                if let Some(bar) = &mut self.current_bar {
                    apply_price(bar, tick.price);
                }
            }
            Some(current_second) if second > current_second => {
                self.roll_current_bar(current_second);

                let last_close = self
                    .completed_bars
                    .back()
                    .map(|(_, bar)| bar.close)
                    .or_else(|| self.latest_tick.map(|latest| latest.price))
                    .unwrap_or(tick.price);

                for missing_second in (current_second + 1)..second {
                    self.push_completed_bar(missing_second, flat_bar(last_close));
                }

                self.current_second = Some(second);
                self.current_bar = Some(new_bar(tick.price));
            }
            Some(_) => return Err(PriceError::OutOfOrderTick),
        }

        self.latest_tick = Some(tick);
        Ok(())
    }

    pub fn snapshot(&self) -> PriceSnapshot {
        PriceSnapshot {
            latest_tick: self.latest_tick,
            latest_server_ts_ms: Some(now_ms()),
            completed_bars: self.completed_bars.iter().map(|(_, bar)| *bar).collect(),
            current_bar: self.current_bar,
        }
    }

    pub fn grid_snapshot(&self, lookback_bars: usize) -> Option<GridMarketSnapshot> {
        self.snapshot().bars_for_grid(lookback_bars)
    }

    pub fn health(&self) -> StreamHealth {
        StreamHealth {
            has_price: self.latest_tick.is_some(),
            latest_tick_ts_ms: self.latest_tick.map(|tick| tick.ts_ms),
            completed_bar_count: self.completed_bars.len(),
        }
    }

    fn roll_current_bar(&mut self, second: u64) {
        if let Some(bar) = self.current_bar.take() {
            self.push_completed_bar(second, bar);
        }
    }

    fn push_completed_bar(&mut self, second: u64, bar: OhlcBar) {
        if self.completed_bars.len() == self.max_completed_bars {
            self.completed_bars.pop_front();
        }
        self.completed_bars.push_back((second, bar));
    }
}

fn new_bar(price: f64) -> OhlcBar {
    OhlcBar {
        open: price,
        high: price,
        low: price,
        close: price,
    }
}

fn flat_bar(price: f64) -> OhlcBar {
    new_bar(price)
}

fn apply_price(bar: &mut OhlcBar, price: f64) {
    bar.high = bar.high.max(price);
    bar.low = bar.low.min(price);
    bar.close = price;
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_ticks_into_one_second_ohlc() {
        let mut cache = PriceCache::new(8);
        cache.apply_tick(PriceTick { ts_ms: 1_000, price: 100.0 }).unwrap();
        cache.apply_tick(PriceTick { ts_ms: 1_200, price: 101.0 }).unwrap();
        cache.apply_tick(PriceTick { ts_ms: 1_800, price: 99.5 }).unwrap();
        cache.apply_tick(PriceTick { ts_ms: 2_000, price: 102.0 }).unwrap();

        let snapshot = cache.snapshot();
        assert_eq!(snapshot.completed_bars.len(), 1);
        assert_eq!(
            snapshot.completed_bars[0],
            OhlcBar {
                open: 100.0,
                high: 101.0,
                low: 99.5,
                close: 99.5,
            }
        );
        assert_eq!(
            snapshot.current_bar.unwrap(),
            OhlcBar {
                open: 102.0,
                high: 102.0,
                low: 102.0,
                close: 102.0,
            }
        );
    }

    #[test]
    fn fills_missing_seconds_with_flat_bars() {
        let mut cache = PriceCache::new(8);
        cache.apply_tick(PriceTick { ts_ms: 1_000, price: 100.0 }).unwrap();
        cache.apply_tick(PriceTick { ts_ms: 4_000, price: 104.0 }).unwrap();

        let snapshot = cache.snapshot();
        assert_eq!(snapshot.completed_bars.len(), 3);
        assert_eq!(snapshot.completed_bars[1].close, 100.0);
        assert_eq!(snapshot.completed_bars[2].close, 100.0);
    }

    #[test]
    fn produces_grid_snapshot_with_current_bar_included() {
        let mut cache = PriceCache::new(8);
        cache.apply_tick(PriceTick { ts_ms: 1_000, price: 100.0 }).unwrap();
        cache.apply_tick(PriceTick { ts_ms: 2_000, price: 101.0 }).unwrap();

        let grid = cache.grid_snapshot(2).unwrap();
        assert_eq!(grid.spot, 101.0);
        assert_eq!(grid.history.len(), 2);
        assert_eq!(grid.history[0].close, 100.0);
        assert_eq!(grid.history[1].close, 101.0);
    }
}
