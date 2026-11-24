use crate::grid::OhlcBar;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriceTick {
    pub ts_ms: u64,
    pub price: f64,
    pub mid_px: Option<f64>,
    pub oracle_px: Option<f64>,
}

impl PriceTick {
    pub fn new(ts_ms: u64, price: f64) -> Self {
        Self {
            ts_ms,
            price,
            mid_px: None,
            oracle_px: None,
        }
    }

    pub fn to_price_tick(self) -> PriceTick {
        self
    }
}

/// Alias kept for offline Hyperliquid `activeAssetCtx` tick consumers (`mid_px`, `oracle_px`
/// now live directly as optional fields on `PriceTick`).
pub type HlTick = PriceTick;

#[derive(Debug, Clone, PartialEq)]
pub struct PriceSnapshot {
    pub latest_tick: Option<PriceTick>,
    pub latest_server_ts_ms: Option<u64>,
    pub completed_bars: Vec<OhlcBar>,
    pub current_bar: Option<OhlcBar>,
}

impl Default for PriceSnapshot {
    fn default() -> Self {
        Self {
            latest_tick: None,
            latest_server_ts_ms: None,
            completed_bars: Vec::new(),
            current_bar: None,
        }
    }
}

impl PriceSnapshot {
    pub fn latest_price(&self) -> Option<f64> {
        self.latest_tick.map(|tick| tick.price)
    }

    pub fn bars_for_grid(&self, lookback_bars: usize) -> Option<GridMarketSnapshot> {
        let spot = self.latest_price()?;
        let mut history = self.completed_bars.clone();
        if let Some(current_bar) = self.current_bar {
            history.push(current_bar);
        }
        if history.is_empty() {
            return None;
        }
        let keep_from = history.len().saturating_sub(lookback_bars.max(1));
        let history = history.split_off(keep_from);

        Some(GridMarketSnapshot {
            spot,
            server_ts_ms: self.latest_server_ts_ms?,
            history,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridMarketSnapshot {
    pub spot: f64,
    pub server_ts_ms: u64,
    pub history: Vec<OhlcBar>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamHealth {
    pub has_price: bool,
    pub latest_tick_ts_ms: Option<u64>,
    pub completed_bar_count: usize,
}

impl Default for StreamHealth {
    fn default() -> Self {
        Self {
            has_price: false,
            latest_tick_ts_ms: None,
            completed_bar_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriceError {
    NonPositivePrice,
    OutOfOrderTick,
}
