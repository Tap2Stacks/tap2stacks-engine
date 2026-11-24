use std::time::Duration;

use crate::grid::OhlcBar;

/// Hyperliquid trade.xyz oracle update period (seconds).
pub const HL_CLOCK_SECS: u64 = 3;
/// Hyperliquid settlement void gap threshold (milliseconds).
pub const HL_VOID_GAP_MS: u64 = 3_000;
/// Hyperliquid feed staleness threshold (seconds).
pub const HL_STALE_AFTER_SECS: u64 = 3;

/// Deribit perp-index update period (seconds).
pub const DERIBIT_CLOCK_SECS: u64 = 1;
/// PROVISIONAL Deribit settlement void gap threshold (milliseconds); lead tunes from recorded data.
pub const DERIBIT_VOID_GAP_MS: u64 = 3_000;
/// PROVISIONAL Deribit feed staleness threshold (seconds); lead tunes from recorded data.
pub const DERIBIT_STALE_AFTER_SECS: u64 = 3;

/// Timing parameters for a given upstream price source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceTiming {
    pub clock_secs: u64,
    pub void_gap_ms: u64,
    pub stale_after: Duration,
}

/// Selects the upstream market-data price source (`hyperliquid` default, or `deribit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PriceSource {
    #[default]
    Hyperliquid,
    Deribit,
}

impl PriceSource {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "hyperliquid" => Some(Self::Hyperliquid),
            "deribit" => Some(Self::Deribit),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hyperliquid => "hyperliquid",
            Self::Deribit => "deribit",
        }
    }

    pub const fn timing(self) -> SourceTiming {
        match self {
            Self::Hyperliquid => SourceTiming {
                clock_secs: HL_CLOCK_SECS,
                void_gap_ms: HL_VOID_GAP_MS,
                stale_after: Duration::from_secs(HL_STALE_AFTER_SECS),
            },
            Self::Deribit => SourceTiming {
                clock_secs: DERIBIT_CLOCK_SECS,
                void_gap_ms: DERIBIT_VOID_GAP_MS,
                stale_after: Duration::from_secs(DERIBIT_STALE_AFTER_SECS),
            },
        }
    }
}

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
