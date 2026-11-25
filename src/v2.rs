//! Odds Model v2: 3-second-clock zero-inflated cent-jump process, exact print-crossing
//! 1-D Dynamic Programming touch probabilities, causal trailing-60s RMS regime detector,
//! far-horizon Markov regime switching, and QTAP pricing without calibration layers.
//!
//! All ticker preset parameters below are direct MLEs from the 7-day recorded tick
//! statistics in `logs/calibration/m1-tick-stats.txt` (2026-11-17..2026-11-23).
//! Tickers without sufficient recorded tick data (AAPL, SP500) return `None` from
//! `v2_ticker_params` and stay on model v1.

use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::grid::{compute_k_skew, GridConfig, PriceBand, TimeWindow};
use crate::horizon::NEAR_WINDOWS;
use crate::price::{PriceSource, PriceTick};

/// Maximum age (seconds) of the last observed non-zero price change for phase inference.
pub const PHASE_MAX_AGE_SECS: u64 = 60;
/// Rolling window (seconds) for the causal realized RMS detector `R_60`.
pub const DETECTOR_WINDOW_SECS: u64 = 60;
/// Minimum valid 1-second steps in `R_60` before leaving cold-start `Normal`.
pub const DETECTOR_MIN_STEPS: usize = 10;
/// Consecutive seconds below `HYSTERESIS_FACTOR * threshold` required to step down one regime.
pub const DOWNGRADE_HOLD_SECS: u64 = 15;
/// Hysteresis multiplier on lower thresholds when stepping down a regime.
pub const HYSTERESIS_FACTOR: f64 = 0.85;
/// Floor for QTAP house margin in v2 (`docs/odds-model-v2.md` §3.3).
pub const V2_M_BASE: f64 = 0.025;
/// Dimensionless slope on `(R_60 / R_ref - 1)+` for `m_risk` (`docs/odds-model-v2.md` §3).
pub const V2_RISK_SLOPE: f64 = 0.005;
/// Cap on `m_risk` added on top of each preset's `m_base` (`docs/odds-model-v2.md` §3.3).
pub const V2_M_RISK_MAX: f64 = 0.025;
/// Pooled dimensionless 2-Geom core weight across all 170 NVDA+TSLA news mins
/// (`m1-tick-stats.txt` lines 84 & 212: `(0.8150 + 0.7712) / 2 = 0.793`).
pub const V2_NEWS_MIX_WEIGHT: f64 = 0.793;
/// Pooled dimensionless 2-Geom tail-to-core mean ratio `rho = m2 / m1` across all 170
/// NVDA+TSLA news mins (`m1-tick-stats.txt` lines 84 & 212: `(2.664 + 2.171) / 2 = 2.42`).
pub const V2_NEWS_TAIL_RATIO: f64 = 2.42;
/// Lower clamp for causal trailing 60s move rate `q_60` in `News` (`docs/odds-model-v2.md` §3.4).
pub const V2_NEWS_Q_MIN: f64 = 0.12;
/// Upper RMS clamp multiplier on `R_med_news` in `dynamic_news_preset` (`docs/odds-model-v2.md` §3.4).
pub const V2_NEWS_R_CAP_MULT: f64 = 2.5;
/// Minimum probability denominator clamp in `M_base = (1 - m_eff) / max(P, eps)`.
pub const V2_P_EPSILON: f64 = 1e-6;
/// PROVISIONAL Deribit order-book lead time `L` in seconds (`docs/odds-model-v2-coinbase-1s.md` §4).
pub const DERIBIT_BOOK_LEAD_SECS: u64 = 3;
/// PROVISIONAL Deribit book-gap catch-up slope `beta` on `(mid_px - oracle_px)` (`docs/odds-model-v2-coinbase-1s.md` §4).
pub const DERIBIT_BOOK_GAP_BETA: f64 = 1.0;
/// PROVISIONAL Deribit NVDA symmetric book-gap clamp `|G|` in cents (`docs/odds-model-v2-coinbase-1s.md` §4).
pub const DERIBIT_NVDA_BOOK_GAP_CLAMP_CENTS: i64 = 20;
/// PROVISIONAL Deribit TSLA symmetric book-gap clamp `|G|` in cents (`docs/odds-model-v2-coinbase-1s.md` §4).
pub const DERIBIT_TSLA_BOOK_GAP_CLAMP_CENTS: i64 = 40;
/// PROVISIONAL Deribit maximum age (ms) of perp book/mark reading before falling back to `G = 0`.
pub const DERIBIT_BOOK_MAX_AGE_MS: u64 = 2_000;
/// Half-width of the integer-cent DP lattice (`[-450c, +450c]`, `901` states).
const DP_HALF_WIDTH: usize = 450;

/// Selects which odds model `QuoteEngine` runs. Default is `V1` until the owner approves v2/m4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PricingModel {
    #[default]
    V1,
    V2,
    M4,
}

impl PricingModel {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "v1" | "1" => Some(Self::V1),
            "v2" | "2" => Some(Self::V2),
            "m4" | "4" => Some(Self::M4),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
            Self::M4 => "m4",
        }
    }
}

/// Four data-derived market regimes partitioned at empirical `p50 / p85 / p99` of `R_60`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Regime {
    Quiet = 0,
    Normal = 1,
    Volatile = 2,
    News = 3,
}

impl Regime {
    pub const ALL: [Regime; 4] = [
        Regime::Quiet,
        Regime::Normal,
        Regime::Volatile,
        Regime::News,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Quiet,
            1 => Self::Normal,
            2 => Self::Volatile,
            _ => Self::News,
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "quiet" => Some(Self::Quiet),
            "normal" => Some(Self::Normal),
            "volatile" => Some(Self::Volatile),
            "news" => Some(Self::News),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quiet => "quiet",
            Self::Normal => "normal",
            Self::Volatile => "volatile",
            Self::News => "news",
        }
    }

    pub fn step_down(self) -> Self {
        match self {
            Self::Quiet => Self::Quiet,
            Self::Normal => Self::Quiet,
            Self::Volatile => Self::Normal,
            Self::News => Self::Volatile,
        }
    }
}

/// Fitted zero-inflated cent-jump parameters for one regime preset.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegimePreset {
    /// Empirical unconditional 1-second move probability `q = P(ΔC != 0)`.
    pub q_1s: f64,
    /// Per-clock-tick move probability `p_move = min(1, clock_secs * q_1s)`.
    pub p_move: f64,
    /// Weight on component 1 (`1.0` for single-Geometric regimes; `< 1.0` for `News`).
    pub mix_weight: f64,
    /// Geometric parameter `p1` on `{1, 2, ...}` (`mean = 1 / p1` cents).
    pub p1: f64,
    /// Tail Geometric parameter `p2` (equals `p1` when `mix_weight == 1.0`).
    pub p2: f64,
    /// Median minute-level `R_60` (cents/s) observed in this regime.
    pub rms_median_cents: f64,
    /// Data-derived per-preset base uncertainty margin `max(0.025, Q_0.75(m_solv) + 0.020)`
    /// (`docs/odds-model-v2.md` §3.3, derived from `logs/calibration/m2-server-logs/m2-summary.txt` §1).
    pub m_base: f64,
}

/// Complete data-derived v2 parameters for one ticker on a specific `PriceSource`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickerV2Params {
    pub ticker: &'static str,
    pub band_cents: u32,
    pub reference_price: f64,
    /// Source update clock period in seconds (Hyperliquid: 3, Deribit: 1).
    pub clock_secs: u64,
    /// Source settlement void gap threshold in milliseconds.
    pub void_gap_ms: u64,
    /// Order-book lead time `L` in seconds (Hyperliquid: `0`, Deribit: `3`).
    pub book_lead_secs: u64,
    /// Book-gap catch-up slope `beta` on `(mid_px - oracle_px)` (Hyperliquid: `0.0`, Deribit: `1.0`).
    pub book_gap_beta: f64,
    /// Symmetric clamp on `G` in integer cents (`|G| <= book_gap_clamp_cents`).
    pub book_gap_clamp_cents: i64,
    /// Maximum age (ms) of perp book/mark reading before falling back to `G = 0`.
    pub book_max_age_ms: u64,
    /// Median `R_60` of the `Normal` regime (used as `R_ref` in `m_risk`).
    pub rms_ref_cents: f64,
    /// Empirical `[p50, p85, p99]` minute `R_60` thresholds in cents/s.
    pub thresholds_rms_cents: [f64; 3],
    /// Single-step shock guard `ceil(4 * p99)` in integer cents (`docs/odds-model-v2.md` §3.4).
    pub shock_single_step_cents: i64,
    /// Ordered `[Quiet, Normal, Volatile, News]`. Drives regime detection, `m_base`,
    /// `rms` medians, and pre-window evolution steps.
    pub presets: [RegimePreset; 4],
    /// Optional separate step presets (`q_1s`, `p_move`, `mix_weight`, `p1`, `p2`) used
    /// strictly INSIDE each window (`None` = use `presets`).
    pub window_presets: Option<[RegimePreset; 4]>,
    /// Empirical 4x4 1-minute regime transition matrix `T[from][to]`, row-stochastic.
    pub transition_1min: [[f64; 4]; 4],
    /// Optional separate table used outside NYSE regular trading hours (`None` = use main table).
    /// An off-hours table's own `off_hours` is always `None`.
    pub off_hours: Option<&'static TickerV2Params>,
}

/// NVDA v2 parameters derived directly from `logs/calibration/m1-tick-stats.txt` and M2 accuracy
/// (`logs/calibration/m2-server-logs/m2-summary.txt` §1):
/// - `reference_price = 139.65` (`m1-tick-stats.txt` line 6)
/// - `thresholds_rms_cents = [1.162, 2.412, 8.390]` (`m1-tick-stats.txt` line 57)
/// - `shock_single_step_cents = ceil(4 * 8.390) = 34` (`docs/odds-model-v2.md` §3.4)
/// - `Quiet`: `m1` lines 59-62 (`q=0.1426`, `Geom p=0.6517`, `med_rms=0.58c`); `m_base=0.025` (`m2-summary.txt` line 7, `Q_0.75=0.00%`)
/// - `Normal`: `m1` lines 66-69 (`q=0.2602`, `Geom p=0.3767`, `med_rms=1.60c`); `m_base=0.098` (`m2-summary.txt` line 10, `7.82%+2.0%`)
/// - `Volatile`: `m1` lines 73-76 (`q=0.3017`, `Geom p=0.1850`, `med_rms=3.36c`); `m_base=0.134` (`m2-summary.txt` line 13, `11.39%+2.0%`)
/// - `News`: `m1` lines 80-84 (`q=0.2949`, `2-Geom w=0.8150, p1=0.0643, p2=0.0242`, `med_rms=11.79c`); `m_base=0.263` (`m2-summary.txt` line 16, `24.31%+2.0%`)
/// - `transition_1min`: `m1` lines 88-91 (Normal diagonal rounded `+0.001` so every row sums to `1.0`).
pub static NVDA_V2_PARAMS: TickerV2Params = TickerV2Params {
    ticker: "NVDA",
    band_cents: 3,
    reference_price: 139.65,
    clock_secs: PriceSource::Hyperliquid.timing().clock_secs,
    void_gap_ms: PriceSource::Hyperliquid.timing().void_gap_ms,
    book_lead_secs: 0,
    book_gap_beta: 0.0,
    book_gap_clamp_cents: 0,
    book_max_age_ms: 0,
    rms_ref_cents: 1.60,
    thresholds_rms_cents: [1.162, 2.412, 8.390],
    shock_single_step_cents: 34,
    presets: [
        RegimePreset {
            q_1s: 0.1426,
            p_move: 0.4278,
            mix_weight: 1.0,
            p1: 0.6517,
            p2: 0.6517,
            rms_median_cents: 0.58,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2602,
            p_move: 0.7806,
            mix_weight: 1.0,
            p1: 0.3767,
            p2: 0.3767,
            rms_median_cents: 1.60,
            m_base: 0.098,
        },
        RegimePreset {
            q_1s: 0.3017,
            p_move: 0.9051,
            mix_weight: 1.0,
            p1: 0.1850,
            p2: 0.1850,
            rms_median_cents: 3.36,
            m_base: 0.134,
        },
        RegimePreset {
            q_1s: 0.2949,
            p_move: 0.8847,
            mix_weight: 0.8150,
            p1: 0.0643,
            p2: 0.0242,
            rms_median_cents: 11.79,
            m_base: 0.263,
        },
    ],
    window_presets: None,
    transition_1min: [
        [0.816, 0.171, 0.013, 0.000],
        [0.252, 0.645, 0.102, 0.001],
        [0.029, 0.273, 0.674, 0.024],
        [0.000, 0.012, 0.353, 0.635],
    ],
    off_hours: None,
};

/// TSLA v2 parameters derived directly from `logs/calibration/m1-tick-stats.txt` and M2 accuracy
/// (`logs/calibration/m2-server-logs/m2-summary.txt` §1):
/// - `reference_price = 363.97` (`m1-tick-stats.txt` line 134)
/// - `thresholds_rms_cents = [1.807, 4.359, 11.505]` (`m1-tick-stats.txt` line 185)
/// - `shock_single_step_cents = ceil(4 * 11.505) = 46` (`docs/odds-model-v2.md` §3.4)
/// - `Quiet`: `m1` lines 187-190 (`q=0.1108`, `Geom p=0.4643`, `med_rms=0.63c`); `m_base=0.025` (`m2-summary.txt` line 41, `0.42%+2.0% -> floor`)
/// - `Normal`: `m1` lines 194-197 (`q=0.2391`, `Geom p=0.2202`, `med_rms=2.77c`); `m_base=0.073` (`m2-summary.txt` line 44, `5.28%+2.0%`)
/// - `Volatile`: `m1` lines 201-204 (`q=0.2900`, `Geom p=0.1183`, `med_rms=5.63c`); `m_base=0.197` (`m2-summary.txt` line 47, `17.69%+2.0%`)
/// - `News`: `m1` lines 208-212 (`q=0.3027`, `2-Geom w=0.7712, p1=0.0645, p2=0.0297`, `med_rms=14.20c`); `m_base=0.185` (`m2-summary.txt` line 50, `16.47%+2.0%`)
/// - `transition_1min`: `m1` lines 216-219 (Quiet diagonal rounded `+0.001` so every row sums to `1.0`).
pub static TSLA_V2_PARAMS: TickerV2Params = TickerV2Params {
    ticker: "TSLA",
    band_cents: 8,
    reference_price: 363.97,
    clock_secs: PriceSource::Hyperliquid.timing().clock_secs,
    void_gap_ms: PriceSource::Hyperliquid.timing().void_gap_ms,
    book_lead_secs: 0,
    book_gap_beta: 0.0,
    book_gap_clamp_cents: 0,
    book_max_age_ms: 0,
    rms_ref_cents: 2.77,
    thresholds_rms_cents: [1.807, 4.359, 11.505],
    shock_single_step_cents: 46,
    presets: [
        RegimePreset {
            q_1s: 0.1108,
            p_move: 0.3324,
            mix_weight: 1.0,
            p1: 0.4643,
            p2: 0.4643,
            rms_median_cents: 0.63,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2391,
            p_move: 0.7173,
            mix_weight: 1.0,
            p1: 0.2202,
            p2: 0.2202,
            rms_median_cents: 2.77,
            m_base: 0.073,
        },
        RegimePreset {
            q_1s: 0.2900,
            p_move: 0.8700,
            mix_weight: 1.0,
            p1: 0.1183,
            p2: 0.1183,
            rms_median_cents: 5.63,
            m_base: 0.197,
        },
        RegimePreset {
            q_1s: 0.3027,
            p_move: 0.9081,
            mix_weight: 0.7712,
            p1: 0.0645,
            p2: 0.0297,
            rms_median_cents: 14.20,
            m_base: 0.185,
        },
    ],
    window_presets: None,
    transition_1min: [
        [0.825, 0.160, 0.015, 0.000],
        [0.236, 0.620, 0.140, 0.004],
        [0.041, 0.366, 0.567, 0.026],
        [0.012, 0.094, 0.412, 0.482],
    ],
    off_hours: None,
};

/// PROVISIONAL NVDA Deribit off-hours v2 parameters: exact copy of `NVDA_DERIBIT_V2_PARAMS`
/// (with `off_hours: None`) until lead fits off-hours values on recorded Deribit data.
pub static NVDA_DERIBIT_V2_PARAMS_OFF: TickerV2Params = TickerV2Params {
    ticker: "NVDA",
    band_cents: 3,
    reference_price: 139.65,
    clock_secs: PriceSource::Deribit.timing().clock_secs,
    void_gap_ms: PriceSource::Deribit.timing().void_gap_ms,
    book_lead_secs: DERIBIT_BOOK_LEAD_SECS,
    book_gap_beta: DERIBIT_BOOK_GAP_BETA,
    book_gap_clamp_cents: DERIBIT_NVDA_BOOK_GAP_CLAMP_CENTS,
    book_max_age_ms: DERIBIT_BOOK_MAX_AGE_MS,
    rms_ref_cents: 1.60,
    thresholds_rms_cents: [1.162, 2.412, 8.390],
    shock_single_step_cents: 34,
    presets: [
        RegimePreset {
            q_1s: 0.1426,
            p_move: 0.1426,
            mix_weight: 1.0,
            p1: 0.6517,
            p2: 0.6517,
            rms_median_cents: 0.58,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2602,
            p_move: 0.2602,
            mix_weight: 1.0,
            p1: 0.3767,
            p2: 0.3767,
            rms_median_cents: 1.60,
            m_base: 0.098,
        },
        RegimePreset {
            q_1s: 0.3017,
            p_move: 0.3017,
            mix_weight: 1.0,
            p1: 0.1850,
            p2: 0.1850,
            rms_median_cents: 3.36,
            m_base: 0.134,
        },
        RegimePreset {
            q_1s: 0.2949,
            p_move: 0.2949,
            mix_weight: 0.8150,
            p1: 0.0643,
            p2: 0.0242,
            rms_median_cents: 11.79,
            m_base: 0.263,
        },
    ],
    window_presets: Some([
        RegimePreset {
            q_1s: 0.1426,
            p_move: 0.1426,
            mix_weight: 1.0,
            p1: 0.6517,
            p2: 0.6517,
            rms_median_cents: 0.58,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2602,
            p_move: 0.2602,
            mix_weight: 1.0,
            p1: 0.3767,
            p2: 0.3767,
            rms_median_cents: 1.60,
            m_base: 0.098,
        },
        RegimePreset {
            q_1s: 0.3017,
            p_move: 0.3017,
            mix_weight: 1.0,
            p1: 0.1850,
            p2: 0.1850,
            rms_median_cents: 3.36,
            m_base: 0.134,
        },
        RegimePreset {
            q_1s: 0.2949,
            p_move: 0.2949,
            mix_weight: 0.8150,
            p1: 0.0643,
            p2: 0.0242,
            rms_median_cents: 11.79,
            m_base: 0.263,
        },
    ]),
    transition_1min: [
        [0.816, 0.171, 0.013, 0.000],
        [0.252, 0.645, 0.102, 0.001],
        [0.029, 0.273, 0.674, 0.024],
        [0.000, 0.012, 0.353, 0.635],
    ],
    off_hours: None,
};

/// PROVISIONAL NVDA Deribit v2 parameters: copies Hyperliquid NVDA preset values
/// (with `p_move = q_1s` for `clock_secs = 1`, PROVISIONAL `window_presets = Some(presets)`,
/// and PROVISIONAL `off_hours = Some(&NVDA_DERIBIT_V2_PARAMS_OFF)`) until lead fits presets
/// on recorded Deribit data.
pub static NVDA_DERIBIT_V2_PARAMS: TickerV2Params = TickerV2Params {
    ticker: "NVDA",
    band_cents: 3,
    reference_price: 139.65,
    clock_secs: PriceSource::Deribit.timing().clock_secs,
    void_gap_ms: PriceSource::Deribit.timing().void_gap_ms,
    book_lead_secs: DERIBIT_BOOK_LEAD_SECS,
    book_gap_beta: DERIBIT_BOOK_GAP_BETA,
    book_gap_clamp_cents: DERIBIT_NVDA_BOOK_GAP_CLAMP_CENTS,
    book_max_age_ms: DERIBIT_BOOK_MAX_AGE_MS,
    rms_ref_cents: 1.60,
    thresholds_rms_cents: [1.162, 2.412, 8.390],
    shock_single_step_cents: 34,
    presets: [
        RegimePreset {
            q_1s: 0.1426,
            p_move: 0.1426,
            mix_weight: 1.0,
            p1: 0.6517,
            p2: 0.6517,
            rms_median_cents: 0.58,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2602,
            p_move: 0.2602,
            mix_weight: 1.0,
            p1: 0.3767,
            p2: 0.3767,
            rms_median_cents: 1.60,
            m_base: 0.098,
        },
        RegimePreset {
            q_1s: 0.3017,
            p_move: 0.3017,
            mix_weight: 1.0,
            p1: 0.1850,
            p2: 0.1850,
            rms_median_cents: 3.36,
            m_base: 0.134,
        },
        RegimePreset {
            q_1s: 0.2949,
            p_move: 0.2949,
            mix_weight: 0.8150,
            p1: 0.0643,
            p2: 0.0242,
            rms_median_cents: 11.79,
            m_base: 0.263,
        },
    ],
    window_presets: Some([
        RegimePreset {
            q_1s: 0.1426,
            p_move: 0.1426,
            mix_weight: 1.0,
            p1: 0.6517,
            p2: 0.6517,
            rms_median_cents: 0.58,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2602,
            p_move: 0.2602,
            mix_weight: 1.0,
            p1: 0.3767,
            p2: 0.3767,
            rms_median_cents: 1.60,
            m_base: 0.098,
        },
        RegimePreset {
            q_1s: 0.3017,
            p_move: 0.3017,
            mix_weight: 1.0,
            p1: 0.1850,
            p2: 0.1850,
            rms_median_cents: 3.36,
            m_base: 0.134,
        },
        RegimePreset {
            q_1s: 0.2949,
            p_move: 0.2949,
            mix_weight: 0.8150,
            p1: 0.0643,
            p2: 0.0242,
            rms_median_cents: 11.79,
            m_base: 0.263,
        },
    ]),
    transition_1min: [
        [0.816, 0.171, 0.013, 0.000],
        [0.252, 0.645, 0.102, 0.001],
        [0.029, 0.273, 0.674, 0.024],
        [0.000, 0.012, 0.353, 0.635],
    ],
    off_hours: Some(&NVDA_DERIBIT_V2_PARAMS_OFF),
};

/// PROVISIONAL TSLA Deribit off-hours v2 parameters: exact copy of `TSLA_DERIBIT_V2_PARAMS`
/// (with `off_hours: None`) until lead fits off-hours values on recorded Deribit data.
pub static TSLA_DERIBIT_V2_PARAMS_OFF: TickerV2Params = TickerV2Params {
    ticker: "TSLA",
    band_cents: 8,
    reference_price: 363.97,
    clock_secs: PriceSource::Deribit.timing().clock_secs,
    void_gap_ms: PriceSource::Deribit.timing().void_gap_ms,
    book_lead_secs: DERIBIT_BOOK_LEAD_SECS,
    book_gap_beta: DERIBIT_BOOK_GAP_BETA,
    book_gap_clamp_cents: DERIBIT_TSLA_BOOK_GAP_CLAMP_CENTS,
    book_max_age_ms: DERIBIT_BOOK_MAX_AGE_MS,
    rms_ref_cents: 2.77,
    thresholds_rms_cents: [1.807, 4.359, 11.505],
    shock_single_step_cents: 46,
    presets: [
        RegimePreset {
            q_1s: 0.1108,
            p_move: 0.1108,
            mix_weight: 1.0,
            p1: 0.4643,
            p2: 0.4643,
            rms_median_cents: 0.63,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2391,
            p_move: 0.2391,
            mix_weight: 1.0,
            p1: 0.2202,
            p2: 0.2202,
            rms_median_cents: 2.77,
            m_base: 0.073,
        },
        RegimePreset {
            q_1s: 0.2900,
            p_move: 0.2900,
            mix_weight: 1.0,
            p1: 0.1183,
            p2: 0.1183,
            rms_median_cents: 5.63,
            m_base: 0.197,
        },
        RegimePreset {
            q_1s: 0.3027,
            p_move: 0.3027,
            mix_weight: 0.7712,
            p1: 0.0645,
            p2: 0.0297,
            rms_median_cents: 14.20,
            m_base: 0.185,
        },
    ],
    window_presets: Some([
        RegimePreset {
            q_1s: 0.1108,
            p_move: 0.1108,
            mix_weight: 1.0,
            p1: 0.4643,
            p2: 0.4643,
            rms_median_cents: 0.63,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2391,
            p_move: 0.2391,
            mix_weight: 1.0,
            p1: 0.2202,
            p2: 0.2202,
            rms_median_cents: 2.77,
            m_base: 0.073,
        },
        RegimePreset {
            q_1s: 0.2900,
            p_move: 0.2900,
            mix_weight: 1.0,
            p1: 0.1183,
            p2: 0.1183,
            rms_median_cents: 5.63,
            m_base: 0.197,
        },
        RegimePreset {
            q_1s: 0.3027,
            p_move: 0.3027,
            mix_weight: 0.7712,
            p1: 0.0645,
            p2: 0.0297,
            rms_median_cents: 14.20,
            m_base: 0.185,
        },
    ]),
    transition_1min: [
        [0.825, 0.160, 0.015, 0.000],
        [0.236, 0.620, 0.140, 0.004],
        [0.041, 0.366, 0.567, 0.026],
        [0.012, 0.094, 0.412, 0.482],
    ],
    off_hours: None,
};

/// PROVISIONAL TSLA Deribit v2 parameters: copies Hyperliquid TSLA preset values
/// (with `p_move = q_1s` for `clock_secs = 1`, PROVISIONAL `window_presets = Some(presets)`,
/// and PROVISIONAL `off_hours = Some(&TSLA_DERIBIT_V2_PARAMS_OFF)`) until lead fits presets
/// on recorded Deribit data.
pub static TSLA_DERIBIT_V2_PARAMS: TickerV2Params = TickerV2Params {
    ticker: "TSLA",
    band_cents: 8,
    reference_price: 363.97,
    clock_secs: PriceSource::Deribit.timing().clock_secs,
    void_gap_ms: PriceSource::Deribit.timing().void_gap_ms,
    book_lead_secs: DERIBIT_BOOK_LEAD_SECS,
    book_gap_beta: DERIBIT_BOOK_GAP_BETA,
    book_gap_clamp_cents: DERIBIT_TSLA_BOOK_GAP_CLAMP_CENTS,
    book_max_age_ms: DERIBIT_BOOK_MAX_AGE_MS,
    rms_ref_cents: 2.77,
    thresholds_rms_cents: [1.807, 4.359, 11.505],
    shock_single_step_cents: 46,
    presets: [
        RegimePreset {
            q_1s: 0.1108,
            p_move: 0.1108,
            mix_weight: 1.0,
            p1: 0.4643,
            p2: 0.4643,
            rms_median_cents: 0.63,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2391,
            p_move: 0.2391,
            mix_weight: 1.0,
            p1: 0.2202,
            p2: 0.2202,
            rms_median_cents: 2.77,
            m_base: 0.073,
        },
        RegimePreset {
            q_1s: 0.2900,
            p_move: 0.2900,
            mix_weight: 1.0,
            p1: 0.1183,
            p2: 0.1183,
            rms_median_cents: 5.63,
            m_base: 0.197,
        },
        RegimePreset {
            q_1s: 0.3027,
            p_move: 0.3027,
            mix_weight: 0.7712,
            p1: 0.0645,
            p2: 0.0297,
            rms_median_cents: 14.20,
            m_base: 0.185,
        },
    ],
    window_presets: Some([
        RegimePreset {
            q_1s: 0.1108,
            p_move: 0.1108,
            mix_weight: 1.0,
            p1: 0.4643,
            p2: 0.4643,
            rms_median_cents: 0.63,
            m_base: 0.025,
        },
        RegimePreset {
            q_1s: 0.2391,
            p_move: 0.2391,
            mix_weight: 1.0,
            p1: 0.2202,
            p2: 0.2202,
            rms_median_cents: 2.77,
            m_base: 0.073,
        },
        RegimePreset {
            q_1s: 0.2900,
            p_move: 0.2900,
            mix_weight: 1.0,
            p1: 0.1183,
            p2: 0.1183,
            rms_median_cents: 5.63,
            m_base: 0.197,
        },
        RegimePreset {
            q_1s: 0.3027,
            p_move: 0.3027,
            mix_weight: 0.7712,
            p1: 0.0645,
            p2: 0.0297,
            rms_median_cents: 14.20,
            m_base: 0.185,
        },
    ]),
    transition_1min: [
        [0.825, 0.160, 0.015, 0.000],
        [0.236, 0.620, 0.140, 0.004],
        [0.041, 0.366, 0.567, 0.026],
        [0.012, 0.094, 0.412, 0.482],
    ],
    off_hours: Some(&TSLA_DERIBIT_V2_PARAMS_OFF),
};

/// Normalizes a ticker, Hyperliquid coin (`xyz:NVDA`), Deribit index (`nvda_usdc`), or
/// Deribit channel (`deribit_price_index.nvda_usdc`) to uppercase base symbol (`NVDA`).
pub fn normalize_v2_ticker(ticker: &str) -> String {
    let trimmed = ticker.trim();
    let stripped = trimmed
        .strip_prefix("deribit_price_index.")
        .unwrap_or(trimmed);
    let base = stripped.split(':').next_back().unwrap_or(stripped);
    let lower = base.to_ascii_lowercase();
    let sym = lower.strip_suffix("_usdc").unwrap_or(&lower);
    let sym = sym.strip_suffix("-perpetual").unwrap_or(sym);
    sym.to_ascii_uppercase()
}

/// Owned JSON representation of `TickerV2Params` used by offline tools
/// (`--v2-params-file` / `--dump-v2-params`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickerV2ParamsJson {
    pub ticker: String,
    pub band_cents: u32,
    pub reference_price: f64,
    pub clock_secs: u64,
    pub void_gap_ms: u64,
    pub book_lead_secs: u64,
    pub book_gap_beta: f64,
    pub book_gap_clamp_cents: i64,
    pub book_max_age_ms: u64,
    pub rms_ref_cents: f64,
    pub thresholds_rms_cents: [f64; 3],
    pub shock_single_step_cents: i64,
    pub presets: [RegimePreset; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_presets: Option<[RegimePreset; 4]>,
    pub transition_1min: [[f64; 4]; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub off_hours: Option<Box<TickerV2ParamsJson>>,
}

fn validate_preset_array(label: &str, presets: &[RegimePreset; 4]) -> Result<(), String> {
    for (idx, preset) in presets.iter().enumerate() {
        let name = Regime::from_index(idx).as_str();
        for (field, val) in [
            ("q_1s", preset.q_1s),
            ("p_move", preset.p_move),
            ("mix_weight", preset.mix_weight),
        ] {
            if !val.is_finite() || !(0.0..=1.0).contains(&val) {
                return Err(format!(
                    "v2 params validation error: {label}[{idx}] ({name}).{field} ({val}) must be in [0, 1]"
                ));
            }
        }
        for (field, val) in [("p1", preset.p1), ("p2", preset.p2)] {
            if !val.is_finite() || val <= 0.0 || val > 1.0 {
                return Err(format!(
                    "v2 params validation error: {label}[{idx}] ({name}).{field} ({val}) must be in (0, 1]"
                ));
            }
        }
        if !preset.rms_median_cents.is_finite() || preset.rms_median_cents <= 0.0 {
            return Err(format!(
                "v2 params validation error: {label}[{idx}] ({name}).rms_median_cents ({}) must be finite and > 0",
                preset.rms_median_cents
            ));
        }
        if !preset.m_base.is_finite() || !(0.0..1.0).contains(&preset.m_base) {
            return Err(format!(
                "v2 params validation error: {label}[{idx}] ({name}).m_base ({}) must be in [0, 1)",
                preset.m_base
            ));
        }
    }
    Ok(())
}

impl TickerV2ParamsJson {
    pub fn from_params(params: &TickerV2Params) -> Self {
        Self {
            ticker: params.ticker.to_string(),
            band_cents: params.band_cents,
            reference_price: params.reference_price,
            clock_secs: params.clock_secs,
            void_gap_ms: params.void_gap_ms,
            book_lead_secs: params.book_lead_secs,
            book_gap_beta: params.book_gap_beta,
            book_gap_clamp_cents: params.book_gap_clamp_cents,
            book_max_age_ms: params.book_max_age_ms,
            rms_ref_cents: params.rms_ref_cents,
            thresholds_rms_cents: params.thresholds_rms_cents,
            shock_single_step_cents: params.shock_single_step_cents,
            presets: params.presets,
            window_presets: params.window_presets,
            transition_1min: params.transition_1min,
            off_hours: params.off_hours.map(|off| Box::new(Self::from_params(off))),
        }
    }

    pub fn matches_coin(&self, coin_or_ticker: &str) -> bool {
        normalize_v2_ticker(&self.ticker) == normalize_v2_ticker(coin_or_ticker)
    }

    /// Validates all parameter bounds and stochastic constraints, failing fast with a clear error.
    pub fn validate(&self) -> Result<(), String> {
        if self.ticker.trim().is_empty() {
            return Err("v2 params validation error: ticker must not be empty".into());
        }
        if self.band_cents == 0 {
            return Err("v2 params validation error: band_cents must be >= 1".into());
        }
        if !self.reference_price.is_finite() || self.reference_price <= 0.0 {
            return Err(format!(
                "v2 params validation error: reference_price ({}) must be finite and > 0",
                self.reference_price
            ));
        }
        if self.clock_secs < 1 {
            return Err(format!(
                "v2 params validation error: clock_secs ({}) must be >= 1",
                self.clock_secs
            ));
        }
        if self.void_gap_ms == 0 {
            return Err("v2 params validation error: void_gap_ms must be >= 1".into());
        }
        if !self.book_gap_beta.is_finite() {
            return Err(format!(
                "v2 params validation error: book_gap_beta ({}) must be finite",
                self.book_gap_beta
            ));
        }
        if self.book_gap_clamp_cents < 0 {
            return Err(format!(
                "v2 params validation error: book_gap_clamp_cents ({}) must be >= 0",
                self.book_gap_clamp_cents
            ));
        }
        if !self.rms_ref_cents.is_finite() || self.rms_ref_cents <= 0.0 {
            return Err(format!(
                "v2 params validation error: rms_ref_cents ({}) must be finite and > 0",
                self.rms_ref_cents
            ));
        }
        if self.shock_single_step_cents <= 0 {
            return Err(format!(
                "v2 params validation error: shock_single_step_cents ({}) must be > 0",
                self.shock_single_step_cents
            ));
        }

        let [t0, t1, t2] = self.thresholds_rms_cents;
        for (idx, t) in [t0, t1, t2].iter().enumerate() {
            if !t.is_finite() || *t <= 0.0 {
                return Err(format!(
                    "v2 params validation error: thresholds_rms_cents[{idx}] ({t}) must be finite and > 0"
                ));
            }
        }
        if !(t0 < t1 && t1 < t2) {
            return Err(format!(
                "v2 params validation error: thresholds_rms_cents ([{t0}, {t1}, {t2}]) must be strictly increasing"
            ));
        }

        validate_preset_array("presets", &self.presets)?;
        if let Some(ref window_presets) = self.window_presets {
            validate_preset_array("window_presets", window_presets)?;
        }

        for (r, row) in self.transition_1min.iter().enumerate() {
            let mut sum = 0.0_f64;
            for (c, &v) in row.iter().enumerate() {
                if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                    return Err(format!(
                        "v2 params validation error: transition_1min[{r}][{c}] ({v}) must be in [0, 1]"
                    ));
                }
                sum += v;
            }
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!(
                    "v2 params validation error: transition_1min[{r}] row sum ({sum:.9}) must equal 1 within 1e-6"
                ));
            }
        }

        if let Some(ref off) = self.off_hours {
            if off.off_hours.is_some() {
                return Err(
                    "v2 params validation error: off_hours must not contain its own off_hours".into(),
                );
            }
            if !off.matches_coin(&self.ticker) {
                return Err(format!(
                    "v2 params validation error: off_hours.ticker ('{}') must match ticker ('{}')",
                    off.ticker, self.ticker
                ));
            }
            off.validate()
                .map_err(|err| format!("off_hours: {err}"))?;
        }

        Ok(())
    }

    /// Converts validated owned JSON params into a leaked `'static` reference for offline tools.
    pub fn leak_static(self) -> &'static TickerV2Params {
        let off_hours: Option<&'static TickerV2Params> =
            self.off_hours.map(|off| (*off).leak_static());
        let ticker: &'static str = Box::leak(self.ticker.into_boxed_str());
        Box::leak(Box::new(TickerV2Params {
            ticker,
            band_cents: self.band_cents,
            reference_price: self.reference_price,
            clock_secs: self.clock_secs,
            void_gap_ms: self.void_gap_ms,
            book_lead_secs: self.book_lead_secs,
            book_gap_beta: self.book_gap_beta,
            book_gap_clamp_cents: self.book_gap_clamp_cents,
            book_max_age_ms: self.book_max_age_ms,
            rms_ref_cents: self.rms_ref_cents,
            thresholds_rms_cents: self.thresholds_rms_cents,
            shock_single_step_cents: self.shock_single_step_cents,
            presets: self.presets,
            window_presets: self.window_presets,
            transition_1min: self.transition_1min,
            off_hours,
        }))
    }
}

impl TickerV2Params {
    /// Returns the active table for `quote_ts_ms`: `off_hours` when configured and
    /// `quote_ts_ms` is outside NYSE regular trading hours, otherwise `self`.
    pub fn for_session<'a>(&'a self, quote_ts_ms: u64) -> &'a TickerV2Params {
        if let Some(off) = self.off_hours {
            if !crate::session::is_us_regular_session(quote_ts_ms) {
                return off;
            }
        }
        self
    }

    /// Serializes this parameter table as pretty JSON matching `TickerV2ParamsJson`.
    pub fn to_json_pretty(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&TickerV2ParamsJson::from_params(self))
            .map_err(|err| format!("failed to serialize v2 params: {err}"))
    }

    /// Formats this parameter table as a paste-ready Rust
    /// `pub const <TICKER>_DERIBIT_V2_PARAMS: TickerV2Params = ...;` block matching `src/v2.rs`.
    /// When `off_hours` is present, emits `<TICKER>_DERIBIT_V2_PARAMS_OFF` first and references it
    /// from the main const.
    pub fn to_rust_deribit_const(&self) -> String {
        fn push_presets(out: &mut String, presets: &[RegimePreset; 4]) {
            for p in presets {
                out.push_str("        RegimePreset {\n");
                out.push_str(&format!("            q_1s: {:?},\n", p.q_1s));
                out.push_str(&format!("            p_move: {:?},\n", p.p_move));
                out.push_str(&format!("            mix_weight: {:?},\n", p.mix_weight));
                out.push_str(&format!("            p1: {:?},\n", p.p1));
                out.push_str(&format!("            p2: {:?},\n", p.p2));
                out.push_str(&format!(
                    "            rms_median_cents: {:?},\n",
                    p.rms_median_cents
                ));
                out.push_str(&format!("            m_base: {:?},\n", p.m_base));
                out.push_str("        },\n");
            }
        }

        fn format_const(
            out: &mut String,
            const_name: &str,
            params: &TickerV2Params,
            off_const_ref: Option<&str>,
        ) {
            let sym = normalize_v2_ticker(params.ticker);
            out.push_str(&format!(
                "pub const {const_name}: TickerV2Params = TickerV2Params {{\n"
            ));
            out.push_str(&format!("    ticker: {:?},\n", sym));
            out.push_str(&format!("    band_cents: {},\n", params.band_cents));
            out.push_str(&format!(
                "    reference_price: {:?},\n",
                params.reference_price
            ));
            out.push_str(&format!("    clock_secs: {},\n", params.clock_secs));
            out.push_str(&format!("    void_gap_ms: {},\n", params.void_gap_ms));
            out.push_str(&format!("    book_lead_secs: {},\n", params.book_lead_secs));
            out.push_str(&format!(
                "    book_gap_beta: {:?},\n",
                params.book_gap_beta
            ));
            out.push_str(&format!(
                "    book_gap_clamp_cents: {},\n",
                params.book_gap_clamp_cents
            ));
            out.push_str(&format!(
                "    book_max_age_ms: {},\n",
                params.book_max_age_ms
            ));
            out.push_str(&format!("    rms_ref_cents: {:?},\n", params.rms_ref_cents));
            out.push_str(&format!(
                "    thresholds_rms_cents: [{:?}, {:?}, {:?}],\n",
                params.thresholds_rms_cents[0],
                params.thresholds_rms_cents[1],
                params.thresholds_rms_cents[2]
            ));
            out.push_str(&format!(
                "    shock_single_step_cents: {},\n",
                params.shock_single_step_cents
            ));
            out.push_str("    presets: [\n");
            push_presets(out, &params.presets);
            out.push_str("    ],\n");
            match &params.window_presets {
                Some(wp) => {
                    out.push_str("    window_presets: Some([\n");
                    push_presets(out, wp);
                    out.push_str("    ]),\n");
                }
                None => {
                    out.push_str("    window_presets: None,\n");
                }
            }
            out.push_str("    transition_1min: [\n");
            for row in &params.transition_1min {
                out.push_str(&format!(
                    "        [{:?}, {:?}, {:?}, {:?}],\n",
                    row[0], row[1], row[2], row[3]
                ));
            }
            out.push_str("    ],\n");
            match off_const_ref {
                Some(off_name) => {
                    out.push_str(&format!("    off_hours: Some(&{off_name}),\n"));
                }
                None => {
                    out.push_str("    off_hours: None,\n");
                }
            }
            out.push_str("};\n");
        }

        let sym = normalize_v2_ticker(self.ticker);
        let main_name = format!("{sym}_DERIBIT_V2_PARAMS");
        let mut out = String::new();
        if let Some(off) = self.off_hours {
            let off_name = format!("{sym}_DERIBIT_V2_PARAMS_OFF");
            format_const(&mut out, &off_name, off, None);
            out.push('\n');
            format_const(&mut out, &main_name, self, Some(&off_name));
        } else {
            format_const(&mut out, &main_name, self, None);
        }
        out
    }

    /// Compact one-line summary for offline tool startup logs.
    pub fn one_line_summary(&self) -> String {
        format!(
            "ticker={} band_cents={} clock={}s void_gap={}ms book_lead={}s beta={:.3} clamp={}c book_max_age={}ms rms_ref={:.3}c thresh=[{:.3},{:.3},{:.3}] shock={}c two_phase={} off_hours={} m_base=[{:.3},{:.3},{:.3},{:.3}]",
            self.ticker,
            self.band_cents,
            self.clock_secs,
            self.void_gap_ms,
            self.book_lead_secs,
            self.book_gap_beta,
            self.book_gap_clamp_cents,
            self.book_max_age_ms,
            self.rms_ref_cents,
            self.thresholds_rms_cents[0],
            self.thresholds_rms_cents[1],
            self.thresholds_rms_cents[2],
            self.shock_single_step_cents,
            self.window_presets.is_some(),
            self.off_hours.is_some(),
            self.presets[0].m_base,
            self.presets[1].m_base,
            self.presets[2].m_base,
            self.presets[3].m_base,
        )
    }
}

/// Parses and validates a JSON string representing `TickerV2ParamsJson`.
pub fn parse_v2_params_json(raw: &str) -> Result<TickerV2ParamsJson, String> {
    let parsed: TickerV2ParamsJson =
        serde_json::from_str(raw).map_err(|err| format!("invalid v2 params JSON: {err}"))?;
    parsed.validate()?;
    Ok(parsed)
}

/// Loads, parses, and validates a `TickerV2ParamsJson` file from disk.
pub fn load_v2_params_file(path: &Path) -> Result<TickerV2ParamsJson, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading v2 params file {}: {err}", path.display()))?;
    parse_v2_params_json(&raw)
        .map_err(|err| format!("{} ({})", err, path.display()))
}

/// Returns source-specific v2 parameters for tickers with configured tables (`NVDA`, `TSLA`).
/// Tickers without v2 parameters (`AAPL`, `SP500`) return `None` and stay on model v1.
pub fn v2_ticker_params_for_source(
    ticker: &str,
    source: PriceSource,
) -> Option<&'static TickerV2Params> {
    match (source, normalize_v2_ticker(ticker).as_str()) {
        (PriceSource::Hyperliquid, "NVDA") => Some(&NVDA_V2_PARAMS),
        (PriceSource::Hyperliquid, "TSLA") => Some(&TSLA_V2_PARAMS),
        (PriceSource::Deribit, "NVDA") => Some(&NVDA_DERIBIT_V2_PARAMS),
        (PriceSource::Deribit, "TSLA") => Some(&TSLA_DERIBIT_V2_PARAMS),
        _ => None,
    }
}

/// Returns Hyperliquid v2 parameters for tickers with recorded history (`NVDA`, `TSLA`).
/// Tickers without sufficient tick data (`AAPL`, `SP500`) return `None` and stay on model v1.
pub fn v2_ticker_params(ticker: &str) -> Option<&'static TickerV2Params> {
    v2_ticker_params_for_source(ticker, PriceSource::Hyperliquid)
}

/// Strictly causal online regime detector and 3-second oracle phase tracker.
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    steps: VecDeque<(u64, i64)>,
    last_tick_ts_ms: Option<u64>,
    last_price_cents: Option<i64>,
    last_change_sec: Option<u64>,
    current_regime: Regime,
    downgrade_streak_secs: u64,
    last_r60: f64,
    last_q60: f64,
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl RegimeDetector {
    pub fn new() -> Self {
        Self {
            steps: VecDeque::with_capacity(DETECTOR_WINDOW_SECS as usize + 4),
            last_tick_ts_ms: None,
            last_price_cents: None,
            last_change_sec: None,
            current_regime: Regime::Normal,
            downgrade_streak_secs: 0,
            last_r60: 0.0,
            last_q60: 0.0,
        }
    }

    pub fn current_regime(&self) -> Regime {
        self.current_regime
    }

    pub fn last_r60(&self) -> f64 {
        self.last_r60
    }

    /// Trailing 60-second causal 1-second move rate `q_60 = (# non-zero steps) / (# valid steps)`.
    pub fn last_q60(&self) -> f64 {
        self.last_q60
    }

    pub fn last_change_sec(&self) -> Option<u64> {
        self.last_change_sec
    }

    /// Infers clock phase `phi in 0..clock_secs` relative to `quote_ts_ms` from the last
    /// observed non-zero price change.
    ///
    /// At `clock_secs <= 1` (e.g. Deribit 1-second index), phase is always `Some(0)`.
    /// At `clock_secs > 1` (e.g. Hyperliquid 3-second oracle), returns `None` if unknown
    /// (restart, gap > `void_gap_ms`, or > 60s since last move), causing the DP to average
    /// uniformly over `0..clock_secs`.
    pub fn inferred_phase(&self, quote_ts_ms: u64, clock_secs: u64) -> Option<usize> {
        let clock = clock_secs.max(1);
        if clock == 1 {
            return Some(0);
        }
        let t_last = self.last_change_sec?;
        let t_now = quote_ts_ms / 1_000;
        if t_now < t_last || t_now.saturating_sub(t_last) > PHASE_MAX_AGE_SECS {
            return None;
        }
        let phi = ((t_last % clock) + clock - (t_now % clock)) % clock;
        Some(phi as usize)
    }

    /// Feeds one raw tick causally into the detector. Idempotent for repeated/out-of-order
    /// timestamps (`tick.ts_ms <= last_tick_ts_ms`).
    pub fn on_tick(&mut self, tick: PriceTick, params: &TickerV2Params) {
        if !tick.price.is_finite() || tick.price <= 0.0 {
            return;
        }
        let curr_cents = (tick.price * 100.0).round() as i64;
        let curr_sec = tick.ts_ms / 1_000;

        let Some(prev_ts_ms) = self.last_tick_ts_ms else {
            self.last_tick_ts_ms = Some(tick.ts_ms);
            self.last_price_cents = Some(curr_cents);
            self.last_r60 = params.rms_ref_cents;
            self.last_q60 = params.presets[Regime::Normal.index()].q_1s;
            return;
        };

        if tick.ts_ms <= prev_ts_ms {
            return;
        }

        let dt_ms = tick.ts_ms.saturating_sub(prev_ts_ms);
        let prev_cents = self.last_price_cents.unwrap_or(curr_cents);
        let prev_sec = prev_ts_ms / 1_000;

        self.last_tick_ts_ms = Some(tick.ts_ms);
        self.last_price_cents = Some(curr_cents);

        // A gap exceeding the source's `void_gap_ms` voids overlapping windows, invalidates
        // phase tracking, and must NOT be counted as a 1-second price jump in R_60.
        if dt_ms > params.void_gap_ms {
            self.last_change_sec = None;
            self.downgrade_streak_secs = 0;
            return;
        }

        let dc = curr_cents - prev_cents;
        if dc != 0 {
            self.last_change_sec = Some(curr_sec);
        }

        // Maintain 1-second resolution steps matching `m1-tick-stats.py` (fill intermediate
        // silent seconds for 2-3s gaps; coalesce sub-second ticks into the same second).
        if curr_sec == prev_sec {
            if let Some(back) = self.steps.back_mut() {
                if back.0 == curr_sec {
                    back.1 += dc;
                } else {
                    self.steps.push_back((curr_sec, dc));
                }
            } else {
                self.steps.push_back((curr_sec, dc));
            }
        } else {
            for mid in (prev_sec + 1)..curr_sec {
                self.steps.push_back((mid, 0));
            }
            self.steps.push_back((curr_sec, dc));
        }

        let cutoff_sec = curr_sec.saturating_sub(DETECTOR_WINDOW_SECS);
        while self
            .steps
            .front()
            .is_some_and(|&(sec, _)| sec <= cutoff_sec)
        {
            self.steps.pop_front();
        }

        let n = self.steps.len();
        if n > 0 {
            let mut sum_sq = 0.0_f64;
            let mut nz_count = 0usize;
            for &(_, step_dc) in &self.steps {
                if step_dc != 0 {
                    nz_count += 1;
                    let v = step_dc as f64;
                    sum_sq += v * v;
                }
            }
            self.last_r60 = (sum_sq / n as f64).sqrt();
            self.last_q60 = nz_count as f64 / n as f64;
        } else {
            self.last_r60 = params.rms_ref_cents;
            self.last_q60 = params.presets[Regime::Normal.index()].q_1s;
        }

        // Single-print shock guard: promote straight to News on print 1.
        if dc.abs() > params.shock_single_step_cents {
            self.current_regime = Regime::News;
            self.downgrade_streak_secs = 0;
            return;
        }

        if n < DETECTOR_MIN_STEPS {
            return;
        }

        let raw_regime = classify_r60(self.last_r60, &params.thresholds_rms_cents);
        if raw_regime > self.current_regime {
            self.current_regime = raw_regime;
            self.downgrade_streak_secs = 0;
            return;
        }

        if self.current_regime > Regime::Quiet {
            let lower_thresh = lower_threshold(self.current_regime, &params.thresholds_rms_cents);
            if self.last_r60 < HYSTERESIS_FACTOR * lower_thresh {
                let elapsed = curr_sec.saturating_sub(prev_sec);
                self.downgrade_streak_secs = self.downgrade_streak_secs.saturating_add(elapsed);
                if self.downgrade_streak_secs >= DOWNGRADE_HOLD_SECS {
                    self.current_regime = self.current_regime.step_down();
                    self.downgrade_streak_secs = 0;
                }
            } else {
                self.downgrade_streak_secs = 0;
            }
        } else {
            self.downgrade_streak_secs = 0;
        }
    }
}

pub fn classify_r60(r60: f64, thresholds: &[f64; 3]) -> Regime {
    if r60 <= thresholds[0] {
        Regime::Quiet
    } else if r60 <= thresholds[1] {
        Regime::Normal
    } else if r60 <= thresholds[2] {
        Regime::Volatile
    } else {
        Regime::News
    }
}

fn lower_threshold(regime: Regime, thresholds: &[f64; 3]) -> f64 {
    match regime {
        Regime::Quiet => 0.0,
        Regime::Normal => thresholds[0],
        Regime::Volatile => thresholds[1],
        Regime::News => thresholds[2],
    }
}

/// Builds a causally scaled `News` preset from trailing `R_60` and `q_60` using the pooled
/// dimensionless 2-Geom shape `(w = 0.793, rho = m2 / m1 = 2.42)` across all 170 NVDA+TSLA
/// news minutes (`docs/odds-model-v2.md` §3.4).
///
/// Solves the exact second-moment equation `q_eff * E[K^2] = R_eff^2` for `m1 = 1 / p1`
/// and `m2 = rho * m1 = 1 / p2`, with `q_eff = clamp(q_60, 0.12, 1/3)` and
/// `R_eff = clamp(R_60, theta_p99, 2.5 * R_med_news)`.
pub fn dynamic_news_preset(params: &TickerV2Params, r60: f64, q60: f64) -> RegimePreset {
    let base_news = params.presets[Regime::News.index()];
    let r_clean = if r60.is_finite() {
        r60
    } else {
        base_news.rms_median_cents
    };
    let q_clean = if q60.is_finite() { q60 } else { base_news.q_1s };

    let clock = params.clock_secs.max(1) as f64;
    let q_max = 1.0 / clock;
    let r_min = params.thresholds_rms_cents[2];
    let r_max = (V2_NEWS_R_CAP_MULT * base_news.rms_median_cents).max(r_min);
    let r_eff = r_clean.clamp(r_min, r_max);
    let q_eff = q_clean.clamp(V2_NEWS_Q_MIN, q_max);
    let p_move = (clock * q_eff).min(1.0);

    let w = V2_NEWS_MIX_WEIGHT;
    let rho = V2_NEWS_TAIL_RATIO;
    let a = w + (1.0 - w) * rho * rho;
    let b = w + (1.0 - w) * rho;
    let e_k2 = (r_eff * r_eff) / q_eff;
    let m1 = (b + (b * b + 8.0 * a * e_k2).sqrt()) / (4.0 * a);

    let p1 = (1.0 / m1.max(1.0)).clamp(1e-4, 1.0 - 1e-6);
    let p2 = (1.0 / (rho * m1).max(1.0)).clamp(1e-4, 1.0 - 1e-6);

    RegimePreset {
        q_1s: q_eff,
        p_move,
        mix_weight: w,
        p1,
        p2,
        rms_median_cents: r_eff,
        m_base: base_news.m_base,
    }
}

/// Computes v2 effective margin `m_eff = m_base + m_risk`, where `m_base` is the preset's
/// data-derived uncertainty margin (`docs/odds-model-v2.md` §3.3) and `m_risk in [0%, 2.5%]`
/// is the QTAP volatility risk premium.
pub fn v2_m_eff(r60: f64, rms_ref: f64, m_base: f64) -> f64 {
    let ratio_excess = (r60 / rms_ref.max(1e-6) - 1.0).max(0.0);
    let m_risk = (V2_RISK_SLOPE * ratio_excess).clamp(0.0, V2_M_RISK_MAX);
    (m_base + m_risk).clamp(V2_M_BASE, 0.95)
}

/// Computes v2 cell multiplier directly from `P(c)` without any calibration layers.
pub fn v2_price_multiplier(
    hit_probability: f64,
    m_eff: f64,
    liability: f64,
    k_safety: f64,
    config: &GridConfig,
) -> f64 {
    let bounded_p = hit_probability.clamp(V2_P_EPSILON, 1.0);
    let base = (1.0 - m_eff) / bounded_p;
    let k_skew = compute_k_skew(liability, config);
    (base * k_skew * k_safety).clamp(config.pricing.min_multiplier, config.pricing.max_multiplier)
}

/// Exact `O(S)` Geometric IIR full-grid propagation step with exterior mass clamped at
/// endpoints `0` and `n - 1` (preserving total probability mass `1.0`).
fn full_step_clamped(u: &[f64], preset: &RegimePreset, out: &mut [f64]) {
    let n = u.len();
    debug_assert_eq!(out.len(), n);
    let stay = 1.0 - preset.p_move;
    for i in 0..n {
        out[i] = stay * u[i];
    }
    let comps: [(f64, f64); 2] = if preset.mix_weight >= 1.0 {
        [(1.0, preset.p1), (0.0, preset.p1)]
    } else {
        [(preset.mix_weight, preset.p1), (1.0 - preset.mix_weight, preset.p2)]
    };
    for &(cw, pg) in &comps {
        if cw <= 0.0 {
            continue;
        }
        let r = 1.0 - pg;
        let half = 0.5 * preset.p_move * cw;
        let mut acc = 0.0_f64;
        for y in 0..n {
            out[y] += half * acc;
            acc = pg * u[y] + r * acc;
        }
        out[n - 1] += half * (acc / pg);

        let mut acc_l = 0.0_f64;
        for y in (0..n).rev() {
            out[y] += half * acc_l;
            acc_l = pg * u[y] + r * acc_l;
        }
        out[0] += half * (acc_l / pg);
    }
}

/// Exact survival probability after 1 oracle tick strictly below `l_idx` and strictly
/// above `u_idx` (any jump entering or crossing `[l_idx, u_idx]` is absorbed as a hit).
fn survive_1tick(u: &[f64], l_idx: usize, u_idx: usize, preset: &RegimePreset) -> f64 {
    let n = u.len();
    let comps: [(f64, f64); 2] = if preset.mix_weight >= 1.0 {
        [(1.0, preset.p1), (0.0, preset.p1)]
    } else {
        [(preset.mix_weight, preset.p1), (1.0 - preset.mix_weight, preset.p2)]
    };

    let mut surv = 0.0_f64;
    if l_idx > 0 {
        let mut mass_b = 0.0_f64;
        for &v in &u[..l_idx] {
            mass_b += v;
        }
        let mut cross_b = 0.0_f64;
        for &(cw, pg) in &comps {
            if cw <= 0.0 {
                continue;
            }
            let r = 1.0 - pg;
            let mut acc = 0.0_f64;
            for &v in &u[..l_idx] {
                acc = v + r * acc;
            }
            cross_b += 0.5 * preset.p_move * cw * acc;
        }
        surv += (mass_b - cross_b).max(0.0);
    }

    if u_idx + 1 < n {
        let mut mass_a = 0.0_f64;
        for &v in &u[(u_idx + 1)..n] {
            mass_a += v;
        }
        let mut cross_a = 0.0_f64;
        for &(cw, pg) in &comps {
            if cw <= 0.0 {
                continue;
            }
            let r = 1.0 - pg;
            let mut acc = 0.0_f64;
            for &v in u[(u_idx + 1)..n].iter().rev() {
                acc = v + r * acc;
            }
            cross_a += 0.5 * preset.p_move * cw * acc;
        }
        surv += (mass_a - cross_a).max(0.0);
    }
    surv
}

/// Exact 1-tick survival mass on `[0..l_idx)` starting from `u[..l_idx]` (interior jumps
/// `>= l_idx` dropped as hits; exterior jumps `< 0` clamped at `0`), followed by 1 more
/// survival tick via closed-form `survive_below_only`.
fn survive_2ticks_below(
    u: &[f64],
    l_idx: usize,
    preset: &RegimePreset,
    scratch: &mut [f64],
) -> f64 {
    if l_idx == 0 {
        return 0.0;
    }
    let b1 = &mut scratch[..l_idx];
    let stay = 1.0 - preset.p_move;
    for i in 0..l_idx {
        b1[i] = stay * u[i];
    }
    let comps: [(f64, f64); 2] = if preset.mix_weight >= 1.0 {
        [(1.0, preset.p1), (0.0, preset.p1)]
    } else {
        [(preset.mix_weight, preset.p1), (1.0 - preset.mix_weight, preset.p2)]
    };
    for &(cw, pg) in &comps {
        if cw <= 0.0 {
            continue;
        }
        let r = 1.0 - pg;
        let half = 0.5 * preset.p_move * cw;
        let mut acc = 0.0_f64;
        for y in 0..l_idx {
            b1[y] += half * acc;
            acc = pg * u[y] + r * acc;
        }
        let mut acc_l = 0.0_f64;
        for y in (0..l_idx).rev() {
            b1[y] += half * acc_l;
            acc_l = pg * u[y] + r * acc_l;
        }
        b1[0] += half * (acc_l / pg);
    }
    // Second tick survival mass strictly < l_idx
    let mut mass = 0.0_f64;
    for &v in b1.iter() {
        mass += v;
    }
    let mut cross = 0.0_f64;
    for &(cw, pg) in &comps {
        if cw <= 0.0 {
            continue;
        }
        let r = 1.0 - pg;
        let mut acc = 0.0_f64;
        for &v in b1.iter() {
            acc = v + r * acc;
        }
        cross += 0.5 * preset.p_move * cw * acc;
    }
    (mass - cross).max(0.0)
}

fn survive_2ticks_above(
    u: &[f64],
    u_idx: usize,
    preset: &RegimePreset,
    scratch: &mut [f64],
) -> f64 {
    let n = u.len();
    if u_idx + 1 >= n {
        return 0.0;
    }
    let off = u_idx + 1;
    let m = n - off;
    let a1 = &mut scratch[..m];
    let stay = 1.0 - preset.p_move;
    for j in 0..m {
        a1[j] = stay * u[off + j];
    }
    let comps: [(f64, f64); 2] = if preset.mix_weight >= 1.0 {
        [(1.0, preset.p1), (0.0, preset.p1)]
    } else {
        [(preset.mix_weight, preset.p1), (1.0 - preset.mix_weight, preset.p2)]
    };
    for &(cw, pg) in &comps {
        if cw <= 0.0 {
            continue;
        }
        let r = 1.0 - pg;
        let half = 0.5 * preset.p_move * cw;
        let mut acc = 0.0_f64;
        for j in 0..m {
            a1[j] += half * acc;
            acc = pg * u[off + j] + r * acc;
        }
        a1[m - 1] += half * (acc / pg);

        let mut acc_l = 0.0_f64;
        for j in (0..m).rev() {
            a1[j] += half * acc_l;
            acc_l = pg * u[off + j] + r * acc_l;
        }
    }
    let mut mass = 0.0_f64;
    for &v in a1.iter() {
        mass += v;
    }
    let mut cross = 0.0_f64;
    for &(cw, pg) in &comps {
        if cw <= 0.0 {
            continue;
        }
        let r = 1.0 - pg;
        let mut acc = 0.0_f64;
        for &v in a1.iter().rev() {
            acc = v + r * acc;
        }
        cross += 0.5 * preset.p_move * cw * acc;
    }
    (mass - cross).max(0.0)
}

fn window_survival_mass(
    u: &[f64],
    l_idx: usize,
    u_idx: usize,
    in_ticks: usize,
    preset: &RegimePreset,
    scratch: &mut [f64],
) -> f64 {
    let n = u.len();
    match in_ticks {
        0 => {
            let mut surv = 0.0_f64;
            for &v in &u[..l_idx] {
                surv += v;
            }
            if u_idx + 1 < n {
                for &v in &u[(u_idx + 1)..n] {
                    surv += v;
                }
            }
            surv
        }
        1 => survive_1tick(u, l_idx, u_idx, preset),
        2 => {
            survive_2ticks_below(u, l_idx, preset, scratch)
                + survive_2ticks_above(u, u_idx, preset, scratch)
        }
        _ => {
            // General path for windows with >= 3 ticks (e.g. 5s window at 1s clock).
            let mut cur_below = u[..l_idx].to_vec();
            let mut cur_above = if u_idx + 1 < n {
                u[(u_idx + 1)..n].to_vec()
            } else {
                Vec::new()
            };
            let mut next_b = vec![0.0_f64; l_idx];
            let mut next_a = vec![0.0_f64; cur_above.len()];
            for _ in 0..in_ticks {
                if l_idx > 0 {
                    step_below_general(&cur_below, preset, &mut next_b);
                    std::mem::swap(&mut cur_below, &mut next_b);
                }
                if !cur_above.is_empty() {
                    step_above_general(&cur_above, preset, &mut next_a);
                    std::mem::swap(&mut cur_above, &mut next_a);
                }
            }
            cur_below.iter().sum::<f64>() + cur_above.iter().sum::<f64>()
        }
    }
}

fn step_below_general(src: &[f64], preset: &RegimePreset, dst: &mut [f64]) {
    let m = src.len();
    let stay = 1.0 - preset.p_move;
    for i in 0..m {
        dst[i] = stay * src[i];
    }
    let comps: [(f64, f64); 2] = if preset.mix_weight >= 1.0 {
        [(1.0, preset.p1), (0.0, preset.p1)]
    } else {
        [(preset.mix_weight, preset.p1), (1.0 - preset.mix_weight, preset.p2)]
    };
    for &(cw, pg) in &comps {
        if cw <= 0.0 {
            continue;
        }
        let r = 1.0 - pg;
        let half = 0.5 * preset.p_move * cw;
        let mut acc = 0.0_f64;
        for y in 0..m {
            dst[y] += half * acc;
            acc = pg * src[y] + r * acc;
        }
        let mut acc_l = 0.0_f64;
        for y in (0..m).rev() {
            dst[y] += half * acc_l;
            acc_l = pg * src[y] + r * acc_l;
        }
        dst[0] += half * (acc_l / pg);
    }
}

fn step_above_general(src: &[f64], preset: &RegimePreset, dst: &mut [f64]) {
    let m = src.len();
    let stay = 1.0 - preset.p_move;
    for i in 0..m {
        dst[i] = stay * src[i];
    }
    let comps: [(f64, f64); 2] = if preset.mix_weight >= 1.0 {
        [(1.0, preset.p1), (0.0, preset.p1)]
    } else {
        [(preset.mix_weight, preset.p1), (1.0 - preset.mix_weight, preset.p2)]
    };
    for &(cw, pg) in &comps {
        if cw <= 0.0 {
            continue;
        }
        let r = 1.0 - pg;
        let half = 0.5 * preset.p_move * cw;
        let mut acc = 0.0_f64;
        for y in 0..m {
            dst[y] += half * acc;
            acc = pg * src[y] + r * acc;
        }
        dst[m - 1] += half * (acc / pg);
        let mut acc_l = 0.0_f64;
        for y in (0..m).rev() {
            dst[y] += half * acc_l;
            acc_l = pg * src[y] + r * acc_l;
        }
    }
}

/// Effective book-lead DP shift `(G, L)` applied for a single quote tick
/// (`docs/odds-model-v2-coinbase-1s.md` §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BookLeadAdjustment {
    /// Clamped integer-cent initial shift `G = clamp(round(beta * (mid - mark) * 100), ±clamp)`.
    pub gap_cents: i64,
    /// Pre-window horizon shortening `L` in seconds (0 when book is absent/stale or HL).
    pub lead_secs: u64,
}

/// Computes the book-lead adjustment `(G, L)` from the latest perp book mid (`mid_px`) and
/// perp mark (`oracle_px`). Returns `(0, 0)` whenever `beta == 0`, either quote is missing or
/// invalid, or `book_age_ms` exceeds `params.book_max_age_ms`.
pub fn compute_book_lead(
    mid_px: Option<f64>,
    oracle_px: Option<f64>,
    book_age_ms: Option<u64>,
    params: &TickerV2Params,
) -> BookLeadAdjustment {
    if params.book_gap_beta == 0.0
        || params.book_gap_clamp_cents <= 0
        || book_age_ms.is_some_and(|age| age > params.book_max_age_ms)
    {
        return BookLeadAdjustment::default();
    }
    let (Some(mid), Some(mark)) = (mid_px, oracle_px) else {
        return BookLeadAdjustment::default();
    };
    if !mid.is_finite() || !mark.is_finite() || mid <= 0.0 || mark <= 0.0 {
        return BookLeadAdjustment::default();
    }
    let raw = (params.book_gap_beta * (mid - mark) * 100.0).round() as i64;
    let gap_cents = raw.clamp(-params.book_gap_clamp_cents, params.book_gap_clamp_cents);
    BookLeadAdjustment {
        gap_cents,
        lead_secs: params.book_lead_secs,
    }
}

/// Computes exact v2 print-crossing touch probabilities `[row][window]` via 1-D Geometric IIR DP
/// without book-lead shift (`G = 0, L = 0`).
pub fn price_grid_v2(
    spot: f64,
    bands: &[PriceBand],
    windows: &[TimeWindow],
    params: &TickerV2Params,
    regime: Regime,
    inferred_phase: Option<usize>,
    news_override: Option<RegimePreset>,
) -> Vec<Vec<f64>> {
    price_grid_v2_with_book(
        spot,
        bands,
        windows,
        params,
        regime,
        inferred_phase,
        news_override,
        BookLeadAdjustment::default(),
    )
}

/// Computes exact v2 print-crossing touch probabilities `[row][window]` via 1-D Geometric IIR DP
/// with optional book-lead adjustment `(G, L)`:
///
/// - Starts the DP at `c0 = round(spot * 100) + G` instead of `C = round(spot * 100)`.
/// - Runs `(window.start_secs - L).max(0)` seconds before each window while preserving the
///   full window duration (`end_secs - start_secs`, i.e. 5 window seconds unchanged).
/// - With `G = 0` and `L = 0`, this is bit-identical to plain v2.
#[allow(clippy::too_many_arguments)]
pub fn price_grid_v2_with_book(
    spot: f64,
    bands: &[PriceBand],
    windows: &[TimeWindow],
    params: &TickerV2Params,
    regime: Regime,
    inferred_phase: Option<usize>,
    news_override: Option<RegimePreset>,
    book_lead: BookLeadAdjustment,
) -> Vec<Vec<f64>> {
    let rows = bands.len();
    let w_count = windows.len();
    let mut probs = vec![vec![0.0_f64; w_count]; rows];
    if rows == 0 || w_count == 0 {
        return probs;
    }

    let spot_cents = (spot * 100.0).round() as i64;
    let start_cents = spot_cents + book_lead.gap_cents;
    let lead = book_lead.lead_secs as usize;
    let half = DP_HALF_WIDTH;
    let n = 2 * half + 1;
    let row_bounds: Vec<(usize, usize)> = bands
        .iter()
        .map(|band| {
            let lc = (band.lower * 100.0).round() as i64;
            let uc = (band.upper * 100.0).round() as i64;
            let li = ((lc - start_cents) + half as i64).clamp(0, n as i64) as usize;
            let ui = ((uc - start_cents) + half as i64).clamp(0, (n - 1) as i64) as usize;
            (li, ui)
        })
        .collect();

    let clock = params.clock_secs.max(1) as usize;
    let far_markov_step_weight = clock as f64 / 60.0;

    // Per-clock-tick Markov transition matrix for far windows (6..23):
    // T_step = (1 - clock/60) * I + (clock/60) * T.
    let mut t_step = [[0.0_f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let eye = if r == c { 1.0 } else { 0.0 };
            t_step[r][c] = (1.0 - far_markov_step_weight) * eye
                + far_markov_step_weight * params.transition_1min[r][c];
        }
    }

    let mut presets = params.presets;
    let mut win_presets = params.window_presets.unwrap_or(params.presets);
    if let Some(np) = news_override {
        presets[Regime::News.index()] = np;
        win_presets[Regime::News.index()] = np;
    }

    let phase_range = if clock <= 1 {
        0..1
    } else if let Some(p) = inferred_phase.filter(|&p| p < clock) {
        p..(p + 1)
    } else {
        0..clock
    };
    let phase_weight = 1.0 / phase_range.len() as f64;
    let r0 = regime.index();
    let preset0 = &presets[r0];
    let win_preset0 = &win_presets[r0];

    let mut u_cur = vec![0.0_f64; n];
    let mut u_next = vec![0.0_f64; n];
    let mut scratch = vec![0.0_f64; n];

    // Lazy buffers for 4-regime far-window Markov DP (allocated only when w_count > NEAR_WINDOWS).
    let mut u_reg = if w_count > NEAR_WINDOWS {
        vec![vec![0.0_f64; n]; 4]
    } else {
        Vec::new()
    };
    let mut u_mix = if w_count > NEAR_WINDOWS {
        vec![vec![0.0_f64; n]; 4]
    } else {
        Vec::new()
    };

    for ph in phase_range {
        u_cur.fill(0.0);
        u_cur[half] = 1.0;
        let mut cur_s: usize = 0;
        let mut far_initialized = false;

        for (w_idx, window) in windows.iter().enumerate() {
            let duration = (window.end_secs as usize).saturating_sub(window.start_secs as usize);
            let ws = (window.start_secs as usize).saturating_sub(lead);
            let we = ws + duration;

            if w_idx < NEAR_WINDOWS {
                // Near horizon (windows 0..5): pure current preset r0 pre-window,
                // then win_preset0 inside the window.
                while cur_s + 1 < ws {
                    cur_s += 1;
                    if (cur_s + clock - ph) % clock == 0 {
                        full_step_clamped(&u_cur, preset0, &mut u_next);
                        std::mem::swap(&mut u_cur, &mut u_next);
                    }
                }
                let in_ticks = (ws..we)
                    .filter(|&s| (s + clock - ph) % clock == 0)
                    .count();
                for (row, &(li, ui)) in row_bounds.iter().enumerate() {
                    let surv =
                        window_survival_mass(&u_cur, li, ui, in_ticks, win_preset0, &mut scratch);
                    let hit = (1.0 - surv).clamp(0.0, 1.0);
                    probs[row][w_idx] += phase_weight * hit;
                }
                while cur_s + 1 < we {
                    cur_s += 1;
                    if (cur_s + clock - ph) % clock == 0 {
                        full_step_clamped(&u_cur, preset0, &mut u_next);
                        std::mem::swap(&mut u_cur, &mut u_next);
                    }
                }
            } else {
                // Far horizon (windows 6..23): Markov-modulated 4-regime DP using T_step
                // with `presets` up to each window start and `win_presets` inside each window.
                if !far_initialized {
                    for slice in &mut u_reg {
                        slice.fill(0.0);
                    }
                    u_reg[r0].copy_from_slice(&u_cur);
                    far_initialized = true;
                }
                while cur_s + 1 < ws {
                    cur_s += 1;
                    if (cur_s + clock - ph) % clock == 0 {
                        step_far_markov(&mut u_reg, &mut u_mix, &t_step, &presets);
                    }
                }
                let in_ticks = (ws..we)
                    .filter(|&s| (s + clock - ph) % clock == 0)
                    .count();
                // Pre-check active regime slices (mass > 1e-12) to skip zero-mass work.
                let active: [bool; 4] = [
                    u_reg[0].iter().sum::<f64>() > 1e-12,
                    u_reg[1].iter().sum::<f64>() > 1e-12,
                    u_reg[2].iter().sum::<f64>() > 1e-12,
                    u_reg[3].iter().sum::<f64>() > 1e-12,
                ];
                for (row, &(li, ui)) in row_bounds.iter().enumerate() {
                    let mut total_surv = 0.0_f64;
                    for r in 0..4 {
                        if active[r] {
                            total_surv += window_survival_mass(
                                &u_reg[r],
                                li,
                                ui,
                                in_ticks,
                                &win_presets[r],
                                &mut scratch,
                            );
                        }
                    }
                    let hit = (1.0 - total_surv).clamp(0.0, 1.0);
                    probs[row][w_idx] += phase_weight * hit;
                }
                while cur_s + 1 < we {
                    cur_s += 1;
                    if (cur_s + clock - ph) % clock == 0 {
                        step_far_markov(&mut u_reg, &mut u_mix, &t_step, &presets);
                    }
                }
            }
        }
    }

    probs
}

fn step_far_markov(
    u_reg: &mut [Vec<f64>],
    u_mix: &mut [Vec<f64>],
    t_step: &[[f64; 4]; 4],
    presets: &[RegimePreset; 4],
) {
    let n = u_reg[0].len();
    for x in 0..n {
        let a0 = u_reg[0][x];
        let a1 = u_reg[1][x];
        let a2 = u_reg[2][x];
        let a3 = u_reg[3][x];
        for dst in 0..4 {
            u_mix[dst][x] = a0 * t_step[0][dst]
                + a1 * t_step[1][dst]
                + a2 * t_step[2][dst]
                + a3 * t_step[3][dst];
        }
    }
    for r in 0..4 {
        full_step_clamped(&u_mix[r], &presets[r], &mut u_reg[r]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::stock_grid_config_for_ticker;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn sample_geom(p: f64, rng: &mut StdRng) -> i64 {
        let mut u: f64 = rng.random();
        while u <= 0.0 {
            u = rng.random();
        }
        1 + (u.ln() / (1.0 - p).ln()).floor() as i64
    }

    fn brute_force_mc_cell_clock(
        spot_c: i64,
        lc: i64,
        uc: i64,
        preset: &RegimePreset,
        w_start: usize,
        w_end: usize,
        phase: Option<usize>,
        clock_secs: usize,
        n_paths: usize,
        seed: u64,
    ) -> f64 {
        let clock = clock_secs.max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut hits = 0usize;
        for _ in 0..n_paths {
            let mut c = spot_c;
            let ph = if clock == 1 {
                0
            } else {
                phase.unwrap_or_else(|| rng.random_range(0..clock))
            };
            let mut hit = false;
            for s in 1..w_end {
                let prev = c;
                if (s + clock - ph) % clock == 0 && rng.random::<f64>() < preset.p_move {
                    let pu = if preset.mix_weight >= 1.0 || rng.random::<f64>() < preset.mix_weight {
                        preset.p1
                    } else {
                        preset.p2
                    };
                    let step = sample_geom(pu, &mut rng);
                    if rng.random::<bool>() {
                        c += step;
                    } else {
                        c -= step;
                    }
                }
                if s >= w_start && s < w_end {
                    let lo = prev.min(c);
                    let hi = prev.max(c);
                    if lo <= uc && hi >= lc {
                        hit = true;
                        break;
                    }
                }
            }
            if hit {
                hits += 1;
            }
        }
        hits as f64 / n_paths as f64
    }

    fn brute_force_mc_cell(
        spot_c: i64,
        lc: i64,
        uc: i64,
        preset: &RegimePreset,
        w_start: usize,
        w_end: usize,
        phase: Option<usize>,
        n_paths: usize,
        seed: u64,
    ) -> f64 {
        brute_force_mc_cell_clock(spot_c, lc, uc, preset, w_start, w_end, phase, 3, n_paths, seed)
    }

    #[test]
    fn exact_dp_matches_discrete_cent_jump_mc_across_presets_and_phases() {
        let cases = [
            ("NVDA_normal_d0_w0_ph1", &NVDA_V2_PARAMS, Regime::Normal, 139.99_f64, 139.98_f64, 140.01_f64, 5u32, 10u32, Some(1)),
            ("NVDA_normal_d1_w0_avg", &NVDA_V2_PARAMS, Regime::Normal, 139.99_f64, 140.01_f64, 140.04_f64, 5u32, 10u32, None),
            ("NVDA_news_d2_w0_ph2", &NVDA_V2_PARAMS, Regime::News, 139.99_f64, 140.04_f64, 140.07_f64, 5u32, 10u32, Some(2)),
            ("TSLA_volatile_d1_w5_avg", &TSLA_V2_PARAMS, Regime::Volatile, 364.04_f64, 364.08_f64, 364.16_f64, 30u32, 35u32, None),
        ];
        for (label, params, regime, spot, lower, upper, ws, we, ph) in cases {
            let bands = vec![PriceBand {
                row_index: 0,
                lower,
                upper,
            }];
            let windows = vec![TimeWindow {
                start_secs: ws,
                end_secs: we,
            }];
            let dp = price_grid_v2(spot, &bands, &windows, params, regime, ph, None)[0][0];
            let mc = brute_force_mc_cell(
                (spot * 100.0).round() as i64,
                (lower * 100.0).round() as i64,
                (upper * 100.0).round() as i64,
                &params.presets[regime.index()],
                ws as usize,
                we as usize,
                ph,
                60_000,
                77,
            );
            let se = (dp * (1.0 - dp) / 60_000.0).sqrt().max(0.002);
            assert!(
                (dp - mc).abs() <= 4.0 * se,
                "{label}: DP={dp:.5} MC={mc:.5} diff={:+.5} (4*se={:.5})",
                dp - mc,
                4.0 * se
            );
        }
    }

    #[test]
    fn dp_transition_matrices_are_row_stochastic_and_far_windows_blend_regimes() {
        for params in [&NVDA_V2_PARAMS, &TSLA_V2_PARAMS] {
            for row in params.transition_1min {
                let sum: f64 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-12, "row sum {sum} != 1 for {}", params.ticker);
                for v in row {
                    assert!(v >= 0.0 && v <= 1.0);
                }
            }
        }

        // Compare Quiet far window (w=11, 60-65s) with real T vs identity T (stationary).
        let mut stationary = NVDA_V2_PARAMS;
        stationary.transition_1min = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let bands = vec![
            PriceBand { row_index: 0, lower: 139.98, upper: 140.01 }, // d=0
            PriceBand { row_index: 1, lower: 140.07, upper: 140.10 }, // d=+3
        ];
        let windows: Vec<TimeWindow> = (0..12)
            .map(|w| TimeWindow {
                start_secs: ((w + 1) * 5) as u32,
                end_secs: ((w + 2) * 5) as u32,
            })
            .collect();
        let blended =
            price_grid_v2(139.99, &bands, &windows, &NVDA_V2_PARAMS, Regime::Quiet, None, None);
        let static_p =
            price_grid_v2(139.99, &bands, &windows, &stationary, Regime::Quiet, None, None);

        // Windows 0-5 MUST be identical (pure current preset).
        for w in 0..NEAR_WINDOWS {
            for r in 0..2 {
                assert!(
                    (blended[r][w] - static_p[r][w]).abs() < 1e-12,
                    "near window {w} row {r} must not switch regimes"
                );
            }
        }
        // Far window w=11 outer band d=+3 must have higher touch probability under Markov blend
        // because Quiet has a 17.1%/min chance of transitioning to Normal.
        assert!(
            blended[1][11] > static_p[1][11] + 0.005,
            "expected far Markov blend {} > stationary {} on Quiet d=3 w=11",
            blended[1][11],
            static_p[1][11]
        );
    }

    #[test]
    fn regime_detector_tracks_phase_escalation_shock_hysteresis_and_gaps() {
        let mut det = RegimeDetector::new();
        assert_eq!(det.current_regime(), Regime::Normal);
        assert_eq!(det.inferred_phase(100_000, 3), None);

        // Quiet tape: 1c move every 9s -> low R_60, holds below 0.85 * 1.162 = 0.988 for 15s -> Quiet
        let mut price = 140.00_f64;
        for sec in 0..40u64 {
            if sec == 10 || sec == 22 {
                price += 0.01;
            }
            det.on_tick(PriceTick { ts_ms: sec * 1_000, price, mid_px: None, oracle_px: None }, &NVDA_V2_PARAMS);
        }
        assert_eq!(det.current_regime(), Regime::Quiet);
        // Last change was at sec=22 -> at quote_ts=40s, phi = ((22%3)+3-(40%3))%3 = (1+3-1)%3 = 0
        assert_eq!(det.inferred_phase(40_000, 3), Some(0));
        assert_eq!(det.inferred_phase(41_000, 3), Some(2));
        assert_eq!(det.inferred_phase(42_000, 3), Some(1));

        // > 3s gap resets phase to None and does NOT trigger a bogus shock
        det.on_tick(PriceTick { ts_ms: 46_000, price: 141.00, mid_px: None, oracle_px: None }, &NVDA_V2_PARAMS);
        assert_eq!(det.inferred_phase(46_000, 3), None);
        assert_eq!(det.current_regime(), Regime::Quiet);

        // Single-print shock > 34c (4 * p99) promotes immediately to News on print 1
        det.on_tick(PriceTick { ts_ms: 47_000, price: 141.40, mid_px: None, oracle_px: None }, &NVDA_V2_PARAMS);
        assert_eq!(det.current_regime(), Regime::News);
        assert_eq!(det.inferred_phase(47_000, 3), Some(0));

        // Downgrade hysteresis: stays News during < 15s of calm before stepping down one level
        for sec in 48..120u64 {
            det.on_tick(PriceTick { ts_ms: sec * 1_000, price: 141.40, mid_px: None, oracle_px: None }, &NVDA_V2_PARAMS);
        }
        assert!(det.current_regime() < Regime::News);
    }

    #[test]
    fn v2_pricing_ignores_v1_calibration_layers_and_respects_configurable_cap() {
        let mut cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        // Verify NVDA has baked v1 layers (calibration factors, off_hours_m_base) that v2 ignores.
        assert!(!cfg.calibration.factors.is_empty());
        assert_eq!(cfg.pricing.off_hours_m_base, Some(0.094));

        let m_eff_quiet = v2_m_eff(
            0.58,
            NVDA_V2_PARAMS.rms_ref_cents,
            NVDA_V2_PARAMS.presets[Regime::Quiet.index()].m_base,
        );
        assert!((m_eff_quiet - 0.025).abs() < 1e-12);
        let m_eff_news = v2_m_eff(
            20.0,
            NVDA_V2_PARAMS.rms_ref_cents,
            NVDA_V2_PARAMS.presets[Regime::News.index()].m_base,
        );
        assert!((m_eff_news - 0.288).abs() < 1e-12);

        // Exact (1 - m_eff) / P at P = 0.015 (fair = 65x): clamped by max_multiplier
        cfg.pricing.max_multiplier = 25.0;
        assert!((v2_price_multiplier(0.015, 0.025, 0.0, 1.0, &cfg) - 25.0).abs() < 1e-12);
        cfg.pricing.max_multiplier = 50.0;
        assert!((v2_price_multiplier(0.015, 0.025, 0.0, 1.0, &cfg) - 50.0).abs() < 1e-12);
        cfg.pricing.max_multiplier = 100.0;
        assert!((v2_price_multiplier(0.015, 0.025, 0.0, 1.0, &cfg) - 65.0).abs() < 1e-12);
    }

    #[test]
    fn dynamic_news_preset_matches_target_second_moment_and_cadence() {
        // Border news minute (R_60 = 9.0c, q_60 = 0.295) vs shock burst (R_60 = 39.0c, q_60 = 0.15).
        for (params, r60, q60, expected_r_eff, expected_q_eff) in [
            (&NVDA_V2_PARAMS, 9.0_f64, 0.295_f64, 9.0_f64, 0.295_f64),
            (&NVDA_V2_PARAMS, 5.0_f64, 0.05_f64, 8.390_f64, V2_NEWS_Q_MIN),
            (
                &NVDA_V2_PARAMS,
                39.05_f64,
                0.15_f64,
                2.5 * 11.79_f64,
                0.15_f64,
            ),
            (&TSLA_V2_PARAMS, 14.20_f64, 0.40_f64, 14.20_f64, 1.0 / 3.0_f64),
        ] {
            let np = dynamic_news_preset(params, r60, q60);
            assert!((np.rms_median_cents - expected_r_eff).abs() < 1e-9);
            assert!((np.q_1s - expected_q_eff).abs() < 1e-9);
            assert!(
                (np.p_move - (params.clock_secs as f64 * expected_q_eff).min(1.0)).abs() < 1e-9
            );
            assert!((np.mix_weight - V2_NEWS_MIX_WEIGHT).abs() < 1e-12);
            assert!((np.m_base - params.presets[Regime::News.index()].m_base).abs() < 1e-12);

            // Verify exact 2-Geom second-moment match: q_eff * E[K^2] == R_eff^2.
            let e_k2 = np.mix_weight * (2.0 - np.p1) / (np.p1 * np.p1)
                + (1.0 - np.mix_weight) * (2.0 - np.p2) / (np.p2 * np.p2);
            let implied_rms = (np.q_1s * e_k2).sqrt();
            assert!(
                (implied_rms - expected_r_eff).abs() < 1e-6,
                "{}: implied_rms={implied_rms:.6} != expected={expected_r_eff:.6}",
                params.ticker
            );
        }

        // Border-regime dynamic news (R_eff=9c, q_eff=0.25) must raise d=0 w=0 touch prob
        // relative to static News, and the DP lift must match an independent Monte Carlo
        // simulation of the same model inputs (R_eff, q_eff, pooled 2-Geom shape).
        let bands = vec![PriceBand {
            row_index: 0,
            lower: 139.98,
            upper: 140.01,
        }];
        let windows = vec![TimeWindow {
            start_secs: 5,
            end_secs: 10,
        }];
        let p_static =
            price_grid_v2(139.99, &bands, &windows, &NVDA_V2_PARAMS, Regime::News, None, None)[0][0];
        let np_border = dynamic_news_preset(&NVDA_V2_PARAMS, 9.0, 0.25);
        let p_dynamic = price_grid_v2(
            139.99,
            &bands,
            &windows,
            &NVDA_V2_PARAMS,
            Regime::News,
            None,
            Some(np_border),
        )[0][0];
        assert!(
            p_dynamic > p_static,
            "border dynamic news d=0 prob {p_dynamic:.4} must exceed static {p_static:.4}"
        );

        let n_paths: usize = 100_000;
        let mc_static = brute_force_mc_cell(
            13_999,
            13_998,
            14_001,
            &NVDA_V2_PARAMS.presets[Regime::News.index()],
            5,
            10,
            None,
            n_paths,
            101,
        );
        let mc_dynamic = brute_force_mc_cell(
            13_999,
            13_998,
            14_001,
            &np_border,
            5,
            10,
            None,
            n_paths,
            102,
        );
        // Tolerance is 4*SE of the difference between two independent binomial MC estimators
        // (N = 100,000 each => SE_diff ~ 0.0022, 4*SE_diff ~ 0.0089, two-sided p < 6.4e-5).
        let se_dyn = (p_dynamic * (1.0 - p_dynamic) / n_paths as f64).sqrt();
        let se_stat = (p_static * (1.0 - p_static) / n_paths as f64).sqrt();
        let se_diff = (se_dyn * se_dyn + se_stat * se_stat).sqrt();
        let dp_lift = p_dynamic - p_static;
        let expected_lift = mc_dynamic - mc_static;
        assert!(
            (p_dynamic - mc_dynamic).abs() <= 4.0 * se_dyn,
            "dynamic news DP={p_dynamic:.5} vs MC={mc_dynamic:.5} (4*se={:.5})",
            4.0 * se_dyn
        );
        assert!(
            (dp_lift - expected_lift).abs() <= 4.0 * se_diff,
            "border dynamic news lift DP={dp_lift:+.5} vs model-input MC={expected_lift:+.5} (4*se_diff={:.5})",
            4.0 * se_diff
        );
    }

    #[test]
    fn shock_trigger_requires_4x_p99_and_tracks_q60() {
        let mut det = RegimeDetector::new();
        let mut price = 140.00_f64;
        // 30s of calm tape with 6 moves (1 every 5s -> q_60 = 6/30 = 0.20)
        for sec in 0..31u64 {
            if sec > 0 && sec % 5 == 0 {
                price += 0.01;
            }
            det.on_tick(PriceTick { ts_ms: sec * 1_000, price, mid_px: None, oracle_px: None }, &NVDA_V2_PARAMS);
        }
        assert!((det.last_q60() - (6.0 / 30.0)).abs() < 1e-9);

        // 30c print (above old 3*p99=26c, below new 4*p99=34c) must NOT force News.
        price += 0.30;
        det.on_tick(PriceTick { ts_ms: 31_000, price, mid_px: None, oracle_px: None }, &NVDA_V2_PARAMS);
        assert_ne!(det.current_regime(), Regime::News);

        // 35c print (> 4*p99=34c) promotes immediately to News on print 1.
        price += 0.35;
        det.on_tick(PriceTick { ts_ms: 32_000, price, mid_px: None, oracle_px: None }, &NVDA_V2_PARAMS);
        assert_eq!(det.current_regime(), Regime::News);
        assert!((det.last_q60() - (8.0 / 32.0)).abs() < 1e-9);
    }

    #[test]
    fn per_preset_m_base_plus_m_risk_matches_m3_derivation() {
        let nvda_expected = [0.025_f64, 0.098, 0.134, 0.263];
        let tsla_expected = [0.025_f64, 0.073, 0.197, 0.185];
        for (idx, (&ne, &te)) in nvda_expected.iter().zip(tsla_expected.iter()).enumerate() {
            assert!((NVDA_V2_PARAMS.presets[idx].m_base - ne).abs() < 1e-12);
            assert!((TSLA_V2_PARAMS.presets[idx].m_base - te).abs() < 1e-12);
            assert!(NVDA_V2_PARAMS.presets[idx].m_base >= V2_M_BASE);
            assert!(TSLA_V2_PARAMS.presets[idx].m_base >= V2_M_BASE);
        }
        // Check m_eff = m_base + m_risk at median R_60 for NVDA Volatile and TSLA Normal.
        let nvda_vol = NVDA_V2_PARAMS.presets[Regime::Volatile.index()];
        let m_nvda_vol = v2_m_eff(nvda_vol.rms_median_cents, NVDA_V2_PARAMS.rms_ref_cents, nvda_vol.m_base);
        let expected_risk = (V2_RISK_SLOPE * (3.36 / 1.60 - 1.0)).clamp(0.0, V2_M_RISK_MAX);
        assert!((m_nvda_vol - (0.134 + expected_risk)).abs() < 1e-12);

        let tsla_norm = TSLA_V2_PARAMS.presets[Regime::Normal.index()];
        let m_tsla_norm = v2_m_eff(tsla_norm.rms_median_cents, TSLA_V2_PARAMS.rms_ref_cents, tsla_norm.m_base);
        assert!((m_tsla_norm - 0.073).abs() < 1e-12);
    }

    #[test]
    fn v2_params_exist_for_nvda_and_tsla_only() {
        assert!(v2_ticker_params("NVDA").is_some());
        assert!(v2_ticker_params("xyz:TSLA").is_some());
        assert!(v2_ticker_params("AAPL").is_none());
        assert!(v2_ticker_params("SP500").is_none());

        let nvda_drb = v2_ticker_params_for_source("nvda_usdc", PriceSource::Deribit).unwrap();
        assert_eq!(nvda_drb.clock_secs, PriceSource::Deribit.timing().clock_secs);
        assert_eq!(nvda_drb.void_gap_ms, PriceSource::Deribit.timing().void_gap_ms);
        assert_eq!(nvda_drb.clock_secs, 1);

        let tsla_drb =
            v2_ticker_params_for_source("deribit_price_index.tsla_usdc", PriceSource::Deribit)
                .unwrap();
        assert_eq!(tsla_drb.clock_secs, 1);
        assert!(v2_ticker_params_for_source("AAPL", PriceSource::Deribit).is_none());
    }

    #[test]
    fn hl_v2_quotes_match_pre_change_golden_values_exactly() {
        let windows: Vec<TimeWindow> = (0..8)
            .map(|i| TimeWindow {
                start_secs: (5 + 5 * i) as u32,
                end_secs: (10 + 5 * i) as u32,
            })
            .collect();
        let nvda_bands = vec![
            PriceBand { row_index: 0, lower: 139.98, upper: 140.01 },
            PriceBand { row_index: 1, lower: 140.01, upper: 140.04 },
            PriceBand { row_index: 2, lower: 140.04, upper: 140.07 },
        ];
        let tsla_bands = vec![
            PriceBand { row_index: 0, lower: 364.00, upper: 364.08 },
            PriceBand { row_index: 1, lower: 364.08, upper: 364.16 },
        ];

        // Exact pre-change IEEE-754 golden values for HL 3-second clock (w0, w5 near; w7 far Markov).
        let nvda_norm_p = price_grid_v2(
            139.99,
            &nvda_bands,
            &windows,
            &NVDA_V2_PARAMS,
            Regime::Normal,
            Some(1),
            None,
        );
        let expected_nvda_norm: [(usize, usize, f64); 9] = [
            (0, 0, 0.540648486121589_f64),
            (0, 5, 0.309469830578935_f64),
            (0, 7, 0.273439432171434_f64),
            (1, 0, 0.369177874392759_f64),
            (1, 5, 0.289560126875915_f64),
            (1, 7, 0.259785505679971_f64),
            (2, 0, 0.176760143990714_f64),
            (2, 5, 0.245739437260077_f64),
            (2, 7, 0.228784446826315_f64),
        ];
        for (r, w, exp) in expected_nvda_norm {
            assert!(
                (nvda_norm_p[r][w] - exp).abs() < 1e-12,
                "HL NVDA Normal r={r} w={w}: got {:.15}, expected {exp:.15}",
                nvda_norm_p[r][w]
            );
        }

        let cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        let m_eff = v2_m_eff(
            NVDA_V2_PARAMS.rms_ref_cents,
            NVDA_V2_PARAMS.rms_ref_cents,
            NVDA_V2_PARAMS.presets[Regime::Normal.index()].m_base,
        );
        let expected_mults = [
            (0usize, 1.668366828270643_f64),
            (1usize, 2.443266681362342_f64),
            (2usize, 5.102960314670179_f64),
        ];
        for (r, exp_m) in expected_mults {
            let mult = v2_price_multiplier(nvda_norm_p[r][0], m_eff, 0.0, 1.0, &cfg);
            assert!(
                (mult - exp_m).abs() < 1e-12,
                "HL NVDA Normal mult r={r}: got {mult:.15}, expected {exp_m:.15}"
            );
        }

        let nvda_vol_p = price_grid_v2(
            139.99,
            &nvda_bands,
            &windows,
            &NVDA_V2_PARAMS,
            Regime::Volatile,
            None,
            None,
        );
        for (r, w, exp) in [
            (0usize, 0usize, 0.506299154013340_f64),
            (0, 5, 0.204724985253390_f64),
            (0, 7, 0.176081838822454_f64),
            (1, 0, 0.437092246605829_f64),
            (2, 7, 0.170157160303662_f64),
        ] {
            assert!(
                (nvda_vol_p[r][w] - exp).abs() < 1e-12,
                "HL NVDA Volatile r={r} w={w}: got {:.15}, expected {exp:.15}",
                nvda_vol_p[r][w]
            );
        }

        let tsla_norm_p = price_grid_v2(
            364.04,
            &tsla_bands,
            &windows,
            &TSLA_V2_PARAMS,
            Regime::Normal,
            Some(0),
            None,
        );
        for (r, w, exp) in [
            (0usize, 0usize, 0.820118831252116_f64),
            (0, 5, 0.370816985963493_f64),
            (0, 7, 0.263920747522168_f64),
            (1, 0, 0.374736187297921_f64),
            (1, 7, 0.238398992289137_f64),
        ] {
            assert!(
                (tsla_norm_p[r][w] - exp).abs() < 1e-12,
                "HL TSLA Normal r={r} w={w}: got {:.15}, expected {exp:.15}",
                tsla_norm_p[r][w]
            );
        }
    }

    #[test]
    fn clock_1_dp_matches_brute_force_enumeration_and_mc() {
        // 1. Exact brute-force enumeration on a small 3-tick clock-1 case (ws=2, we=4):
        //    s=1 pre-tick, s=2 & s=3 in-window ticks, single-Geom(p1=0.85), p_move=0.40.
        //    Truncating at K=11 leaves tail mass (1 - 0.85)^11 < 9e-10 per step.
        let mut small_params = NVDA_DERIBIT_V2_PARAMS;
        let test_preset = RegimePreset {
            q_1s: 0.40,
            p_move: 0.40,
            mix_weight: 1.0,
            p1: 0.85,
            p2: 0.85,
            rms_median_cents: 1.0,
            m_base: 0.025,
        };
        small_params.presets[Regime::Normal.index()] = test_preset;
        small_params.window_presets = None;

        let bands = vec![PriceBand {
            row_index: 0,
            lower: 140.01,
            upper: 140.03,
        }];
        let windows = vec![TimeWindow {
            start_secs: 2,
            end_secs: 4,
        }];
        let dp_small = price_grid_v2(
            140.00,
            &bands,
            &windows,
            &small_params,
            Regime::Normal,
            Some(0),
            None,
        )[0][0];

        let mut steps: Vec<(i64, f64)> = vec![(0, 1.0 - test_preset.p_move)];
        for k in 1..=11_i64 {
            let pk = 0.5
                * test_preset.p_move
                * test_preset.p1
                * (1.0 - test_preset.p1).powi((k - 1) as i32);
            steps.push((k, pk));
            steps.push((-k, pk));
        }
        let mut exact_hit = 0.0_f64;
        for &(d1, w1) in &steps {
            let x1 = d1;
            for &(d2, w2) in &steps {
                let x2 = x1 + d2;
                if x1.min(x2) <= 3 && x1.max(x2) >= 1 {
                    exact_hit += w1 * w2;
                    continue;
                }
                for &(d3, w3) in &steps {
                    let x3 = x2 + d3;
                    if x2.min(x3) <= 3 && x2.max(x3) >= 1 {
                        exact_hit += w1 * w2 * w3;
                    }
                }
            }
        }
        assert!(
            (dp_small - exact_hit).abs() < 1e-6,
            "clock-1 DP={dp_small:.12} != brute-force enumeration={exact_hit:.12}"
        );

        // 2. Standard 5-second window [5, 10) (in_ticks=5) at clock=1 vs MC across Normal & News.
        for (label, regime, lower, upper) in [
            ("DRB_NVDA_normal_d0_w0", Regime::Normal, 139.98_f64, 140.01_f64),
            ("DRB_NVDA_normal_d1_w0", Regime::Normal, 140.01_f64, 140.04_f64),
            ("DRB_NVDA_news_d2_w0", Regime::News, 140.04_f64, 140.07_f64),
        ] {
            let b = vec![PriceBand {
                row_index: 0,
                lower,
                upper,
            }];
            let w = vec![TimeWindow {
                start_secs: 5,
                end_secs: 10,
            }];
            let dp = price_grid_v2(
                139.99,
                &b,
                &w,
                &NVDA_DERIBIT_V2_PARAMS,
                regime,
                None,
                None,
            )[0][0];
            let mc = brute_force_mc_cell_clock(
                13_999,
                (lower * 100.0).round() as i64,
                (upper * 100.0).round() as i64,
                &NVDA_DERIBIT_V2_PARAMS.presets[regime.index()],
                5,
                10,
                Some(0),
                1,
                60_000,
                99,
            );
            let se = (dp * (1.0 - dp) / 60_000.0).sqrt().max(0.002);
            assert!(
                (dp - mc).abs() <= 4.0 * se,
                "{label}: DP={dp:.5} MC={mc:.5} diff={:+.5} (4*se={:.5})",
                dp - mc,
                4.0 * se
            );
        }

        // 3. dynamic_news_preset at clock=1 allows q_eff up to 1.0 (not clamped at 1/3).
        let np_drb = dynamic_news_preset(&NVDA_DERIBIT_V2_PARAMS, 12.0, 0.65);
        assert!((np_drb.q_1s - 0.65).abs() < 1e-9);
        assert!((np_drb.p_move - 0.65).abs() < 1e-9);
        let e_k2 = np_drb.mix_weight * (2.0 - np_drb.p1) / (np_drb.p1 * np_drb.p1)
            + (1.0 - np_drb.mix_weight) * (2.0 - np_drb.p2) / (np_drb.p2 * np_drb.p2);
        assert!(((np_drb.q_1s * e_k2).sqrt() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn phase_is_always_zero_at_clock_1() {
        let mut det = RegimeDetector::new();
        // Cold start (no move observed yet): clock 1 still yields phase 0.
        assert_eq!(det.inferred_phase(100_000, 1), Some(0));

        det.on_tick(PriceTick::new(10_000, 140.00), &NVDA_DERIBIT_V2_PARAMS);
        det.on_tick(PriceTick::new(11_000, 140.02), &NVDA_DERIBIT_V2_PARAMS);
        for sec in 11..15_u64 {
            assert_eq!(det.inferred_phase(sec * 1_000, 1), Some(0));
        }

        // After > 60s silence or > void_gap_ms gap, clock 1 still yields phase 0.
        assert_eq!(det.inferred_phase(200_000, 1), Some(0));
        det.on_tick(PriceTick::new(205_000, 140.10), &NVDA_DERIBIT_V2_PARAMS);
        assert_eq!(det.inferred_phase(205_000, 1), Some(0));
    }

    #[test]
    fn regime_detector_void_gap_uses_source_params() {
        let mut tight_params = NVDA_V2_PARAMS;
        tight_params.void_gap_ms = 1_500;

        let mut det_tight = RegimeDetector::new();
        let mut det_hl = RegimeDetector::new();

        for (det, params) in [(&mut det_tight, &tight_params), (&mut det_hl, &NVDA_V2_PARAMS)] {
            det.on_tick(PriceTick::new(10_000, 140.00), params);
            det.on_tick(PriceTick::new(11_000, 140.01), params);
            // 2_000 ms gap with a 50c shock: > 1_500 ms (voided on tight), <= 3_000 ms (accepted on HL).
            det.on_tick(PriceTick::new(13_000, 140.51), params);
        }

        assert_eq!(det_tight.inferred_phase(13_000, 3), None);
        assert_eq!(det_tight.current_regime(), Regime::Normal);

        assert_eq!(det_hl.inferred_phase(13_000, 3), Some(0));
        assert_eq!(det_hl.current_regime(), Regime::News);
    }

    #[test]
    fn book_lead_zero_matches_plain_v2_exactly_and_hl_stays_disabled() {
        let windows: Vec<TimeWindow> = (0..8)
            .map(|i| TimeWindow {
                start_secs: (5 + 5 * i) as u32,
                end_secs: (10 + 5 * i) as u32,
            })
            .collect();
        let bands = vec![
            PriceBand { row_index: 0, lower: 139.95, upper: 139.98 },
            PriceBand { row_index: 1, lower: 139.98, upper: 140.01 },
            PriceBand { row_index: 2, lower: 140.01, upper: 140.04 },
        ];

        // HL params have L = 0, beta = 0 -> compute_book_lead is always zero even with a large gap.
        assert_eq!(NVDA_V2_PARAMS.book_lead_secs, 0);
        assert_eq!(NVDA_V2_PARAMS.book_gap_beta, 0.0);
        assert_eq!(TSLA_V2_PARAMS.book_lead_secs, 0);
        assert_eq!(TSLA_V2_PARAMS.book_gap_beta, 0.0);
        assert_eq!(
            compute_book_lead(Some(140.20), Some(140.00), Some(100), &NVDA_V2_PARAMS),
            BookLeadAdjustment::default()
        );

        let plain = price_grid_v2(
            139.99,
            &bands,
            &windows,
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::Normal,
            Some(0),
            None,
        );
        let with_zero = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows,
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::Normal,
            Some(0),
            None,
            BookLeadAdjustment {
                gap_cents: 0,
                lead_secs: 0,
            },
        );
        assert_eq!(plain, with_zero);
    }

    #[test]
    fn book_lead_positive_gap_shifts_probs_horizon_shortens_and_stale_or_clamp_works() {
        let windows: Vec<TimeWindow> = (0..8)
            .map(|i| TimeWindow {
                start_secs: (5 + 5 * i) as u32,
                end_secs: (10 + 5 * i) as u32,
            })
            .collect();
        let bands = vec![
            PriceBand { row_index: 0, lower: 139.92, upper: 139.95 }, // below spot
            PriceBand { row_index: 1, lower: 139.95, upper: 139.98 }, // below spot
            PriceBand { row_index: 2, lower: 139.98, upper: 140.01 }, // center
            PriceBand { row_index: 3, lower: 140.01, upper: 140.04 }, // above spot
            PriceBand { row_index: 4, lower: 140.04, upper: 140.07 }, // above spot
        ];

        // 1. G = +5c raises P(touch) for rows above spot and lowers it for rows below spot.
        let p_base = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows,
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::Normal,
            Some(0),
            None,
            BookLeadAdjustment {
                gap_cents: 0,
                lead_secs: 0,
            },
        );
        let p_up5 = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows,
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::Normal,
            Some(0),
            None,
            BookLeadAdjustment {
                gap_cents: 5,
                lead_secs: 0,
            },
        );
        assert!(p_up5[0][0] < p_base[0][0], "row below must drop with G=+5c");
        assert!(p_up5[1][0] < p_base[1][0], "row below must drop with G=+5c");
        assert!(p_up5[3][0] > p_base[3][0], "row above must rise with G=+5c");
        assert!(p_up5[4][0] > p_base[4][0], "row above must rise with G=+5c");

        // 2. L = 2 shortens pre-window horizon by 2 s while keeping all 5 window seconds unchanged.
        let shifted_windows: Vec<TimeWindow> = windows
            .iter()
            .map(|w| TimeWindow {
                start_secs: w.start_secs - 2,
                end_secs: w.end_secs - 2,
            })
            .collect();
        let p_lead2 = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows,
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::Normal,
            Some(0),
            None,
            BookLeadAdjustment {
                gap_cents: 0,
                lead_secs: 2,
            },
        );
        let p_shifted_plain = price_grid_v2(
            139.99,
            &bands,
            &shifted_windows,
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::Normal,
            Some(0),
            None,
        );
        for r in 0..bands.len() {
            for w in 0..windows.len() {
                assert!(
                    (p_lead2[r][w] - p_shifted_plain[r][w]).abs() < 1e-12,
                    "r={r} w={w}: lead=2 {:.12} != shifted {:.12}",
                    p_lead2[r][w],
                    p_shifted_plain[r][w]
                );
            }
        }
        // Shortening pre-window horizon concentrates mass at center and reduces far-row touch prob.
        assert!(p_lead2[2][0] > p_base[2][0]);
        assert!(p_lead2[4][0] < p_base[4][0]);

        // Floor at 0 when window_start <= L: [2, 7) with L=2 has 0 pre-steps and 5 window ticks.
        let floor_win = vec![TimeWindow {
            start_secs: 2,
            end_secs: 7,
        }];
        let zero_pre_win = vec![TimeWindow {
            start_secs: 1,
            end_secs: 6,
        }];
        let p_floored = price_grid_v2_with_book(
            139.99,
            &bands,
            &floor_win,
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::Normal,
            Some(0),
            None,
            BookLeadAdjustment {
                gap_cents: 0,
                lead_secs: 2,
            },
        );
        let p_zero_pre = price_grid_v2(
            139.99,
            &bands,
            &zero_pre_win,
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::Normal,
            Some(0),
            None,
        );
        for r in 0..bands.len() {
            assert!((p_floored[r][0] - p_zero_pre[r][0]).abs() < 1e-12);
        }

        // 3. Stale or missing book -> G = 0, L = 0; fresh book -> exact gap & clamp.
        assert_eq!(
            compute_book_lead(None, Some(140.00), Some(100), &NVDA_DERIBIT_V2_PARAMS),
            BookLeadAdjustment::default()
        );
        assert_eq!(
            compute_book_lead(Some(140.05), None, Some(100), &NVDA_DERIBIT_V2_PARAMS),
            BookLeadAdjustment::default()
        );
        assert_eq!(
            compute_book_lead(
                Some(140.05),
                Some(140.00),
                Some(DERIBIT_BOOK_MAX_AGE_MS + 1),
                &NVDA_DERIBIT_V2_PARAMS,
            ),
            BookLeadAdjustment::default()
        );
        assert_eq!(
            compute_book_lead(
                Some(140.05),
                Some(140.00),
                Some(DERIBIT_BOOK_MAX_AGE_MS),
                &NVDA_DERIBIT_V2_PARAMS,
            ),
            BookLeadAdjustment {
                gap_cents: 5,
                lead_secs: DERIBIT_BOOK_LEAD_SECS,
            }
        );

        // 4. Symmetric clamp: NVDA ±20c, TSLA ±40c, with provisional L = DERIBIT_BOOK_LEAD_SECS (3 s).
        assert_eq!(DERIBIT_BOOK_LEAD_SECS, 3);
        assert_eq!(
            compute_book_lead(Some(140.50), Some(140.00), Some(0), &NVDA_DERIBIT_V2_PARAMS),
            BookLeadAdjustment {
                gap_cents: 20,
                lead_secs: DERIBIT_BOOK_LEAD_SECS,
            }
        );
        assert_eq!(
            compute_book_lead(Some(139.50), Some(140.00), Some(0), &NVDA_DERIBIT_V2_PARAMS),
            BookLeadAdjustment {
                gap_cents: -20,
                lead_secs: DERIBIT_BOOK_LEAD_SECS,
            }
        );
        assert_eq!(
            compute_book_lead(Some(365.00), Some(364.00), Some(0), &TSLA_DERIBIT_V2_PARAMS),
            BookLeadAdjustment {
                gap_cents: 40,
                lead_secs: DERIBIT_BOOK_LEAD_SECS,
            }
        );
        assert_eq!(
            compute_book_lead(Some(363.00), Some(364.00), Some(0), &TSLA_DERIBIT_V2_PARAMS),
            BookLeadAdjustment {
                gap_cents: -40,
                lead_secs: DERIBIT_BOOK_LEAD_SECS,
            }
        );
    }

    #[test]
    fn v2_params_json_round_trip_matches_compiled_tables_for_both_sources() {
        let windows: Vec<TimeWindow> = (0..8)
            .map(|i| TimeWindow {
                start_secs: (5 + 5 * i) as u32,
                end_secs: (10 + 5 * i) as u32,
            })
            .collect();
        for (ticker, source, spot, band_c, mid, mark) in [
            ("NVDA", PriceSource::Hyperliquid, 139.99_f64, 3_i64, 140.04_f64, 139.99_f64),
            ("TSLA", PriceSource::Hyperliquid, 364.04_f64, 8_i64, 364.14_f64, 364.04_f64),
            ("NVDA", PriceSource::Deribit, 139.99_f64, 3_i64, 140.04_f64, 139.99_f64),
            ("TSLA", PriceSource::Deribit, 364.04_f64, 8_i64, 364.14_f64, 364.04_f64),
        ] {
            let compiled = v2_ticker_params_for_source(ticker, source).unwrap();
            let dumped = compiled.to_json_pretty().unwrap();
            let parsed = parse_v2_params_json(&dumped).unwrap();
            assert!(parsed.matches_coin(ticker));
            let loaded = parsed.leak_static();
            assert_eq!(loaded, compiled);

            let spot_c = (spot * 100.0).round() as i64;
            let bands: Vec<PriceBand> = (-2..=2_i64)
                .enumerate()
                .map(|(row_index, k)| PriceBand {
                    row_index,
                    lower: (spot_c + k * band_c) as f64 / 100.0,
                    upper: (spot_c + (k + 1) * band_c) as f64 / 100.0,
                })
                .collect();
            let adj_compiled = compute_book_lead(Some(mid), Some(mark), Some(0), compiled);
            let adj_loaded = compute_book_lead(Some(mid), Some(mark), Some(0), loaded);
            assert_eq!(adj_compiled, adj_loaded);

            let p_compiled = price_grid_v2_with_book(
                spot,
                &bands,
                &windows,
                compiled,
                Regime::Normal,
                None,
                None,
                adj_compiled,
            );
            let p_loaded = price_grid_v2_with_book(
                spot,
                &bands,
                &windows,
                loaded,
                Regime::Normal,
                None,
                None,
                adj_loaded,
            );
            assert_eq!(p_compiled, p_loaded);
        }
    }

    #[test]
    fn v2_params_json_validation_catches_every_error() {
        let base = TickerV2ParamsJson::from_params(&NVDA_DERIBIT_V2_PARAMS);
        assert!(base.validate().is_ok());

        // Coin matching across naming variants.
        assert!(base.matches_coin("NVDA"));
        assert!(base.matches_coin("xyz:NVDA"));
        assert!(base.matches_coin("nvda_usdc"));
        assert!(base.matches_coin("deribit_price_index.nvda_usdc"));
        assert!(!base.matches_coin("TSLA"));
        assert!(!base.matches_coin("xyz:TSLA"));

        // clock_secs < 1
        let mut bad = base.clone();
        bad.clock_secs = 0;
        assert!(bad.validate().unwrap_err().contains("clock_secs"));

        // Probabilities outside [0, 1]
        let prob_mutators: [fn(&mut TickerV2ParamsJson); 9] = [
            |p| p.presets[0].q_1s = -0.01,
            |p| p.presets[0].q_1s = 1.01,
            |p| p.presets[1].p_move = -0.1,
            |p| p.presets[1].p_move = 1.5,
            |p| p.presets[3].mix_weight = -0.01,
            |p| p.presets[3].mix_weight = 1.1,
            |p| p.presets[2].m_base = -0.01,
            |p| p.presets[2].m_base = 1.0,
            |p| p.transition_1min[0][0] = -0.1,
        ];
        for mutator in prob_mutators {
            let mut c = base.clone();
            mutator(&mut c);
            assert!(c.validate().is_err(), "expected probability error");
        }

        // p1, p2 must be in (0, 1] (0.0 rejected, 1.0 accepted, >1.0 rejected)
        for bad_p in [0.0_f64, -0.2, 1.01, f64::NAN] {
            let mut c1 = base.clone();
            c1.presets[0].p1 = bad_p;
            assert!(c1.validate().unwrap_err().contains("p1"));
            let mut c2 = base.clone();
            c2.presets[3].p2 = bad_p;
            assert!(c2.validate().unwrap_err().contains("p2"));
        }
        let mut ok_one = base.clone();
        ok_one.presets[0].p1 = 1.0;
        ok_one.presets[0].p2 = 1.0;
        assert!(ok_one.validate().is_ok());

        // Thresholds must be strictly increasing and > 0
        for bad_thresh in [
            [0.0_f64, 2.0, 5.0],
            [-1.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            [3.0, 2.0, 5.0],
            [1.0, 5.0, 4.0],
        ] {
            let mut c = base.clone();
            c.thresholds_rms_cents = bad_thresh;
            assert!(c.validate().unwrap_err().contains("thresholds_rms_cents"));
        }

        // Transition row sum must equal 1 within 1e-6
        let mut bad_row = base.clone();
        bad_row.transition_1min[1][1] += 2e-6;
        assert!(bad_row.validate().unwrap_err().contains("row sum"));

        let mut ok_row = base.clone();
        ok_row.transition_1min[1][1] += 5e-7;
        assert!(ok_row.validate().is_ok());

        // Basic scalar sanity checks
        let mut empty_ticker = base.clone();
        empty_ticker.ticker = "  ".to_string();
        assert!(empty_ticker.validate().is_err());

        let mut zero_band = base.clone();
        zero_band.band_cents = 0;
        assert!(zero_band.validate().is_err());

        let mut bad_ref = base.clone();
        bad_ref.reference_price = 0.0;
        assert!(bad_ref.validate().is_err());

        let mut bad_void = base.clone();
        bad_void.void_gap_ms = 0;
        assert!(bad_void.validate().is_err());

        let mut bad_clamp = base.clone();
        bad_clamp.book_gap_clamp_cents = -1;
        assert!(bad_clamp.validate().is_err());

        // window_presets uses the exact same validation rules as presets.
        let win_mutators: [fn(&mut [RegimePreset; 4]); 7] = [
            |wp| wp[0].q_1s = -0.01,
            |wp| wp[1].p_move = 1.05,
            |wp| wp[2].mix_weight = 1.1,
            |wp| wp[0].p1 = 0.0,
            |wp| wp[3].p2 = 1.01,
            |wp| wp[1].rms_median_cents = 0.0,
            |wp| wp[2].m_base = 1.0,
        ];
        for mutator in win_mutators {
            let mut c = base.clone();
            let mut wp = c.presets;
            mutator(&mut wp);
            c.window_presets = Some(wp);
            assert!(
                c.validate().unwrap_err().contains("window_presets"),
                "expected window_presets validation error"
            );
        }

        // Wrong-length window_presets array in JSON fails deserialization.
        let mut raw_val: serde_json::Value =
            serde_json::to_value(&base).expect("serialize base to Value");
        let short_wp = raw_val["presets"].as_array().unwrap()[..3].to_vec();
        raw_val["window_presets"] = serde_json::Value::Array(short_wp);
        assert!(parse_v2_params_json(&raw_val.to_string()).is_err());

        // off_hours validation: nested off_hours rejected, ticker mismatch rejected, inner field errors caught.
        let mut plain_off = base.clone();
        plain_off.off_hours = None;

        let mut nested_off = base.clone();
        let mut mid_off = plain_off.clone();
        mid_off.off_hours = Some(Box::new(plain_off.clone()));
        nested_off.off_hours = Some(Box::new(mid_off));
        assert!(nested_off.validate().unwrap_err().contains("off_hours"));

        let mut wrong_ticker_off = base.clone();
        let mut tsla_off = plain_off.clone();
        tsla_off.ticker = "TSLA".to_string();
        wrong_ticker_off.off_hours = Some(Box::new(tsla_off));
        assert!(wrong_ticker_off.validate().unwrap_err().contains("off_hours.ticker"));

        let mut bad_inner_off = base.clone();
        let mut broken_off = plain_off.clone();
        broken_off.presets[0].p1 = 0.0;
        bad_inner_off.off_hours = Some(Box::new(broken_off));
        let err = bad_inner_off.validate().unwrap_err();
        assert!(err.contains("off_hours") && err.contains("p1"));

        assert!(parse_v2_params_json("{not json").is_err());
    }

    #[test]
    fn emit_rust_table_reproduces_compiled_nvda_and_tsla_deribit_tables() {
        for (expected_const, params) in [
            ("NVDA_DERIBIT_V2_PARAMS", &NVDA_DERIBIT_V2_PARAMS),
            ("TSLA_DERIBIT_V2_PARAMS", &TSLA_DERIBIT_V2_PARAMS),
        ] {
            let expected_off_const = format!("{expected_const}_OFF");
            let rendered = params.to_rust_deribit_const();
            assert!(
                rendered.starts_with(&format!(
                    "pub const {expected_off_const}: TickerV2Params = TickerV2Params {{"
                )),
                "expected off-hours const first:\n{rendered}"
            );
            let main_header = format!(
                "\npub const {expected_const}: TickerV2Params = TickerV2Params {{"
            );
            let (off_section, main_section) = rendered
                .split_once(&main_header)
                .unwrap_or_else(|| panic!("missing main const header {main_header}:\n{rendered}"));
            assert!(off_section.contains("    off_hours: None,\n"));
            assert!(main_section.contains(&format!(
                "    off_hours: Some(&{expected_off_const}),\n"
            )));
            assert!(rendered.ends_with("};\n"));
            assert!(main_section.contains(&format!("    ticker: {:?},", params.ticker)));
            assert!(main_section.contains(&format!("    band_cents: {},", params.band_cents)));
            assert!(main_section.contains(&format!(
                "    reference_price: {:?},",
                params.reference_price
            )));
            assert!(main_section.contains(&format!("    clock_secs: {},", params.clock_secs)));
            assert!(main_section.contains(&format!("    void_gap_ms: {},", params.void_gap_ms)));
            assert!(main_section.contains(&format!(
                "    book_lead_secs: {},",
                params.book_lead_secs
            )));
            assert!(main_section.contains(&format!(
                "    book_gap_beta: {:?},",
                params.book_gap_beta
            )));
            assert!(main_section.contains(&format!(
                "    book_gap_clamp_cents: {},",
                params.book_gap_clamp_cents
            )));
            assert!(main_section.contains(&format!(
                "    book_max_age_ms: {},",
                params.book_max_age_ms
            )));
            assert!(main_section.contains(&format!(
                "    rms_ref_cents: {:?},",
                params.rms_ref_cents
            )));
            assert!(main_section.contains(&format!(
                "    thresholds_rms_cents: [{:?}, {:?}, {:?}],",
                params.thresholds_rms_cents[0],
                params.thresholds_rms_cents[1],
                params.thresholds_rms_cents[2]
            )));
            assert!(main_section.contains(&format!(
                "    shock_single_step_cents: {},",
                params.shock_single_step_cents
            )));
            assert!(main_section.contains("    window_presets: Some([\n"));
            assert!(main_section.contains("    ]),\n"));
            for preset in &params.presets {
                assert!(main_section.contains(&format!("            q_1s: {:?},", preset.q_1s)));
                assert!(main_section.contains(&format!("            p_move: {:?},", preset.p_move)));
                assert!(main_section.contains(&format!(
                    "            mix_weight: {:?},",
                    preset.mix_weight
                )));
                assert!(main_section.contains(&format!("            p1: {:?},", preset.p1)));
                assert!(main_section.contains(&format!("            p2: {:?},", preset.p2)));
                assert!(main_section.contains(&format!(
                    "            rms_median_cents: {:?},",
                    preset.rms_median_cents
                )));
                assert!(main_section.contains(&format!("            m_base: {:?},", preset.m_base)));
            }
            for row in &params.transition_1min {
                assert!(main_section.contains(&format!(
                    "        [{:?}, {:?}, {:?}, {:?}],",
                    row[0], row[1], row[2], row[3]
                )));
            }
        }

        // Distinct window_presets and distinct off_hours values are emitted separately.
        let mut custom_json = TickerV2ParamsJson::from_params(&NVDA_DERIBIT_V2_PARAMS);
        custom_json.window_presets.as_mut().unwrap()[Regime::Normal.index()].p1 = 0.5123;
        custom_json.window_presets.as_mut().unwrap()[Regime::Normal.index()].p2 = 0.5123;
        custom_json.off_hours.as_mut().unwrap().book_gap_beta = 0.68;
        custom_json.off_hours.as_mut().unwrap().book_lead_secs = 2;
        let custom = custom_json.leak_static();
        let rendered_two_phase = custom.to_rust_deribit_const();
        let (off_part, main_part) = rendered_two_phase
            .split_once("\npub const NVDA_DERIBIT_V2_PARAMS: TickerV2Params = TickerV2Params {")
            .expect("expected _OFF followed by main const");
        assert!(off_part.contains("    book_gap_beta: 0.68,\n"));
        assert!(off_part.contains("    book_lead_secs: 2,\n"));
        assert!(off_part.contains("    off_hours: None,\n"));
        assert!(main_part.contains("    book_gap_beta: 1.0,\n"));
        assert!(main_part.contains("    off_hours: Some(&NVDA_DERIBIT_V2_PARAMS_OFF),\n"));
        let (_, wp_section) = main_part
            .split_once("    window_presets: Some([\n")
            .expect("expected window_presets: Some([ section");
        assert!(wp_section.contains("            p1: 0.5123,"));
        assert!(wp_section.contains("            p2: 0.5123,"));

        // When off_hours is None (e.g. HL), no _OFF const is emitted and both window_presets & off_hours are None.
        let rendered_none = NVDA_V2_PARAMS.to_rust_deribit_const();
        assert!(rendered_none.starts_with(
            "pub const NVDA_DERIBIT_V2_PARAMS: TickerV2Params = TickerV2Params {"
        ));
        assert!(!rendered_none.contains("_DERIBIT_V2_PARAMS_OFF"));
        assert!(rendered_none.contains("    window_presets: None,\n"));
        assert!(!rendered_none.contains("window_presets: Some"));
        assert!(rendered_none.contains("    off_hours: None,\n"));
    }

    #[test]
    fn two_phase_dp_window_presets_news_override_far_windows_and_json_round_trip() {
        // 1. Compiled table invariants: HL has None; Deribit has provisional Some(presets) and L = 3.
        assert!(NVDA_V2_PARAMS.window_presets.is_none());
        assert!(TSLA_V2_PARAMS.window_presets.is_none());
        assert_eq!(
            NVDA_DERIBIT_V2_PARAMS.window_presets,
            Some(NVDA_DERIBIT_V2_PARAMS.presets)
        );
        assert_eq!(
            TSLA_DERIBIT_V2_PARAMS.window_presets,
            Some(TSLA_DERIBIT_V2_PARAMS.presets)
        );
        assert_eq!(NVDA_DERIBIT_V2_PARAMS.book_lead_secs, 3);
        assert_eq!(TSLA_DERIBIT_V2_PARAMS.book_lead_secs, 3);

        let windows: Vec<TimeWindow> = (0..10)
            .map(|i| TimeWindow {
                start_secs: (5 + 5 * i) as u32,
                end_secs: (10 + 5 * i) as u32,
            })
            .collect();
        let bands: Vec<PriceBand> = (-4..=4_i64)
            .enumerate()
            .map(|(row_index, k)| PriceBand {
                row_index,
                lower: (13_998 + k * 3) as f64 / 100.0,
                upper: (14_001 + k * 3) as f64 / 100.0,
            })
            .collect();
        let adj = BookLeadAdjustment {
            gap_cents: 4,
            lead_secs: DERIBIT_BOOK_LEAD_SECS,
        };

        // 2. window_presets = Some(presets) is bit-identical to window_presets = None across all regimes,
        //    near & far windows, and with/without news_override.
        let mut drb_none = NVDA_DERIBIT_V2_PARAMS;
        drb_none.window_presets = None;
        let dyn_news = dynamic_news_preset(&NVDA_DERIBIT_V2_PARAMS, 12.0, 0.70);
        for regime in Regime::ALL {
            for news_ov in [None, Some(dyn_news)] {
                let p_some = price_grid_v2_with_book(
                    139.99,
                    &bands,
                    &windows,
                    &NVDA_DERIBIT_V2_PARAMS,
                    regime,
                    Some(0),
                    news_ov,
                    adj,
                );
                let p_none = price_grid_v2_with_book(
                    139.99,
                    &bands,
                    &windows,
                    &drb_none,
                    regime,
                    Some(0),
                    news_ov,
                    adj,
                );
                assert_eq!(
                    p_some, p_none,
                    "window_presets == Some(presets) must be bit-identical to None for {regime:?}"
                );
            }
        }

        // 3. Two-phase process with book pre-window steps and smaller index in-window steps:
        //    verify directional physics on near & far windows AND exact agreement with two-phase MC.
        let book_pre = RegimePreset {
            q_1s: 0.85,
            p_move: 0.85,
            mix_weight: 1.0,
            p1: 0.25,
            p2: 0.25,
            rms_median_cents: 3.5,
            m_base: 0.05,
        };
        let index_win = RegimePreset {
            q_1s: 0.72,
            p_move: 0.72,
            mix_weight: 1.0,
            p1: 0.42,
            p2: 0.42,
            rms_median_cents: 2.5,
            m_base: 0.05,
        };

        let mut all_book = NVDA_DERIBIT_V2_PARAMS;
        all_book.presets = [book_pre; 4];
        all_book.window_presets = None;
        all_book.off_hours = None;

        let mut two_phase = all_book;
        two_phase.window_presets = Some([index_win; 4]);

        let mut all_index = NVDA_DERIBIT_V2_PARAMS;
        all_index.presets = [index_win; 4];
        all_index.window_presets = None;
        all_index.off_hours = None;

        let no_gap = BookLeadAdjustment {
            gap_cents: 0,
            lead_secs: DERIBIT_BOOK_LEAD_SECS,
        };
        let p_book = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows,
            &all_book,
            Regime::Normal,
            Some(0),
            None,
            no_gap,
        );
        let p_two = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows,
            &two_phase,
            Regime::Normal,
            Some(0),
            None,
            no_gap,
        );
        let p_idx = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows,
            &all_index,
            Regime::Normal,
            Some(0),
            None,
            no_gap,
        );

        let d0 = 4usize; // [139.98, 140.01] centered on 139.99
        let d_far_up = 8usize; // +4 rows
        let d_far_dn = 0usize; // -4 rows

        for w in [0usize, 1, 5, 7, 9] {
            // Smaller in-window steps (two_phase vs all_book, same pre-window) strictly lower far-row touch prob
            // and concentrate probability on d0 relative to far rows (lower p(far)/p(d0)).
            assert!(
                p_two[d_far_up][w] < p_book[d_far_up][w],
                "w={w}: smaller in-window steps must lower far-up prob ({:.5} vs {:.5})",
                p_two[d_far_up][w],
                p_book[d_far_up][w]
            );
            assert!(
                p_two[d_far_dn][w] < p_book[d_far_dn][w],
                "w={w}: smaller in-window steps must lower far-dn prob ({:.5} vs {:.5})",
                p_two[d_far_dn][w],
                p_book[d_far_dn][w]
            );
            assert!(
                p_two[d_far_up][w] / p_two[d0][w] < p_book[d_far_up][w] / p_book[d0][w],
                "w={w}: smaller in-window steps must reduce far/center ratio"
            );
            // Smaller pre-window steps (all_index vs two_phase, same in-window) raise price-row d0 prob.
            assert!(
                p_idx[d0][w] > p_two[d0][w],
                "w={w}: smaller pre-window steps must raise d0 prob ({:.5} vs {:.5})",
                p_idx[d0][w],
                p_two[d0][w]
            );
        }
        // Comparing all_index vs all_book on near windows: smaller steps lower far row and raise price row d0.
        assert!(p_idx[d_far_up][0] < p_book[d_far_up][0]);
        assert!(p_idx[d0][0] > p_book[d0][0]);

        // Exact two-phase Monte Carlo verification on w0 (ws=5-3=2, we=7) and w1 (ws=10-3=7, we=12) with G=+4c.
        let p_two_gap = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows[..2],
            &two_phase,
            Regime::Normal,
            Some(0),
            None,
            adj,
        );
        let n_paths: usize = 60_000;
        let mut rng = StdRng::seed_from_u64(2026_1125);
        for (w_idx, ws, we) in [(0usize, 2usize, 7usize), (1usize, 7usize, 12usize)] {
            for row in [d0, d_far_up] {
                let lc = (bands[row].lower * 100.0).round() as i64;
                let uc = (bands[row].upper * 100.0).round() as i64;
                let mut hits = 0usize;
                for _ in 0..n_paths {
                    let mut c = 13_999 + adj.gap_cents;
                    let mut hit = false;
                    for s in 1..we {
                        let prev = c;
                        let step_preset = if s < ws { &book_pre } else { &index_win };
                        if rng.random::<f64>() < step_preset.p_move {
                            let k = sample_geom(step_preset.p1, &mut rng);
                            if rng.random::<bool>() {
                                c += k;
                            } else {
                                c -= k;
                            }
                        }
                        if s >= ws && prev.min(c) <= uc && prev.max(c) >= lc {
                            hit = true;
                            break;
                        }
                    }
                    if hit {
                        hits += 1;
                    }
                }
                let mc = hits as f64 / n_paths as f64;
                let dp = p_two_gap[row][w_idx];
                let se = (dp * (1.0 - dp) / n_paths as f64).sqrt().max(0.002);
                assert!(
                    (dp - mc).abs() <= 4.0 * se,
                    "two-phase MC mismatch w={w_idx} row={row}: DP={dp:.5} MC={mc:.5} (4*se={:.5})",
                    4.0 * se
                );
            }
        }

        // 4. news_override applies to BOTH phases (near Regime::News ignores window_presets[News]).
        let mut news_split = NVDA_DERIBIT_V2_PARAMS;
        let mut wp_news = news_split.presets;
        wp_news[Regime::News.index()].p1 = 0.50;
        wp_news[Regime::News.index()].p2 = 0.50;
        wp_news[Regime::News.index()].mix_weight = 1.0;
        news_split.window_presets = Some(wp_news);

        let p_news_no_ov_base = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows[..NEAR_WINDOWS],
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::News,
            Some(0),
            None,
            no_gap,
        );
        let p_news_no_ov_split = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows[..NEAR_WINDOWS],
            &news_split,
            Regime::News,
            Some(0),
            None,
            no_gap,
        );
        assert_ne!(p_news_no_ov_base, p_news_no_ov_split);

        let p_news_ov_base = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows[..NEAR_WINDOWS],
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::News,
            Some(0),
            Some(dyn_news),
            no_gap,
        );
        let p_news_ov_split = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows[..NEAR_WINDOWS],
            &news_split,
            Regime::News,
            Some(0),
            Some(dyn_news),
            no_gap,
        );
        assert_eq!(
            p_news_ov_base, p_news_ov_split,
            "news_override must override both pre-window and in-window News presets"
        );

        // 5. Far windows (Markov regime blend) use window_presets[r] for every active regime r.
        let mut far_split = NVDA_DERIBIT_V2_PARAMS;
        let mut wp_far = far_split.presets;
        // Shrink only Normal's in-window steps, starting in Quiet so Normal only appears in far windows.
        wp_far[Regime::Normal.index()].p1 = 0.80;
        wp_far[Regime::Normal.index()].p2 = 0.80;
        far_split.window_presets = Some(wp_far);

        let p_far_base = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows,
            &NVDA_DERIBIT_V2_PARAMS,
            Regime::Quiet,
            Some(0),
            None,
            no_gap,
        );
        let p_far_split = price_grid_v2_with_book(
            139.99,
            &bands,
            &windows,
            &far_split,
            Regime::Quiet,
            Some(0),
            None,
            no_gap,
        );
        for w in 0..NEAR_WINDOWS {
            for r in 0..bands.len() {
                assert_eq!(p_far_base[r][w], p_far_split[r][w]);
            }
        }
        assert!(
            p_far_split[d_far_up][9] < p_far_base[d_far_up][9],
            "far Markov window must reflect shrunken Normal window_presets"
        );

        // 6. JSON round trip WITH distinct window_presets and WITHOUT window_presets (absent / null -> None).
        let json_with = two_phase.to_json_pretty().unwrap();
        assert!(json_with.contains("\"window_presets\""));
        let parsed_with = parse_v2_params_json(&json_with).unwrap();
        assert_eq!(parsed_with.window_presets, two_phase.window_presets);
        let loaded_with = parsed_with.leak_static();
        assert_eq!(loaded_with, &two_phase);
        assert_eq!(
            price_grid_v2_with_book(139.99, &bands, &windows, loaded_with, Regime::Normal, Some(0), None, adj),
            price_grid_v2_with_book(139.99, &bands, &windows, &two_phase, Regime::Normal, Some(0), None, adj)
        );

        // Absent field in JSON deserializes to None (and HL dump omits the field).
        let json_without = all_book.to_json_pretty().unwrap();
        assert!(!json_without.contains("\"window_presets\""));
        let parsed_without = parse_v2_params_json(&json_without).unwrap();
        assert!(parsed_without.window_presets.is_none());
        let loaded_without = parsed_without.leak_static();
        assert_eq!(loaded_without, &all_book);

        // Explicit `"window_presets": null` also deserializes cleanly to None.
        let mut val_null: serde_json::Value = serde_json::from_str(&json_without).unwrap();
        val_null["window_presets"] = serde_json::Value::Null;
        let parsed_null = parse_v2_params_json(&val_null.to_string()).unwrap();
        assert!(parsed_null.window_presets.is_none());
    }

    #[test]
    fn off_hours_session_selection_and_json_round_trip() {
        // 1. Compiled table invariants: HL has None; Deribit has provisional Some(&*_OFF).
        assert!(NVDA_V2_PARAMS.off_hours.is_none());
        assert!(TSLA_V2_PARAMS.off_hours.is_none());
        assert_eq!(NVDA_DERIBIT_V2_PARAMS.off_hours, Some(&NVDA_DERIBIT_V2_PARAMS_OFF));
        assert_eq!(TSLA_DERIBIT_V2_PARAMS.off_hours, Some(&TSLA_DERIBIT_V2_PARAMS_OFF));
        assert!(NVDA_DERIBIT_V2_PARAMS_OFF.off_hours.is_none());
        assert!(TSLA_DERIBIT_V2_PARAMS_OFF.off_hours.is_none());

        // 2. Build a distinct off-hours table and verify `for_session` across all session boundaries.
        let mut base_json = TickerV2ParamsJson::from_params(&NVDA_DERIBIT_V2_PARAMS);
        base_json.off_hours = None;
        let mut off_json = base_json.clone();
        off_json.book_gap_beta = 0.68;
        off_json.book_lead_secs = 2;
        off_json.book_gap_clamp_cents = 12;
        off_json.rms_ref_cents = 0.90;
        off_json.thresholds_rms_cents = [0.60, 1.30, 3.20];
        off_json.shock_single_step_cents = 15;
        off_json.presets[Regime::Normal.index()].m_base = 0.14;

        let mut split_json = base_json.clone();
        split_json.off_hours = Some(Box::new(off_json.clone()));
        let split = split_json.clone().leak_static();

        // US regular hours (2026-11-17 15:00 UTC = 10:00 ET) -> main table
        assert_eq!(split.for_session(1_794_927_600_000).book_gap_beta, 1.0);
        // Night (2026-11-18 03:00 UTC = 22:00 ET) -> off table
        assert_eq!(split.for_session(1_794_970_800_000).book_gap_beta, 0.68);
        // Weekend (2026-11-21 15:00 UTC = Sat 10:00 ET) -> off table
        assert_eq!(split.for_session(1_795_273_200_000).book_gap_beta, 0.68);
        // Holiday (2026-11-26 15:00 UTC = Thanksgiving) -> off table
        assert_eq!(split.for_session(1_795_705_200_000).book_gap_beta, 0.68);
        // Early-close boundary 2026-11-27 12:59:59 ET (17:59:59 UTC) vs 13:00:00 ET (18:00:00 UTC)
        assert_eq!(split.for_session(1_795_802_399_000).book_gap_beta, 1.0);
        assert_eq!(split.for_session(1_795_802_400_000).book_gap_beta, 0.68);
        // Early-close boundary 2026-12-24 12:59:59 ET vs 13:00:00 ET
        assert_eq!(split.for_session(1_798_135_199_000).book_gap_beta, 1.0);
        assert_eq!(split.for_session(1_798_135_200_000).book_gap_beta, 0.68);

        // 3. JSON round trip WITH distinct off_hours, WITHOUT off_hours (omitted), and `"off_hours": null`.
        let json_with = split.to_json_pretty().unwrap();
        assert!(json_with.contains("\"off_hours\""));
        let parsed_with = parse_v2_params_json(&json_with).unwrap();
        assert_eq!(parsed_with, split_json);
        let loaded_with = parsed_with.leak_static();
        assert_eq!(loaded_with, split);
        assert_eq!(loaded_with.for_session(1_795_802_399_000).book_gap_beta, 1.0);
        assert_eq!(loaded_with.for_session(1_795_802_400_000).book_gap_beta, 0.68);

        let no_off = base_json.clone().leak_static();
        let json_without = no_off.to_json_pretty().unwrap();
        assert!(!json_without.contains("\"off_hours\""));
        let parsed_without = parse_v2_params_json(&json_without).unwrap();
        assert!(parsed_without.off_hours.is_none());
        assert_eq!(parsed_without.leak_static(), no_off);

        let mut val_null: serde_json::Value = serde_json::from_str(&json_without).unwrap();
        val_null["off_hours"] = serde_json::Value::Null;
        let parsed_null = parse_v2_params_json(&val_null.to_string()).unwrap();
        assert!(parsed_null.off_hours.is_none());
    }
}
