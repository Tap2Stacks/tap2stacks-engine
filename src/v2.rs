//! Odds Model v2: 3-second-clock zero-inflated cent-jump process, exact print-crossing
//! 1-D Dynamic Programming touch probabilities, causal trailing-60s RMS regime detector,
//! far-horizon Markov regime switching, and QTAP pricing without calibration layers.
//!
//! All ticker preset parameters below are direct MLEs from the 7-day recorded tick
//! statistics in `logs/calibration/m1-tick-stats.txt` (2026-11-17..2026-11-23).
//! Tickers without sufficient recorded tick data (AAPL, SP500) return `None` from
//! `v2_ticker_params` and stay on model v1.

use std::collections::VecDeque;

use crate::grid::{compute_k_skew, GridConfig, PriceBand, TimeWindow};
use crate::horizon::NEAR_WINDOWS;
use crate::price::PriceTick;

/// Oracle mark-price refresh period in seconds (`m1-tick-stats.txt` lines 47-50, 175-178).
pub const ORACLE_CLOCK_SECS: u64 = 3;
/// Settlement void gap threshold in milliseconds (`src/calibrate/eval.rs:3`).
pub const VOID_GAP_MS: u64 = 3_000;
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
/// Upper clamp for causal trailing 60s move rate `q_60` in `News` (`3 * q_60 <= 1.0`).
pub const V2_NEWS_Q_MAX: f64 = 1.0 / 3.0;
/// Upper RMS clamp multiplier on `R_med_news` in `dynamic_news_preset` (`docs/odds-model-v2.md` §3.4).
pub const V2_NEWS_R_CAP_MULT: f64 = 2.5;
/// Minimum probability denominator clamp in `M_base = (1 - m_eff) / max(P, eps)`.
pub const V2_P_EPSILON: f64 = 1e-6;
/// Per-3-second-tick generator weight `3 s / 60 s = 0.05` for far-horizon Markov switching.
const FAR_MARKOV_STEP_WEIGHT: f64 = 3.0 / 60.0;
/// Half-width of the integer-cent DP lattice (`[-450c, +450c]`, `901` states).
const DP_HALF_WIDTH: usize = 450;

/// Selects which odds model `QuoteEngine` runs. Default is `V1` until the owner approves v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PricingModel {
    #[default]
    V1,
    V2,
}

impl PricingModel {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "v1" | "1" => Some(Self::V1),
            "v2" | "2" => Some(Self::V2),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegimePreset {
    /// Empirical unconditional 1-second move probability `q = P(ΔC != 0)`.
    pub q_1s: f64,
    /// Per-3-second oracle clock move probability `p_move = min(1, 3 * q_1s)`.
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

/// Complete data-derived v2 parameters for one ticker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickerV2Params {
    pub ticker: &'static str,
    pub band_cents: u32,
    pub reference_price: f64,
    /// Median `R_60` of the `Normal` regime (used as `R_ref` in `m_risk`).
    pub rms_ref_cents: f64,
    /// Empirical `[p50, p85, p99]` minute `R_60` thresholds in cents/s.
    pub thresholds_rms_cents: [f64; 3],
    /// Single-step shock guard `ceil(4 * p99)` in integer cents (`docs/odds-model-v2.md` §3.4).
    pub shock_single_step_cents: i64,
    /// Ordered `[Quiet, Normal, Volatile, News]`.
    pub presets: [RegimePreset; 4],
    /// Empirical 4x4 1-minute regime transition matrix `T[from][to]`, row-stochastic.
    pub transition_1min: [[f64; 4]; 4],
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
    transition_1min: [
        [0.816, 0.171, 0.013, 0.000],
        [0.252, 0.645, 0.102, 0.001],
        [0.029, 0.273, 0.674, 0.024],
        [0.000, 0.012, 0.353, 0.635],
    ],
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
    transition_1min: [
        [0.825, 0.160, 0.015, 0.000],
        [0.236, 0.620, 0.140, 0.004],
        [0.041, 0.366, 0.567, 0.026],
        [0.012, 0.094, 0.412, 0.482],
    ],
};

/// Returns data-derived v2 parameters for tickers with recorded history (`NVDA`, `TSLA`).
/// Tickers without sufficient tick data (`AAPL`, `SP500`) return `None` and stay on model v1.
pub fn v2_ticker_params(ticker: &str) -> Option<&'static TickerV2Params> {
    let symbol = ticker.split(':').next_back().unwrap_or(ticker);
    if symbol.eq_ignore_ascii_case("NVDA") {
        Some(&NVDA_V2_PARAMS)
    } else if symbol.eq_ignore_ascii_case("TSLA") {
        Some(&TSLA_V2_PARAMS)
    } else {
        None
    }
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

    /// Infers oracle phase `phi in {0, 1, 2}` relative to `quote_ts_ms` from the last
    /// observed non-zero price change. Returns `None` if unknown (restart, gap > 3s, or
    /// > 60s since last move), causing the DP to average uniformly over `{0, 1, 2}`.
    pub fn inferred_phase(&self, quote_ts_ms: u64) -> Option<usize> {
        let t_last = self.last_change_sec?;
        let t_now = quote_ts_ms / 1_000;
        if t_now < t_last || t_now.saturating_sub(t_last) > PHASE_MAX_AGE_SECS {
            return None;
        }
        let phi = ((t_last % ORACLE_CLOCK_SECS) + ORACLE_CLOCK_SECS - (t_now % ORACLE_CLOCK_SECS))
            % ORACLE_CLOCK_SECS;
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

        // A > 3 s feed gap voids overlapping windows, invalidates oracle phase tracking,
        // and must NOT be counted as a 1-second price jump in R_60.
        if dt_ms > VOID_GAP_MS {
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

    let r_min = params.thresholds_rms_cents[2];
    let r_max = (V2_NEWS_R_CAP_MULT * base_news.rms_median_cents).max(r_min);
    let r_eff = r_clean.clamp(r_min, r_max);
    let q_eff = q_clean.clamp(V2_NEWS_Q_MIN, V2_NEWS_Q_MAX);
    let p_move = (ORACLE_CLOCK_SECS as f64 * q_eff).min(1.0);

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
            // General fallback for windows wider than 5s (not used by standard 5s grid).
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

/// Computes exact v2 print-crossing touch probabilities `[row][window]` via 1-D Geometric IIR DP.
///
/// - Windows `0..min(6, windows.len())`: evolve purely in `regime`.
/// - Windows `6..windows.len()`: blend in `params.transition_1min` via per-3s-tick Markov step
///   `T_step = 0.95 * I + 0.05 * T` starting after window 5.
/// - `inferred_phase`: if `Some(phi)`, evaluates exact conditional probability at `phi`;
///   if `None`, averages uniformly over `phi in {0, 1, 2}`.
/// - `news_override`: when currently in `Regime::News`, replaces `presets[News]` with the
///   causally scaled `dynamic_news_preset(params, r60, q60)`.
pub fn price_grid_v2(
    spot: f64,
    bands: &[PriceBand],
    windows: &[TimeWindow],
    params: &TickerV2Params,
    regime: Regime,
    inferred_phase: Option<usize>,
    news_override: Option<RegimePreset>,
) -> Vec<Vec<f64>> {
    let rows = bands.len();
    let w_count = windows.len();
    let mut probs = vec![vec![0.0_f64; w_count]; rows];
    if rows == 0 || w_count == 0 {
        return probs;
    }

    let spot_cents = (spot * 100.0).round() as i64;
    let half = DP_HALF_WIDTH;
    let n = 2 * half + 1;
    let row_bounds: Vec<(usize, usize)> = bands
        .iter()
        .map(|band| {
            let lc = (band.lower * 100.0).round() as i64;
            let uc = (band.upper * 100.0).round() as i64;
            let li = ((lc - spot_cents) + half as i64).clamp(0, n as i64) as usize;
            let ui = ((uc - spot_cents) + half as i64).clamp(0, (n - 1) as i64) as usize;
            (li, ui)
        })
        .collect();

    // Per-3s-tick Markov transition matrix for far windows (6..23): T_step = 0.95 I + 0.05 T.
    let mut t_step = [[0.0_f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let eye = if r == c { 1.0 } else { 0.0 };
            t_step[r][c] =
                (1.0 - FAR_MARKOV_STEP_WEIGHT) * eye + FAR_MARKOV_STEP_WEIGHT * params.transition_1min[r][c];
        }
    }

    let mut presets = params.presets;
    if let Some(np) = news_override {
        presets[Regime::News.index()] = np;
    }

    let phase_list: &[usize] = match inferred_phase {
        Some(0) => &[0],
        Some(1) => &[1],
        Some(2) => &[2],
        _ => &[0, 1, 2],
    };
    let phase_weight = 1.0 / phase_list.len() as f64;
    let r0 = regime.index();
    let preset0 = &presets[r0];

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

    for &ph in phase_list {
        u_cur.fill(0.0);
        u_cur[half] = 1.0;
        let mut cur_s: usize = 0;
        let mut far_initialized = false;

        for (w_idx, window) in windows.iter().enumerate() {
            let ws = window.start_secs as usize;
            let we = window.end_secs as usize;

            if w_idx < NEAR_WINDOWS {
                // Near horizon (windows 0..5): pure current preset r0.
                while cur_s + 1 < ws {
                    cur_s += 1;
                    if (cur_s + ORACLE_CLOCK_SECS as usize - ph) % ORACLE_CLOCK_SECS as usize == 0 {
                        full_step_clamped(&u_cur, preset0, &mut u_next);
                        std::mem::swap(&mut u_cur, &mut u_next);
                    }
                }
                let in_ticks = (ws..we)
                    .filter(|&s| (s + ORACLE_CLOCK_SECS as usize - ph) % ORACLE_CLOCK_SECS as usize == 0)
                    .count();
                for (row, &(li, ui)) in row_bounds.iter().enumerate() {
                    let surv = window_survival_mass(&u_cur, li, ui, in_ticks, preset0, &mut scratch);
                    let hit = (1.0 - surv).clamp(0.0, 1.0);
                    probs[row][w_idx] += phase_weight * hit;
                }
                while cur_s + 1 < we {
                    cur_s += 1;
                    if (cur_s + ORACLE_CLOCK_SECS as usize - ph) % ORACLE_CLOCK_SECS as usize == 0 {
                        full_step_clamped(&u_cur, preset0, &mut u_next);
                        std::mem::swap(&mut u_cur, &mut u_next);
                    }
                }
            } else {
                // Far horizon (windows 6..23): Markov-modulated 4-regime DP using T_step.
                if !far_initialized {
                    for slice in &mut u_reg {
                        slice.fill(0.0);
                    }
                    u_reg[r0].copy_from_slice(&u_cur);
                    far_initialized = true;
                }
                while cur_s + 1 < ws {
                    cur_s += 1;
                    if (cur_s + ORACLE_CLOCK_SECS as usize - ph) % ORACLE_CLOCK_SECS as usize == 0 {
                        step_far_markov(&mut u_reg, &mut u_mix, &t_step, &presets);
                    }
                }
                let in_ticks = (ws..we)
                    .filter(|&s| (s + ORACLE_CLOCK_SECS as usize - ph) % ORACLE_CLOCK_SECS as usize == 0)
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
                                &presets[r],
                                &mut scratch,
                            );
                        }
                    }
                    let hit = (1.0 - total_surv).clamp(0.0, 1.0);
                    probs[row][w_idx] += phase_weight * hit;
                }
                while cur_s + 1 < we {
                    cur_s += 1;
                    if (cur_s + ORACLE_CLOCK_SECS as usize - ph) % ORACLE_CLOCK_SECS as usize == 0 {
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
        let mut rng = StdRng::seed_from_u64(seed);
        let mut hits = 0usize;
        for _ in 0..n_paths {
            let mut c = spot_c;
            let ph = phase.unwrap_or_else(|| rng.random_range(0..3));
            let mut hit = false;
            for s in 1..w_end {
                let prev = c;
                if (s + 3 - ph) % 3 == 0 && rng.random::<f64>() < preset.p_move {
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
        assert_eq!(det.inferred_phase(100_000), None);

        // Quiet tape: 1c move every 9s -> low R_60, holds below 0.85 * 1.162 = 0.988 for 15s -> Quiet
        let mut price = 140.00_f64;
        for sec in 0..40u64 {
            if sec == 10 || sec == 22 {
                price += 0.01;
            }
            det.on_tick(PriceTick { ts_ms: sec * 1_000, price }, &NVDA_V2_PARAMS);
        }
        assert_eq!(det.current_regime(), Regime::Quiet);
        // Last change was at sec=22 -> at quote_ts=40s, phi = ((22%3)+3-(40%3))%3 = (1+3-1)%3 = 0
        assert_eq!(det.inferred_phase(40_000), Some(0));
        assert_eq!(det.inferred_phase(41_000), Some(2));
        assert_eq!(det.inferred_phase(42_000), Some(1));

        // > 3s gap resets phase to None and does NOT trigger a bogus shock
        det.on_tick(PriceTick { ts_ms: 46_000, price: 141.00 }, &NVDA_V2_PARAMS);
        assert_eq!(det.inferred_phase(46_000), None);
        assert_eq!(det.current_regime(), Regime::Quiet);

        // Single-print shock > 34c (4 * p99) promotes immediately to News on print 1
        det.on_tick(PriceTick { ts_ms: 47_000, price: 141.40 }, &NVDA_V2_PARAMS);
        assert_eq!(det.current_regime(), Regime::News);
        assert_eq!(det.inferred_phase(47_000), Some(0));

        // Downgrade hysteresis: stays News during < 15s of calm before stepping down one level
        for sec in 48..120u64 {
            det.on_tick(PriceTick { ts_ms: sec * 1_000, price: 141.40 }, &NVDA_V2_PARAMS);
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
            (&TSLA_V2_PARAMS, 14.20_f64, 0.40_f64, 14.20_f64, V2_NEWS_Q_MAX),
        ] {
            let np = dynamic_news_preset(params, r60, q60);
            assert!((np.rms_median_cents - expected_r_eff).abs() < 1e-9);
            assert!((np.q_1s - expected_q_eff).abs() < 1e-9);
            assert!((np.p_move - (3.0 * expected_q_eff).min(1.0)).abs() < 1e-9);
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
            det.on_tick(PriceTick { ts_ms: sec * 1_000, price }, &NVDA_V2_PARAMS);
        }
        assert!((det.last_q60() - (6.0 / 30.0)).abs() < 1e-9);

        // 30c print (above old 3*p99=26c, below new 4*p99=34c) must NOT force News.
        price += 0.30;
        det.on_tick(PriceTick { ts_ms: 31_000, price }, &NVDA_V2_PARAMS);
        assert_ne!(det.current_regime(), Regime::News);

        // 35c print (> 4*p99=34c) promotes immediately to News on print 1.
        price += 0.35;
        det.on_tick(PriceTick { ts_ms: 32_000, price }, &NVDA_V2_PARAMS);
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
    }
}
