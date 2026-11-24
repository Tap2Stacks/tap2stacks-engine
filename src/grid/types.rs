use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OhlcBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeWindow {
    pub start_secs: u32,
    pub end_secs: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriceBand {
    pub row_index: usize,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolatilitySnapshot {
    pub variance: f64,
    pub sigma_raw: f64,
    pub sigma: f64,
    pub jump_intensity: f64,
    pub jump_flag: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeState {
    pub variance: Option<f64>,
    pub last_price: Option<f64>,
    pub volatility: VolatilitySnapshot,
    pub jump_sampler: JumpSamplerState,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            variance: None,
            last_price: None,
            volatility: VolatilitySnapshot {
                variance: 0.0,
                sigma_raw: 0.0,
                sigma: 0.0,
                jump_intensity: 0.0,
                jump_flag: false,
            },
            jump_sampler: JumpSamplerState::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellQuote {
    pub server_ts_ms: u64,
    pub band: PriceBand,
    pub window: TimeWindow,
    pub hit_probability: f64,
    pub multiplier: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LiabilityBook {
    by_cell: HashMap<usize, f64>,
}

impl LiabilityBook {
    pub fn set_liability(&mut self, cell_index: usize, value: f64) {
        self.by_cell.insert(cell_index, value.max(0.0));
    }

    pub fn add_liability(&mut self, cell_index: usize, value: f64) {
        let entry = self.by_cell.entry(cell_index).or_insert(0.0);
        *entry = (*entry + value).max(0.0);
    }

    pub fn get_liability(&self, cell_index: usize) -> f64 {
        self.by_cell.get(&cell_index).copied().unwrap_or(0.0)
    }

    pub fn snapshot(&self) -> HashMap<usize, f64> {
        self.by_cell.clone()
    }

    pub fn clear(&mut self) {
        self.by_cell.clear();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridSnapshot {
    pub server_ts_ms: u64,
    pub spot: f64,
    pub anchor: f64,
    pub volatility: VolatilitySnapshot,
    pub cells: Vec<CellQuote>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JumpConfig {
    pub enabled: bool,
    pub hawkes_mu0: f64,
    pub alpha: f64,
    pub beta: f64,
    pub jump_mean: f64,
    pub jump_stddev: f64,
    pub kappa0: f64,
    pub kappa_q: f64,
    pub kappa_min: f64,
    pub kappa_max: f64,
}

impl Default for JumpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hawkes_mu0: 0.02,
            alpha: 0.15,
            beta: 0.5,
            jump_mean: 0.0,
            jump_stddev: 0.012,
            kappa0: 3.3,
            kappa_q: 0.5,
            kappa_min: 2.7,
            kappa_max: 6.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JumpSamplerState {
    magnitudes: VecDeque<f64>,
    signs: VecDeque<i8>,
    max_samples: usize,
}

impl JumpSamplerState {
    pub fn new(max_samples: usize) -> Self {
        Self {
            magnitudes: VecDeque::with_capacity(max_samples),
            signs: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    pub fn record(&mut self, log_return: f64) {
        if log_return == 0.0 {
            return;
        }
        if self.magnitudes.len() == self.max_samples {
            self.magnitudes.pop_front();
        }
        if self.signs.len() == self.max_samples {
            self.signs.pop_front();
        }
        self.magnitudes.push_back(log_return.abs());
        self.signs.push_back(if log_return > 0.0 { 1 } else { -1 });
    }

    pub fn sample(&self, n: usize, rng: &mut impl rand::Rng, fallback_sigma: f64) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }

        let magnitudes = if self.magnitudes.is_empty() {
            (0..n)
                .map(|_| {
                    let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
                    fallback_sigma.abs() * z.abs()
                })
                .collect::<Vec<_>>()
        } else {
            (0..n)
                .map(|_| {
                    let index = rng.random_range(0..self.magnitudes.len());
                    self.magnitudes[index]
                })
                .collect::<Vec<_>>()
        };

        let p_up = if self.signs.is_empty() {
            0.5
        } else {
            let up_count = self.signs.iter().filter(|sign| **sign > 0).count() as f64;
            (up_count + 1.0) / (self.signs.len() as f64 + 2.0)
        };

        magnitudes
            .into_iter()
            .map(|magnitude| {
                let sign = if rng.random_bool(p_up) { 1.0 } else { -1.0 };
                magnitude * sign
            })
            .collect()
    }
}

impl Default for JumpSamplerState {
    fn default() -> Self {
        Self::new(5_000)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridSpec {
    pub rows: usize,
    pub center_row: usize,
    pub band_width: f64,
    pub windows: Vec<TimeWindow>,
}

impl Default for GridSpec {
    fn default() -> Self {
        let rows = 20;
        Self {
            rows,
            center_row: (rows - 1) / 2,
            band_width: 10.0,
            windows: vec![
                TimeWindow {
                    start_secs: 5,
                    end_secs: 10,
                },
                TimeWindow {
                    start_secs: 10,
                    end_secs: 15,
                },
                TimeWindow {
                    start_secs: 15,
                    end_secs: 20,
                },
                TimeWindow {
                    start_secs: 20,
                    end_secs: 25,
                },
                TimeWindow {
                    start_secs: 25,
                    end_secs: 30,
                },
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TouchMode {
    #[default]
    Continuous,
    Discrete,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulationConfig {
    pub mu: f64,
    pub seed: Option<u64>,
    pub use_parallel: bool,
    pub parallel_workers: Option<usize>,
    pub lock_offset: u32,
    pub bridge_epsilon: f64,
    pub use_antithetic: bool,
    pub mc_n_min: usize,
    pub mc_n_max: usize,
    pub se_abs: f64,
    pub se_rel: f64,
    pub p_floor: f64,
    pub adaptive_mc: bool,
    pub touch_mode: TouchMode,
    pub quote_window_count: usize,
    pub far_refresh_ms: u64,
    pub far_trigger_sigma: f64,
    pub far_phase_ms: u64,
    pub pricing_model: crate::v2::PricingModel,
    pub v2_forced_regime: Option<crate::v2::Regime>,
    pub price_source: crate::price::PriceSource,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            mu: 0.0,
            seed: None,
            use_parallel: true,
            parallel_workers: Some(8),
            lock_offset: 1,
            bridge_epsilon: 1e-6,
            use_antithetic: true,
            mc_n_min: 5_000,
            mc_n_max: 10_000,
            se_abs: 0.001,
            se_rel: 0.10,
            p_floor: 0.01,
            adaptive_mc: false,
            touch_mode: TouchMode::Continuous,
            quote_window_count: crate::horizon::DEFAULT_WINDOWS,
            far_refresh_ms: crate::horizon::FAR_REFRESH_MS,
            far_trigger_sigma: crate::horizon::DEFAULT_TRIGGER_SIGMA,
            far_phase_ms: 0,
            pricing_model: crate::v2::PricingModel::V1,
            v2_forced_regime: None,
            price_source: crate::price::PriceSource::Hyperliquid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolatilityConfig {
    pub sigma_scale: f64,
    pub sigma_ref: f64,
    pub ewma_half_life_seconds: f64,
    pub parkinson_weight: f64,
    pub variance_floor: f64,
    pub price_adj_reference: f64,
    pub price_adj_min_price: f64,
    pub price_adj_min: f64,
    pub price_adj_max: f64,
    pub jump: JumpConfig,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            sigma_scale: 0.35,
            sigma_ref: 0.002,
            ewma_half_life_seconds: 30.0,
            parkinson_weight: 0.6,
            variance_floor: 1e-8,
            price_adj_reference: 100_000.0,
            price_adj_min_price: 10_000.0,
            price_adj_min: 0.7,
            price_adj_max: 1.3,
            jump: JumpConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PricingConfig {
    pub m_base: f64,
    pub off_hours_m_base: Option<f64>,
    pub m_risk_max: f64,
    pub lambda_ref: f64,
    pub a_sigma: f64,
    pub a_lambda: f64,
    pub beta_skew: f64,
    pub pool_cap: f64,
    pub risk_budget: f64,
    pub k_safety_ttl_ms: u64,
    pub liability_rho: f64,
    pub p_raw_gamma: f64,
    pub p_near_boost: f64,
    pub p_time_boost: f64,
    pub calibration_dist_max: usize,
    pub calibration_sigma_edges: (f64, f64),
    pub current_row_logit_shift: f64,
    pub adjacent_row_logit_shift: f64,
    pub zone_vol_shift_scale: f64,
    pub min_multiplier: f64,
    pub max_multiplier: f64,
    pub skew_factor: f64,
    pub safety_factor: f64,
}

impl Default for PricingConfig {
    fn default() -> Self {
        Self {
            m_base: 0.02,
            off_hours_m_base: None,
            m_risk_max: 0.025,
            lambda_ref: 0.1,
            a_sigma: 0.5,
            a_lambda: 0.5,
            beta_skew: 4.0,
            pool_cap: 5_000.0,
            risk_budget: 50_000.0,
            k_safety_ttl_ms: 1_000,
            liability_rho: 0.7,
            p_raw_gamma: 2.5,
            p_near_boost: 0.0,
            p_time_boost: -0.35,
            calibration_dist_max: 6,
            calibration_sigma_edges: (0.8, 1.2),
            current_row_logit_shift: 1.0,
            adjacent_row_logit_shift: 1.6,
            zone_vol_shift_scale: 0.2,
            min_multiplier: 1.05,
            max_multiplier: 25.0,
            skew_factor: 1.0,
            safety_factor: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CalibrationKey {
    pub window_index: usize,
    pub dist_bucket: usize,
    pub sigma_bucket: usize,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CalibrationConfig {
    pub factors: HashMap<CalibrationKey, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridConfig {
    pub spec: GridSpec,
    pub simulation: SimulationConfig,
    pub volatility: VolatilityConfig,
    pub pricing: PricingConfig,
    pub calibration: CalibrationConfig,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            spec: GridSpec::default(),
            simulation: SimulationConfig::default(),
            volatility: VolatilityConfig::default(),
            pricing: PricingConfig::default(),
            calibration: CalibrationConfig::default(),
        }
    }
}

impl GridConfig {
    pub fn validate(&self) -> Result<(), GridError> {
        if self.spec.rows == 0 {
            return Err(GridError::InvalidConfig("rows must be > 0"));
        }
        if self.spec.center_row >= self.spec.rows {
            return Err(GridError::InvalidConfig(
                "center_row must be strictly smaller than rows",
            ));
        }
        if self.spec.band_width <= 0.0 {
            return Err(GridError::InvalidConfig("band_width must be > 0"));
        }
        if self.spec.windows.is_empty() {
            return Err(GridError::InvalidConfig("windows must not be empty"));
        }
        if self.simulation.mc_n_min == 0 {
            return Err(GridError::InvalidConfig("simulation.mc_n_min must be > 0"));
        }
        if self.simulation.mc_n_max == 0 {
            return Err(GridError::InvalidConfig("simulation.mc_n_max must be > 0"));
        }
        if self.simulation.mc_n_max < self.simulation.mc_n_min {
            return Err(GridError::InvalidConfig(
                "simulation.mc_n_max must be >= simulation.mc_n_min",
            ));
        }
        if !(0.0..=1.0).contains(&self.volatility.parkinson_weight) {
            return Err(GridError::InvalidConfig(
                "parkinson_weight must be between 0 and 1",
            ));
        }
        if !(0.0..=1.0).contains(&self.pricing.liability_rho) {
            return Err(GridError::InvalidConfig(
                "pricing.liability_rho must be between 0 and 1",
            ));
        }
        if let Some(off_m_base) = self.pricing.off_hours_m_base {
            if !off_m_base.is_finite() || !(0.0..1.0).contains(&off_m_base) {
                return Err(GridError::InvalidConfig(
                    "pricing.off_hours_m_base must be in [0, 1)",
                ));
            }
        }
        for window in &self.spec.windows {
            if window.start_secs >= window.end_secs {
                return Err(GridError::InvalidConfig(
                    "window start must be strictly smaller than end",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridError {
    InvalidSpot,
    InvalidHistory(&'static str),
    InvalidConfig(&'static str),
}

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GridError::InvalidSpot => write!(f, "spot price must be > 0"),
            GridError::InvalidHistory(message) => write!(f, "invalid history: {message}"),
            GridError::InvalidConfig(message) => write!(f, "invalid config: {message}"),
        }
    }
}

impl std::error::Error for GridError {}
