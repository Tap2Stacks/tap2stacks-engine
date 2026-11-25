mod book;

use serde::{Deserialize, Serialize};

use crate::grid::{
    GridConfig, OhlcBar, PriceBand, TimeWindow, VolatilitySnapshot, compute_k_safety,
    compute_quote_probability, effective_m_base, estimate_runtime_state_with_seed_prior,
    price_multiplier_with_margin, simulate_summary, stock_grid_config_for_ticker,
};
use crate::horizon::{
    NEAR_WINDOWS, RepriceStats, cadence_slot_due, move_trigger_due_sigma,
};
use crate::m4::{MarkFormulaTracker, TickerM4Params, m4_ticker_params, price_grid_m4};
use crate::price::{PriceSource, PriceTick};
use crate::session::is_us_regular_session;
use crate::v2::{
    PricingModel, Regime, RegimeDetector, TickerV2Params, V2_M_BASE, compute_book_lead,
    dynamic_news_preset, normalize_v2_ticker, price_grid_v2_with_book, v2_m_eff,
    v2_price_multiplier, v2_ticker_params_for_source,
};

pub use book::{CellKey, ExposureBook};

const WINDOW_WIDTH_MS: u64 = 5_000;
const WINDOW_WIDTH_SECS: u32 = 5;
const MIN_LEAD_MS: u64 = 5_000;

/// Quote snapshot for a single ticker produced from cached probabilities and live exposure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StockQuote {
    pub ticker: String,
    pub server_ts_ms: u64,
    pub spot: f64,
    pub band_cents: u32,
    pub cells: Vec<StockCell>,
}

/// Single quoted cell within a `StockQuote`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StockCell {
    pub window_start_ms: u64,
    pub window_end_ms: u64,
    pub lower_cents: i64,
    pub upper_cents: i64,
    pub hit_probability: f64,
    pub multiplier: f64,
}

#[derive(Debug, Clone, Copy)]
struct CachedCell {
    key: CellKey,
    window_end_ms: u64,
    upper_cents: i64,
    hit_probability: f64,
}

#[derive(Debug, Clone, Copy)]
struct CachedSafety {
    value: f64,
    computed_at_ms: u64,
}

/// Self-contained Monte Carlo sample job that can run off the async executor (e.g. `spawn_blocking`).
#[derive(Debug, Clone)]
pub struct SampleJob {
    tick: PriceTick,
    history: Vec<OhlcBar>,
    band_cents: u32,
    base_config: GridConfig,
    seed_variance_prior: Option<f64>,
    exposure: ExposureBook,
    cached_k_safety: Option<CachedSafety>,
    last_full_reprice_ms: Option<u64>,
    far_cache_end_window_ms: Option<u64>,
    last_full_price: f64,
    last_full_sigma: f64,
    v2_params: Option<&'static TickerV2Params>,
    v2_detector: RegimeDetector,
    m4_params: Option<&'static TickerM4Params>,
    m4_tracker: MarkFormulaTracker,
}

/// Computed Monte Carlo sample update ready to apply to `QuoteEngine`.
#[derive(Debug, Clone)]
pub struct SampleResult {
    tick: PriceTick,
    config: GridConfig,
    volatility: VolatilitySnapshot,
    new_cells: Vec<CachedCell>,
    far_cache_update: Option<(Vec<CachedCell>, Option<u64>)>,
    safety_update: Option<(Vec<f64>, CachedSafety)>,
    quote_windows: usize,
    full: bool,
    move_due: bool,
    cadence_due: bool,
    far_start: u64,
    far_end: u64,
    elapsed_ns: u128,
    v2_m_eff: Option<f64>,
    v2_regime: Option<Regime>,
}

impl SampleJob {
    /// Executes the Monte Carlo (v1) or exact DP (v2) simulation for this sample without touching `QuoteEngine`.
    pub fn run(mut self) -> Option<SampleResult> {
        let started = std::time::Instant::now();
        let quote_windows = self.base_config.simulation.quote_window_count.max(1);
        let far_refresh_ms = self.base_config.simulation.far_refresh_ms;
        let far_phase_ms = self.base_config.simulation.far_phase_ms;
        let far_trigger_sigma = self.base_config.simulation.far_trigger_sigma;

        let full_windows = aligned_windows(self.tick.ts_ms, quote_windows, WINDOW_WIDTH_MS, MIN_LEAD_MS);
        if full_windows.is_empty() {
            return None;
        }
        let far_start = full_windows
            .get(NEAR_WINDOWS)
            .map(|window| window.0)
            .unwrap_or(u64::MAX);
        let far_end = full_windows.last().map(|window| window.0).unwrap_or(0);

        if let Some(base_v2_params) = self.v2_params {
            let v2_params = base_v2_params.for_session(self.tick.ts_ms);
            self.v2_detector.on_tick(self.tick, v2_params);
            let regime = self
                .base_config
                .simulation
                .v2_forced_regime
                .unwrap_or_else(|| self.v2_detector.current_regime());
            let r60 = self
                .base_config
                .simulation
                .v2_forced_regime
                .map(|r| v2_params.presets[r.index()].rms_median_cents)
                .unwrap_or_else(|| self.v2_detector.last_r60().max(0.0));
            let inferred_phase = self
                .v2_detector
                .inferred_phase(self.tick.ts_ms, v2_params.clock_secs);

            let floor_sec_ms = (self.tick.ts_ms / 1_000) * 1_000;
            let relative_windows: Vec<TimeWindow> = full_windows
                .iter()
                .map(|&(start_ms, _)| {
                    let start_secs = ((start_ms - floor_sec_ms) / 1_000) as u32;
                    TimeWindow {
                        start_secs,
                        end_secs: start_secs + WINDOW_WIDTH_SECS,
                    }
                })
                .collect();

            let mut config = self.base_config.clone();
            config.spec.band_width = self.band_cents as f64 / 100.0;
            config.spec.windows = relative_windows;
            if config.validate().is_err() {
                return None;
            }

            let bands = build_cent_bands(
                self.tick.price,
                config.spec.rows,
                config.spec.center_row,
                self.band_cents,
            );
            let (probs, m_eff) = if let Some(m4_params) = self.m4_params {
                self.m4_tracker.on_tick(self.tick);
                let state = self.m4_tracker.state(self.tick.price);
                let regime_m4 = &m4_params.presets[regime.index()];
                let mc_n = self.base_config.simulation.mc_n_min.max(1);
                let base_seed = self
                    .base_config
                    .simulation
                    .seed
                    .unwrap_or(0x517C_C1B7_2722_0A95);
                let seed = base_seed
                    ^ self.tick.ts_ms.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (state.mark_cents as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let probs = price_grid_m4(
                    &bands,
                    &config.spec.windows,
                    state,
                    regime_m4,
                    inferred_phase,
                    mc_n,
                    seed,
                );
                let m_eff = v2_m_eff(r60, v2_params.rms_ref_cents, regime_m4.m_base);
                (probs, m_eff)
            } else {
                let q60 = self
                    .base_config
                    .simulation
                    .v2_forced_regime
                    .map(|r| v2_params.presets[r.index()].q_1s)
                    .unwrap_or_else(|| self.v2_detector.last_q60());
                let news_override = if regime == Regime::News {
                    Some(dynamic_news_preset(v2_params, r60, q60))
                } else {
                    None
                };
                let book_lead = compute_book_lead(
                    self.tick.mid_px,
                    self.tick.oracle_px,
                    None,
                    v2_params,
                );
                let probs = price_grid_v2_with_book(
                    self.tick.price,
                    &bands,
                    &config.spec.windows,
                    v2_params,
                    regime,
                    inferred_phase,
                    news_override,
                    book_lead,
                );
                let preset_m_base = v2_params.presets[regime.index()].m_base;
                let m_eff = v2_m_eff(r60, v2_params.rms_ref_cents, preset_m_base);
                (probs, m_eff)
            };

            let mut new_cells = Vec::with_capacity(bands.len() * quote_windows);
            for (row_index, band) in bands.iter().enumerate() {
                let lower_cents = (band.lower * 100.0).round() as i64;
                let upper_cents = (band.upper * 100.0).round() as i64;
                for (window_index, &(window_start_ms, window_end_ms)) in
                    full_windows.iter().enumerate()
                {
                    new_cells.push(CachedCell {
                        key: CellKey {
                            window_start_ms,
                            lower_cents,
                        },
                        window_end_ms,
                        upper_cents,
                        hit_probability: probs[row_index][window_index],
                    });
                }
            }
            let sigma_eff = (r60 / (100.0 * self.tick.price)).max(1e-8);
            let volatility = VolatilitySnapshot {
                variance: sigma_eff * sigma_eff,
                sigma_raw: sigma_eff,
                sigma: sigma_eff,
                jump_intensity: 0.0,
                jump_flag: false,
            };

            return Some(SampleResult {
                tick: self.tick,
                config,
                volatility,
                new_cells,
                far_cache_update: None,
                safety_update: Some((
                    Vec::new(),
                    CachedSafety {
                        value: 1.0,
                        computed_at_ms: self.tick.ts_ms,
                    },
                )),
                quote_windows,
                full: true,
                move_due: false,
                cadence_due: false,
                far_start,
                far_end,
                elapsed_ns: started.elapsed().as_nanos(),
                v2_m_eff: Some(m_eff),
                v2_regime: Some(regime),
            });
        }

        let far_covered = self
            .far_cache_end_window_ms
            .is_some_and(|end_w| end_w >= far_end);
        let cadence_due = self.last_full_reprice_ms.is_none_or(|last_ms| {
            cadence_slot_due(self.tick.ts_ms, last_ms, far_refresh_ms, far_phase_ms)
        }) || !far_covered;
        let move_due = self.last_full_reprice_ms.is_some_and(|last_ms| {
            move_trigger_due_sigma(
                self.tick.ts_ms,
                self.tick.price,
                last_ms,
                self.last_full_price,
                self.last_full_sigma,
                far_trigger_sigma,
            )
        });
        let full = quote_windows <= NEAR_WINDOWS
            || self.last_full_reprice_ms.is_none()
            || move_due
            || cadence_due;

        // When split repricing is active (quote_windows > NEAR_WINDOWS), pre-compute +1 lookahead
        // window on full reprices so staggered 5s refreshes never drop the tail window across a 5s roll.
        let sim_windows_ms = if full {
            if quote_windows > NEAR_WINDOWS && far_refresh_ms > 0 {
                aligned_windows(
                    self.tick.ts_ms,
                    quote_windows.saturating_add(1),
                    WINDOW_WIDTH_MS,
                    MIN_LEAD_MS,
                )
            } else {
                full_windows.clone()
            }
        } else {
            full_windows[..NEAR_WINDOWS.min(full_windows.len())].to_vec()
        };

        let floor_sec_ms = (self.tick.ts_ms / 1_000) * 1_000;
        let relative_windows: Vec<TimeWindow> = sim_windows_ms
            .iter()
            .map(|&(start_ms, _)| {
                let start_secs = ((start_ms - floor_sec_ms) / 1_000) as u32;
                TimeWindow {
                    start_secs,
                    end_secs: start_secs + WINDOW_WIDTH_SECS,
                }
            })
            .collect();

        let mut config = self.base_config.clone();
        config.spec.band_width = self.band_cents as f64 / 100.0;
        config.spec.windows = relative_windows;
        if config.validate().is_err() {
            return None;
        }

        let Ok(runtime_state) = estimate_runtime_state_with_seed_prior(
            self.tick.price,
            &self.history,
            &config,
            self.seed_variance_prior,
        ) else {
            return None;
        };

        let bands = build_cent_bands(
            self.tick.price,
            config.spec.rows,
            config.spec.center_row,
            self.band_cents,
        );
        let max_horizon = config
            .spec
            .windows
            .iter()
            .map(|window| window.end_secs)
            .max()
            .unwrap_or(0)
            .saturating_add(config.simulation.lock_offset) as usize;

        let window_count = sim_windows_ms.len();
        let mut liabilities_vector = vec![0.0; bands.len() * window_count];
        for (row_index, band) in bands.iter().enumerate() {
            let lower_cents = (band.lower * 100.0).round() as i64;
            let row_offset = row_index * window_count;
            for (window_index, &(window_start_ms, _)) in sim_windows_ms.iter().enumerate() {
                let key = CellKey {
                    window_start_ms,
                    lower_cents,
                };
                liabilities_vector[row_offset + window_index] = self.exposure.get(&key);
            }
        }

        let k_safety_fresh = self.cached_k_safety.is_some_and(|cached| {
            self.tick.ts_ms >= cached.computed_at_ms
                && self.tick.ts_ms - cached.computed_at_ms < config.pricing.k_safety_ttl_ms
        });
        let refresh_safety = full && !k_safety_fresh;

        let simulation = simulate_summary(
            self.tick.price,
            &bands,
            &config,
            &runtime_state,
            max_horizon,
            if refresh_safety {
                Some(&liabilities_vector)
            } else {
                None
            },
        );

        let safety_update = if refresh_safety {
            let value = compute_k_safety(&simulation.scenario_payouts, &config);
            Some((
                simulation.scenario_payouts,
                CachedSafety {
                    value,
                    computed_at_ms: self.tick.ts_ms,
                },
            ))
        } else {
            None
        };

        let current_window0_row_index = bands
            .iter()
            .min_by(|left, right| {
                let left_center = (left.lower + left.upper) / 2.0;
                let right_center = (right.lower + right.upper) / 2.0;
                (left_center - self.tick.price)
                    .abs()
                    .partial_cmp(&(right_center - self.tick.price).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|band| band.row_index)
            .unwrap_or(config.spec.center_row);

        let volatility = runtime_state.volatility;
        let max_factor_window = quote_windows.saturating_sub(1);
        let mut new_cells = Vec::with_capacity(bands.len() * quote_windows);
        let mut far_cache_cells = if full && quote_windows > NEAR_WINDOWS {
            Vec::with_capacity(bands.len() * window_count)
        } else {
            Vec::new()
        };

        for (row_index, band) in bands.iter().enumerate() {
            let lower_cents = (band.lower * 100.0).round() as i64;
            let upper_cents = (band.upper * 100.0).round() as i64;
            for (window_index, window) in config.spec.windows.iter().enumerate() {
                let (window_start_ms, window_end_ms) = sim_windows_ms[window_index];
                let raw_hit_probability = simulation.mean_probabilities[row_index][window_index];
                let factor_window_index = window_index.min(max_factor_window);
                let hit_probability = compute_quote_probability(
                    raw_hit_probability,
                    row_index,
                    factor_window_index,
                    *window,
                    volatility,
                    current_window0_row_index,
                    &config,
                );
                let cell = CachedCell {
                    key: CellKey {
                        window_start_ms,
                        lower_cents,
                    },
                    window_end_ms,
                    upper_cents,
                    hit_probability,
                };
                if full && quote_windows > NEAR_WINDOWS {
                    far_cache_cells.push(cell);
                }
                if window_index < quote_windows {
                    new_cells.push(cell);
                }
            }
        }

        let far_cache_update = if full && quote_windows > NEAR_WINDOWS {
            let cache_end = sim_windows_ms.last().map(|w| w.0);
            Some((far_cache_cells, cache_end))
        } else {
            None
        };

        Some(SampleResult {
            tick: self.tick,
            config,
            volatility,
            new_cells,
            far_cache_update,
            safety_update,
            quote_windows,
            full,
            move_due,
            cadence_due,
            far_start,
            far_end,
            elapsed_ns: started.elapsed().as_nanos(),
            v2_m_eff: None,
            v2_regime: None,
        })
    }
}

/// Computes `count` contiguous `width_ms`-aligned `[start, end)` windows starting at or after `now_ms + min_lead_ms`.
pub fn aligned_windows(
    now_ms: u64,
    count: usize,
    width_ms: u64,
    min_lead_ms: u64,
) -> Vec<(u64, u64)> {
    if count == 0 || width_ms == 0 {
        return Vec::new();
    }
    let earliest = now_ms.saturating_add(min_lead_ms);
    let first_start = earliest.div_ceil(width_ms).saturating_mul(width_ms);
    (0..count)
        .map(|idx| {
            let start = first_start.saturating_add((idx as u64).saturating_mul(width_ms));
            let end = start.saturating_add(width_ms);
            (start, end)
        })
        .collect()
}

/// Per-ticker quoting engine that runs Monte Carlo once per sample and re-prices from cache on `price()`.
#[derive(Debug, Clone)]
pub struct QuoteEngine {
    ticker: String,
    band_cents: u32,
    spot: f64,
    base_config: GridConfig,
    config: GridConfig,
    seed_variance_prior: Option<f64>,
    volatility: Option<VolatilitySnapshot>,
    cached_cells: Vec<CachedCell>,
    far_cache: Vec<CachedCell>,
    cached_scenario_payouts: Vec<f64>,
    cached_k_safety: Option<CachedSafety>,
    exposure: ExposureBook,
    last_full_reprice_ms: Option<u64>,
    far_cache_end_window_ms: Option<u64>,
    last_full_price: f64,
    last_full_sigma: f64,
    stats: RepriceStats,
    v2_params: Option<&'static TickerV2Params>,
    v2_detector: RegimeDetector,
    m4_params: Option<&'static TickerM4Params>,
    m4_tracker: MarkFormulaTracker,
    v2_last_m_eff: f64,
    v2_last_regime: Option<Regime>,
}

impl QuoteEngine {
    /// Creates a new `QuoteEngine` for `ticker` with `band_cents`-wide price bands.
    pub fn new(ticker: String, band_cents: u32) -> Self {
        let base_config = stock_grid_config_for_ticker(&ticker, band_cents, Vec::new());
        Self::with_config(ticker, band_cents, base_config)
    }

    /// Creates a `QuoteEngine` using an explicit `GridConfig` template (for calibration/tuning).
    pub fn with_config(ticker: String, band_cents: u32, base_config: GridConfig) -> Self {
        let (v2_params, m4_params) = match base_config.simulation.pricing_model {
            PricingModel::V1 => (None, None),
            PricingModel::V2 => {
                let params = base_config
                    .simulation
                    .v2_params_override
                    .filter(|p| normalize_v2_ticker(p.ticker) == normalize_v2_ticker(&ticker))
                    .or_else(|| {
                        v2_ticker_params_for_source(&ticker, base_config.simulation.price_source)
                    });
                (params, None)
            }
            PricingModel::M4 => match (
                v2_ticker_params_for_source(&ticker, PriceSource::Hyperliquid),
                m4_ticker_params(&ticker),
            ) {
                (Some(v2), Some(m4)) => (Some(v2), Some(m4)),
                _ => (None, None),
            },
        };
        Self {
            ticker,
            band_cents,
            spot: 0.0,
            config: base_config.clone(),
            base_config,
            seed_variance_prior: None,
            volatility: None,
            cached_cells: Vec::new(),
            far_cache: Vec::new(),
            cached_scenario_payouts: Vec::new(),
            cached_k_safety: None,
            exposure: ExposureBook::default(),
            last_full_reprice_ms: None,
            far_cache_end_window_ms: None,
            last_full_price: 0.0,
            last_full_sigma: 0.0,
            stats: RepriceStats::default(),
            v2_params,
            v2_detector: RegimeDetector::new(),
            m4_params,
            m4_tracker: MarkFormulaTracker::new(),
            v2_last_m_eff: V2_M_BASE,
            v2_last_regime: None,
        }
    }

    /// Feeds a raw tick to the causal v2/m4 trackers (no-op on v1).
    pub fn observe_tick(&mut self, tick: PriceTick) {
        if let Some(params) = self.v2_params {
            self.v2_detector
                .on_tick(tick, params.for_session(tick.ts_ms));
        }
        if self.m4_params.is_some() {
            self.m4_tracker.on_tick(tick);
        }
    }

    /// Returns `true` when odds model v2 is active for this ticker.
    pub fn is_v2_active(&self) -> bool {
        self.v2_params.is_some() && self.m4_params.is_none()
    }

    /// Returns `true` when odds model m4 (trade.xyz mark formula) is active for this ticker.
    pub fn is_m4_active(&self) -> bool {
        self.m4_params.is_some()
    }

    /// Returns the latest v2 regime if v2 is active for this ticker.
    pub fn current_v2_regime(&self) -> Option<Regime> {
        self.v2_last_regime
            .or_else(|| self.v2_params.map(|_| self.v2_detector.current_regime()))
    }

    /// Sets a measured variance used instead of `sigma_ref²` as the cold-start prior.
    pub fn set_seed_variance_prior(&mut self, variance: Option<f64>) {
        self.seed_variance_prior = variance.filter(|value| (*value).is_finite() && *value > 0.0);
    }

    /// Quoted volatility from the latest sample, if one has been taken.
    pub fn volatility(&self) -> Option<VolatilitySnapshot> {
        self.volatility
    }

    pub fn reprice_stats(&self) -> RepriceStats {
        self.stats.clone()
    }

    /// Prepares an owned `SampleJob` that can execute Monte Carlo on a blocking worker thread.
    pub fn prepare_sample_job(&self, tick: PriceTick, history: Vec<OhlcBar>) -> Option<SampleJob> {
        if !tick.price.is_finite() || tick.price <= 0.0 || self.band_cents == 0 {
            return None;
        }
        Some(SampleJob {
            tick,
            history,
            band_cents: self.band_cents,
            base_config: self.base_config.clone(),
            seed_variance_prior: self.seed_variance_prior,
            exposure: self.exposure.clone(),
            cached_k_safety: self.cached_k_safety,
            last_full_reprice_ms: self.last_full_reprice_ms,
            far_cache_end_window_ms: self.far_cache_end_window_ms,
            last_full_price: self.last_full_price,
            last_full_sigma: self.last_full_sigma,
            v2_params: self.v2_params,
            v2_detector: self.v2_detector.clone(),
            m4_params: self.m4_params,
            m4_tracker: self.m4_tracker.clone(),
        })
    }

    /// Applies the output of `SampleJob::run` back into this engine's quote cache.
    pub fn apply_sample_result(&mut self, result: SampleResult) {
        if let Some(m_eff) = result.v2_m_eff {
            self.v2_last_m_eff = m_eff;
        }
        self.v2_last_regime = result.v2_regime;
        if let Some((payouts, safety)) = result.safety_update {
            self.cached_scenario_payouts = payouts;
            self.cached_k_safety = Some(safety);
        }

        let mut cached_cells = result.new_cells;
        if !result.full {
            for old in &self.far_cache {
                if old.key.window_start_ms >= result.far_start
                    && old.key.window_start_ms <= result.far_end
                {
                    cached_cells.push(*old);
                }
            }
            self.stats.near_reprices += 1;
            self.stats.near_ns += result.elapsed_ns;
        } else {
            self.stats.full_reprices += 1;
            self.stats.full_ns += result.elapsed_ns;
            if result.quote_windows > NEAR_WINDOWS && self.last_full_reprice_ms.is_some() {
                if result.move_due && !result.cadence_due {
                    self.stats.trigger_reprices += 1;
                    self.stats.trigger_timestamps_ms.push(result.tick.ts_ms);
                } else {
                    self.stats.cadence_reprices += 1;
                }
            }
            if let Some((far_cells, cache_end)) = result.far_cache_update {
                self.far_cache = far_cells;
                self.far_cache_end_window_ms = cache_end;
            }
            self.last_full_reprice_ms = Some(result.tick.ts_ms);
            self.last_full_price = result.tick.price;
            self.last_full_sigma = result.volatility.sigma.max(1e-8);
        }

        self.spot = result.tick.price;
        self.config = result.config;
        self.volatility = Some(result.volatility);
        self.cached_cells = cached_cells;
        self.stats.sample_count += 1;
        self.stats.sample_ns += result.elapsed_ns;
    }

    /// Runs Monte Carlo (v1) or exact DP (v2) for a new price sample and caches per-cell probabilities.
    pub fn on_sample(&mut self, tick: PriceTick, history: &[OhlcBar]) {
        self.observe_tick(tick);
        if let Some(job) = self.prepare_sample_job(tick, history.to_vec()) {
            if let Some(result) = job.run() {
                self.apply_sample_result(result);
            }
        }
    }

    /// Replaces the ticker's whole live exposure map.
    pub fn set_exposure(&mut self, cells: Vec<(CellKey, f64)>) {
        self.exposure.replace(cells);
    }

    /// Re-prices all cached cells still outside the lead time using live exposure, without running Monte Carlo.
    pub fn price(&mut self, now_ms: u64) -> StockQuote {
        let Some(volatility) = self.volatility else {
            return StockQuote {
                ticker: self.ticker.clone(),
                server_ts_ms: now_ms,
                spot: self.spot,
                band_cents: self.band_cents,
                cells: Vec::new(),
            };
        };

        let k_safety = match self.cached_k_safety {
            Some(cached)
                if now_ms >= cached.computed_at_ms
                    && now_ms - cached.computed_at_ms <= self.config.pricing.k_safety_ttl_ms =>
            {
                cached.value
            }
            _ => {
                let value = compute_k_safety(&self.cached_scenario_payouts, &self.config);
                self.cached_k_safety = Some(CachedSafety {
                    value,
                    computed_at_ms: now_ms,
                });
                value
            }
        };

        let min_start_ms = now_ms.saturating_add(MIN_LEAD_MS);
        let mut cells = Vec::with_capacity(self.cached_cells.len());
        for cell in &self.cached_cells {
            if cell.key.window_start_ms < min_start_ms {
                continue;
            }
            let liability = self.exposure.get(&cell.key);
            let multiplier = if self.v2_params.is_some() {
                v2_price_multiplier(
                    cell.hit_probability,
                    self.v2_last_m_eff,
                    liability,
                    k_safety,
                    &self.config,
                )
            } else {
                let m_base =
                    effective_m_base(&self.config, is_us_regular_session(cell.key.window_start_ms));
                price_multiplier_with_margin(
                    cell.hit_probability,
                    volatility,
                    liability,
                    k_safety,
                    m_base,
                    &self.config,
                )
            };
            cells.push(StockCell {
                window_start_ms: cell.key.window_start_ms,
                window_end_ms: cell.window_end_ms,
                lower_cents: cell.key.lower_cents,
                upper_cents: cell.upper_cents,
                hit_probability: cell.hit_probability,
                multiplier,
            });
        }

        StockQuote {
            ticker: self.ticker.clone(),
            server_ts_ms: now_ms,
            spot: self.spot,
            band_cents: self.band_cents,
            cells,
        }
    }
}

/// Builds `rows` contiguous `band_cents`-wide price bands around `floor(spot / band) * band`.
fn build_cent_bands(
    spot: f64,
    rows: usize,
    center_row: usize,
    band_cents: u32,
) -> Vec<PriceBand> {
    let band_c = band_cents as i64;
    let spot_cents = (spot * 100.0 + 1e-9).floor() as i64;
    let anchor_cents = spot_cents.div_euclid(band_c) * band_c;
    // Offset by floor(band_c / 2) so the center row straddles anchor_cents with exact integer-cent edges.
    let center_lower_cents = anchor_cents - (band_c / 2);

    (0..rows)
        .map(|row_index| {
            let lower_cents =
                center_lower_cents + (row_index as i64 - center_row as i64) * band_c;
            let upper_cents = lower_cents + band_c;
            PriceBand {
                row_index,
                lower: lower_cents as f64 / 100.0,
                upper: upper_cents as f64 / 100.0,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::horizon::{FAR_REFRESH_MS, stagger_phase_ms};

    fn sample_history(spot: f64) -> Vec<OhlcBar> {
        vec![
            OhlcBar {
                open: spot - 0.12,
                high: spot - 0.12,
                low: spot - 0.12,
                close: spot - 0.12,
            },
            OhlcBar {
                open: spot - 0.05,
                high: spot - 0.05,
                low: spot - 0.05,
                close: spot - 0.05,
            },
            OhlcBar {
                open: spot,
                high: spot,
                low: spot,
                close: spot,
            },
        ]
    }

    #[test]
    fn windows_align_to_5s_and_lead_at_least_5s() {
        let w = aligned_windows(1_795_000_001_234, 6, 5_000, 5_000);
        assert_eq!(w[0], (1_795_000_010_000, 1_795_000_015_000));
        assert_eq!(w.len(), 6);
        assert!(w.iter().all(|(s, e)| s % 5_000 == 0 && e - s == 5_000));
    }

    #[test]
    fn exposure_lowers_only_that_cell() {
        let mut engine = QuoteEngine::new("NVDA".to_string(), 3);
        let ts = 1_795_000_001_234;
        let spot = 157.53;
        engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));

        let before = engine.price(ts);
        let target = before
            .cells
            .iter()
            .find(|c| c.multiplier > 1.2 && c.multiplier < 25.0)
            .expect("expected a cell with multiplier in (1.2, 25.0)")
            .clone();
        let target_key = CellKey {
            window_start_ms: target.window_start_ms,
            lower_cents: target.lower_cents,
        };

        engine.set_exposure(vec![(target_key, 5_000.0)]);
        let after = engine.price(ts);

        assert_eq!(before.cells.len(), after.cells.len());
        for (b, a) in before.cells.iter().zip(after.cells.iter()) {
            let k = CellKey {
                window_start_ms: b.window_start_ms,
                lower_cents: b.lower_cents,
            };
            if k == target_key {
                assert!(
                    a.multiplier < b.multiplier,
                    "expected targeted cell multiplier {} < {}",
                    a.multiplier,
                    b.multiplier
                );
            } else {
                assert_eq!(
                    a.multiplier, b.multiplier,
                    "non-target cell {:?} multiplier changed",
                    k
                );
            }
        }
    }

    #[test]
    fn exposure_for_unknown_cell_is_ignored() {
        let mut engine = QuoteEngine::new("NVDA".to_string(), 3);
        let ts = 1_795_000_001_234;
        let spot = 157.53;
        engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));

        let before = engine.price(ts);
        engine.set_exposure(vec![
            (
                CellKey {
                    window_start_ms: 1_795_000_000_000,
                    lower_cents: 15_700,
                },
                10_000.0,
            ),
            (
                CellKey {
                    window_start_ms: 1_795_000_010_000,
                    lower_cents: 999_999,
                },
                10_000.0,
            ),
        ]);
        let after = engine.price(ts);

        assert_eq!(before.cells, after.cells);
    }

    #[test]
    fn multipliers_stay_within_bounds() {
        let mut engine = QuoteEngine::new("TSLA".to_string(), 8);
        let ts = 1_795_000_001_234;
        let spot = 250.00;
        engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));

        let q1 = engine.price(ts);
        assert_eq!(q1.cells.len(), 20 * 6);
        assert!(
            q1.cells
                .iter()
                .all(|c| c.multiplier >= 1.05 && c.multiplier <= 25.0)
        );

        let heavy: Vec<(CellKey, f64)> = q1
            .cells
            .iter()
            .map(|c| {
                (
                    CellKey {
                        window_start_ms: c.window_start_ms,
                        lower_cents: c.lower_cents,
                    },
                    100_000.0,
                )
            })
            .collect();
        engine.set_exposure(heavy);
        let q2 = engine.price(ts);
        assert!(
            q2.cells
                .iter()
                .all(|c| c.multiplier >= 1.05 && c.multiplier <= 25.0)
        );
    }

    #[test]
    fn price_without_new_sample_does_not_rerun_monte_carlo() {
        let mut engine = QuoteEngine::new("NVDA".to_string(), 3);
        let ts = 1_795_000_001_234;
        let spot = 157.53;
        engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));

        let start = Instant::now();
        for i in 0..100 {
            let q = engine.price(ts + i);
            assert_eq!(q.cells.len(), 20 * 6);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100 price() calls took {:?}, expected < 50 ms",
            elapsed
        );
    }

    #[test]
    fn cells_inside_lead_time_drop_out_between_samples() {
        let mut engine = QuoteEngine::new("NVDA".to_string(), 3);
        let ts = 1_795_000_004_000;
        let spot = 157.53;
        engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));

        // First window starts at 1_795_000_010_000. At now = 1_795_000_004_500, lead is 5500 ms -> all 6 windows kept.
        let q_all = engine.price(1_795_000_004_500);
        assert_eq!(q_all.cells.len(), 20 * 6);

        // At now = 1_795_000_005_000, window_start_ms == now + 5000 -> still kept.
        let q_boundary = engine.price(1_795_000_005_000);
        assert_eq!(q_boundary.cells.len(), 20 * 6);

        // At now = 1_795_000_005_001, first window (1_795_000_010_000 < now + 5000) drops out.
        let q_dropped = engine.price(1_795_000_005_001);
        assert_eq!(q_dropped.cells.len(), 20 * 5);
        assert!(
            q_dropped
                .cells
                .iter()
                .all(|c| c.window_start_ms >= 1_795_000_015_000)
        );
    }

    #[test]
    fn price_before_first_sample_returns_empty_cells() {
        let mut engine = QuoteEngine::new("NVDA".to_string(), 3);
        let q = engine.price(1_795_000_001_000);
        assert_eq!(q.ticker, "NVDA");
        assert_eq!(q.server_ts_ms, 1_795_000_001_000);
        assert_eq!(q.band_cents, 3);
        assert!(q.cells.is_empty());
    }

    #[test]
    fn set_exposure_replaces_entire_map() {
        let mut engine = QuoteEngine::new("NVDA".to_string(), 3);
        let ts = 1_795_000_001_234;
        let spot = 157.53;
        engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));

        let baseline = engine.price(ts);
        let target = baseline
            .cells
            .iter()
            .find(|c| c.multiplier > 1.2 && c.multiplier < 25.0)
            .unwrap()
            .clone();
        let target_key = CellKey {
            window_start_ms: target.window_start_ms,
            lower_cents: target.lower_cents,
        };

        engine.set_exposure(vec![(target_key, 5_000.0)]);
        let loaded = engine.price(ts);
        assert!(
            loaded
                .cells
                .iter()
                .find(|c| c.window_start_ms == target.window_start_ms
                    && c.lower_cents == target.lower_cents)
                .unwrap()
                .multiplier
                < target.multiplier
        );

        engine.set_exposure(Vec::new());
        let cleared = engine.price(ts);
        assert_eq!(baseline.cells, cleared.cells);
    }

    #[test]
    fn adjacent_rows_share_edge_cents_and_serialize() {
        let mut engine = QuoteEngine::new("NVDA".to_string(), 3);
        let ts = 1_795_000_001_234;
        let spot = 157.53;
        engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));

        let q = engine.price(ts);
        let first_window = q.cells[0].window_start_ms;
        let col: Vec<&StockCell> = q
            .cells
            .iter()
            .filter(|c| c.window_start_ms == first_window)
            .collect();
        assert_eq!(col.len(), 20);
        for cell in &col {
            assert_eq!(cell.upper_cents - cell.lower_cents, 3);
        }
        for pair in col.windows(2) {
            assert_eq!(pair[0].upper_cents, pair[1].lower_cents);
        }

        let json = serde_json::to_string(&q).expect("StockQuote should serialize");
        let roundtrip: StockQuote = serde_json::from_str(&json).expect("StockQuote should deserialize");
        assert_eq!(q.ticker, roundtrip.ticker);
        assert_eq!(q.server_ts_ms, roundtrip.server_ts_ms);
        assert_eq!(q.band_cents, roundtrip.band_cents);
        assert_eq!(q.cells.len(), roundtrip.cells.len());
        for (a, b) in q.cells.iter().zip(roundtrip.cells.iter()) {
            assert_eq!(a.window_start_ms, b.window_start_ms);
            assert_eq!(a.window_end_ms, b.window_end_ms);
            assert_eq!(a.lower_cents, b.lower_cents);
            assert_eq!(a.upper_cents, b.upper_cents);
            assert!((a.hit_probability - b.hit_probability).abs() < 1e-12);
            assert!((a.multiplier - b.multiplier).abs() < 1e-12);
        }

        let key = CellKey {
            window_start_ms: first_window,
            lower_cents: col[0].lower_cents,
        };
        let key_json = serde_json::to_string(&key).expect("CellKey should serialize");
        let key_rt: CellKey = serde_json::from_str(&key_json).expect("CellKey should deserialize");
        assert_eq!(key, key_rt);
    }

    #[test]
    fn off_hours_cell_uses_higher_margin_than_us_regular() {
        let us_ts = 1_794_931_200_000_u64; // 2026-11-17 16:00:00Z = 11:00 EST (US regular)
        let off_ts = 1_794_916_800_000_u64; // 2026-11-17 12:00:00Z = 07:00 EST (off-hours)
        assert!(is_us_regular_session(us_ts + 5_000));
        assert!(!is_us_regular_session(off_ts + 5_000));

        let spot = 157.53;
        let mut cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        cfg.simulation.seed = Some(42);
        let mut us_engine = QuoteEngine::with_config("NVDA".to_string(), 3, cfg.clone());
        us_engine.on_sample(PriceTick { ts_ms: us_ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));
        let us_quote = us_engine.price(us_ts);

        let mut off_engine = QuoteEngine::with_config("NVDA".to_string(), 3, cfg);
        off_engine.on_sample(PriceTick { ts_ms: off_ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));
        let off_quote = off_engine.price(off_ts);

        assert_eq!(us_quote.cells.len(), off_quote.cells.len());
        let mut checked = 0usize;
        for (us_cell, off_cell) in us_quote.cells.iter().zip(off_quote.cells.iter()) {
            assert!((us_cell.hit_probability - off_cell.hit_probability).abs() < 1e-12);
            if us_cell.multiplier < 25.0 && us_cell.multiplier > 1.05 {
                assert!(
                    off_cell.multiplier < us_cell.multiplier,
                    "expected off-hours multiplier ({}) < US multiplier ({}) for p={}",
                    off_cell.multiplier,
                    us_cell.multiplier,
                    us_cell.hit_probability,
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn us_regular_quote_ignores_off_hours_m_base() {
        let us_ts = 1_794_931_200_000_u64; // 2026-11-17 16:00:00Z = 11:00 EST
        let spot = 157.53;
        let history = sample_history(spot);

        let mut cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        cfg.simulation.seed = Some(42);
        let mut with_off_hours = QuoteEngine::with_config("NVDA".to_string(), 3, cfg.clone());
        with_off_hours.on_sample(PriceTick { ts_ms: us_ts, price: spot, mid_px: None, oracle_px: None }, &history);
        let q1 = with_off_hours.price(us_ts);

        cfg.pricing.off_hours_m_base = None;
        let mut baseline = QuoteEngine::with_config("NVDA".to_string(), 3, cfg);
        baseline.on_sample(PriceTick { ts_ms: us_ts, price: spot, mid_px: None, oracle_px: None }, &history);
        let q2 = baseline.price(us_ts);

        assert_eq!(q1.cells, q2.cells);
    }

    #[test]
    fn far_windows_stay_cached_until_a_two_sigma_move() {
        let mut cfg = crate::grid::stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        cfg.simulation.quote_window_count = 24;
        cfg.simulation.mc_n_min = 20;
        cfg.simulation.mc_n_max = 20;
        cfg.simulation.seed = Some(3);
        cfg.simulation.use_parallel = false;
        assert!((cfg.simulation.far_trigger_sigma - 2.0).abs() < 1e-12);

        let mut engine = QuoteEngine::with_config("NVDA".to_string(), 3, cfg);
        let ts = 1_795_000_001_000;
        let spot = 140.0;
        engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));
        let sigma = engine.volatility().unwrap().sigma;
        let first = engine.price(ts);
        assert_eq!(first.cells.len(), 20 * 24);
        let far_window = first.cells.iter().map(|c| c.window_start_ms).max().unwrap();
        let far_before: Vec<(i64, f64)> = first
            .cells
            .iter()
            .filter(|c| c.window_start_ms == far_window)
            .map(|c| (c.lower_cents, c.hit_probability))
            .collect();

        // 1.5-sigma move stays cached under the 2-sigma default trigger.
        let mid_move = spot * (1.5 * sigma).exp();
        engine.on_sample(
            PriceTick {
                ts_ms: ts + 1_000,
                price: mid_move,
                mid_px: None,
                oracle_px: None,
            },
            &sample_history(spot),
        );
        let stats = engine.reprice_stats();
        assert_eq!(stats.full_reprices, 1);
        assert_eq!(stats.near_reprices, 1);
        assert_eq!(stats.trigger_reprices, 0);
        let second = engine.price(ts + 1_000);
        assert_eq!(second.cells.len(), 20 * 24);
        let far_mid: Vec<(i64, f64)> = second
            .cells
            .iter()
            .filter(|c| c.window_start_ms == far_window)
            .map(|c| (c.lower_cents, c.hit_probability))
            .collect();
        assert_eq!(far_before, far_mid);

        // News-sized (+2%) move exceeds 2-sigma and triggers an early full reprice.
        let jumped = spot * 1.02;
        engine.on_sample(
            PriceTick {
                ts_ms: ts + 2_000,
                price: jumped,
                mid_px: None,
                oracle_px: None,
            },
            &sample_history(jumped),
        );
        let stats = engine.reprice_stats();
        assert_eq!(stats.full_reprices, 2);
        assert_eq!(stats.trigger_reprices, 1);
        assert_eq!(stats.trigger_timestamps_ms, vec![ts + 2_000]);
        let third = engine.price(ts + 2_000);
        assert_eq!(third.cells.len(), 20 * 24);
        let far_after: Vec<(i64, f64)> = third
            .cells
            .iter()
            .filter(|c| c.window_start_ms == far_window)
            .map(|c| (c.lower_cents, c.hit_probability))
            .collect();
        assert_ne!(far_before, far_after);
    }

    #[test]
    fn staggered_engines_preserve_24_windows_across_rolls_and_never_collide() {
        let mut engines: Vec<QuoteEngine> = (0..4)
            .map(|idx| {
                let mut cfg = crate::grid::stock_grid_config_for_ticker("NVDA", 3, Vec::new());
                cfg.simulation.quote_window_count = 24;
                cfg.simulation.far_phase_ms = stagger_phase_ms(idx, 4, FAR_REFRESH_MS);
                cfg.simulation.mc_n_min = 20;
                cfg.simulation.mc_n_max = 20;
                cfg.simulation.seed = Some(3);
                cfg.simulation.use_parallel = false;
                QuoteEngine::with_config(format!("T{idx}"), 3, cfg)
            })
            .collect();

        let base_ts = 1_795_000_000_000_u64;
        let spot = 140.0;
        let hist = sample_history(spot);
        for engine in &mut engines {
            engine.on_sample(PriceTick { ts_ms: base_ts, price: spot, mid_px: None, oracle_px: None }, &hist);
            assert_eq!(engine.price(base_ts).cells.len(), 20 * 24);
        }

        // Steady-state 1 Hz quiet ticks across 20 seconds: every ticker always keeps 24 windows,
        // even across 5s window rolls, and at most 1 ticker full-reprices on any second.
        for sec in 1..=20u64 {
            let ts = base_ts + sec * 1_000;
            let mut full_this_sec = 0usize;
            for engine in &mut engines {
                let before_full = engine.reprice_stats().full_reprices;
                engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &hist);
                assert_eq!(engine.price(ts).cells.len(), 20 * 24, "sec={sec}");
                if engine.reprice_stats().full_reprices > before_full {
                    full_this_sec += 1;
                }
            }
            assert!(
                full_this_sec <= 1,
                "expected staggered full reprices <= 1 at sec={sec}, got {full_this_sec}"
            );
        }
    }

    #[test]
    fn sample_job_matches_on_sample_output() {
        let mut cfg = crate::grid::stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        cfg.simulation.quote_window_count = 24;
        cfg.simulation.mc_n_min = 40;
        cfg.simulation.mc_n_max = 40;
        cfg.simulation.seed = Some(11);
        cfg.simulation.use_parallel = false;

        let mut sync_engine = QuoteEngine::with_config("NVDA".to_string(), 3, cfg.clone());
        let mut async_engine = QuoteEngine::with_config("NVDA".to_string(), 3, cfg);

        let ts = 1_795_000_001_000;
        let spot = 140.0;
        let hist = sample_history(spot);
        sync_engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &hist);

        let job = async_engine
            .prepare_sample_job(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, hist)
            .unwrap();
        let res = job.run().unwrap();
        async_engine.apply_sample_result(res);

        assert_eq!(sync_engine.price(ts), async_engine.price(ts));
    }

    #[test]
    fn default_engine_stays_on_six_windows_with_no_split_reprice() {
        let mut engine = QuoteEngine::new("NVDA".to_string(), 3);
        let ts = 1_795_000_001_000;
        let spot = 140.0;
        engine.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &sample_history(spot));
        engine.on_sample(
            PriceTick {
                ts_ms: ts + 1_000,
                price: spot * 1.02,
                mid_px: None,
                oracle_px: None,
            },
            &sample_history(spot * 1.02),
        );
        let stats = engine.reprice_stats();
        assert_eq!(stats.full_reprices, 2);
        assert_eq!(stats.near_reprices, 0);
        assert_eq!(stats.trigger_reprices, 0);
        assert_eq!(engine.price(ts + 1_000).cells.len(), 20 * 6);
    }

    #[test]
    fn v2_quote_engine_ignores_v1_calibration_layers_and_respects_cap() {
        let mut cfg_clean = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        cfg_clean.simulation.pricing_model = PricingModel::V2;
        cfg_clean.simulation.quote_window_count = 24;
        cfg_clean.pricing.max_multiplier = 50.0;
        cfg_clean.calibration.factors.clear();
        cfg_clean.pricing.off_hours_m_base = None;
        cfg_clean.pricing.current_row_logit_shift = 0.0;

        let mut cfg_dirty = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        cfg_dirty.simulation.pricing_model = PricingModel::V2;
        cfg_dirty.simulation.quote_window_count = 24;
        cfg_dirty.pricing.max_multiplier = 50.0;
        cfg_dirty.pricing.p_raw_gamma = 2.0;
        cfg_dirty.pricing.current_row_logit_shift = 0.8;
        cfg_dirty.pricing.off_hours_m_base = Some(0.25);

        let mut e_clean = QuoteEngine::with_config("NVDA".to_string(), 3, cfg_clean);
        let mut e_dirty = QuoteEngine::with_config("NVDA".to_string(), 3, cfg_dirty);
        assert!(e_clean.is_v2_active());
        assert!(e_dirty.is_v2_active());

        // Off-hours timestamp (03:00 UTC) so off_hours_m_base would apply if v1 ran.
        let ts = 1_795_057_200_000_u64;
        let spot = 139.99;
        let hist = sample_history(spot);
        e_clean.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &hist);
        e_dirty.on_sample(PriceTick { ts_ms: ts, price: spot, mid_px: None, oracle_px: None }, &hist);

        let q_clean = e_clean.price(ts);
        let q_dirty = e_dirty.price(ts);
        assert_eq!(q_clean.cells.len(), 20 * 24);
        assert_eq!(q_clean, q_dirty);
        assert!(q_clean.cells.iter().any(|c| c.multiplier > 25.0 && c.multiplier <= 50.0));
    }

    #[test]
    fn v2_falls_back_to_v1_for_tickers_without_recorded_data() {
        for ticker in ["AAPL", "SP500"] {
            let mut cfg = stock_grid_config_for_ticker(ticker, 8, Vec::new());
            cfg.simulation.pricing_model = PricingModel::V2;
            let engine = QuoteEngine::with_config(ticker.to_string(), 8, cfg);
            assert!(
                !engine.is_v2_active(),
                "{ticker} has no recorded tick history and must stay on v1"
            );
        }
    }

    #[test]
    fn m4_switch_leaves_v1_and_v2_identical_and_activates_for_nvda_tsla() {
        let ts = 1_795_057_200_000_u64;
        let spot = 145.98;
        let hist = sample_history(spot);
        let tick_bare = PriceTick::new(ts, spot);
        let tick_full = PriceTick {
            ts_ms: ts,
            price: spot,
            mid_px: Some(146.04),
            oracle_px: Some(145.92),
        };

        // Adding optional mid_px / oracle_px to PriceTick must leave v1 and v2 outputs identical.
        for model in [PricingModel::V1, PricingModel::V2] {
            let mut cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
            cfg.simulation.pricing_model = model;
            cfg.simulation.seed = Some(42);
            cfg.simulation.mc_n_min = 500;
            cfg.simulation.mc_n_max = 500;
            let mut e_bare = QuoteEngine::with_config("NVDA".to_string(), 3, cfg.clone());
            let mut e_full = QuoteEngine::with_config("NVDA".to_string(), 3, cfg);
            assert!(!e_bare.is_m4_active());
            e_bare.on_sample(tick_bare, &hist);
            e_full.on_sample(tick_full, &hist);
            assert_eq!(e_bare.price(ts), e_full.price(ts));
        }

        // M4 activates on NVDA & TSLA, falls back to v1 on AAPL & SP500.
        let mut cfg_m4 = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        cfg_m4.simulation.pricing_model = PricingModel::M4;
        cfg_m4.simulation.seed = Some(42);
        cfg_m4.simulation.mc_n_min = 2_000;
        cfg_m4.simulation.mc_n_max = 2_000;
        let mut e_m4 = QuoteEngine::with_config("NVDA".to_string(), 3, cfg_m4);
        assert!(e_m4.is_m4_active());
        assert!(!e_m4.is_v2_active());
        e_m4.on_sample(tick_full, &hist);
        let q_m4 = e_m4.price(ts);
        assert_eq!(q_m4.cells.len(), 20 * 6);
        assert!(
            q_m4.cells
                .iter()
                .all(|c| c.multiplier >= 1.05 && c.multiplier <= 25.0)
        );

        for ticker in ["AAPL", "SP500"] {
            let mut cfg = stock_grid_config_for_ticker(ticker, 8, Vec::new());
            cfg.simulation.pricing_model = PricingModel::M4;
            let engine = QuoteEngine::with_config(ticker.to_string(), 8, cfg);
            assert!(!engine.is_m4_active());
            assert!(!engine.is_v2_active());
        }
    }

    #[test]
    fn deribit_v2_quote_engine_applies_book_lead_and_supports_custom_band_heights() {
        let ts = 1_795_057_200_000_u64;
        let spot = 146.00;
        let hist = sample_history(spot);

        // NVDA with 5c rows on Deribit v2: +5c book gap shifts w0 touch probs up above spot and down below.
        let mut cfg = stock_grid_config_for_ticker("NVDA", 5, Vec::new());
        cfg.simulation.pricing_model = PricingModel::V2;
        cfg.simulation.price_source = PriceSource::Deribit;
        let mut eng_flat = QuoteEngine::with_config("NVDA".to_string(), 5, cfg.clone());
        let mut eng_up = QuoteEngine::with_config("NVDA".to_string(), 5, cfg);

        eng_flat.on_sample(
            PriceTick {
                ts_ms: ts,
                price: spot,
                mid_px: Some(146.00),
                oracle_px: Some(146.00),
            },
            &hist,
        );
        eng_up.on_sample(
            PriceTick {
                ts_ms: ts,
                price: spot,
                mid_px: Some(146.05),
                oracle_px: Some(146.00),
            },
            &hist,
        );

        let q_flat = eng_flat.price(ts);
        let q_up = eng_up.price(ts);
        assert_eq!(q_flat.band_cents, 5);
        assert_eq!(q_up.band_cents, 5);

        let w0_start = q_flat.cells.iter().map(|c| c.window_start_ms).min().unwrap();
        let above_flat = q_flat
            .cells
            .iter()
            .find(|c| c.window_start_ms == w0_start && c.lower_cents > 14_605)
            .unwrap();
        let above_up = q_up
            .cells
            .iter()
            .find(|c| c.window_start_ms == w0_start && c.lower_cents == above_flat.lower_cents)
            .unwrap();
        assert!(above_up.hit_probability > above_flat.hit_probability);

        let below_flat = q_flat
            .cells
            .iter()
            .find(|c| c.window_start_ms == w0_start && c.upper_cents < 14_595)
            .unwrap();
        let below_up = q_up
            .cells
            .iter()
            .find(|c| c.window_start_ms == w0_start && c.lower_cents == below_flat.lower_cents)
            .unwrap();
        assert!(below_up.hit_probability < below_flat.hit_probability);

        // TSLA with 10c rows on Deribit v2 quotes cleanly.
        let mut tsla_cfg = stock_grid_config_for_ticker("TSLA", 10, Vec::new());
        tsla_cfg.simulation.pricing_model = PricingModel::V2;
        tsla_cfg.simulation.price_source = PriceSource::Deribit;
        let mut tsla_eng = QuoteEngine::with_config("TSLA".to_string(), 10, tsla_cfg);
        tsla_eng.on_sample(
            PriceTick {
                ts_ms: ts,
                price: 364.00,
                mid_px: Some(364.10),
                oracle_px: Some(364.00),
            },
            &sample_history(364.00),
        );
        let q_tsla = tsla_eng.price(ts);
        assert_eq!(q_tsla.band_cents, 10);
        assert_eq!(q_tsla.cells.len(), 20 * 6);
    }

    #[test]
    fn v2_params_json_round_trip_produces_bit_identical_quote_engine_quotes_for_both_sources() {
        use crate::v2::parse_v2_params_json;

        let ts = 1_795_057_200_000_u64;
        for (ticker, band_cents, spot, mid_px, oracle_px, source) in [
            ("NVDA", 3_u32, 139.99_f64, 140.04_f64, 139.99_f64, PriceSource::Hyperliquid),
            ("TSLA", 8_u32, 364.04_f64, 364.14_f64, 364.04_f64, PriceSource::Hyperliquid),
            ("NVDA", 3_u32, 139.99_f64, 140.04_f64, 139.99_f64, PriceSource::Deribit),
            ("TSLA", 8_u32, 364.04_f64, 364.14_f64, 364.04_f64, PriceSource::Deribit),
        ] {
            let compiled = v2_ticker_params_for_source(ticker, source).unwrap();
            let dumped = compiled.to_json_pretty().unwrap();
            let loaded = parse_v2_params_json(&dumped).unwrap().leak_static();

            let mut cfg_compiled = stock_grid_config_for_ticker(ticker, band_cents, Vec::new());
            cfg_compiled.simulation.pricing_model = PricingModel::V2;
            cfg_compiled.simulation.price_source = source;

            let mut cfg_loaded = cfg_compiled.clone();
            cfg_loaded.simulation.v2_params_override = Some(loaded);

            let mut eng_compiled =
                QuoteEngine::with_config(ticker.to_string(), band_cents, cfg_compiled);
            let mut eng_loaded =
                QuoteEngine::with_config(ticker.to_string(), band_cents, cfg_loaded);

            let hist = sample_history(spot);
            let tick = PriceTick {
                ts_ms: ts,
                price: spot,
                mid_px: Some(mid_px),
                oracle_px: Some(oracle_px),
            };
            eng_compiled.on_sample(tick, &hist);
            eng_loaded.on_sample(tick, &hist);

            assert_eq!(
                eng_compiled.price(ts),
                eng_loaded.price(ts),
                "round-trip quote mismatch for {ticker} on {}",
                source.as_str()
            );
        }
    }

    #[test]
    fn v2_off_hours_table_switches_by_session_and_identical_copy_is_bit_identical() {
        use crate::v2::{NVDA_DERIBIT_V2_PARAMS, TickerV2ParamsJson};

        const TS_US_REGULAR: u64 = 1_794_927_600_000; // 2026-11-17 15:00:00 UTC (10:00:00 ET)
        const TS_NIGHT: u64 = 1_794_970_800_000; // 2026-11-18 03:00:00 UTC (22:00:00 ET)
        const TS_WEEKEND: u64 = 1_795_273_200_000; // 2026-11-21 15:00:00 UTC (Sat 10:00:00 ET)
        const TS_HOLIDAY: u64 = 1_795_705_200_000; // 2026-11-26 15:00:00 UTC (Thanksgiving)
        const TS_EARLY_BEFORE: u64 = 1_795_802_399_000; // 2026-11-27 17:59:59 UTC (12:59:59 ET)
        const TS_EARLY_AFTER: u64 = 1_795_802_400_000; // 2026-11-27 18:00:00 UTC (13:00:00 ET)
        const TS_XMAS_EVE_AFTER: u64 = 1_798_135_200_000; // 2026-12-24 18:00:00 UTC (13:00:00 ET)

        let make_engine = |params: &'static TickerV2Params| {
            let mut cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
            cfg.simulation.pricing_model = PricingModel::V2;
            cfg.simulation.price_source = PriceSource::Deribit;
            cfg.simulation.v2_params_override = Some(params);
            QuoteEngine::with_config("NVDA".to_string(), 3, cfg)
        };

        let quote_at = |params: &'static TickerV2Params, ts: u64| {
            let mut eng = make_engine(params);
            let tick = PriceTick {
                ts_ms: ts,
                price: 140.00,
                mid_px: Some(140.10),
                oracle_px: Some(140.00),
            };
            eng.on_sample(tick, &sample_history(140.00));
            eng.price(ts)
        };

        // 1. off_hours = None vs Some(identical copy) give bit-identical quotes in both sessions.
        let mut main_only_json = TickerV2ParamsJson::from_params(&NVDA_DERIBIT_V2_PARAMS);
        main_only_json.off_hours = None;
        let main_only = main_only_json.clone().leak_static();
        let mut identical_off_json = main_only_json.clone();
        identical_off_json.off_hours = Some(Box::new(main_only_json.clone()));
        let identical_off = identical_off_json.leak_static();

        for ts in [TS_US_REGULAR, TS_NIGHT, TS_EARLY_BEFORE, TS_EARLY_AFTER] {
            assert_eq!(
                quote_at(main_only, ts),
                quote_at(identical_off, ts),
                "identical off_hours copy must be bit-identical at ts={ts}"
            );
        }

        // 2. Distinct off_hours table: US timestamps use main, off-hours timestamps use off.
        let mut off_json = main_only_json.clone();
        off_json.book_gap_beta = 0.30;
        off_json.book_lead_secs = 1;
        off_json.book_gap_clamp_cents = 10;
        off_json.rms_ref_cents = 0.80;
        off_json.thresholds_rms_cents = [0.50, 1.00, 2.00];
        off_json.shock_single_step_cents = 12;
        for p in &mut off_json.presets {
            p.p_move = (p.p_move * 0.5).clamp(0.05, 1.0);
            p.q_1s = p.p_move;
            p.m_base = (p.m_base + 0.04).min(0.5);
        }
        off_json.window_presets = Some(off_json.presets);
        let off_only = off_json.clone().leak_static();

        let mut split_json = main_only_json.clone();
        split_json.off_hours = Some(Box::new(off_json));
        let split = split_json.leak_static();

        // US regular session & 12:59:59 ET on early close day -> main table.
        for ts in [TS_US_REGULAR, TS_EARLY_BEFORE] {
            let q_split = quote_at(split, ts);
            let q_main = quote_at(main_only, ts);
            let q_off = quote_at(off_only, ts);
            assert_eq!(q_split, q_main, "US timestamp {ts} must use main table");
            assert_ne!(q_split, q_off, "US timestamp {ts} must differ from off table");
        }

        // Night, weekend, holiday, and 13:00:00 ET early closes -> off table.
        for ts in [TS_NIGHT, TS_WEEKEND, TS_HOLIDAY, TS_EARLY_AFTER, TS_XMAS_EVE_AFTER] {
            let q_split = quote_at(split, ts);
            let q_main = quote_at(main_only, ts);
            let q_off = quote_at(off_only, ts);
            assert_eq!(q_split, q_off, "off-hours timestamp {ts} must use off table");
            assert_ne!(q_split, q_main, "off-hours timestamp {ts} must differ from main table");
        }

        // 3. Crossing the 13:00 ET early-close boundary keeps the R_60 buffer and classifies
        //    immediately with the off-hours thresholds (3c alternating steps -> R_60 ~ 3c:
        //    Volatile under US thresholds [1.162, 2.412, 8.390], News under off [0.50, 1.00, 2.00]).
        let mut eng = make_engine(split);
        let hist = sample_history(140.00);
        for sec in 0..16_u64 {
            // 17:59:44..17:59:59 UTC (12:59:44..12:59:59 ET, still US session)
            let ts = TS_EARLY_BEFORE - (15 - sec) * 1_000;
            let price = if sec % 2 == 0 { 140.00 } else { 140.03 };
            eng.observe_tick(PriceTick::new(ts, price));
        }
        assert_eq!(eng.current_v2_regime(), Some(Regime::Volatile));
        // Next tick at 13:00:00 ET (18:00:00 UTC) enters off hours: same R_60 (~3c) > off p99 (2.0c) -> News.
        eng.on_sample(PriceTick::new(TS_EARLY_AFTER, 140.00), &hist);
        assert_eq!(eng.current_v2_regime(), Some(Regime::News));
    }
}
