mod pricing;
mod simulator;
mod state;
mod stock;
mod touch;
mod types;

use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

pub use pricing::{
    apply_quote_logit_shift, compute_k_safety, compute_k_skew, compute_m_eff,
    compute_m_eff_with_base, compute_quote_probability, effective_m_base, price_multiplier,
    price_multiplier_with_margin, smooth_used_liability,
};
pub use simulator::{
    debug_reconstruct_paths_from_returns, debug_reconstruct_paths_from_shocks,
    debug_simulate_paths, simulate_summary, DebugPathTrace, SimulationSummary,
};
pub use state::{estimate_runtime_state, estimate_volatility, update_runtime_state};
pub use stock::{stock_grid_config, stock_grid_config_for_ticker};
pub use touch::{
    interval_hit_probabilities, interval_hit_probability, window_hit_discrete,
    window_hit_probability,
};
pub use types::{
    CalibrationConfig, CalibrationKey, CellQuote, GridConfig, GridError, GridSnapshot, GridSpec,
    JumpConfig, LiabilityBook, OhlcBar, PriceBand, PricingConfig, RuntimeState, SimulationConfig,
    TimeWindow, TouchMode, VolatilityConfig, VolatilitySnapshot,
};

#[derive(Debug, Clone)]
pub struct GridEngine {
    pub config: GridConfig,
    liabilities: LiabilityBook,
    liability_used: std::collections::HashMap<usize, f64>,
    cached_k_safety: Option<CachedSafetyFactor>,
}

#[derive(Debug, Clone, Copy)]
struct CachedSafetyFactor {
    value: f64,
    computed_at: Instant,
}

impl GridEngine {
    pub fn new(config: GridConfig) -> Self {
        Self {
            config,
            liabilities: LiabilityBook::default(),
            liability_used: std::collections::HashMap::new(),
            cached_k_safety: None,
        }
    }

    pub fn set_liability(&mut self, cell_index: usize, value: f64) {
        self.liabilities.set_liability(cell_index, value);
    }

    pub fn add_liability(&mut self, cell_index: usize, value: f64) {
        self.liabilities.add_liability(cell_index, value);
    }

    pub fn get_liability(&self, cell_index: usize) -> f64 {
        self.liabilities.get_liability(cell_index)
    }

    pub fn get_used_liability(&self, cell_index: usize) -> f64 {
        self.liability_used.get(&cell_index).copied().unwrap_or(0.0)
    }

    pub fn liability_snapshot(&self) -> std::collections::HashMap<usize, f64> {
        self.liabilities.snapshot()
    }

    pub fn clear_liabilities(&mut self) {
        self.liabilities.clear();
        self.liability_used.clear();
        self.cached_k_safety = None;
    }

    pub fn quote_grid(
        &mut self,
        spot: f64,
        history: &[OhlcBar],
    ) -> Result<GridSnapshot, GridError> {
        quote_grid_with_internal_liabilities(
            spot,
            history,
            &self.config,
            &self.liabilities,
            &mut self.liability_used,
            &mut self.cached_k_safety,
        )
    }
}

pub fn quote_grid(
    spot: f64,
    history: &[OhlcBar],
    config: &GridConfig,
) -> Result<GridSnapshot, GridError> {
    quote_grid_with_liabilities(spot, history, config, None)
}

pub fn quote_grid_with_liabilities(
    spot: f64,
    history: &[OhlcBar],
    config: &GridConfig,
    liabilities: Option<&[f64]>,
) -> Result<GridSnapshot, GridError> {
    quote_grid_internal(spot, history, config, liabilities, None)
}

fn quote_grid_with_internal_liabilities(
    spot: f64,
    history: &[OhlcBar],
    config: &GridConfig,
    liabilities: &LiabilityBook,
    liability_used: &mut std::collections::HashMap<usize, f64>,
    cached_k_safety: &mut Option<CachedSafetyFactor>,
) -> Result<GridSnapshot, GridError> {
    quote_grid_internal(
        spot,
        history,
        config,
        None,
        Some((liabilities, liability_used, cached_k_safety)),
    )
}

fn quote_grid_internal(
    spot: f64,
    history: &[OhlcBar],
    config: &GridConfig,
    external_liabilities: Option<&[f64]>,
    mut internal_liability_state: Option<(
        &LiabilityBook,
        &mut std::collections::HashMap<usize, f64>,
        &mut Option<CachedSafetyFactor>,
    )>,
) -> Result<GridSnapshot, GridError> {
    if spot <= 0.0 {
        return Err(GridError::InvalidSpot);
    }
    config.validate()?;

    let anchor = aligned_anchor(spot, config.spec.band_width);
    let bands = build_bands(
        anchor,
        config.spec.rows,
        config.spec.center_row,
        config.spec.band_width,
    );
    let runtime_state = estimate_runtime_state(spot, history, config)?;
    let volatility = runtime_state.volatility;
    let max_horizon = config
        .spec
        .windows
        .iter()
        .map(|window| window.end_secs)
        .max()
        .unwrap_or(0)
        .saturating_add(config.simulation.lock_offset) as usize;

    let mut cells = Vec::with_capacity(bands.len() * config.spec.windows.len());
    let server_ts_ms = now_ms();
    let expected_cells = bands.len() * config.spec.windows.len();
    if let Some(values) = external_liabilities {
        if values.len() != expected_cells {
            return Err(GridError::InvalidConfig(
                "liabilities length must match the number of grid cells",
            ));
        }
    }
    let mut liabilities_vector = vec![0.0; expected_cells];
    if let Some(values) = external_liabilities {
        liabilities_vector.copy_from_slice(values);
    } else if let Some((book, _, _)) = &internal_liability_state {
        for (cell_index, liability) in liabilities_vector.iter_mut().enumerate() {
            *liability = book.get_liability(cell_index);
        }
    }
    let now = Instant::now();
    let cached_k_safety_value = internal_liability_state
        .as_ref()
        .and_then(|(_, _, cached)| cached.as_ref())
        .filter(|cached| {
            now.duration_since(cached.computed_at)
                <= Duration::from_millis(config.pricing.k_safety_ttl_ms)
        })
        .map(|cached| cached.value);
    let simulation = simulate_summary(
        spot,
        &bands,
        config,
        &runtime_state,
        max_horizon,
        if cached_k_safety_value.is_some() {
            None
        } else {
            Some(&liabilities_vector)
        },
    );
    let k_safety = if let Some(value) = cached_k_safety_value {
        value
    } else {
        let value = compute_k_safety(&simulation.scenario_payouts, config);
        if let Some((_, _, cached)) = &mut internal_liability_state {
            **cached = Some(CachedSafetyFactor {
                value,
                computed_at: now,
            });
        }
        value
    };
    let current_window0_row_index = bands
        .iter()
        .min_by(|left, right| {
            let left_center = (left.lower + left.upper) / 2.0;
            let right_center = (right.lower + right.upper) / 2.0;
            (left_center - spot)
                .abs()
                .partial_cmp(&(right_center - spot).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|band| band.row_index)
        .unwrap_or(config.spec.center_row);

    for (row_index, band) in bands.iter().enumerate() {
        for (window_index, window) in config.spec.windows.iter().enumerate() {
            let cell_index = row_index * config.spec.windows.len() + window_index;
            let raw_hit_probability = simulation.mean_probabilities[row_index][window_index];
            let hit_probability = compute_quote_probability(
                raw_hit_probability,
                row_index,
                window_index,
                *window,
                volatility,
                current_window0_row_index,
                config,
            );
            let liability = if let Some((_, liability_used, _)) = &mut internal_liability_state {
                let instant = liabilities_vector[cell_index];
                let previous = liability_used.get(&cell_index).copied().unwrap_or(0.0);
                let used = smooth_used_liability(previous, instant, config);
                liability_used.insert(cell_index, used);
                used
            } else {
                liabilities_vector[cell_index]
            };
            let multiplier =
                price_multiplier(hit_probability, volatility, liability, k_safety, config);

            cells.push(CellQuote {
                server_ts_ms,
                band: *band,
                window: *window,
                hit_probability,
                multiplier,
            });
        }
    }

    Ok(GridSnapshot {
        server_ts_ms,
        spot,
        anchor,
        volatility,
        cells,
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn aligned_anchor(spot: f64, band_width: f64) -> f64 {
    (spot / band_width).floor() * band_width
}

fn build_bands(anchor: f64, rows: usize, center_row: usize, band_width: f64) -> Vec<PriceBand> {
    (0..rows)
        .map(|row_index| {
            let center = anchor + ((row_index as f64) - (center_row as f64)) * band_width;
            PriceBand {
                row_index,
                lower: center - band_width / 2.0,
                upper: center + band_width / 2.0,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_history() -> Vec<OhlcBar> {
        vec![
            OhlcBar {
                open: 50_000.0,
                high: 50_020.0,
                low: 49_980.0,
                close: 50_010.0,
            },
            OhlcBar {
                open: 50_010.0,
                high: 50_045.0,
                low: 49_995.0,
                close: 50_030.0,
            },
            OhlcBar {
                open: 50_030.0,
                high: 50_060.0,
                low: 50_010.0,
                close: 50_040.0,
            },
        ]
    }

    fn sample_volatility(config: &GridConfig) -> VolatilitySnapshot {
        VolatilitySnapshot {
            variance: 0.0,
            sigma_raw: config.volatility.sigma_ref,
            sigma: config.volatility.sigma_ref,
            jump_intensity: config.pricing.lambda_ref,
            jump_flag: false,
        }
    }

    #[test]
    fn builds_expected_number_of_cells() {
        let config = GridConfig {
            simulation: SimulationConfig {
                mc_n_min: 100,
                mc_n_max: 100,
                ..SimulationConfig::default()
            },
            volatility: VolatilityConfig {
                jump: JumpConfig {
                    enabled: false,
                    ..JumpConfig::default()
                },
                ..VolatilityConfig::default()
            },
            ..GridConfig::default()
        };

        let snapshot = quote_grid(50_000.0, &sample_history(), &config).unwrap();

        assert_eq!(
            snapshot.cells.len(),
            config.spec.rows * config.spec.windows.len()
        );
    }

    #[test]
    fn anchor_uses_floor_alignment_like_python_mc_branch() {
        let config = GridConfig {
            simulation: SimulationConfig {
                mc_n_min: 100,
                mc_n_max: 100,
                ..SimulationConfig::default()
            },
            volatility: VolatilityConfig {
                jump: JumpConfig {
                    enabled: false,
                    ..JumpConfig::default()
                },
                ..VolatilityConfig::default()
            },
            ..GridConfig::default()
        };

        let snapshot = quote_grid(50_006.0, &sample_history(), &config).unwrap();

        assert_eq!(snapshot.anchor, 50_000.0);
    }

    #[test]
    fn center_row_band_straddles_anchor_like_python_mc_branch() {
        let config = GridConfig {
            simulation: SimulationConfig {
                mc_n_min: 100,
                mc_n_max: 100,
                ..SimulationConfig::default()
            },
            volatility: VolatilityConfig {
                jump: JumpConfig {
                    enabled: false,
                    ..JumpConfig::default()
                },
                ..VolatilityConfig::default()
            },
            ..GridConfig::default()
        };

        let snapshot = quote_grid(50_006.0, &sample_history(), &config).unwrap();
        let center_band = snapshot
            .cells
            .iter()
            .find(|cell| {
                cell.band.row_index == config.spec.center_row
                    && cell.window == config.spec.windows[0]
            })
            .map(|cell| cell.band)
            .unwrap();

        assert_eq!(center_band.lower, 49_995.0);
        assert_eq!(center_band.upper, 50_005.0);
    }

    #[test]
    fn jump_sampler_records_and_replays_empirical_jumps() {
        let mut sampler = super::types::JumpSamplerState::new(4);
        let mut rng = rand::rng();

        sampler.record(0.01);
        sampler.record(-0.02);
        sampler.record(0.03);

        let samples = sampler.sample(16, &mut rng, 0.005);

        assert_eq!(samples.len(), 16);
        assert!(samples.iter().all(|sample| sample.abs() > 0.0));
        assert!(samples.iter().any(|sample| *sample > 0.0));
        assert!(samples.iter().any(|sample| *sample < 0.0));
    }

    #[test]
    fn effective_margin_increases_with_market_risk() {
        let config = GridConfig::default();
        let calm = sample_volatility(&config);
        let stressed = VolatilitySnapshot {
            variance: 0.0,
            sigma_raw: config.volatility.sigma_ref * 2.0,
            sigma: config.volatility.sigma_ref * 2.0,
            jump_intensity: config.pricing.lambda_ref * 2.0,
            jump_flag: true,
        };

        assert!(compute_m_eff(stressed, &config) > compute_m_eff(calm, &config));
        assert!(compute_m_eff(calm, &config) >= config.pricing.m_base);
    }

    #[test]
    fn liability_skew_reduces_multiplier_for_loaded_cell() {
        let config = GridConfig {
            simulation: SimulationConfig {
                mc_n_min: 100,
                mc_n_max: 100,
                ..SimulationConfig::default()
            },
            volatility: VolatilityConfig {
                jump: JumpConfig {
                    enabled: false,
                    ..JumpConfig::default()
                },
                ..VolatilityConfig::default()
            },
            ..GridConfig::default()
        };
        let volatility = sample_volatility(&config);

        let unloaded = price_multiplier(0.1, volatility, 0.0, 1.0, &config);
        let loaded = price_multiplier(0.1, volatility, config.pricing.pool_cap, 1.0, &config);

        assert!(compute_k_skew(config.pricing.pool_cap, &config) < 1.0);
        assert!(loaded < unloaded);
    }

    #[test]
    fn safety_factor_reduces_global_multiplier_when_board_liability_is_high() {
        let config = GridConfig {
            pricing: PricingConfig {
                risk_budget: 50.0,
                ..PricingConfig::default()
            },
            ..GridConfig::default()
        };
        let volatility = sample_volatility(&config);
        let scenario_payouts = vec![150.0, 75.0, 75.0];
        let k_safety = compute_k_safety(&scenario_payouts, &config);
        let safe = price_multiplier(0.1, volatility, 0.0, 1.0, &config);
        let stressed = price_multiplier(0.1, volatility, 0.0, k_safety, &config);

        assert!(k_safety < 1.0);
        assert!(stressed < safe);
    }

    #[test]
    fn smooth_used_liability_holds_on_way_down() {
        let config = GridConfig {
            pricing: PricingConfig {
                liability_rho: 0.7,
                ..PricingConfig::default()
            },
            ..GridConfig::default()
        };

        let used = super::smooth_used_liability(100.0, 20.0, &config);

        assert_eq!(used, 76.0);
    }

    #[test]
    fn grid_engine_applies_internal_liability_smoothing() {
        let config = GridConfig {
            simulation: SimulationConfig {
                mc_n_min: 100,
                mc_n_max: 100,
                ..SimulationConfig::default()
            },
            volatility: VolatilityConfig {
                jump: JumpConfig {
                    enabled: false,
                    ..JumpConfig::default()
                },
                ..VolatilityConfig::default()
            },
            pricing: PricingConfig {
                liability_rho: 0.7,
                ..PricingConfig::default()
            },
            ..GridConfig::default()
        };
        let mut engine = GridEngine::new(config);

        engine.set_liability(0, 100.0);
        let _ = engine.quote_grid(50_000.0, &sample_history()).unwrap();
        assert_eq!(engine.get_used_liability(0), 100.0);

        engine.set_liability(0, 20.0);
        let _ = engine.quote_grid(50_000.0, &sample_history()).unwrap();
        assert_eq!(engine.get_used_liability(0), 76.0);
    }

    #[test]
    fn distance_tempering_penalizes_far_rows_more_than_center_row() {
        let config = GridConfig {
            pricing: PricingConfig {
                p_near_boost: 0.0,
                current_row_logit_shift: 0.0,
                adjacent_row_logit_shift: 0.0,
                zone_vol_shift_scale: 0.0,
                ..PricingConfig::default()
            },
            ..GridConfig::default()
        };
        let raw_probability = 0.4;
        let volatility = sample_volatility(&config);
        let far = compute_quote_probability(
            raw_probability,
            0,
            0,
            config.spec.windows[0],
            volatility,
            config.spec.center_row,
            &config,
        );
        let center = compute_quote_probability(
            raw_probability,
            config.spec.center_row,
            0,
            config.spec.windows[0],
            volatility,
            config.spec.center_row,
            &config,
        );

        assert!(far < center);
        assert!(center <= raw_probability);
    }

    #[test]
    fn near_boost_prefers_center_row_over_far_row() {
        let config = GridConfig {
            pricing: PricingConfig {
                p_raw_gamma: 0.0,
                p_near_boost: 0.15,
                current_row_logit_shift: 0.0,
                adjacent_row_logit_shift: 0.0,
                zone_vol_shift_scale: 0.0,
                ..PricingConfig::default()
            },
            ..GridConfig::default()
        };
        let raw_probability = 0.4;
        let volatility = sample_volatility(&config);
        let center = compute_quote_probability(
            raw_probability,
            config.spec.center_row,
            0,
            config.spec.windows[0],
            volatility,
            config.spec.center_row,
            &config,
        );
        let edge_row = config.spec.rows - 1;
        let far = compute_quote_probability(
            raw_probability,
            edge_row,
            0,
            config.spec.windows[0],
            volatility,
            config.spec.center_row,
            &config,
        );

        assert!(center > raw_probability);
        assert_eq!(far, raw_probability);
        assert!(center > far);
    }

    #[test]
    fn negative_time_boost_penalizes_later_windows() {
        let config = GridConfig {
            pricing: PricingConfig {
                p_raw_gamma: 0.0,
                p_near_boost: 0.0,
                p_time_boost: -0.35,
                current_row_logit_shift: 0.0,
                adjacent_row_logit_shift: 0.0,
                zone_vol_shift_scale: 0.0,
                ..PricingConfig::default()
            },
            ..GridConfig::default()
        };
        let raw_probability = 0.4;
        let row = config.spec.center_row;
        let volatility = sample_volatility(&config);
        let near = compute_quote_probability(
            raw_probability,
            row,
            0,
            config.spec.windows[0],
            volatility,
            row,
            &config,
        );
        let late = compute_quote_probability(
            raw_probability,
            row,
            4,
            config.spec.windows[4],
            volatility,
            row,
            &config,
        );

        assert_eq!(near, raw_probability);
        assert!(late < raw_probability);
    }

    #[test]
    fn calibration_factor_adjusts_quote_probability_by_bucket() {
        let mut calibration = CalibrationConfig::default();
        calibration.factors.insert(
            CalibrationKey {
                window_index: 0,
                dist_bucket: 0,
                sigma_bucket: 1,
            },
            1.1,
        );
        let config = GridConfig {
            pricing: PricingConfig {
                p_raw_gamma: 0.0,
                p_near_boost: 0.0,
                p_time_boost: 0.0,
                current_row_logit_shift: 0.0,
                adjacent_row_logit_shift: 0.0,
                zone_vol_shift_scale: 0.0,
                ..PricingConfig::default()
            },
            calibration,
            ..GridConfig::default()
        };
        let raw_probability = 0.4;
        let volatility = sample_volatility(&config);
        let calibrated = compute_quote_probability(
            raw_probability,
            config.spec.center_row,
            0,
            config.spec.windows[0],
            volatility,
            config.spec.center_row,
            &config,
        );

        assert!(calibrated > raw_probability);
    }

    #[test]
    fn zone_logit_shift_reduces_current_row_quote_probability() {
        let config = GridConfig {
            pricing: PricingConfig {
                p_raw_gamma: 0.0,
                p_near_boost: 0.0,
                p_time_boost: 0.0,
                current_row_logit_shift: 1.0,
                adjacent_row_logit_shift: 0.0,
                zone_vol_shift_scale: 0.0,
                ..PricingConfig::default()
            },
            ..GridConfig::default()
        };
        let raw_probability = 0.65;
        let volatility = sample_volatility(&config);
        let shifted = compute_quote_probability(
            raw_probability,
            config.spec.center_row,
            0,
            config.spec.windows[0],
            volatility,
            config.spec.center_row,
            &config,
        );

        assert!(shifted < raw_probability);
    }

    #[test]
    fn rejects_invalid_history() {
        let config = GridConfig::default();
        let history = vec![OhlcBar {
            open: 1.0,
            high: 0.5,
            low: 1.5,
            close: 1.0,
        }];

        let error = quote_grid(50_000.0, &history, &config).unwrap_err();

        assert!(matches!(error, GridError::InvalidHistory(_)));
    }

    #[test]
    fn discrete_probability_not_above_continuous() {
        let windows = vec![
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
        ];
        let mut discrete_cfg = stock_grid_config(3, windows.clone());
        discrete_cfg.simulation.seed = Some(42);
        discrete_cfg.simulation.mc_n_min = 2_000;
        discrete_cfg.simulation.mc_n_max = 2_000;

        let mut continuous_cfg = discrete_cfg.clone();
        continuous_cfg.simulation.touch_mode = TouchMode::Continuous;

        let history = vec![
            OhlcBar {
                open: 157.40,
                high: 157.40,
                low: 157.40,
                close: 157.40,
            },
            OhlcBar {
                open: 157.45,
                high: 157.45,
                low: 157.45,
                close: 157.45,
            },
            OhlcBar {
                open: 157.53,
                high: 157.53,
                low: 157.53,
                close: 157.53,
            },
        ];

        let spot = 157.53;
        let anchor = aligned_anchor(spot, discrete_cfg.spec.band_width);
        let bands = build_bands(
            anchor,
            discrete_cfg.spec.rows,
            discrete_cfg.spec.center_row,
            discrete_cfg.spec.band_width,
        );
        let runtime_state = estimate_runtime_state(spot, &history, &discrete_cfg).unwrap();
        let max_horizon = 20;

        let discrete_sim = simulate_summary(spot, &bands, &discrete_cfg, &runtime_state, max_horizon, None);
        let continuous_sim = simulate_summary(spot, &bands, &continuous_cfg, &runtime_state, max_horizon, None);

        for (row_d, row_c) in discrete_sim
            .mean_probabilities
            .iter()
            .zip(continuous_sim.mean_probabilities.iter())
        {
            for (&p_d, &p_c) in row_d.iter().zip(row_c.iter()) {
                assert!(
                    p_d <= p_c + 1e-9,
                    "expected discrete probability {p_d} <= continuous {p_c}"
                );
            }
        }
    }
}
