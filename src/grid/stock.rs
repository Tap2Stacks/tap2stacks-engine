use super::types::{
    CalibrationKey, GridConfig, GridSpec, PricingConfig, SimulationConfig, TimeWindow, TouchMode,
    VolatilityConfig,
};

const CALIBRATION_WINDOWS: usize = 6;
const CALIBRATION_SIGMA_BUCKETS: usize = 3;

fn set_distance_factors(cfg: &mut GridConfig, dist_factors: &[(usize, f64)]) {
    cfg.calibration.factors.clear();
    for window_index in 0..CALIBRATION_WINDOWS {
        for sigma_bucket in 0..CALIBRATION_SIGMA_BUCKETS {
            for &(dist_bucket, factor) in dist_factors {
                cfg.calibration.factors.insert(
                    CalibrationKey {
                        window_index,
                        dist_bucket,
                        sigma_bucket,
                    },
                    factor,
                );
            }
        }
    }
}

/// Builds a `GridConfig` configured for US equity 1 Hz discrete-sample quoting.
pub fn stock_grid_config(band_cents: u32, windows: Vec<TimeWindow>) -> GridConfig {
    let default = GridConfig::default();
    let mut volatility = VolatilityConfig {
        parkinson_weight: 0.0,
        price_adj_min: 1.0,
        price_adj_max: 1.0,
        variance_floor: 1e-10,
        sigma_scale: 1.005,
        ..default.volatility
    };
    volatility.jump.enabled = false;

    GridConfig {
        spec: GridSpec {
            rows: 20,
            center_row: 9,
            band_width: band_cents as f64 / 100.0,
            windows,
        },
        simulation: SimulationConfig {
            touch_mode: TouchMode::Continuous,
            lock_offset: 0,
            bridge_epsilon: 1e-12,
            ..default.simulation
        },
        volatility,
        pricing: PricingConfig {
            m_base: 0.020,
            p_raw_gamma: 0.0,
            p_time_boost: 0.0,
            current_row_logit_shift: 0.32,
            adjacent_row_logit_shift: 0.0,
            zone_vol_shift_scale: 0.0,
            ..default.pricing
        },
        ..default
    }
}

/// Builds a ticker-specific `GridConfig` for US equities (`NVDA`, `TSLA`).
pub fn stock_grid_config_for_ticker(
    ticker: &str,
    band_cents: u32,
    windows: Vec<TimeWindow>,
) -> GridConfig {
    let mut cfg = stock_grid_config(band_cents, windows);
    let symbol = ticker.split(':').next_back().unwrap_or(ticker);
    if symbol.eq_ignore_ascii_case("TSLA") {
        cfg.volatility.sigma_ref = 0.00011;
        cfg.volatility.sigma_scale = 1.07;
        cfg.pricing.m_base = 0.012;
        cfg.pricing.current_row_logit_shift = 0.64;
        set_distance_factors(&mut cfg, &[(1, 0.71), (2, 0.49), (3, 0.51), (4, 0.94)]);
    } else if symbol.eq_ignore_ascii_case("NVDA") {
        cfg.volatility.sigma_ref = 0.00023;
        set_distance_factors(&mut cfg, &[(1, 0.72), (2, 0.545), (3, 0.505), (4, 0.96)]);
    } else {
        cfg.volatility.sigma_ref = 0.00020;
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::state::estimate_volatility;
    use crate::grid::touch::{window_hit_discrete, window_hit_probability};
    use crate::grid::types::{OhlcBar, PriceBand};

    #[test]
    fn stock_config_uses_bridge_touch() {
        let windows = vec![
            TimeWindow {
                start_secs: 5,
                end_secs: 10,
            },
            TimeWindow {
                start_secs: 10,
                end_secs: 15,
            },
        ];
        let c = stock_grid_config(3, windows);
        assert_eq!(c.simulation.touch_mode, TouchMode::Continuous);
    }

    #[test]
    fn stock_config_prices_straddle_as_certain_hit() {
        let window = TimeWindow {
            start_secs: 0,
            end_secs: 1,
        };
        let c = stock_grid_config(3, vec![window]);
        let path = [157.40, 157.70];
        let band = PriceBand {
            row_index: 0,
            lower: 157.52,
            upper: 157.55,
        };
        let p = match c.simulation.touch_mode {
            TouchMode::Continuous => window_hit_probability(
                &path,
                band,
                window,
                0.01,
                c.simulation.lock_offset,
                c.simulation.bridge_epsilon,
            ),
            TouchMode::Discrete => {
                window_hit_discrete(&path, &band, &window, c.simulation.lock_offset)
            }
        };
        assert_eq!(p, 1.0);
    }

    #[test]
    fn stock_config_neutralises_btc_adjustments() {
        let windows = vec![
            TimeWindow {
                start_secs: 5,
                end_secs: 10,
            },
            TimeWindow {
                start_secs: 10,
                end_secs: 15,
            },
        ];
        let c = stock_grid_config(3, windows);
        assert_eq!(c.volatility.parkinson_weight, 0.0);
        assert_eq!(c.volatility.price_adj_min, 1.0);
        assert_eq!(c.volatility.price_adj_max, 1.0);
        assert_eq!(c.spec.band_width, 0.03);
        assert_eq!(c.spec.rows, 20);
        assert_eq!(c.spec.center_row, 9);
        assert_eq!(c.simulation.lock_offset, 0);
        assert_eq!(c.simulation.touch_mode, TouchMode::Continuous);

        // Verify 1s flat bars do not suffer Parkinson dilution or BTC price adjustment.
        let bars = vec![
            OhlcBar {
                open: 157.0,
                high: 157.0,
                low: 157.0,
                close: 157.0,
            },
            OhlcBar {
                open: 157.5,
                high: 157.5,
                low: 157.5,
                close: 157.5,
            },
            OhlcBar {
                open: 158.0,
                high: 158.0,
                low: 158.0,
                close: 158.0,
            },
        ];
        let stock_vol = estimate_volatility(158.0, &bars, &c).unwrap();
        let btc_vol = estimate_volatility(158.0, &bars, &GridConfig::default()).unwrap();
        assert!(stock_vol.sigma_raw > btc_vol.sigma_raw);
        assert!(stock_vol.sigma > btc_vol.sigma);
        assert!(
            (stock_vol.sigma - stock_vol.sigma_raw * c.volatility.sigma_scale).abs() < 1e-12
        );
    }

    #[test]
    fn ticker_configs_set_calibrated_stock_defaults() {
        let windows = vec![TimeWindow {
            start_secs: 5,
            end_secs: 10,
        }];
        let nvda = stock_grid_config_for_ticker("NVDA", 3, windows.clone());
        let tsla = stock_grid_config_for_ticker("xyz:TSLA", 8, windows);
        assert_eq!(nvda.volatility.sigma_ref, 0.00023);
        assert_eq!(nvda.volatility.sigma_scale, 1.005);
        assert_eq!(nvda.volatility.variance_floor, 1e-10);
        assert!(!nvda.volatility.jump.enabled);
        assert_eq!(nvda.simulation.bridge_epsilon, 1e-12);
        assert_eq!(nvda.pricing.m_base, 0.020);
        assert_eq!(nvda.pricing.p_raw_gamma, 0.0);
        assert_eq!(nvda.pricing.current_row_logit_shift, 0.32);
        assert_eq!(
            nvda.calibration.factors.get(&CalibrationKey {
                window_index: 0,
                dist_bucket: 2,
                sigma_bucket: 1,
            }),
            Some(&0.545)
        );

        assert_eq!(tsla.volatility.sigma_ref, 0.00011);
        assert_eq!(tsla.volatility.sigma_scale, 1.07);
        assert_eq!(tsla.volatility.variance_floor, 1e-10);
        assert!(!tsla.volatility.jump.enabled);
        assert_eq!(tsla.simulation.bridge_epsilon, 1e-12);
        assert_eq!(tsla.pricing.m_base, 0.012);
        assert_eq!(tsla.pricing.p_raw_gamma, 0.0);
        assert_eq!(tsla.pricing.current_row_logit_shift, 0.64);
        assert_eq!(
            tsla.calibration.factors.get(&CalibrationKey {
                window_index: 0,
                dist_bucket: 2,
                sigma_bucket: 1,
            }),
            Some(&0.49)
        );
    }
}
