use super::types::{
    CalibrationKey, GridConfig, GridSpec, PricingConfig, SimulationConfig, TimeWindow, TouchMode,
    VolatilityConfig,
};

const CALIBRATION_WINDOWS: usize = 6;
const CALIBRATION_SIGMA_BUCKETS: usize = 3;

/// 7-day pooled (2026-11-17..23) calibrated distance factors for NVDA 3c windows 6..24.
/// Format: `(window_index, [dist0, dist1, dist2, dist3, dist4, dist5, dist6])`.
const NVDA_FAR_FACTORS: [(usize, [f64; 7]); 18] = [
    (6, [0.915500, 0.787886, 0.674303, 0.676690, 0.698808, 0.708506, 0.716205]),
    (7, [0.956309, 0.838050, 0.714763, 0.720958, 0.746935, 0.754675, 0.761753]),
    (8, [0.920529, 0.833251, 0.698605, 0.684425, 0.698700, 0.725891, 0.713776]),
    (9, [0.975982, 0.849490, 0.702724, 0.679485, 0.696976, 0.728635, 0.722569]),
    (10, [0.997099, 0.884279, 0.731311, 0.703358, 0.735471, 0.746692, 0.773285]),
    (11, [1.046875, 0.893161, 0.718590, 0.678903, 0.699242, 0.740150, 0.748076]),
    (12, [0.990748, 0.868789, 0.704594, 0.674567, 0.682855, 0.712492, 0.727833]),
    (13, [0.975552, 0.887015, 0.746451, 0.708756, 0.720100, 0.723786, 0.757471]),
    (14, [0.921093, 0.861437, 0.747703, 0.701401, 0.697334, 0.730425, 0.723374]),
    (15, [0.962366, 0.857728, 0.736418, 0.683781, 0.694551, 0.709820, 0.745036]),
    (16, [0.988912, 0.872927, 0.749080, 0.688375, 0.716163, 0.748739, 0.750772]),
    (17, [0.953921, 0.883206, 0.766248, 0.709945, 0.714758, 0.755511, 0.763527]),
    (18, [0.918614, 0.850272, 0.747562, 0.676167, 0.674508, 0.696607, 0.747554]),
    (19, [0.917869, 0.860163, 0.772573, 0.714645, 0.703067, 0.734216, 0.785153]),
    (20, [0.901802, 0.817703, 0.745006, 0.692211, 0.687672, 0.721056, 0.745266]),
    (21, [0.896949, 0.817724, 0.750544, 0.713878, 0.707060, 0.730446, 0.752610]),
    (22, [0.961336, 0.861987, 0.786015, 0.729680, 0.711046, 0.754437, 0.772309]),
    (23, [0.964201, 0.867056, 0.790175, 0.712352, 0.700964, 0.711032, 0.737530]),
];

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

fn set_far_distance_factors(cfg: &mut GridConfig, table: &[(usize, [f64; 7])]) {
    for &(window_index, factors) in table {
        for (dist_bucket, &factor) in factors.iter().enumerate() {
            for sigma_bucket in 0..CALIBRATION_SIGMA_BUCKETS {
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

/// Builds a ticker-specific `GridConfig` for US equities (`NVDA`, `TSLA`, `AAPL`, `SP500`).
pub fn stock_grid_config_for_ticker(
    ticker: &str,
    band_cents: u32,
    windows: Vec<TimeWindow>,
) -> GridConfig {
    let mut cfg = stock_grid_config(band_cents, windows);
    let symbol = ticker.split(':').next_back().unwrap_or(ticker);
    if symbol.eq_ignore_ascii_case("TSLA") {
        cfg.volatility.sigma_ref = 0.00011;
        cfg.volatility.sigma_scale = 1.073;
        cfg.volatility.variance_floor = 2.08e-9;
        cfg.pricing.m_base = 0.012;
        cfg.pricing.current_row_logit_shift = 0.0;
        set_distance_factors(&mut cfg, &[(1, 0.71), (2, 0.49), (3, 0.51), (4, 0.94)]);
    } else if symbol.eq_ignore_ascii_case("NVDA") {
        cfg.volatility.sigma_ref = 0.00023;
        cfg.volatility.sigma_scale = 1.035;
        cfg.volatility.variance_floor = 1.8e-9;
        cfg.pricing.m_base = 0.015;
        cfg.pricing.off_hours_m_base = Some(0.094);
        cfg.pricing.current_row_logit_shift = 0.0;
        set_distance_factors(&mut cfg, &[(1, 0.72), (2, 0.545), (3, 0.505), (4, 0.96)]);
        set_far_distance_factors(&mut cfg, &NVDA_FAR_FACTORS);
    } else if symbol.eq_ignore_ascii_case("AAPL") {
        // AAPL/SP500 priors derive from HL xyz 1m candles scaled to existing xyz priors; other parameters copy TSLA and are not calibrated.
        cfg.volatility.sigma_ref = 0.0000933;
        cfg.volatility.sigma_scale = 1.073;
        cfg.volatility.variance_floor = 1.50e-9;
        cfg.pricing.m_base = 0.012;
        cfg.pricing.current_row_logit_shift = 0.0;
        set_distance_factors(&mut cfg, &[(1, 0.71), (2, 0.49), (3, 0.51), (4, 0.94)]);
    } else if symbol.eq_ignore_ascii_case("SP500") {
        cfg.volatility.sigma_ref = 0.0000655;
        cfg.volatility.sigma_scale = 1.073;
        cfg.volatility.variance_floor = 7.37e-10;
        cfg.pricing.m_base = 0.012;
        cfg.pricing.current_row_logit_shift = 0.0;
        set_distance_factors(&mut cfg, &[(1, 0.71), (2, 0.49), (3, 0.51), (4, 0.94)]);
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
        let tsla = stock_grid_config_for_ticker("xyz:TSLA", 8, windows.clone());
        let aapl = stock_grid_config_for_ticker("xyz:AAPL", 8, windows.clone());
        let sp500 = stock_grid_config_for_ticker("SP500", 150, windows);
        assert_eq!(nvda.volatility.sigma_ref, 0.00023);
        assert_eq!(nvda.volatility.sigma_scale, 1.035);
        assert_eq!(nvda.volatility.variance_floor, 1.8e-9);
        assert!(!nvda.volatility.jump.enabled);
        assert_eq!(nvda.simulation.bridge_epsilon, 1e-12);
        assert_eq!(nvda.pricing.m_base, 0.015);
        assert_eq!(nvda.pricing.off_hours_m_base, Some(0.094));
        assert_eq!(nvda.pricing.p_raw_gamma, 0.0);
        assert_eq!(nvda.pricing.current_row_logit_shift, 0.0);
        assert_eq!(
            nvda.calibration.factors.get(&CalibrationKey {
                window_index: 0,
                dist_bucket: 2,
                sigma_bucket: 1,
            }),
            Some(&0.545)
        );
        assert_eq!(
            nvda.calibration.factors.get(&CalibrationKey {
                window_index: 6,
                dist_bucket: 0,
                sigma_bucket: 1,
            }),
            Some(&0.915500)
        );
        assert_eq!(
            nvda.calibration.factors.get(&CalibrationKey {
                window_index: 23,
                dist_bucket: 6,
                sigma_bucket: 2,
            }),
            Some(&0.737530)
        );

        assert_eq!(tsla.volatility.sigma_ref, 0.00011);
        assert_eq!(tsla.volatility.sigma_scale, 1.073);
        assert_eq!(tsla.volatility.variance_floor, 2.08e-9);
        assert!(!tsla.volatility.jump.enabled);
        assert_eq!(tsla.simulation.bridge_epsilon, 1e-12);
        assert_eq!(tsla.pricing.m_base, 0.012);
        assert_eq!(tsla.pricing.off_hours_m_base, None);
        assert_eq!(tsla.pricing.p_raw_gamma, 0.0);
        assert_eq!(tsla.pricing.current_row_logit_shift, 0.0);
        assert_eq!(
            tsla.calibration.factors.get(&CalibrationKey {
                window_index: 0,
                dist_bucket: 2,
                sigma_bucket: 1,
            }),
            Some(&0.49)
        );

        assert_eq!(aapl.volatility.sigma_ref, 0.0000933);
        assert_eq!(aapl.volatility.sigma_scale, 1.073);
        assert_eq!(aapl.volatility.variance_floor, 1.50e-9);
        assert_eq!(aapl.pricing.m_base, 0.012);
        assert_eq!(aapl.pricing.off_hours_m_base, None);
        assert_eq!(
            aapl.calibration.factors.get(&CalibrationKey {
                window_index: 0,
                dist_bucket: 2,
                sigma_bucket: 1,
            }),
            Some(&0.49)
        );

        assert_eq!(sp500.volatility.sigma_ref, 0.0000655);
        assert_eq!(sp500.volatility.sigma_scale, 1.073);
        assert_eq!(sp500.volatility.variance_floor, 7.37e-10);
        assert_eq!(sp500.pricing.m_base, 0.012);
        assert_eq!(sp500.pricing.off_hours_m_base, None);
        assert_eq!(
            sp500.calibration.factors.get(&CalibrationKey {
                window_index: 0,
                dist_bucket: 2,
                sigma_bucket: 1,
            }),
            Some(&0.49)
        );
    }
}
