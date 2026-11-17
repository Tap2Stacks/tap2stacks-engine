use super::types::{
    GridConfig, GridSpec, SimulationConfig, TimeWindow, TouchMode, VolatilityConfig,
};

/// Builds a `GridConfig` configured for US equity 1 Hz discrete-sample quoting.
pub fn stock_grid_config(band_cents: u32, windows: Vec<TimeWindow>) -> GridConfig {
    let default = GridConfig::default();
    GridConfig {
        spec: GridSpec {
            rows: 20,
            center_row: 9,
            band_width: band_cents as f64 / 100.0,
            windows,
        },
        simulation: SimulationConfig {
            touch_mode: TouchMode::Discrete,
            lock_offset: 0,
            ..default.simulation
        },
        volatility: VolatilityConfig {
            parkinson_weight: 0.0,
            price_adj_min: 1.0,
            price_adj_max: 1.0,
            ..default.volatility
        },
        ..default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::state::estimate_volatility;
    use crate::grid::types::OhlcBar;

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
        assert_eq!(c.simulation.touch_mode, TouchMode::Discrete);

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
}
