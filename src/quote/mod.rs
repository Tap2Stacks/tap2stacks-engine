mod book;

use serde::{Deserialize, Serialize};

use crate::grid::{
    GridConfig, OhlcBar, PriceBand, TimeWindow, VolatilitySnapshot, compute_k_safety,
    compute_quote_probability, estimate_runtime_state, price_multiplier, simulate_summary,
    stock_grid_config,
};
use crate::price::PriceTick;

pub use book::{CellKey, ExposureBook};

const WINDOW_COUNT: usize = 6;
const WINDOW_WIDTH_MS: u64 = 5_000;
const WINDOW_WIDTH_SECS: u32 = 5;
const MIN_LEAD_MS: u64 = 5_000;

/// Quote snapshot for a single ticker produced from cached probabilities and live exposure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StockQuote {
    pub ticker: String,
    pub server_ts_ms: u64,
    pub spot: f64,
    pub band_cents: u32,
    pub cells: Vec<StockCell>,
}

/// Single quoted cell within a `StockQuote`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    config: GridConfig,
    volatility: Option<VolatilitySnapshot>,
    cached_cells: Vec<CachedCell>,
    cached_scenario_payouts: Vec<f64>,
    cached_k_safety: Option<CachedSafety>,
    exposure: ExposureBook,
}

impl QuoteEngine {
    /// Creates a new `QuoteEngine` for `ticker` with `band_cents`-wide price bands.
    pub fn new(ticker: String, band_cents: u32) -> Self {
        Self {
            ticker,
            band_cents,
            spot: 0.0,
            config: stock_grid_config(band_cents, Vec::new()),
            volatility: None,
            cached_cells: Vec::new(),
            cached_scenario_payouts: Vec::new(),
            cached_k_safety: None,
            exposure: ExposureBook::default(),
        }
    }

    /// Runs Monte Carlo for a new price sample and caches per-cell probabilities and scenario payouts.
    pub fn on_sample(&mut self, tick: PriceTick, history: &[OhlcBar]) {
        if !tick.price.is_finite() || tick.price <= 0.0 || self.band_cents == 0 {
            return;
        }

        let windows_ms = aligned_windows(tick.ts_ms, WINDOW_COUNT, WINDOW_WIDTH_MS, MIN_LEAD_MS);
        let floor_sec_ms = (tick.ts_ms / 1_000) * 1_000;
        let relative_windows: Vec<TimeWindow> = windows_ms
            .iter()
            .map(|&(start_ms, _)| {
                let start_secs = ((start_ms - floor_sec_ms) / 1_000) as u32;
                TimeWindow {
                    start_secs,
                    end_secs: start_secs + WINDOW_WIDTH_SECS,
                }
            })
            .collect();

        let config = stock_grid_config(self.band_cents, relative_windows);
        if config.validate().is_err() {
            return;
        }

        let Ok(runtime_state) = estimate_runtime_state(tick.price, history, &config) else {
            return;
        };

        let bands = build_cent_bands(
            tick.price,
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

        let window_count = windows_ms.len();
        let mut liabilities_vector = vec![0.0; bands.len() * window_count];
        for (row_index, band) in bands.iter().enumerate() {
            let lower_cents = (band.lower * 100.0).round() as i64;
            let row_offset = row_index * window_count;
            for (window_index, &(window_start_ms, _)) in windows_ms.iter().enumerate() {
                let key = CellKey {
                    window_start_ms,
                    lower_cents,
                };
                liabilities_vector[row_offset + window_index] = self.exposure.get(&key);
            }
        }

        let k_safety_fresh = self.cached_k_safety.is_some_and(|cached| {
            tick.ts_ms >= cached.computed_at_ms
                && tick.ts_ms - cached.computed_at_ms < config.pricing.k_safety_ttl_ms
        });

        let simulation = simulate_summary(
            tick.price,
            &bands,
            &config,
            &runtime_state,
            max_horizon,
            if k_safety_fresh {
                None
            } else {
                Some(&liabilities_vector)
            },
        );

        if !k_safety_fresh {
            self.cached_scenario_payouts = simulation.scenario_payouts;
            let value = compute_k_safety(&self.cached_scenario_payouts, &config);
            self.cached_k_safety = Some(CachedSafety {
                value,
                computed_at_ms: tick.ts_ms,
            });
        }

        let current_window0_row_index = bands
            .iter()
            .min_by(|left, right| {
                let left_center = (left.lower + left.upper) / 2.0;
                let right_center = (right.lower + right.upper) / 2.0;
                (left_center - tick.price)
                    .abs()
                    .partial_cmp(&(right_center - tick.price).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|band| band.row_index)
            .unwrap_or(config.spec.center_row);

        let volatility = runtime_state.volatility;
        let mut cached_cells = Vec::with_capacity(bands.len() * window_count);
        for (row_index, band) in bands.iter().enumerate() {
            let lower_cents = (band.lower * 100.0).round() as i64;
            let upper_cents = (band.upper * 100.0).round() as i64;
            for (window_index, window) in config.spec.windows.iter().enumerate() {
                let (window_start_ms, window_end_ms) = windows_ms[window_index];
                let raw_hit_probability = simulation.mean_probabilities[row_index][window_index];
                let hit_probability = compute_quote_probability(
                    raw_hit_probability,
                    row_index,
                    window_index,
                    *window,
                    volatility,
                    current_window0_row_index,
                    &config,
                );
                cached_cells.push(CachedCell {
                    key: CellKey {
                        window_start_ms,
                        lower_cents,
                    },
                    window_end_ms,
                    upper_cents,
                    hit_probability,
                });
            }
        }

        self.spot = tick.price;
        self.config = config;
        self.volatility = Some(volatility);
        self.cached_cells = cached_cells;
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
            let multiplier = price_multiplier(
                cell.hit_probability,
                volatility,
                liability,
                k_safety,
                &self.config,
            );
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
        engine.on_sample(PriceTick { ts_ms: ts, price: spot }, &sample_history(spot));

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
        engine.on_sample(PriceTick { ts_ms: ts, price: spot }, &sample_history(spot));

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
        engine.on_sample(PriceTick { ts_ms: ts, price: spot }, &sample_history(spot));

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
        engine.on_sample(PriceTick { ts_ms: ts, price: spot }, &sample_history(spot));

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
        engine.on_sample(PriceTick { ts_ms: ts, price: spot }, &sample_history(spot));

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
        engine.on_sample(PriceTick { ts_ms: ts, price: spot }, &sample_history(spot));

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
        engine.on_sample(PriceTick { ts_ms: ts, price: spot }, &sample_history(spot));

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
}
