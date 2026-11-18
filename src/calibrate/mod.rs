pub mod eval;
pub mod stats;

use crate::grid::GridConfig;
use crate::price::{PriceCache, PriceTick};
use crate::quote::QuoteEngine;

use eval::{VOID_GAP_MS, collect_gaps_over_3s, window_hit_bands, window_is_void};
pub use stats::{
    Bucket, TailBucket, brier_score, implied_house_edge, is_us_regular_session_utc,
    reliability_table, tail_reliability_table,
};

const MAX_COMPLETED_BARS: usize = 600;
const GRID_LOOKBACK_BARS: usize = 180;
const WINDOW_WIDTH_MS: u64 = 5_000;

#[derive(Debug, Clone)]
pub struct CalibrateOptions {
    pub ticker: String,
    pub band_cents: u32,
    pub base_config: GridConfig,
    pub warmup_ms: u64,
    pub restart_gap_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct CalibrationSample {
    pub window_start_ms: u64,
    pub predicted: f64,
    pub hit: bool,
    pub multiplier: f64,
    pub is_warmup: bool,
    pub is_us_session: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CalibrationReport {
    pub samples: Vec<CalibrationSample>,
    pub voided_windows: usize,
    pub evaluated_windows: usize,
    pub gaps_over_3s: Vec<(u64, u64)>,
}

pub fn run_calibration(ticks: &[PriceTick], opts: &CalibrateOptions) -> CalibrationReport {
    let mut report = CalibrationReport::default();
    if ticks.len() < 2 {
        return report;
    }
    report.gaps_over_3s = collect_gaps_over_3s(ticks);

    let first_ts = ticks[0].ts_ms;
    let last_ts = ticks[ticks.len() - 1].ts_ms;
    let mut window_start_ms = first_ts.saturating_add(WINDOW_WIDTH_MS).div_ceil(WINDOW_WIDTH_MS)
        * WINDOW_WIDTH_MS;

    let mut state = EngineReplayState::new(opts, first_ts);
    let mut tick_idx = 0usize;

    while window_start_ms.saturating_add(WINDOW_WIDTH_MS) <= last_ts {
        let quote_ts_ms = window_start_ms - WINDOW_WIDTH_MS;
        tick_idx = state.advance_to(ticks, tick_idx, quote_ts_ms, opts);
        evaluate_window(
            &mut state,
            ticks,
            tick_idx,
            quote_ts_ms,
            window_start_ms,
            opts,
            &mut report,
        );
        window_start_ms = window_start_ms.saturating_add(WINDOW_WIDTH_MS);
    }

    report
}

struct EngineReplayState {
    cache: PriceCache,
    engine: QuoteEngine,
    feed_start_ms: u64,
    last_tick: Option<PriceTick>,
    last_quoted_tick_ts: Option<u64>,
}

impl EngineReplayState {
    fn new(opts: &CalibrateOptions, first_ts: u64) -> Self {
        Self {
            cache: PriceCache::new(MAX_COMPLETED_BARS),
            engine: QuoteEngine::with_config(
                opts.ticker.clone(),
                opts.band_cents,
                opts.base_config.clone(),
            ),
            feed_start_ms: first_ts,
            last_tick: None,
            last_quoted_tick_ts: None,
        }
    }

    fn advance_to(
        &mut self,
        ticks: &[PriceTick],
        mut tick_idx: usize,
        quote_ts_ms: u64,
        opts: &CalibrateOptions,
    ) -> usize {
        while tick_idx < ticks.len() && ticks[tick_idx].ts_ms <= quote_ts_ms {
            let tick = ticks[tick_idx];
            if let Some(prev) = self.last_tick {
                if tick.ts_ms.saturating_sub(prev.ts_ms) > opts.restart_gap_ms {
                    self.cache = PriceCache::new(MAX_COMPLETED_BARS);
                    self.engine = QuoteEngine::with_config(
                        opts.ticker.clone(),
                        opts.band_cents,
                        opts.base_config.clone(),
                    );
                    self.feed_start_ms = tick.ts_ms;
                    self.last_quoted_tick_ts = None;
                }
            }
            let _ = self.cache.apply_tick(tick);
            self.last_tick = Some(tick);
            tick_idx += 1;
        }
        tick_idx
    }
}

fn evaluate_window(
    state: &mut EngineReplayState,
    ticks: &[PriceTick],
    tick_idx: usize,
    quote_ts_ms: u64,
    window_start_ms: u64,
    opts: &CalibrateOptions,
    report: &mut CalibrationReport,
) {
    let Some(tick) = state.last_tick else {
        return;
    };
    if quote_ts_ms.saturating_sub(tick.ts_ms) > VOID_GAP_MS {
        return;
    }

    let window_end_ms = window_start_ms + WINDOW_WIDTH_MS;
    if window_is_void(ticks, tick_idx, window_start_ms, window_end_ms) {
        report.voided_windows += 1;
        return;
    }

    if state.last_quoted_tick_ts != Some(tick.ts_ms) {
        let Some(grid_snap) = state.cache.snapshot().bars_for_grid(GRID_LOOKBACK_BARS) else {
            return;
        };
        state.engine.on_sample(tick, &grid_snap.history);
        state.last_quoted_tick_ts = Some(tick.ts_ms);
    }

    let quote = state.engine.price(quote_ts_ms);
    let window_cells: Vec<_> = quote
        .cells
        .into_iter()
        .filter(|cell| cell.window_start_ms == window_start_ms)
        .collect();
    if window_cells.is_empty() {
        return;
    }

    let bands_cents: Vec<(i64, i64)> = window_cells
        .iter()
        .map(|cell| (cell.lower_cents, cell.upper_cents))
        .collect();
    let hits = window_hit_bands(ticks, tick_idx, window_start_ms, window_end_ms, &bands_cents);
    let is_warmup = window_start_ms.saturating_sub(state.feed_start_ms) < opts.warmup_ms;
    let is_us_session = is_us_regular_session_utc(window_start_ms);

    report.evaluated_windows += 1;
    for (cell, hit) in window_cells.into_iter().zip(hits) {
        report.samples.push(CalibrationSample {
            window_start_ms,
            predicted: cell.hit_probability,
            hit,
            multiplier: cell.multiplier,
            is_warmup,
            is_us_session,
        });
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::StandardNormal;

    use super::eval::{VOID_GAP_MS, window_hit_bands, window_is_void};
    use super::*;
    use crate::grid::stock_grid_config;

    #[test]
    fn reliability_buckets_cover_all_samples() {
        let samples: Vec<(f64, bool)> = (0..=100)
            .map(|i| (i as f64 / 100.0, i % 2 == 0))
            .collect();
        let table = reliability_table(&samples, 10);
        assert_eq!(table.len(), 10);
        let total: usize = table.iter().map(|b| b.count).sum();
        assert_eq!(total, samples.len());
        assert!((table[0].lo - 0.0).abs() < 1e-12);
        assert!((table[9].hi - 1.0).abs() < 1e-12);
    }

    #[test]
    fn segment_hit_catches_crossing_and_voids_gap_over_3000ms() {
        let ticks = vec![
            PriceTick { ts_ms: 9_500, price: 157.40 },
            PriceTick { ts_ms: 10_500, price: 157.70 },
            PriceTick { ts_ms: 12_500, price: 157.75 },
            PriceTick { ts_ms: 14_500, price: 157.80 },
            PriceTick { ts_ms: 15_100, price: 157.80 },
            PriceTick { ts_ms: 18_600, price: 157.90 }, // gap 3500ms > 3000ms in [15000, 20000)
        ];
        assert!(!window_is_void(&ticks, 0, 10_000, 15_000));
        let hits = window_hit_bands(&ticks, 0, 10_000, 15_000, &[(15752, 15755), (15785, 15790)]);
        assert_eq!(hits, vec![true, false]);
        assert!(window_is_void(&ticks, 0, 15_000, 20_000));
        assert_eq!(VOID_GAP_MS, 3_000);
    }

    #[test]
    fn gbm_ticks_with_known_sigma_calibrate_within_5_points() {
        let sigma_1s = 0.00025_f64;
        let mut rng = StdRng::seed_from_u64(42);
        let mut ticks = Vec::with_capacity(4_000);
        let mut price = 157.50_f64;
        let base_ts = 1_794_936_000_000_u64;
        for sec in 0..4_000_u64 {
            let z: f64 = rng.sample(StandardNormal);
            price *= (sigma_1s * z).exp();
            ticks.push(PriceTick {
                ts_ms: base_ts + sec * 1_000,
                price,
            });
        }

        let mut cfg = stock_grid_config(3, Vec::new());
        cfg.volatility.sigma_scale = 0.88;
        cfg.volatility.sigma_ref = sigma_1s;
        cfg.volatility.variance_floor = 1e-10;
        cfg.volatility.jump.enabled = false;
        cfg.simulation.bridge_epsilon = 1e-12;
        cfg.pricing.p_raw_gamma = 0.0;
        cfg.pricing.p_near_boost = 0.0;
        cfg.pricing.p_time_boost = 0.0;
        cfg.pricing.current_row_logit_shift = -0.35;
        cfg.pricing.adjacent_row_logit_shift = -0.25;
        cfg.pricing.zone_vol_shift_scale = 0.0;
        cfg.simulation.seed = Some(7);

        let report = run_calibration(
            &ticks,
            &CalibrateOptions {
                ticker: "NVDA".to_string(),
                band_cents: 3,
                base_config: cfg,
                warmup_ms: 300_000,
                restart_gap_ms: 30_000,
            },
        );

        let steady: Vec<(f64, bool)> = report
            .samples
            .iter()
            .filter(|s| !s.is_warmup)
            .map(|s| (s.predicted, s.hit))
            .collect();
        assert!(!steady.is_empty());
        assert!(report.samples.iter().any(|s| s.is_warmup));
        assert!(report.samples.iter().all(|s| s.is_us_session));

        let buckets = reliability_table(&steady, 10);
        let large_buckets: Vec<&Bucket> = buckets.iter().filter(|b| b.count >= 200).collect();
        assert!(!large_buckets.is_empty());
        for bucket in large_buckets {
            let diff = (bucket.mean_predicted - bucket.hit_rate).abs();
            assert!(
                diff < 0.05,
                "bucket [{:.1}, {:.1}) count={} pred={:.4} hit={:.4} diff={:.4}",
                bucket.lo,
                bucket.hi,
                bucket.count,
                bucket.mean_predicted,
                bucket.hit_rate,
                diff
            );
        }
    }
}
