pub mod eval;
pub mod stats;

use std::collections::BTreeMap;

use crate::grid::GridConfig;
use crate::horizon::{RepriceStats, row_dist_bucket};
use crate::price::{PriceCache, PriceTick};
use crate::quote::{QuoteEngine, StockCell};
use crate::v2::{Regime, RegimeDetector, TickerV2Params, v2_ticker_params};

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
    pub all_windows: bool,
    pub quote_step_ms: u64,
    pub reprice_every_tick: bool,
    pub reprice_from_ms: Option<u64>,
    pub reprice_to_ms: Option<u64>,
    pub session_starts_ms: Option<Vec<u64>>,
    pub session_burst_ms: u64,
    pub include_minute_quotes: bool,
    pub skip_eval: bool,
}

impl CalibrateOptions {
    pub fn standard(ticker: String, band_cents: u32, base_config: GridConfig) -> Self {
        Self {
            ticker,
            band_cents,
            base_config,
            warmup_ms: 300_000,
            restart_gap_ms: 30_000,
            all_windows: false,
            quote_step_ms: WINDOW_WIDTH_MS,
            reprice_every_tick: false,
            reprice_from_ms: None,
            reprice_to_ms: None,
            session_starts_ms: None,
            session_burst_ms: 30_000,
            include_minute_quotes: false,
            skip_eval: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CalibrationSample {
    pub quote_ts_ms: u64,
    pub window_start_ms: u64,
    pub window_index: usize,
    pub dist_bucket: usize,
    pub predicted: f64,
    pub hit: bool,
    pub multiplier: f64,
    pub center_distance: usize,
    pub signed_distance: i64,
    pub is_warmup: bool,
    pub is_us_session: bool,
    pub spot: f64,
    pub sigma: f64,
    pub sigma_raw: f64,
    pub regime: Option<Regime>,
}

#[derive(Debug, Clone, Default)]
pub struct CalibrationReport {
    pub samples: Vec<CalibrationSample>,
    pub voided_windows: usize,
    pub evaluated_windows: usize,
    pub gaps_over_3s: Vec<(u64, u64)>,
    pub voided_window_starts: Vec<u64>,
    pub stale_quote_skips: usize,
    pub reprice_stats: RepriceStats,
}

pub fn run_calibration(ticks: &[PriceTick], opts: &CalibrateOptions) -> CalibrationReport {
    let mut report = CalibrationReport::default();
    if ticks.len() < 2 {
        return report;
    }
    report.gaps_over_3s = collect_gaps_over_3s(ticks);

    let first_ts = ticks[0].ts_ms;
    let last_ts = ticks[ticks.len() - 1].ts_ms;
    let mut window_start_ms =
        first_ts.saturating_add(WINDOW_WIDTH_MS).div_ceil(WINDOW_WIDTH_MS) * WINDOW_WIDTH_MS;
    let quote_step_ms = opts.quote_step_ms.max(WINDOW_WIDTH_MS);

    let mut state = EngineReplayState::new(opts, first_ts);
    let mut tick_idx = 0usize;
    let mut session_cursor = 0usize;
    let mut next_full_step_ms = window_start_ms;

    while window_start_ms.saturating_add(WINDOW_WIDTH_MS) <= last_ts {
        let quote_ts_ms = window_start_ms - WINDOW_WIDTH_MS;
        tick_idx = state.advance_to(ticks, tick_idx, quote_ts_ms, opts, &mut report);

        let window_end_ms = window_start_ms + WINDOW_WIDTH_MS;
        let stale = state
            .last_tick
            .is_some_and(|tick| quote_ts_ms.saturating_sub(tick.ts_ms) > VOID_GAP_MS);
        let window0_void = !stale
            && state.last_tick.is_some()
            && window_is_void(ticks, tick_idx, window_start_ms, window_end_ms);

        if stale {
            report.stale_quote_skips += 1;
            window_start_ms = window_start_ms.saturating_add(WINDOW_WIDTH_MS);
            continue;
        }
        if window0_void {
            report.voided_windows += 1;
            report.voided_window_starts.push(window_start_ms);
            if !opts.all_windows {
                window_start_ms = window_start_ms.saturating_add(WINDOW_WIDTH_MS);
                continue;
            }
        }

        if opts.skip_eval {
            window_start_ms = window_start_ms.saturating_add(WINDOW_WIDTH_MS);
            continue;
        }

        let in_reprice_range = opts.reprice_every_tick
            && opts.reprice_from_ms.is_none_or(|from| quote_ts_ms >= from)
            && opts.reprice_to_ms.is_none_or(|to| quote_ts_ms < to);
        let on_minute = opts.include_minute_quotes && quote_ts_ms % 60_000 == 0;
        let should_quote = if let Some(starts) = opts.session_starts_ms.as_deref() {
            while session_cursor < starts.len()
                && starts[session_cursor].saturating_add(opts.session_burst_ms) <= quote_ts_ms
            {
                session_cursor += 1;
            }
            let needed_for_session = starts.get(session_cursor).is_some_and(|&s| {
                quote_ts_ms >= s && quote_ts_ms < s.saturating_add(opts.session_burst_ms)
            });
            needed_for_session || in_reprice_range || on_minute
        } else if window_start_ms >= next_full_step_ms {
            next_full_step_ms = window_start_ms.saturating_add(quote_step_ms);
            true
        } else {
            in_reprice_range
        };

        if should_quote {
            let produced = evaluate_window(
                &mut state,
                ticks,
                tick_idx,
                quote_ts_ms,
                window_start_ms,
                last_ts,
                window0_void,
                opts,
                &mut report,
            );
            if produced {
                if let Some(starts) = opts.session_starts_ms.as_deref() {
                    while session_cursor < starts.len() && starts[session_cursor] <= quote_ts_ms {
                        session_cursor += 1;
                    }
                }
            }
        }

        window_start_ms = window_start_ms.saturating_add(WINDOW_WIDTH_MS);
    }

    // Drain any remaining ticks for 1 Hz repricing (e.g. bench_only tail).
    if opts.reprice_every_tick && tick_idx < ticks.len() {
        let _ = state.advance_to(ticks, tick_idx, last_ts, opts, &mut report);
    }
    report.reprice_stats.merge(&state.engine.reprice_stats());
    report
}

struct EngineReplayState {
    cache: PriceCache,
    engine: QuoteEngine,
    causal_detector: Option<(RegimeDetector, &'static TickerV2Params)>,
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
            causal_detector: v2_ticker_params(&opts.ticker)
                .map(|params| (RegimeDetector::new(), params)),
            feed_start_ms: first_ts,
            last_tick: None,
            last_quoted_tick_ts: None,
        }
    }

    fn advance_to(
        &mut self,
        ticks: &[PriceTick],
        mut tick_idx: usize,
        until_ts_ms: u64,
        opts: &CalibrateOptions,
        report: &mut CalibrationReport,
    ) -> usize {
        while tick_idx < ticks.len() && ticks[tick_idx].ts_ms <= until_ts_ms {
            let tick = ticks[tick_idx];
            if let Some(prev) = self.last_tick {
                if tick.ts_ms.saturating_sub(prev.ts_ms) > opts.restart_gap_ms {
                    report.reprice_stats.merge(&self.engine.reprice_stats());
                    self.cache = PriceCache::new(MAX_COMPLETED_BARS);
                    self.engine = QuoteEngine::with_config(
                        opts.ticker.clone(),
                        opts.band_cents,
                        opts.base_config.clone(),
                    );
                    self.causal_detector = v2_ticker_params(&opts.ticker)
                        .map(|params| (RegimeDetector::new(), params));
                    self.feed_start_ms = tick.ts_ms;
                    self.last_quoted_tick_ts = None;
                }
            }
            let _ = self.cache.apply_tick(tick);
            self.engine.observe_tick(tick);
            if let Some((ref mut detector, params)) = self.causal_detector {
                detector.on_tick(tick, params);
            }
            self.last_tick = Some(tick);
            if opts.reprice_every_tick
                && opts.reprice_from_ms.is_none_or(|from| tick.ts_ms >= from)
                && opts.reprice_to_ms.is_none_or(|to| tick.ts_ms < to)
            {
                if let Some(grid_snap) = self.cache.snapshot().bars_for_grid(GRID_LOOKBACK_BARS) {
                    self.engine.on_sample(tick, &grid_snap.history);
                    self.last_quoted_tick_ts = Some(tick.ts_ms);
                }
            }
            tick_idx += 1;
        }
        tick_idx
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_window(
    state: &mut EngineReplayState,
    ticks: &[PriceTick],
    tick_idx: usize,
    quote_ts_ms: u64,
    window_start_ms: u64,
    last_ts: u64,
    window0_void: bool,
    opts: &CalibrateOptions,
    report: &mut CalibrationReport,
) -> bool {
    let Some(tick) = state.last_tick else {
        return false;
    };

    if state.last_quoted_tick_ts != Some(tick.ts_ms) {
        let Some(grid_snap) = state.cache.snapshot().bars_for_grid(GRID_LOOKBACK_BARS) else {
            return false;
        };
        state.engine.on_sample(tick, &grid_snap.history);
        state.last_quoted_tick_ts = Some(tick.ts_ms);
    }

    let quote = state.engine.price(quote_ts_ms);
    let volatility = state.engine.volatility();
    let sigma = volatility.map(|snap| snap.sigma).unwrap_or(0.0);
    let sigma_raw = volatility.map(|snap| snap.sigma_raw).unwrap_or(0.0);
    let regime = state
        .causal_detector
        .as_ref()
        .map(|(detector, _)| detector.current_regime());
    let spot_cents = (quote.spot * 100.0).round() as i64;
    let dist_max = opts.base_config.pricing.calibration_dist_max;

    if !opts.all_windows {
        let window_cells: Vec<StockCell> = quote
            .cells
            .into_iter()
            .filter(|cell| cell.window_start_ms == window_start_ms)
            .collect();
        if window_cells.is_empty() {
            return false;
        }
        let window_end_ms = window_start_ms + WINDOW_WIDTH_MS;
        let bands_cents: Vec<(i64, i64)> = window_cells
            .iter()
            .map(|cell| (cell.lower_cents, cell.upper_cents))
            .collect();
        let hits = window_hit_bands(ticks, tick_idx, window_start_ms, window_end_ms, &bands_cents);
        let is_warmup = window_start_ms.saturating_sub(state.feed_start_ms) < opts.warmup_ms;
        let is_us_session = crate::session::is_us_regular_session(window_start_ms);

        report.evaluated_windows += 1;
        for (cell, hit) in window_cells.into_iter().zip(hits) {
            let signed_distance =
                (cell.lower_cents - spot_cents).div_euclid(opts.band_cents as i64);
            let center_distance = cell
                .lower_cents
                .abs_diff(spot_cents)
                .div_ceil(opts.band_cents as u64) as usize;
            let dist_bucket = row_dist_bucket(quote.spot, opts.band_cents, cell.lower_cents, dist_max);
            report.samples.push(CalibrationSample {
                quote_ts_ms,
                window_start_ms,
                window_index: 0,
                dist_bucket,
                predicted: cell.hit_probability,
                hit,
                multiplier: cell.multiplier,
                center_distance,
                signed_distance,
                is_warmup,
                is_us_session,
                spot: quote.spot,
                sigma,
                sigma_raw,
                regime,
            });
        }
        return true;
    }

    let mut by_window: BTreeMap<u64, Vec<StockCell>> = BTreeMap::new();
    for cell in quote.cells {
        if cell.window_start_ms >= window_start_ms
            && cell.window_start_ms.saturating_add(WINDOW_WIDTH_MS) <= last_ts
        {
            by_window.entry(cell.window_start_ms).or_default().push(cell);
        }
    }
    if by_window.is_empty() {
        return false;
    }

    let is_warmup = quote_ts_ms.saturating_sub(state.feed_start_ms) < opts.warmup_ms;
    let mut evaluated_any = false;
    for (w_start, window_cells) in by_window {
        let w_end = w_start + WINDOW_WIDTH_MS;
        let window_index = ((w_start - window_start_ms) / WINDOW_WIDTH_MS) as usize;
        if window_index == 0 && window0_void {
            continue;
        }
        if window_index > 0 && window_is_void(ticks, tick_idx, w_start, w_end) {
            continue;
        }
        let bands_cents: Vec<(i64, i64)> = window_cells
            .iter()
            .map(|cell| (cell.lower_cents, cell.upper_cents))
            .collect();
        let hits = window_hit_bands(ticks, tick_idx, w_start, w_end, &bands_cents);
        let is_us_session = crate::session::is_us_regular_session(w_start);
        if window_index == 0 {
            report.evaluated_windows += 1;
        }
        evaluated_any = true;
        for (cell, hit) in window_cells.into_iter().zip(hits) {
            let signed_distance =
                (cell.lower_cents - spot_cents).div_euclid(opts.band_cents as i64);
            let center_distance = cell
                .lower_cents
                .abs_diff(spot_cents)
                .div_ceil(opts.band_cents as u64) as usize;
            let dist_bucket = row_dist_bucket(quote.spot, opts.band_cents, cell.lower_cents, dist_max);
            report.samples.push(CalibrationSample {
                quote_ts_ms,
                window_start_ms: w_start,
                window_index,
                dist_bucket,
                predicted: cell.hit_probability,
                hit,
                multiplier: cell.multiplier,
                center_distance,
                signed_distance,
                is_warmup,
                is_us_session,
                spot: quote.spot,
                sigma,
                sigma_raw,
                regime,
            });
        }
    }
    evaluated_any
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

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
            &CalibrateOptions::standard("NVDA".to_string(), 3, cfg),
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
        assert!(
            report
                .samples
                .iter()
                .any(|s| s.sigma.is_finite() && s.sigma > 0.0 && s.spot > 0.0)
        );

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

    #[test]
    fn calibration_records_sigma_and_void_windows() {
        let mut ticks = Vec::new();
        let mut price = 140.0_f64;
        let base_ts = 1_795_392_000_000_u64;
        for sec in 0..400_u64 {
            ticks.push(PriceTick {
                ts_ms: base_ts + sec * 1_000,
                price,
            });
            price += 0.001;
        }
        let gap_start = ticks.last().unwrap().ts_ms;
        ticks.push(PriceTick {
            ts_ms: gap_start + 16_000,
            price: 141.0,
        });
        for sec in 1..30_u64 {
            ticks.push(PriceTick {
                ts_ms: gap_start + 16_000 + sec * 1_000,
                price: 141.0,
            });
        }
        let mut cfg = stock_grid_config(3, Vec::new());
        cfg.simulation.mc_n_min = 20;
        cfg.simulation.mc_n_max = 20;
        cfg.simulation.seed = Some(1);
        let report = run_calibration(
            &ticks,
            &CalibrateOptions::standard("NVDA".to_string(), 3, cfg),
        );
        assert!(report.samples.iter().any(|sample| {
            sample.sigma.is_finite()
                && sample.sigma > 0.0
                && sample.sigma_raw > 0.0
                && sample.spot > 0.0
        }));
        assert!(report.voided_windows > 0);
        assert_eq!(report.voided_window_starts.len(), report.voided_windows);
        assert!(
            report
                .gaps_over_3s
                .iter()
                .any(|(start, end)| end.saturating_sub(*start) >= 16_000)
        );
        assert!(report.stale_quote_skips > 0);
    }

    #[test]
    fn all_windows_calibration_evaluates_all_24_windows_and_matches_lazy_sessions() {
        let base_ts = 1_795_392_000_000_u64;
        let ticks: Vec<PriceTick> = (0..500u64)
            .map(|sec| PriceTick {
                ts_ms: base_ts + sec * 1_000,
                price: 140.0 + ((sec % 11) as f64 - 5.0) * 0.01,
            })
            .collect();
        let mut cfg = stock_grid_config(3, Vec::new());
        cfg.simulation.quote_window_count = 24;
        cfg.simulation.mc_n_min = 20;
        cfg.simulation.mc_n_max = 20;
        cfg.simulation.seed = Some(5);
        cfg.simulation.use_parallel = false;

        let mut opts_full = CalibrateOptions::standard("NVDA".to_string(), 3, cfg.clone());
        opts_full.all_windows = true;
        opts_full.warmup_ms = 60_000;
        let full_report = run_calibration(&ticks, &opts_full);

        let seen_windows: BTreeSet<usize> = full_report
            .samples
            .iter()
            .filter(|s| !s.is_warmup)
            .map(|s| s.window_index)
            .collect();
        assert_eq!(seen_windows.len(), 24);
        assert_eq!(*seen_windows.iter().next().unwrap(), 0);
        assert_eq!(*seen_windows.iter().next_back().unwrap(), 23);

        // Lazy session quoting must produce identical samples at the requested quote timestamp.
        let target_quote = base_ts + 120_000;
        let mut opts_lazy = opts_full.clone();
        opts_lazy.session_starts_ms = Some(vec![target_quote]);
        let lazy_report = run_calibration(&ticks, &opts_lazy);
        let full_at_target: Vec<(usize, f64, bool)> = full_report
            .samples
            .iter()
            .filter(|s| s.quote_ts_ms == target_quote)
            .map(|s| (s.window_index, s.multiplier, s.hit))
            .collect();
        let lazy_at_target: Vec<(usize, f64, bool)> = lazy_report
            .samples
            .iter()
            .filter(|s| s.quote_ts_ms == target_quote)
            .map(|s| (s.window_index, s.multiplier, s.hit))
            .collect();
        assert_eq!(full_at_target.len(), 20 * 24);
        assert_eq!(full_at_target, lazy_at_target);
    }

    #[test]
    fn reprice_every_tick_fires_jump_trigger_on_shock() {
        let base_ts = 1_795_392_001_000_u64;
        let mut ticks = Vec::new();
        for sec in 0..40u64 {
            let price = if sec < 22 {
                140.0
            } else {
                // +3% sudden jump mid-5s-window (base_ts + 22s has same 5s window bucket as +20s/+21s).
                140.0 * 1.03
            };
            ticks.push(PriceTick {
                ts_ms: base_ts + sec * 1_000,
                price,
            });
        }
        let mut cfg = stock_grid_config(3, Vec::new());
        cfg.simulation.quote_window_count = 24;
        cfg.simulation.mc_n_min = 20;
        cfg.simulation.mc_n_max = 20;
        cfg.simulation.seed = Some(7);
        cfg.simulation.use_parallel = false;

        let mut opts = CalibrateOptions::standard("NVDA".to_string(), 3, cfg);
        opts.all_windows = true;
        opts.reprice_every_tick = true;
        opts.skip_eval = true;
        let report = run_calibration(&ticks, &opts);
        assert!(report.reprice_stats.near_reprices > 0);
        assert!(report.reprice_stats.trigger_reprices >= 1);
        assert!(
            report
                .reprice_stats
                .trigger_timestamps_ms
                .contains(&(base_ts + 22_000))
        );
    }
}
