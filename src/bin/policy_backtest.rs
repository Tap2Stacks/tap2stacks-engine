use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use st_engine::calibrate::{CalibrateOptions, CalibrationReport, run_calibration};
use st_engine::grid::{GridConfig, stock_grid_config_for_ticker};
use st_engine::horizon::{
    CAP25_EPS, ExposureInterval, NEAR_WINDOWS, RepriceStats, apply_factor_table,
    compute_outstanding_exposure, parse_factor_table,
};
use st_engine::price::{PriceTick, read_hl_jsonl};
use st_engine::stress::{
    CellTap, POLICIES, QUOTE_LEAD_MS, SHOCK_HORIZONS_MS, SIGMA_LAG_OFFSETS_MS, WINDOW_MS,
    WindowQuote, audit_feed_gaps, bucket_quote, choose_taps, dense_session_starts, format_utc,
    hour_starts, log_move, max_gap_overlapping, merge_starts, minute_starts, move_sign,
    normal_session_starts, parse_period, parse_utc, price_at_or_before, quoted_sigma_at,
    realized_step_sigma, stake_stats, tripwire_k, worst_edge_minute, worst_minute,
};

const STAKE: f64 = 10.0;
const BURST_MS: u64 = 30_000;
const WATCH_MS: u64 = 120_000;
const RETURN_MIN_MS: u64 = 60_000;
const RETURN_MAX_MS: u64 = 300_000;
const TAPS_PER_SESSION: usize = 18;
const WARMUP_MS: u64 = 300_000;
const NEWS_MIN_1356_FROM_MS: u64 = 1_795_442_160_000; // 2026-11-23T13:56:00Z
const NEWS_MIN_1356_TO_MS: u64 = 1_795_442_220_000; // 2026-11-23T13:57:00Z

struct Args {
    files: Vec<PathBuf>,
    coin: String,
    band_cents: u32,
    m_base: Option<f64>,
    a_sigma: Option<f64>,
    ewma_half_life: Option<f64>,
    p_raw_gamma: Option<f64>,
    current_shift: Option<f64>,
    adjacent_shift: Option<f64>,
    sessions: usize,
    seed: u64,
    from_utc: Option<u64>,
    to_utc: Option<u64>,
    dense_from: Option<u64>,
    dense_to: Option<u64>,
    dense_step_ms: u64,
    minute_from: Option<u64>,
    minute_to: Option<u64>,
    shocks: Vec<u64>,
    periods: Vec<st_engine::stress::Period>,
    windows: Option<usize>,
    factors_file: Option<PathBuf>,
    far_refresh_ms: Option<u64>,
    mc_n: usize,
    tick_reprice: bool,
    reprice_from: Option<u64>,
    reprice_to: Option<u64>,
    bench_only: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        files: Vec::new(),
        coin: String::new(),
        band_cents: 0,
        m_base: None,
        a_sigma: None,
        ewma_half_life: None,
        p_raw_gamma: None,
        current_shift: None,
        adjacent_shift: None,
        sessions: 200,
        seed: 42,
        from_utc: None,
        to_utc: None,
        dense_from: None,
        dense_to: None,
        dense_step_ms: 10_000,
        minute_from: None,
        minute_to: None,
        shocks: Vec::new(),
        periods: Vec::new(),
        windows: None,
        factors_file: None,
        far_refresh_ms: None,
        mc_n: 1000,
        tick_reprice: false,
        reprice_from: None,
        reprice_to: None,
        bench_only: false,
    };
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let key = raw[i].as_str();
        let val = raw
            .get(i + 1)
            .unwrap_or_else(|| panic!("missing value for {key}"));
        match key {
            "--file" => args.files.push(PathBuf::from(val)),
            "--coin" => args.coin = val.to_string(),
            "--band-cents" => args.band_cents = val.parse().expect("bad --band-cents"),
            "--m-base" => args.m_base = Some(val.parse().expect("bad --m-base")),
            "--a-sigma" => args.a_sigma = Some(val.parse().expect("bad --a-sigma")),
            "--ewma-half-life" => {
                args.ewma_half_life = Some(val.parse().expect("bad --ewma-half-life"))
            }
            "--p-raw-gamma" => args.p_raw_gamma = Some(val.parse().expect("bad --p-raw-gamma")),
            "--current-shift" => {
                args.current_shift = Some(val.parse().expect("bad --current-shift"))
            }
            "--adjacent-shift" => {
                args.adjacent_shift = Some(val.parse().expect("bad --adjacent-shift"))
            }
            "--sessions" => args.sessions = val.parse().expect("bad --sessions"),
            "--seed" => args.seed = val.parse().expect("bad --seed"),
            "--from-utc" => {
                args.from_utc = Some(parse_utc(val).unwrap_or_else(|err| panic!("{err}")))
            }
            "--to-utc" => args.to_utc = Some(parse_utc(val).unwrap_or_else(|err| panic!("{err}"))),
            "--dense-from-utc" => {
                args.dense_from = Some(parse_utc(val).unwrap_or_else(|err| panic!("{err}")))
            }
            "--dense-to-utc" => {
                args.dense_to = Some(parse_utc(val).unwrap_or_else(|err| panic!("{err}")))
            }
            "--dense-step-secs" => {
                args.dense_step_ms = val.parse::<u64>().expect("bad --dense-step-secs") * 1_000
            }
            "--minute-from-utc" => {
                args.minute_from = Some(parse_utc(val).unwrap_or_else(|err| panic!("{err}")))
            }
            "--minute-to-utc" => {
                args.minute_to = Some(parse_utc(val).unwrap_or_else(|err| panic!("{err}")))
            }
            "--shock-utc" => {
                args.shocks
                    .push(parse_utc(val).unwrap_or_else(|err| panic!("{err}")))
            }
            "--period" => args
                .periods
                .push(parse_period(val).unwrap_or_else(|err| panic!("{err}"))),
            "--windows" => {
                let n: usize = val.parse().expect("bad --windows");
                assert!(n >= 1, "--windows must be >= 1");
                args.windows = Some(n);
            }
            "--factors-file" => args.factors_file = Some(PathBuf::from(val)),
            "--far-refresh-ms" => {
                args.far_refresh_ms = Some(val.parse().expect("bad --far-refresh-ms"))
            }
            "--mc-n" => args.mc_n = val.parse().expect("bad --mc-n"),
            "--tick-reprice" => args.tick_reprice = val.parse().expect("bad --tick-reprice"),
            "--reprice-from-utc" => {
                args.reprice_from = Some(parse_utc(val).unwrap_or_else(|err| panic!("{err}")))
            }
            "--reprice-to-utc" => {
                args.reprice_to = Some(parse_utc(val).unwrap_or_else(|err| panic!("{err}")))
            }
            "--bench-only" => args.bench_only = val.parse().expect("bad --bench-only"),
            _ => panic!("unknown arg {key}"),
        }
        i += 2;
    }
    if args.files.is_empty() || args.coin.is_empty() || args.band_cents == 0 {
        panic!(
            "usage: policy_backtest --file <jsonl> --coin <coin> --band-cents <n> [--windows <n>] [--factors-file <path>] [--far-refresh-ms <ms>] [--mc-n <n>] [--tick-reprice true] [--reprice-from-utc <ts> --reprice-to-utc <ts>] [--bench-only true] [--from-utc <ts>] [--to-utc <ts>] [--dense-from-utc <ts> --dense-to-utc <ts>] [--minute-from-utc <ts> --minute-to-utc <ts>] [--shock-utc <ts>] [--period name,start,end] [--sessions 0]"
        );
    }
    if args.dense_from.is_some() != args.dense_to.is_some() {
        panic!("--dense-from-utc and --dense-to-utc must be set together");
    }
    if args.minute_from.is_some() != args.minute_to.is_some() {
        panic!("--minute-from-utc and --minute-to-utc must be set together");
    }
    args
}

fn event_mode(args: &Args) -> bool {
    args.from_utc.is_some()
        || args.to_utc.is_some()
        || args.dense_from.is_some()
        || args.minute_from.is_some()
        || !args.shocks.is_empty()
        || !args.periods.is_empty()
}

fn apply(cfg: &mut GridConfig, args: &Args) {
    if let Some(v) = args.m_base {
        cfg.pricing.m_base = v;
    }
    if let Some(v) = args.a_sigma {
        cfg.pricing.a_sigma = v;
    }
    if let Some(v) = args.ewma_half_life {
        cfg.volatility.ewma_half_life_seconds = v;
    }
    if let Some(v) = args.p_raw_gamma {
        cfg.pricing.p_raw_gamma = v;
    }
    if let Some(v) = args.current_shift {
        cfg.pricing.current_row_logit_shift = v;
    }
    if let Some(v) = args.adjacent_shift {
        cfg.pricing.adjacent_row_logit_shift = v;
    }
    if let Some(windows) = args.windows {
        cfg.simulation.quote_window_count = windows;
    }
    if let Some(far_refresh_ms) = args.far_refresh_ms {
        cfg.simulation.far_refresh_ms = far_refresh_ms;
    }
    if let Some(ref path) = args.factors_file {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));
        let entries = parse_factor_table(&text).unwrap_or_else(|err| panic!("{err}"));
        apply_factor_table(cfg, &entries);
    }
    cfg.simulation.mc_n_min = args.mc_n;
    cfg.simulation.mc_n_max = args.mc_n;
    cfg.simulation.seed = Some(args.seed);
}

fn load_ticks(args: &Args) -> Vec<PriceTick> {
    let mut ticks = Vec::new();
    for path in &args.files {
        let mut part = read_hl_jsonl(path, &args.coin).unwrap_or_else(|err| panic!("{err}"));
        ticks.append(&mut part);
    }
    ticks.sort_by_key(|tick| tick.ts_ms);
    ticks
}

#[derive(Default, Clone)]
struct Acc {
    taps: usize,
    hits: usize,
    volume: f64,
    payout: f64,
    mult_sum: f64,
    cap25_taps: usize,
    minutes: BTreeMap<u64, (f64, f64)>,
}

#[derive(Default, Clone)]
struct Book {
    taps: usize,
    volume: f64,
    payout: f64,
}

fn add_fill(acc: &mut Acc, tap: &CellTap, max_m: f64, attr_ms: u64) {
    let payout = if tap.hit { STAKE * tap.multiplier } else { 0.0 };
    acc.taps += 1;
    acc.volume += STAKE;
    acc.payout += payout;
    acc.mult_sum += tap.multiplier;
    if tap.hit {
        acc.hits += 1;
    }
    if tap.multiplier >= max_m - CAP25_EPS {
        acc.cap25_taps += 1;
    }
    let minute = attr_ms / 60_000 * 60_000;
    let entry = acc.minutes.entry(minute).or_insert((0.0, 0.0));
    entry.0 += STAKE;
    entry.1 += payout;
}

fn quotes_from(report: &CalibrationReport) -> Vec<WindowQuote> {
    let mut by_quote: BTreeMap<u64, WindowQuote> = BTreeMap::new();
    for sample in &report.samples {
        let start_ms = sample.quote_ts_ms.saturating_add(QUOTE_LEAD_MS);
        by_quote
            .entry(start_ms)
            .and_modify(|entry| {
                entry.max_mult = entry.max_mult.max(sample.multiplier);
                entry.sigma = sample.sigma;
                entry.sigma_raw = sample.sigma_raw;
            })
            .or_insert(WindowQuote {
                start_ms,
                sigma: sample.sigma,
                sigma_raw: sample.sigma_raw,
                max_mult: sample.multiplier,
            });
    }
    by_quote.into_values().collect()
}

fn build_taps(report: &CalibrationReport, ticks: &[PriceTick]) -> Vec<CellTap> {
    let mut taps = Vec::new();
    for sample in &report.samples {
        if sample.is_warmup {
            continue;
        }
        taps.push(CellTap {
            quote_ts_ms: sample.quote_ts_ms,
            start_ms: sample.window_start_ms,
            window_index: sample.window_index,
            signed_dist: sample.signed_distance,
            dist: sample.center_distance as i64,
            multiplier: sample.multiplier,
            hit: sample.hit,
            us: sample.is_us_session,
            move_sign: move_sign(ticks, sample.quote_ts_ms, 10_000),
            sigma: sample.sigma,
            sigma_raw: sample.sigma_raw,
        });
    }
    taps.sort_by_key(|tap| (tap.start_ms, tap.window_index, tap.dist));
    taps
}

fn voids_in(voided: &[u64], from_ms: u64, to_ms: u64) -> usize {
    voided
        .iter()
        .filter(|start| **start >= from_ms && **start < to_ms)
        .count()
}

fn max_tripwire(ticks: &[PriceTick], quotes: &[WindowQuote], from_ms: u64, to_ms: u64) -> (f64, u64) {
    let mut best = 0.0_f64;
    let mut at = from_ms;
    for minute in minute_starts(from_ms, to_ms) {
        let Some(move_log) = log_move(ticks, minute, minute.saturating_add(60_000)) else {
            continue;
        };
        let Some((sigma, _)) = quoted_sigma_at(quotes, minute, QUOTE_LEAD_MS) else {
            continue;
        };
        let Some(k) = tripwire_k(move_log.abs(), sigma, 60.0) else {
            continue;
        };
        if k > best {
            best = k;
            at = minute;
        }
    }
    (best, at)
}

fn print_board_cap25(taps: &[CellTap], max_m: f64) {
    let mut all = (0usize, 0usize);
    let mut near = (0usize, 0usize);
    let mut far = (0usize, 0usize);
    for tap in taps {
        let capped = usize::from(tap.multiplier >= max_m - CAP25_EPS);
        all.0 += 1;
        all.1 += capped;
        if tap.window_index < NEAR_WINDOWS {
            near.0 += 1;
            near.1 += capped;
        } else {
            far.0 += 1;
            far.1 += capped;
        }
    }
    for (bucket, (cells, capped)) in [("all", all), ("near", near), ("far", far)] {
        let pct = if cells == 0 {
            0.0
        } else {
            capped as f64 / cells as f64 * 100.0
        };
        println!("BOARD_CAP25 bucket={bucket} cells={cells} cap25_cells={capped} cap25_pct={pct:.2}");
    }
}

fn print_horizon_policy_line(name: &str, bucket: &str, acc: &Acc) {
    let edge_pct = if acc.volume > 0.0 {
        (acc.volume - acc.payout) / acc.volume * 100.0
    } else {
        0.0
    };
    let cap25_pct = if acc.taps > 0 {
        acc.cap25_taps as f64 / acc.taps as f64 * 100.0
    } else {
        0.0
    };
    let avg_mult = if acc.taps > 0 {
        acc.mult_sum / acc.taps as f64
    } else {
        0.0
    };
    println!(
        "HORIZON_POLICY name={name} bucket={bucket} taps={} hits={} volume={:.2} payout={:.2} edge_pct={edge_pct:.2} cap25_taps={} cap25_pct={cap25_pct:.2} avg_mult={avg_mult:.2}",
        acc.taps, acc.hits, acc.volume, acc.payout, acc.cap25_taps
    );
}

fn window_band_index(window_index: usize) -> usize {
    match window_index {
        0..=5 => 0,
        6..=11 => 1,
        12..=17 => 2,
        _ => 3,
    }
}

fn print_window_buckets(bands: &[Acc; 4]) {
    let mut far_total = Acc::default();
    for b in &bands[1..4] {
        far_total.taps += b.taps;
        far_total.hits += b.hits;
        far_total.volume += b.volume;
        far_total.payout += b.payout;
        far_total.mult_sum += b.mult_sum;
        far_total.cap25_taps += b.cap25_taps;
    }
    let rows = [
        ("0-5", &bands[0]),
        ("6-11", &bands[1]),
        ("12-17", &bands[2]),
        ("18-23", &bands[3]),
        ("6-23", &far_total),
    ];
    for (label, acc) in rows {
        let edge_pct = if acc.volume > 0.0 {
            (acc.volume - acc.payout) / acc.volume * 100.0
        } else {
            0.0
        };
        let cap25_pct = if acc.taps > 0 {
            acc.cap25_taps as f64 / acc.taps as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "WINDOW_BUCKET band={label} taps={} hits={} volume={:.2} payout={:.2} edge_pct={edge_pct:.2} cap25_taps={} cap25_pct={cap25_pct:.2}",
            acc.taps, acc.hits, acc.volume, acc.payout, acc.cap25_taps
        );
    }
}

fn print_outstanding_exposure(combined: &[ExposureInterval], per_policy: &[Vec<ExposureInterval>]) {
    let summary = compute_outstanding_exposure(combined);
    println!(
        "OUTSTANDING_EXPOSURE scope=combined active_minutes={} total_minutes={} mean_active_stake={:.2} p95_peak_stake={:.2} peak_stake={:.2} peak_stake_at={} mean_active_liability={:.2} p95_peak_liability={:.2} peak_liability={:.2} peak_liability_at={} mean_open_taps={:.2} peak_open_taps={}",
        summary.active_minutes,
        summary.total_minutes,
        summary.mean_active_stake,
        summary.p95_active_peak_stake,
        summary.peak_stake,
        format_utc(summary.peak_stake_ms),
        summary.mean_active_liability,
        summary.p95_active_peak_liability,
        summary.peak_liability,
        format_utc(summary.peak_liability_ms),
        summary.mean_active_open_taps,
        summary.peak_open_taps
    );
    let mut sum_mean_stake = 0.0;
    let mut sum_peak_stake = 0.0;
    let mut sum_mean_liab = 0.0;
    let mut sum_peak_liab = 0.0;
    let mut sum_mean_taps = 0.0;
    let mut count = 0usize;
    for intervals in per_policy {
        if intervals.is_empty() {
            continue;
        }
        let s = compute_outstanding_exposure(intervals);
        sum_mean_stake += s.mean_active_stake;
        sum_peak_stake += s.peak_stake;
        sum_mean_liab += s.mean_active_liability;
        sum_peak_liab += s.peak_liability;
        sum_mean_taps += s.mean_active_open_taps;
        count += 1;
    }
    if count > 0 {
        let n = count as f64;
        println!(
            "OUTSTANDING_EXPOSURE scope=per_policy_mean policies={count} mean_active_stake={:.2} mean_peak_stake={:.2} mean_active_liability={:.2} mean_peak_liability={:.2} mean_open_taps={:.2}",
            sum_mean_stake / n,
            sum_peak_stake / n,
            sum_mean_liab / n,
            sum_peak_liab / n,
            sum_mean_taps / n
        );
    }
}

fn print_reprice_and_triggers(
    cfg: &GridConfig,
    args: &Args,
    stats: &RepriceStats,
    periods: &[(String, u64, u64)],
) {
    println!(
        "REPRICE_STATS windows={} far_refresh_ms={} mc_n={} tick_reprice={} samples={} full={} near={} trigger={} cadence={} mean_us={} full_mean_us={} near_mean_us={}",
        cfg.simulation.quote_window_count,
        cfg.simulation.far_refresh_ms,
        args.mc_n,
        args.tick_reprice,
        stats.sample_count,
        stats.full_reprices,
        stats.near_reprices,
        stats.trigger_reprices,
        stats.cadence_reprices,
        stats.mean_us(),
        stats.full_mean_us(),
        stats.near_mean_us(),
    );
    let format_first = |items: &[u64]| -> String {
        if items.is_empty() {
            "-".to_string()
        } else {
            items
                .iter()
                .take(10)
                .map(|&ts| format_utc(ts))
                .collect::<Vec<_>>()
                .join(",")
        }
    };
    println!(
        "JUMP_TRIGGER period=all triggers={} first_triggers={}",
        stats.trigger_timestamps_ms.len(),
        format_first(&stats.trigger_timestamps_ms)
    );
    for (name, from_ms, to_ms) in periods {
        let in_period: Vec<u64> = stats
            .trigger_timestamps_ms
            .iter()
            .copied()
            .filter(|&ts| ts >= *from_ms && ts < *to_ms)
            .collect();
        println!(
            "JUMP_TRIGGER period={name} triggers={} first_triggers={}",
            in_period.len(),
            format_first(&in_period)
        );
    }
    let in_1356: Vec<u64> = stats
        .trigger_timestamps_ms
        .iter()
        .copied()
        .filter(|&ts| ts >= NEWS_MIN_1356_FROM_MS && ts < NEWS_MIN_1356_TO_MS)
        .collect();
    if !in_1356.is_empty()
        || periods
            .iter()
            .any(|(_, f, t)| *f <= NEWS_MIN_1356_FROM_MS && *t >= NEWS_MIN_1356_TO_MS)
    {
        println!(
            "JUMP_TRIGGER period=news-min-1356 triggers={} first_triggers={}",
            in_1356.len(),
            format_first(&in_1356)
        );
    }
}

fn print_policy(name: &str, period: &str, acc: &Acc) {
    let stats = stake_stats(acc.volume, acc.payout);
    let minutes: Vec<(u64, f64, f64)> = acc
        .minutes
        .iter()
        .map(|(minute, (volume, payout))| (*minute, *volume, *payout))
        .collect();
    let worst = worst_minute(&minutes);
    let worst_edge = worst_edge_minute(&minutes, 100.0);
    let hit_rate = if acc.taps == 0 {
        0.0
    } else {
        acc.hits as f64 / acc.taps as f64
    };
    let avg_mult = if acc.taps == 0 {
        0.0
    } else {
        acc.mult_sum / acc.taps as f64
    };
    let worst_ts = worst
        .map(|(ms, _)| format_utc(ms))
        .unwrap_or_else(|| "-".to_string());
    let worst_pnl = worst.map(|(_, pnl)| pnl).unwrap_or(0.0);
    let worst_edge_ts = worst_edge
        .map(|(ms, _)| format_utc(ms))
        .unwrap_or_else(|| "-".to_string());
    let worst_edge_pct = worst_edge.map(|(_, edge)| edge).unwrap_or(0.0);
    let house = acc.volume - acc.payout;
    println!(
        "POLICY name={name} period={period} taps={} volume={:.2} payout={:.2} house_pnl={:.2} edge_pct={:.2} payout_ratio={:.4} loss_per_1000={:.2} house_per_1000={:.2} hit_rate={:.3} avg_mult={:.2} worst_minute={worst_ts} worst_pnl={worst_pnl:.2} worst_edge_minute={worst_edge_ts} worst_edge_pct={worst_edge_pct:.2}",
        acc.taps,
        acc.volume,
        acc.payout,
        house,
        stats.edge_pct,
        stats.payout_ratio,
        stats.loss_per_1000,
        stats.house_per_1000,
        hit_rate,
        avg_mult
    );
}

#[allow(clippy::too_many_arguments)]
fn print_book(
    kind: &str,
    start_ms: u64,
    width_ms: u64,
    book: &Book,
    quotes: &[WindowQuote],
    ticks: &[PriceTick],
    voided: &[u64],
    max_k: Option<(f64, u64)>,
) {
    let end = start_ms.saturating_add(width_ms);
    let (sigma, raw, max_mult, _) = bucket_quote(quotes, start_ms, end);
    let realized = realized_step_sigma(ticks, start_ms, start_ms.saturating_add(60_000))
        .map(|(value, _)| value)
        .unwrap_or(0.0);
    let raw_over = if realized > 0.0 { raw / realized } else { 0.0 };
    let move_log = log_move(ticks, start_ms, start_ms.saturating_add(60_000)).unwrap_or(0.0);
    let move_pct = (move_log.exp() - 1.0) * 100.0;
    let k = tripwire_k(move_log.abs(), sigma, 60.0).unwrap_or(0.0);
    let px = price_at_or_before(ticks, start_ms).unwrap_or(0.0);
    let house = book.volume - book.payout;
    let (hour_k, hour_at) = max_k.unwrap_or((0.0, start_ms));
    println!(
        "BOOK {kind} ts={} taps={} volume={:.2} payout={:.2} house_pnl={:.2} voids={} max_mult={:.2} quoted_sigma={:.6} quoted_raw={:.6} realized_fwd_60s={:.6} raw_over_realized={:.3} px={:.4} move_60s_pct={:.3} move_over_sigma={:.2} max_move_over_sigma={:.2} max_move_at={}",
        format_utc(start_ms),
        book.taps,
        book.volume,
        book.payout,
        house,
        voids_in(voided, start_ms, end),
        max_mult,
        sigma,
        raw,
        realized,
        raw_over,
        px,
        move_pct,
        k,
        hour_k,
        format_utc(hour_at)
    );
}

fn board_snapshot_at<'a>(taps_by_quote: &'a [CellTap], start_ms: u64, burst_ms: u64) -> &'a [CellTap] {
    let lo = taps_by_quote.partition_point(|tap| tap.quote_ts_ms < start_ms);
    let Some(first) = taps_by_quote.get(lo) else {
        return &[];
    };
    if first.quote_ts_ms >= start_ms.saturating_add(burst_ms) {
        return &[];
    }
    let target_quote = first.quote_ts_ms;
    let hi = taps_by_quote[lo..].partition_point(|tap| tap.quote_ts_ms == target_quote) + lo;
    &taps_by_quote[lo..hi]
}

fn precompute_legacy_board_starts(ticks: &[PriceTick], args: &Args) -> (Vec<Vec<u64>>, Vec<u64>) {
    if ticks.len() < 2 {
        return (vec![Vec::new(); POLICIES.len()], Vec::new());
    }
    let first_ts = ticks[0].ts_ms;
    let last_ts = ticks[ticks.len() - 1].ts_ms;
    let first_non_warmup =
        first_ts.saturating_add(WARMUP_MS).div_ceil(WINDOW_MS).saturating_mul(WINDOW_MS);
    let mut per_policy = Vec::with_capacity(POLICIES.len());
    let mut all = Vec::new();
    for policy_index in 0..POLICIES.len() {
        let mut rng = StdRng::seed_from_u64(args.seed + policy_index as u64);
        let mut starts = Vec::new();
        let mut start = first_non_warmup.saturating_add(WATCH_MS);
        while starts.len() < args.sessions && start.saturating_add(BURST_MS) < last_ts {
            starts.push(start);
            all.push(start);
            start = start.saturating_add(rng.random_range(RETURN_MIN_MS..RETURN_MAX_MS));
        }
        per_policy.push(starts);
    }
    all.sort_unstable();
    all.dedup();
    (per_policy, all)
}

fn run_legacy_board(
    args: &Args,
    cfg: &GridConfig,
    taps: &[CellTap],
    per_policy_starts: &[Vec<u64>],
    report: &CalibrationReport,
) {
    if taps.is_empty() {
        println!("no taps");
        return;
    }
    let max_m = cfg.pricing.max_multiplier;
    let mut by_quote = taps.to_vec();
    by_quote.sort_by_key(|tap| (tap.quote_ts_ms, tap.window_index, tap.dist));

    print_board_cap25(&by_quote, max_m);

    let mut combined_intervals: Vec<ExposureInterval> = Vec::new();
    let mut per_policy_intervals: Vec<Vec<ExposureInterval>> =
        vec![Vec::new(); POLICIES.len()];
    let mut window_bands: [Acc; 4] = Default::default();

    for (policy_index, policy) in POLICIES.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(args.seed.wrapping_add(10_000 + policy_index as u64));
        let mut volume = [0.0; 2];
        let mut payout = [0.0; 2];
        let mut count = [0usize; 2];
        let mut multiplier_sum = 0.0;
        let mut by_dist = vec![(0.0_f64, 0.0_f64, 0usize); 11];
        let mut all_acc = Acc::default();
        let mut near_acc = Acc::default();
        let mut far_acc = Acc::default();

        let starts = &per_policy_starts[policy_index];
        for &start in starts {
            let board = board_snapshot_at(&by_quote, start, BURST_MS);
            if board.is_empty() {
                continue;
            }
            for tap in choose_taps(policy, board, TAPS_PER_SESSION, &mut rng) {
                let side = usize::from(!tap.us);
                volume[side] += STAKE;
                count[side] += 1;
                multiplier_sum += tap.multiplier;
                let bucket = (tap.dist as usize).min(10);
                by_dist[bucket].0 += STAKE;
                by_dist[bucket].2 += 1;
                if tap.hit {
                    payout[side] += STAKE * tap.multiplier;
                    by_dist[bucket].1 += STAKE * tap.multiplier;
                }
                add_fill(&mut all_acc, &tap, max_m, tap.quote_ts_ms);
                if tap.window_index < NEAR_WINDOWS {
                    add_fill(&mut near_acc, &tap, max_m, tap.quote_ts_ms);
                } else {
                    add_fill(&mut far_acc, &tap, max_m, tap.quote_ts_ms);
                }
                add_fill(
                    &mut window_bands[window_band_index(tap.window_index)],
                    &tap,
                    max_m,
                    tap.quote_ts_ms,
                );
                let interval = ExposureInterval {
                    open_ms: tap.quote_ts_ms,
                    close_ms: tap.start_ms.saturating_add(WINDOW_MS),
                    stake: STAKE,
                    liability: STAKE * tap.multiplier,
                };
                combined_intervals.push(interval);
                per_policy_intervals[policy_index].push(interval);
            }
        }
        for (side, name) in ["US", "OFF"].iter().enumerate() {
            let edge = if volume[side] > 0.0 {
                (volume[side] - payout[side]) / volume[side] * 100.0
            } else {
                0.0
            };
            println!(
                "{policy} {name} sessions={} taps={} volume={:.0} payout={:.0} edge={edge:.2}%",
                starts.len(),
                count[side],
                volume[side],
                payout[side]
            );
        }
        let taps_total = count[0] + count[1];
        let avg_multiplier = if taps_total > 0 {
            multiplier_sum / taps_total as f64
        } else {
            0.0
        };
        println!("{policy} avg_multiplier={avg_multiplier:.2}");
        print_horizon_policy_line(policy, "all", &all_acc);
        print_horizon_policy_line(policy, "near", &near_acc);
        print_horizon_policy_line(policy, "far", &far_acc);
        if *policy == "random_near" {
            for (dist, (vol, pay, n)) in by_dist.iter().enumerate() {
                if *n == 0 {
                    continue;
                }
                let edge = (vol - pay) / vol * 100.0;
                println!("random_near dist={dist} taps={n} volume={vol:.0} edge={edge:.2}%");
            }
        }
    }
    print_window_buckets(&window_bands);
    print_outstanding_exposure(&combined_intervals, &per_policy_intervals);
    print_reprice_and_triggers(cfg, args, &report.reprice_stats, &[]);
    println!(
        "note: board-snapshot mode quotes all {} windows at each session start; exposure skew and safety factor are off.",
        cfg.simulation.quote_window_count
    );
}

fn run_legacy(args: &Args, cfg: &GridConfig, taps: &[CellTap], report: &CalibrationReport) {
    if taps.is_empty() {
        println!("no taps");
        return;
    }
    let first = taps[0].start_ms;
    let last = taps[taps.len() - 1].start_ms;
    for (policy_index, policy) in POLICIES.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(args.seed + policy_index as u64);
        let mut volume = [0.0; 2];
        let mut payout = [0.0; 2];
        let mut count = [0usize; 2];
        let mut multiplier_sum = 0.0;
        let mut by_dist = vec![(0.0_f64, 0.0_f64, 0usize); 11];
        let mut start = first + WATCH_MS;
        let mut sessions = 0usize;
        while sessions < args.sessions && start + BURST_MS < last {
            let end = start + BURST_MS;
            let lo = taps.partition_point(|tap| tap.start_ms < start);
            let hi = taps.partition_point(|tap| tap.start_ms < end);
            for tap in choose_taps(policy, &taps[lo..hi], TAPS_PER_SESSION, &mut rng) {
                let side = usize::from(!tap.us);
                volume[side] += STAKE;
                count[side] += 1;
                multiplier_sum += tap.multiplier;
                let bucket = (tap.dist as usize).min(10);
                by_dist[bucket].0 += STAKE;
                by_dist[bucket].2 += 1;
                if tap.hit {
                    payout[side] += STAKE * tap.multiplier;
                    by_dist[bucket].1 += STAKE * tap.multiplier;
                }
            }
            sessions += 1;
            start += rng.random_range(RETURN_MIN_MS..RETURN_MAX_MS);
        }
        for (side, name) in ["US", "OFF"].iter().enumerate() {
            let edge = if volume[side] > 0.0 {
                (volume[side] - payout[side]) / volume[side] * 100.0
            } else {
                0.0
            };
            println!(
                "{policy} {name} sessions={sessions} taps={} volume={:.0} payout={:.0} edge={edge:.2}%",
                count[side], volume[side], payout[side]
            );
        }
        let taps_total = count[0] + count[1];
        let avg_multiplier = if taps_total > 0 {
            multiplier_sum / taps_total as f64
        } else {
            0.0
        };
        println!("{policy} avg_multiplier={avg_multiplier:.2}");
        if *policy == "random_near" {
            for (dist, (vol, pay, n)) in by_dist.iter().enumerate() {
                if *n == 0 {
                    continue;
                }
                let edge = (vol - pay) / vol * 100.0;
                println!("random_near dist={dist} taps={n} volume={vol:.0} edge={edge:.2}%");
            }
        }
    }
    print_reprice_and_triggers(cfg, args, &report.reprice_stats, &[]);
    println!(
        "note: quotes cached every 5s; production reprices every 100ms. exposure skew and safety factor are off, so this slightly understates house edge."
    );
}

#[allow(clippy::too_many_arguments)]
fn run_event(
    args: &Args,
    cfg: &GridConfig,
    taps: &[CellTap],
    ticks: &[PriceTick],
    quotes: &[WindowQuote],
    report: &CalibrationReport,
    precomputed_starts: Option<(Vec<u64>, Vec<u64>, Vec<u64>)>,
) {
    let board_mode = args.windows.is_some();
    let max_m = cfg.pricing.max_multiplier;
    let file_first = ticks.first().map(|tick| tick.ts_ms).unwrap_or(0);
    let file_last = ticks.last().map(|tick| tick.ts_ms).unwrap_or(0);
    let from_ms = args.from_utc.unwrap_or(file_first);
    let to_ms = args.to_utc.unwrap_or(file_last.saturating_add(1));

    let mut selected: Vec<CellTap> = taps
        .iter()
        .copied()
        .filter(|tap| {
            let ts = if board_mode {
                tap.quote_ts_ms
            } else {
                tap.start_ms
            };
            ts >= from_ms && ts < to_ms
        })
        .collect();
    if board_mode {
        selected.sort_by_key(|tap| (tap.quote_ts_ms, tap.window_index, tap.dist));
    }

    let first_tap = selected
        .first()
        .map(|tap| if board_mode { tap.quote_ts_ms } else { tap.start_ms })
        .unwrap_or(from_ms);
    let last_tap = selected
        .last()
        .map(|tap| if board_mode { tap.quote_ts_ms } else { tap.start_ms })
        .unwrap_or(from_ms);
    if args.sessions > 0 {
        eprintln!(
            "warning: --sessions {} caps normal density; pass 0 to cover the whole window",
            args.sessions
        );
    }
    let (normal, dense, starts) = if let Some(pre) = precomputed_starts {
        pre
    } else {
        let mut schedule_rng = StdRng::seed_from_u64(args.seed);
        let normal = if selected.len() < 2 {
            Vec::new()
        } else {
            normal_session_starts(
                first_tap,
                last_tap,
                from_ms,
                to_ms,
                WATCH_MS,
                BURST_MS,
                RETURN_MIN_MS,
                RETURN_MAX_MS,
                args.sessions,
                &mut schedule_rng,
            )
        };
        let dense = match (args.dense_from, args.dense_to) {
            (Some(start), Some(end)) => {
                dense_session_starts(start, end, args.dense_step_ms, BURST_MS, last_tap)
            }
            _ => Vec::new(),
        };
        let merged = merge_starts(normal.clone(), dense.clone());
        (normal, dense, merged)
    };

    println!(
        "replay from={} to_exclusive={} file_first={} file_last={} px_first={:.4} px_last={:.4} ticks={} samples={} quotes={} voids={} stale_skips={}",
        format_utc(from_ms),
        format_utc(to_ms),
        format_utc(file_first),
        format_utc(file_last),
        ticks.first().map(|tick| tick.price).unwrap_or(0.0),
        ticks.last().map(|tick| tick.price).unwrap_or(0.0),
        ticks.len(),
        selected.len(),
        quotes.len(),
        report.voided_windows,
        report.stale_quote_skips
    );
    println!(
        "schedule=shared watch_s={} burst_s={} return_s={}-{} normal_sessions={} dense_sessions={} merged_sessions={} dense_step_s={}",
        WATCH_MS / 1000,
        BURST_MS / 1000,
        RETURN_MIN_MS / 1000,
        RETURN_MAX_MS / 1000,
        normal.len(),
        dense.len(),
        starts.len(),
        args.dense_step_ms / 1000
    );
    println!("period_note=named periods are slices and may overlap; day is the replay window");

    let mut trip_minutes = 0usize;
    let mut trip_counts = [0usize; 3];
    let thresholds = [4.0_f64, 6.0, 8.0];
    let mut max_k = 0.0_f64;
    let mut max_k_at = from_ms;
    for minute in minute_starts(from_ms, to_ms) {
        let Some(move_log) = log_move(ticks, minute, minute.saturating_add(60_000)) else {
            continue;
        };
        let Some((sigma, _)) = quoted_sigma_at(quotes, minute, QUOTE_LEAD_MS) else {
            continue;
        };
        let Some(k) = tripwire_k(move_log.abs(), sigma, 60.0) else {
            continue;
        };
        trip_minutes += 1;
        if k > max_k {
            max_k = k;
            max_k_at = minute;
        }
        for (index, threshold) in thresholds.iter().enumerate() {
            if k > *threshold {
                trip_counts[index] += 1;
            }
        }
    }
    println!(
        "TRIPWIRE minutes={trip_minutes} k4={} k6={} k8={} max_k={max_k:.2} max_at={}",
        trip_counts[0],
        trip_counts[1],
        trip_counts[2],
        format_utc(max_k_at)
    );

    for audit in audit_feed_gaps(ticks, 5_000) {
        let caught =
            audit.overlapping_windows > 0 && audit.voided_windows == audit.overlapping_windows;
        let in_replay = audit.end_ms > from_ms && audit.start_ms < to_ms;
        println!(
            "GAP start={} end={} gap_ms={} overlapping={} voided={} caught={} in_replay={}",
            format_utc(audit.start_ms),
            format_utc(audit.end_ms),
            audit.gap_ms,
            audit.overlapping_windows,
            audit.voided_windows,
            if caught { "yes" } else { "no" },
            if in_replay { "yes" } else { "no" }
        );
    }

    let mut periods = vec![("day".to_string(), from_ms, to_ms)];
    for period in &args.periods {
        periods.push((period.name.clone(), period.from_ms, period.to_ms));
    }
    let mut books: Vec<Vec<Acc>> = (0..POLICIES.len())
        .map(|_| vec![Acc::default(); periods.len()])
        .collect();
    let mut near_books: Vec<Acc> = vec![Acc::default(); POLICIES.len()];
    let mut far_books: Vec<Acc> = vec![Acc::default(); POLICIES.len()];
    let mut window_bands: [Acc; 4] = Default::default();
    let mut combined_intervals: Vec<ExposureInterval> = Vec::new();
    let mut per_policy_intervals: Vec<Vec<ExposureInterval>> = vec![Vec::new(); POLICIES.len()];

    let mut hour_book: BTreeMap<u64, Book> = BTreeMap::new();
    let mut minute_book: BTreeMap<u64, Book> = BTreeMap::new();
    let mut fills: Vec<(u64, f64, f64)> = Vec::new();
    let mut cell_stake: HashMap<(u64, i64), f64> = HashMap::new();
    let mut policy_rngs: Vec<StdRng> = POLICIES
        .iter()
        .enumerate()
        .map(|(index, _)| StdRng::seed_from_u64(args.seed.wrapping_add(10_000 + index as u64)))
        .collect();
    let minute_from = args.minute_from.unwrap_or(u64::MAX);
    let minute_to = args.minute_to.unwrap_or(0);

    for &start in &starts {
        let window: &[CellTap] = if board_mode {
            board_snapshot_at(&selected, start, BURST_MS)
        } else {
            let end = start.saturating_add(BURST_MS);
            let lo = selected.partition_point(|tap| tap.start_ms < start);
            let hi = selected.partition_point(|tap| tap.start_ms < end);
            &selected[lo..hi]
        };
        if window.is_empty() {
            continue;
        }
        for (index, policy) in POLICIES.iter().enumerate() {
            for tap in choose_taps(policy, window, TAPS_PER_SESSION, &mut policy_rngs[index]) {
                let payout = if tap.hit { STAKE * tap.multiplier } else { 0.0 };
                let attr_ms = if board_mode {
                    tap.quote_ts_ms
                } else {
                    tap.start_ms
                };
                fills.push((attr_ms, STAKE, payout));
                *cell_stake.entry((tap.start_ms, tap.signed_dist)).or_insert(0.0) += STAKE;
                for (period_index, (_, period_from, period_to)) in periods.iter().enumerate() {
                    if attr_ms >= *period_from && attr_ms < *period_to {
                        add_fill(&mut books[index][period_index], &tap, max_m, attr_ms);
                    }
                }
                if tap.window_index < NEAR_WINDOWS {
                    add_fill(&mut near_books[index], &tap, max_m, attr_ms);
                } else {
                    add_fill(&mut far_books[index], &tap, max_m, attr_ms);
                }
                add_fill(
                    &mut window_bands[window_band_index(tap.window_index)],
                    &tap,
                    max_m,
                    attr_ms,
                );
                let interval = ExposureInterval {
                    open_ms: tap.quote_ts_ms,
                    close_ms: tap.start_ms.saturating_add(WINDOW_MS),
                    stake: STAKE,
                    liability: STAKE * tap.multiplier,
                };
                combined_intervals.push(interval);
                per_policy_intervals[index].push(interval);

                let hour = attr_ms / 3_600_000 * 3_600_000;
                let hour_entry = hour_book.entry(hour).or_default();
                hour_entry.taps += 1;
                hour_entry.volume += STAKE;
                hour_entry.payout += payout;
                if attr_ms >= minute_from && attr_ms < minute_to {
                    let minute = attr_ms / 60_000 * 60_000;
                    let minute_entry = minute_book.entry(minute).or_default();
                    minute_entry.taps += 1;
                    minute_entry.volume += STAKE;
                    minute_entry.payout += payout;
                }
            }
        }
    }

    let mut best_stake = 0.0_f64;
    let mut best_cell = (0_u64, 0_i64);
    for (cell, stake) in &cell_stake {
        if *stake > best_stake {
            best_stake = *stake;
            best_cell = *cell;
        }
    }
    println!(
        "EXPOSURE max_cell_stake={best_stake:.2} max_cell_ts={} max_cell_dist={} cells={}",
        format_utc(best_cell.0),
        best_cell.1,
        cell_stake.len()
    );

    for shock in &args.shocks {
        let px0 = price_at_or_before(ticks, *shock).unwrap_or(0.0);
        for horizon in SHOCK_HORIZONS_MS {
            let end = shock.saturating_add(horizon);
            let px1 = price_at_or_before(ticks, end).unwrap_or(px0);
            let move_pct = if px0 > 0.0 {
                (px1 / px0 - 1.0) * 100.0
            } else {
                0.0
            };
            let quoted = quoted_sigma_at(quotes, *shock, QUOTE_LEAD_MS).unwrap_or((0.0, 0.0));
            let quoted_end = quoted_sigma_at(quotes, end, QUOTE_LEAD_MS).unwrap_or((0.0, 0.0));
            let realized = realized_step_sigma(ticks, *shock, end).unwrap_or((0.0, 0));
            let mut volume = 0.0;
            let mut payout = 0.0;
            let mut tap_count = 0usize;
            for (attr_ms, stake, pay) in &fills {
                if *attr_ms >= *shock && *attr_ms < end {
                    volume += stake;
                    payout += pay;
                    tap_count += 1;
                }
            }
            println!(
                "SHOCK t0={} horizon_s={} px0={px0:.4} px1={px1:.4} move_pct={move_pct:.3} max_gap_ms={} quoted_sigma={:.6} quoted_sigma_end={:.6} realized_sigma={:.6} realized_n={} voids={} taps={tap_count} volume={volume:.2} payout={payout:.2} house_pnl={:.2}",
                format_utc(*shock),
                horizon / 1000,
                max_gap_overlapping(ticks, *shock, end),
                quoted.0,
                quoted_end.0,
                realized.0,
                realized.1,
                voids_in(&report.voided_window_starts, *shock, end),
                volume - payout
            );
        }
        for offset in SIGMA_LAG_OFFSETS_MS {
            let ts = shock.saturating_add(offset);
            let (quoted, raw) = quoted_sigma_at(quotes, ts, QUOTE_LEAD_MS).unwrap_or((0.0, 0.0));
            let back = realized_step_sigma(ticks, ts.saturating_sub(60_000), ts)
                .map(|(value, _)| value)
                .unwrap_or(0.0);
            let fwd = realized_step_sigma(ticks, ts, ts.saturating_add(60_000))
                .map(|(value, _)| value)
                .unwrap_or(0.0);
            let fwd_over_raw = if raw > 0.0 { fwd / raw } else { 0.0 };
            println!(
                "SIGMA t0={} offset_s={} quoted_sigma={quoted:.6} quoted_raw={raw:.6} realized_back_60s={back:.6} realized_fwd_60s={fwd:.6} fwd_over_raw={fwd_over_raw:.2}",
                format_utc(*shock),
                offset / 1000
            );
        }
    }

    for (index, policy) in POLICIES.iter().enumerate() {
        for (period_index, (name, _, _)) in periods.iter().enumerate() {
            print_policy(policy, name, &books[index][period_index]);
        }
    }
    if board_mode {
        print_board_cap25(&selected, max_m);
        for (index, policy) in POLICIES.iter().enumerate() {
            print_horizon_policy_line(policy, "all", &books[index][0]);
            print_horizon_policy_line(policy, "near", &near_books[index]);
            print_horizon_policy_line(policy, "far", &far_books[index]);
        }
        print_window_buckets(&window_bands);
    }
    print_outstanding_exposure(&combined_intervals, &per_policy_intervals);
    print_reprice_and_triggers(cfg, args, &report.reprice_stats, &periods);

    let voided = &report.voided_window_starts;
    for hour in hour_starts(from_ms, to_ms) {
        let book = hour_book.get(&hour).cloned().unwrap_or_default();
        let hour_end = hour.saturating_add(3_600_000).min(to_ms);
        let max_k = Some(max_tripwire(ticks, quotes, hour, hour_end));
        print_book(
            "hour",
            hour,
            hour_end.saturating_sub(hour),
            &book,
            quotes,
            ticks,
            voided,
            max_k,
        );
    }
    if let (Some(start), Some(end)) = (args.minute_from, args.minute_to) {
        for minute in minute_starts(start, end) {
            let book = minute_book.get(&minute).cloned().unwrap_or_default();
            print_book("minute", minute, 60_000, &book, quotes, ticks, voided, None);
        }
    }
    println!(
        "note: quotes cached every 5s (or 1 Hz inside reprice range); production reprices every 100ms. exposure skew and safety factor are off. shock policies use the 10s move at quote time. a gap over 3s inside a 5s window is a void."
    );
}

fn main() {
    let args = parse_args();
    let ticker = args
        .coin
        .split(':')
        .next_back()
        .unwrap_or(&args.coin)
        .to_string();
    let mut cfg = stock_grid_config_for_ticker(&ticker, args.band_cents, Vec::new());
    apply(&mut cfg, &args);
    let cur_shift = cfg.pricing.current_row_logit_shift;
    let adj_shift = cfg.pricing.adjacent_row_logit_shift;
    let mut ticks = load_ticks(&args);

    let mut opts = CalibrateOptions::standard(ticker.clone(), args.band_cents, cfg.clone());
    opts.reprice_every_tick = args.tick_reprice;
    opts.reprice_from_ms = args.reprice_from;
    opts.reprice_to_ms = args.reprice_to;

    if args.bench_only {
        if let (Some(reprice_from), Some(reprice_to)) = (args.reprice_from, args.reprice_to) {
            let warmup_start = reprice_from.saturating_sub(600_000);
            ticks.retain(|t| t.ts_ms >= warmup_start && t.ts_ms <= reprice_to);
        }
        opts.all_windows = args.windows.is_some();
        opts.skip_eval = true;
        let report = run_calibration(&ticks, &opts);
        let periods: Vec<(String, u64, u64)> = args
            .periods
            .iter()
            .map(|p| (p.name.clone(), p.from_ms, p.to_ms))
            .collect();
        println!(
            "ticker={ticker} band={} windows={} far_refresh_ms={} mc_n={} bench_only=true ticks={}",
            args.band_cents,
            cfg.simulation.quote_window_count,
            cfg.simulation.far_refresh_ms,
            args.mc_n,
            ticks.len()
        );
        print_reprice_and_triggers(&cfg, &args, &report.reprice_stats, &periods);
        return;
    }

    let is_event = event_mode(&args);
    let mut legacy_per_policy_starts: Option<Vec<Vec<u64>>> = None;
    let mut event_precomputed_starts: Option<(Vec<u64>, Vec<u64>, Vec<u64>)> = None;

    if args.windows.is_some() {
        opts.all_windows = true;
        if is_event {
            if ticks.len() >= 2 {
                let file_first = ticks[0].ts_ms;
                let file_last = ticks[ticks.len() - 1].ts_ms;
                let from_ms = args.from_utc.unwrap_or(file_first);
                let to_ms = args.to_utc.unwrap_or(file_last.saturating_add(1));
                let first_non_warmup = file_first
                    .saturating_add(WARMUP_MS)
                    .div_ceil(WINDOW_MS)
                    .saturating_mul(WINDOW_MS);
                let mut schedule_rng = StdRng::seed_from_u64(args.seed);
                let normal = normal_session_starts(
                    first_non_warmup,
                    file_last,
                    from_ms,
                    to_ms,
                    WATCH_MS,
                    BURST_MS,
                    RETURN_MIN_MS,
                    RETURN_MAX_MS,
                    args.sessions,
                    &mut schedule_rng,
                );
                let dense = match (args.dense_from, args.dense_to) {
                    (Some(start), Some(end)) => {
                        dense_session_starts(start, end, args.dense_step_ms, BURST_MS, file_last)
                    }
                    _ => Vec::new(),
                };
                let merged = merge_starts(normal.clone(), dense.clone());
                opts.session_starts_ms = Some(merged.clone());
                opts.session_burst_ms = BURST_MS;
                opts.include_minute_quotes = true;
                event_precomputed_starts = Some((normal, dense, merged));
            }
        } else {
            let (per_policy, all_starts) = precompute_legacy_board_starts(&ticks, &args);
            opts.session_starts_ms = Some(all_starts);
            opts.session_burst_ms = BURST_MS;
            legacy_per_policy_starts = Some(per_policy);
        }
    }

    let report = run_calibration(&ticks, &opts);
    let taps = build_taps(&report, &ticks);
    println!(
        "ticker={ticker} band={} windows={} cur_shift={cur_shift} adj_shift={adj_shift} samples={}",
        args.band_cents,
        cfg.simulation.quote_window_count,
        taps.len()
    );
    if is_event {
        let quotes = quotes_from(&report);
        run_event(
            &args,
            &cfg,
            &taps,
            &ticks,
            &quotes,
            &report,
            event_precomputed_starts,
        );
    } else if let Some(per_policy_starts) = legacy_per_policy_starts {
        run_legacy_board(&args, &cfg, &taps, &per_policy_starts, &report);
    } else {
        run_legacy(&args, &cfg, &taps, &report);
    }
}
