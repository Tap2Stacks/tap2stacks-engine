use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use st_engine::calibrate::{
    CalibrateOptions, CalibrationSample, brier_score, implied_house_edge, reliability_table,
    run_calibration, tail_reliability_table,
};
use st_engine::grid::{CalibrationKey, GridConfig, stock_grid_config_for_ticker};
use st_engine::horizon::{
    CAP25_EPS, MIN_CALIBRATION_COUNT, NEAR_WINDOWS, apply_factor_table, factor_from_counts,
    parse_factor_table,
};
use st_engine::price::{PriceSource, PriceTick, deribit_ticker_symbol, read_ticks_for_source};
use st_engine::v2::{PricingModel, load_v2_params_file, v2_ticker_params_for_source};

struct CliArgs {
    files: Vec<PathBuf>,
    coin: String,
    source: PriceSource,
    band_cents: u32,
    model: PricingModel,
    v2_params_file: Option<PathBuf>,
    dump_v2_params: bool,
    distance_factors: [Option<f64>; 6],
    sigma_scale: Option<f64>,
    sigma_ref: Option<f64>,
    ewma_half_life: Option<f64>,
    p_raw_gamma: Option<f64>,
    p_time_boost: Option<f64>,
    current_shift: Option<f64>,
    adjacent_shift: Option<f64>,
    zone_vol_scale: Option<f64>,
    m_base: Option<f64>,
    off_hours_m_base: Option<f64>,
    m_risk_max: Option<f64>,
    a_sigma: Option<f64>,
    jump_enabled: Option<bool>,
    bridge_epsilon: Option<f64>,
    variance_floor: Option<f64>,
    mc_n: Option<usize>,
    seed: Option<u64>,
    windows: Option<usize>,
    quote_step_secs: Option<u64>,
    clear_factors: bool,
    factors_file: Option<PathBuf>,
    dump_counts: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            coin: String::new(),
            source: PriceSource::Hyperliquid,
            band_cents: 0,
            model: PricingModel::V1,
            v2_params_file: None,
            dump_v2_params: false,
            distance_factors: [None; 6],
            sigma_scale: None,
            sigma_ref: None,
            ewma_half_life: None,
            p_raw_gamma: None,
            p_time_boost: None,
            current_shift: None,
            adjacent_shift: None,
            zone_vol_scale: None,
            m_base: None,
            off_hours_m_base: None,
            m_risk_max: None,
            a_sigma: None,
            jump_enabled: None,
            bridge_epsilon: None,
            variance_floor: None,
            mc_n: None,
            seed: Some(42),
            windows: None,
            quote_step_secs: None,
            clear_factors: false,
            factors_file: None,
            dump_counts: false,
        }
    }
}

fn apply_arg(out: &mut CliArgs, key: &str, val: &str) -> Result<(), String> {
    match key {
        "--file" => out.files.push(PathBuf::from(val)),
        "--coin" => out.coin = val.to_string(),
        "--source" => {
            out.source = PriceSource::parse(val)
                .ok_or_else(|| "bad --source (expected hyperliquid or deribit)".to_string())?;
        }
        "--band-cents" => out.band_cents = val.parse().map_err(|_| "bad --band-cents")?,
        "--distance-factor-1"
        | "--distance-factor-2"
        | "--distance-factor-3"
        | "--distance-factor-4"
        | "--distance-factor-5"
        | "--distance-factor-6" => {
            let idx = key.as_bytes()[key.len() - 1] as usize - b'1' as usize;
            let factor: f64 = val.parse().map_err(|_| "bad distance factor")?;
            if !factor.is_finite() || factor <= 0.0 {
                return Err("distance factor must be finite and positive".into());
            }
            out.distance_factors[idx] = Some(factor);
        }
        "--sigma-scale" => out.sigma_scale = Some(val.parse().map_err(|_| "bad --sigma-scale")?),
        "--sigma-ref" => out.sigma_ref = Some(val.parse().map_err(|_| "bad --sigma-ref")?),
        "--ewma-half-life" => {
            out.ewma_half_life = Some(val.parse().map_err(|_| "bad --ewma-half-life")?)
        }
        "--p-raw-gamma" => out.p_raw_gamma = Some(val.parse().map_err(|_| "bad --p-raw-gamma")?),
        "--p-time-boost" => out.p_time_boost = Some(val.parse().map_err(|_| "bad --p-time-boost")?),
        "--current-shift" => {
            out.current_shift = Some(val.parse().map_err(|_| "bad --current-shift")?)
        }
        "--adjacent-shift" => {
            out.adjacent_shift = Some(val.parse().map_err(|_| "bad --adjacent-shift")?)
        }
        "--zone-vol-scale" => {
            out.zone_vol_scale = Some(val.parse().map_err(|_| "bad --zone-vol-scale")?)
        }
        "--m-base" => out.m_base = Some(val.parse().map_err(|_| "bad --m-base")?),
        "--off-hours-m-base" => {
            out.off_hours_m_base = Some(val.parse().map_err(|_| "bad --off-hours-m-base")?)
        }
        "--m-risk-max" => out.m_risk_max = Some(val.parse().map_err(|_| "bad --m-risk-max")?),
        "--a-sigma" => out.a_sigma = Some(val.parse().map_err(|_| "bad --a-sigma")?),
        "--jump-enabled" => out.jump_enabled = Some(val.parse().map_err(|_| "bad --jump-enabled")?),
        "--bridge-epsilon" => {
            out.bridge_epsilon = Some(val.parse().map_err(|_| "bad --bridge-epsilon")?)
        }
        "--variance-floor" => {
            out.variance_floor = Some(val.parse().map_err(|_| "bad --variance-floor")?)
        }
        "--mc-n" => out.mc_n = Some(val.parse().map_err(|_| "bad --mc-n")?),
        "--seed" => out.seed = Some(val.parse().map_err(|_| "bad --seed")?),
        "--windows" => {
            let n: usize = val.parse().map_err(|_| "bad --windows")?;
            if n == 0 {
                return Err("--windows must be >= 1".into());
            }
            out.windows = Some(n);
        }
        "--quote-step-secs" => {
            let secs: u64 = val.parse().map_err(|_| "bad --quote-step-secs")?;
            if secs == 0 {
                return Err("--quote-step-secs must be >= 1".into());
            }
            out.quote_step_secs = Some(secs);
        }
        "--clear-factors" => {
            out.clear_factors = val.parse().map_err(|_| "bad --clear-factors")?;
        }
        "--factors-file" => out.factors_file = Some(PathBuf::from(val)),
        "--dump-counts" => {
            out.dump_counts = val.parse().map_err(|_| "bad --dump-counts")?;
        }
        "--model" => {
            out.model = PricingModel::parse(val)
                .ok_or_else(|| "bad --model (expected v1, v2, or m4)".to_string())?;
        }
        "--v2-params-file" => out.v2_params_file = Some(PathBuf::from(val)),
        "--dump-v2-params" => {
            out.dump_v2_params = val.parse().map_err(|_| "bad --dump-v2-params")?;
        }
        _ => return Err(format!("unknown arg: {key}")),
    }
    Ok(())
}

fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut out = CliArgs::default();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "--dump-v2-params" {
            if let Some(next) = args.get(i + 1) {
                if next == "true" || next == "false" {
                    out.dump_v2_params = next == "true";
                    i += 2;
                    continue;
                }
            }
            out.dump_v2_params = true;
            i += 1;
            continue;
        }
        let val = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        apply_arg(&mut out, key, val)?;
        i += 2;
    }
    if out.dump_v2_params {
        if out.coin.is_empty() {
            return Err(
                "usage: calibrate --dump-v2-params --coin <coin> [--source hyperliquid|deribit]"
                    .into(),
            );
        }
        return Ok(out);
    }
    if out.files.is_empty() || out.coin.is_empty() || out.band_cents == 0 {
        return Err(
            "usage: calibrate --file <jsonl> --coin <coin> --band-cents <n> [--source hyperliquid|deribit] [--model v1|v2|m4] [--v2-params-file <path.json>] [--dump-v2-params]"
                .into(),
        );
    }
    Ok(out)
}

fn apply_overrides(cfg: &mut GridConfig, cli: &CliArgs) -> Result<(), String> {
    cfg.simulation.price_source = cli.source;
    cfg.simulation.pricing_model = cli.model;
    if let Some(windows) = cli.windows {
        cfg.simulation.quote_window_count = windows;
    }
    if cli.clear_factors {
        cfg.calibration.factors.clear();
    }
    if let Some(ref path) = cli.factors_file {
        let text = fs::read_to_string(path)
            .map_err(|err| format!("failed reading factors file {}: {err}", path.display()))?;
        let entries = parse_factor_table(&text)?;
        apply_factor_table(cfg, &entries);
    }
    let factor_windows = cfg.simulation.quote_window_count.max(6);
    for (index, factor) in cli.distance_factors.iter().enumerate() {
        if let Some(value) = factor {
            for window_index in 0..factor_windows {
                for sigma_bucket in 0..3 {
                    cfg.calibration.factors.insert(
                        CalibrationKey {
                            window_index,
                            dist_bucket: index + 1,
                            sigma_bucket,
                        },
                        *value,
                    );
                }
            }
        }
    }
    if let Some(v) = cli.sigma_scale {
        cfg.volatility.sigma_scale = v;
    }
    if let Some(v) = cli.sigma_ref {
        cfg.volatility.sigma_ref = v;
    }
    if let Some(v) = cli.ewma_half_life {
        cfg.volatility.ewma_half_life_seconds = v;
    }
    if let Some(v) = cli.p_raw_gamma {
        cfg.pricing.p_raw_gamma = v;
    }
    if let Some(v) = cli.p_time_boost {
        cfg.pricing.p_time_boost = v;
    }
    if let Some(v) = cli.current_shift {
        cfg.pricing.current_row_logit_shift = v;
    }
    if let Some(v) = cli.adjacent_shift {
        cfg.pricing.adjacent_row_logit_shift = v;
    }
    if let Some(v) = cli.zone_vol_scale {
        cfg.pricing.zone_vol_shift_scale = v;
    }
    if let Some(v) = cli.m_base {
        cfg.pricing.m_base = v;
    }
    if let Some(v) = cli.off_hours_m_base {
        cfg.pricing.off_hours_m_base = Some(v);
    }
    if let Some(v) = cli.m_risk_max {
        cfg.pricing.m_risk_max = v;
    }
    if let Some(v) = cli.a_sigma {
        cfg.pricing.a_sigma = v;
    }
    if let Some(v) = cli.jump_enabled {
        cfg.volatility.jump.enabled = v;
    }
    if let Some(v) = cli.bridge_epsilon {
        cfg.simulation.bridge_epsilon = v;
    }
    if let Some(v) = cli.variance_floor {
        cfg.volatility.variance_floor = v;
    }
    if let Some(v) = cli.mc_n {
        cfg.simulation.mc_n_min = v;
        cfg.simulation.mc_n_max = v;
    }
    cfg.simulation.seed = cli.seed;
    Ok(())
}

fn load_ticks(cli: &CliArgs) -> Vec<PriceTick> {
    let mut ticks = Vec::new();
    for path in &cli.files {
        let mut part = read_ticks_for_source(path, &cli.coin, cli.source).unwrap_or_else(|e| {
            eprintln!("failed reading {}: {e}", path.display());
            std::process::exit(1);
        });
        ticks.append(&mut part);
    }
    ticks.sort_by_key(|t| t.ts_ms);
    ticks
}

fn print_section(label: &str, subset: &[&CalibrationSample], max_multiplier: f64) {
    if subset.is_empty() {
        println!("\n=== {label} (n=0) ===");
        return;
    }
    let pairs: Vec<(f64, bool)> = subset.iter().map(|s| (s.predicted, s.hit)).collect();
    let edge_tuples: Vec<(f64, bool, f64)> = subset
        .iter()
        .map(|s| (s.predicted, s.hit, s.multiplier))
        .collect();
    let unclamped: Vec<(f64, bool, f64)> = subset
        .iter()
        .filter(|s| s.multiplier < max_multiplier - CAP25_EPS)
        .map(|s| (s.predicted, s.hit, s.multiplier))
        .collect();
    let table = reliability_table(&pairs, 10);
    let brier = brier_score(&pairs);
    let edge = implied_house_edge(&edge_tuples);
    let unclamped_edge = implied_house_edge(&unclamped);
    let worst_gap = table
        .iter()
        .filter(|b| b.count >= 200)
        .map(|b| (b.mean_predicted - b.hit_rate).abs())
        .fold(0.0_f64, f64::max);
    let worst_gap_100 = table
        .iter()
        .filter(|b| b.count >= 100)
        .map(|b| (b.mean_predicted - b.hit_rate).abs())
        .fold(0.0_f64, f64::max);

    println!(
        "\n=== {label} (samples={}, unclamped={}, brier={brier:.6}, edge={:.2}%, unclamped_edge={:.2}%, worst_gap_200={:.2}pp, worst_gap_100={:.2}pp) ===",
        subset.len(),
        unclamped.len(),
        edge * 100.0,
        unclamped_edge * 100.0,
        worst_gap * 100.0,
        worst_gap_100 * 100.0
    );
    println!("bucket       |   count | mean_pred |  hit_rate |   diff_pp");
    for b in &table {
        if b.count == 0 {
            continue;
        }
        println!(
            "[{:.1}, {:.1}{}| {:7} | {:9.4} | {:9.4} | {:+8.2}",
            b.lo,
            b.hi,
            if (b.hi - 1.0).abs() < 1e-9 { "]" } else { ")" },
            b.count,
            b.mean_predicted,
            b.hit_rate,
            (b.mean_predicted - b.hit_rate) * 100.0
        );
    }

    print_distance_slice("dist 0.30-0.50", subset, |sample| {
        (0.3..0.5).contains(&sample.predicted)
    });
    print_distance_slice("dist tail 0.05-0.10", subset, |sample| {
        (0.05..0.10).contains(&sample.predicted)
    });

    let tail = tail_reliability_table(&edge_tuples);
    println!("tail p<0.1     |   count | mean_pred |  hit_rate | hit/pred |   edge_%");
    for t in &tail {
        if t.count == 0 {
            continue;
        }
        println!(
            "[{:.3}, {:.3}) | {:7} | {:9.5} | {:9.5} | {:8.3} | {:+7.2}%",
            t.lo,
            t.hi,
            t.count,
            t.mean_predicted,
            t.hit_rate,
            t.hit_to_pred_ratio,
            t.implied_edge * 100.0
        );
    }
}

fn print_distance_slice(
    label: &str,
    subset: &[&CalibrationSample],
    include: impl Fn(&CalibrationSample) -> bool,
) {
    let mut counts = [0usize; 11];
    let mut predicted = [0.0f64; 11];
    let mut hits = [0.0f64; 11];
    for sample in subset.iter().copied().filter(|sample| include(sample)) {
        let dist = sample.center_distance.min(10);
        counts[dist] += 1;
        predicted[dist] += sample.predicted;
        if sample.hit {
            hits[dist] += 1.0;
        }
    }
    println!("{label} | dist | count | mean_pred | hit_rate | diff_pp");
    for dist in 0..=10 {
        if counts[dist] < 50 {
            continue;
        }
        let n = counts[dist] as f64;
        let mean = predicted[dist] / n;
        let hit = hits[dist] / n;
        println!(
            "{label} | {dist:4} | {count:7} | {mean:9.4} | {hit:8.4} | {diff:+7.2}",
            count = counts[dist],
            diff = (mean - hit) * 100.0,
        );
    }
}

#[derive(Default, Clone, Copy)]
struct BucketAcc {
    hits: usize,
    count: usize,
    pred_sum: f64,
    cap25: usize,
    payout_sum: f64,
}

impl BucketAcc {
    fn add(&mut self, sample: &CalibrationSample, max_m: f64) {
        self.count += 1;
        self.pred_sum += sample.predicted;
        if sample.hit {
            self.hits += 1;
            self.payout_sum += sample.multiplier;
        }
        if sample.multiplier >= max_m - CAP25_EPS {
            self.cap25 += 1;
        }
    }
}

fn print_horizon_counts(samples: &[CalibrationSample], max_m: f64) {
    let mut by_wd: BTreeMap<(usize, usize), BucketAcc> = BTreeMap::new();
    let mut by_window: BTreeMap<usize, BucketAcc> = BTreeMap::new();
    let mut near = BucketAcc::default();
    let mut far = BucketAcc::default();

    for sample in samples.iter().filter(|s| !s.is_warmup) {
        by_wd
            .entry((sample.window_index, sample.dist_bucket))
            .or_default()
            .add(sample, max_m);
        by_window
            .entry(sample.window_index)
            .or_default()
            .add(sample, max_m);
        if sample.window_index < NEAR_WINDOWS {
            near.add(sample, max_m);
        } else {
            far.add(sample, max_m);
        }
    }

    for ((window, dist), acc) in &by_wd {
        let factor = factor_from_counts(acc.hits, acc.count, acc.pred_sum, MIN_CALIBRATION_COUNT)
            .unwrap_or(1.0);
        println!(
            "FACTOR_COUNT window={window} dist={dist} hits={} count={} pred_sum={:.6} factor={:.6} cap25={} payout_sum={:.6}",
            acc.hits, acc.count, acc.pred_sum, factor, acc.cap25, acc.payout_sum
        );
    }
    for (window, acc) in &by_window {
        if acc.count == 0 {
            continue;
        }
        let n = acc.count as f64;
        let mean_pred = acc.pred_sum / n;
        let hit_rate = acc.hits as f64 / n;
        let factor = factor_from_counts(acc.hits, acc.count, acc.pred_sum, MIN_CALIBRATION_COUNT)
            .unwrap_or(1.0);
        let edge_pct = (1.0 - acc.payout_sum / n) * 100.0;
        let cap25_pct = acc.cap25 as f64 / n * 100.0;
        println!(
            "WINDOW_CALIB window={window} samples={} mean_pred={mean_pred:.5} hit_rate={hit_rate:.5} factor={factor:.4} edge_pct={edge_pct:.2} cap25_pct={cap25_pct:.2}",
            acc.count
        );
    }
    for (label, acc) in [("0-5", near), ("6-23", far)] {
        if acc.count == 0 {
            continue;
        }
        let n = acc.count as f64;
        let mean_pred = acc.pred_sum / n;
        let hit_rate = acc.hits as f64 / n;
        let factor = factor_from_counts(acc.hits, acc.count, acc.pred_sum, MIN_CALIBRATION_COUNT)
            .unwrap_or(1.0);
        let edge_pct = (1.0 - acc.payout_sum / n) * 100.0;
        let cap25_pct = acc.cap25 as f64 / n * 100.0;
        println!(
            "HORIZON_GROUP group={label} samples={} mean_pred={mean_pred:.5} hit_rate={hit_rate:.5} factor={factor:.4} edge_pct={edge_pct:.2} cap25_pct={cap25_pct:.2}",
            acc.count
        );
    }
}

fn print_report(cli: &CliArgs, cfg: &GridConfig, ticker: &str, ticks: &[PriceTick]) {
    println!("distance_factor_overrides={:?}", cli.distance_factors);
    println!(
        "coin={} ticker={} band_cents={} windows={} clear_factors={} ticks={} sigma_scale={} sigma_ref={} ewma_hl={} p_raw_gamma={} p_time_boost={} cur_shift={} adj_shift={} zone_vol={} m_base={} off_m_base={:?} m_risk_max={} a_sigma={} jump={} bridge_eps={:e} var_floor={:e}",
        cli.coin,
        ticker,
        cli.band_cents,
        cfg.simulation.quote_window_count,
        cli.clear_factors,
        ticks.len(),
        cfg.volatility.sigma_scale,
        cfg.volatility.sigma_ref,
        cfg.volatility.ewma_half_life_seconds,
        cfg.pricing.p_raw_gamma,
        cfg.pricing.p_time_boost,
        cfg.pricing.current_row_logit_shift,
        cfg.pricing.adjacent_row_logit_shift,
        cfg.pricing.zone_vol_shift_scale,
        cfg.pricing.m_base,
        cfg.pricing.off_hours_m_base,
        cfg.pricing.m_risk_max,
        cfg.pricing.a_sigma,
        cfg.volatility.jump.enabled,
        cfg.simulation.bridge_epsilon,
        cfg.volatility.variance_floor,
    );
    let mut opts = CalibrateOptions::standard(ticker.to_string(), cli.band_cents, cfg.clone());
    if cli.windows.is_some() {
        opts.all_windows = true;
    }
    if let Some(secs) = cli.quote_step_secs {
        opts.quote_step_ms = secs * 1_000;
    }
    let report = run_calibration(ticks, &opts);
    println!(
        "evaluated_windows={} voided_windows={} gaps_over_3s={}",
        report.evaluated_windows,
        report.voided_windows,
        report.gaps_over_3s.len()
    );
    for (from, to) in report
        .gaps_over_3s
        .iter()
        .filter(|(f, t)| t - f > 10_000)
        .take(20)
    {
        println!(
            "  major_gap: {from} -> {to} ({:.1}s)",
            (to - from) as f64 / 1000.0
        );
    }
    let max_m = cfg.pricing.max_multiplier;
    if cli.dump_counts {
        print_horizon_counts(&report.samples, max_m);
    }
    let us_steady: Vec<_> = report
        .samples
        .iter()
        .filter(|s| s.is_us_session && !s.is_warmup)
        .collect();
    let us_warmup: Vec<_> = report
        .samples
        .iter()
        .filter(|s| s.is_us_session && s.is_warmup)
        .collect();
    let off_steady: Vec<_> = report
        .samples
        .iter()
        .filter(|s| !s.is_us_session && !s.is_warmup)
        .collect();
    let off_warmup: Vec<_> = report
        .samples
        .iter()
        .filter(|s| !s.is_us_session && s.is_warmup)
        .collect();
    let all_samples: Vec<_> = report.samples.iter().collect();

    print_section(
        "US REGULAR (09:30-16:00 NY / 14:30-21:00 EST) STEADY (>5m)",
        &us_steady,
        max_m,
    );
    print_section(
        "US REGULAR (09:30-16:00 NY / 14:30-21:00 EST) WARMUP (0-5m)",
        &us_warmup,
        max_m,
    );
    print_section("OFF-HOURS STEADY (>5m)", &off_steady, max_m);
    print_section("OFF-HOURS WARMUP (0-5m)", &off_warmup, max_m);
    print_section("ALL COMBINED", &all_samples, max_m);
}

fn main() {
    let cli = parse_args().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    if cli.dump_v2_params {
        let Some(params) = v2_ticker_params_for_source(&cli.coin, cli.source) else {
            eprintln!(
                "no compiled v2 params for coin '{}' on source '{}'",
                cli.coin,
                cli.source.as_str()
            );
            std::process::exit(2);
        };
        let json = params.to_json_pretty().unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(2);
        });
        println!("{json}");
        return;
    }
    let ticks = load_ticks(&cli);
    let ticker = match cli.source {
        PriceSource::Deribit => deribit_ticker_symbol(&cli.coin),
        PriceSource::Hyperliquid => cli
            .coin
            .split(':')
            .next_back()
            .unwrap_or(&cli.coin)
            .to_string(),
    };
    let mut cfg = stock_grid_config_for_ticker(&ticker, cli.band_cents, Vec::new());
    if let Err(err) = apply_overrides(&mut cfg, &cli) {
        eprintln!("{err}");
        std::process::exit(2);
    }
    if let Some(ref path) = cli.v2_params_file {
        let loaded = load_v2_params_file(path).unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(2);
        });
        if !loaded.matches_coin(&cli.coin) {
            eprintln!(
                "v2 params file {} has ticker '{}', which does not match --coin '{}'",
                path.display(),
                loaded.ticker,
                cli.coin
            );
            std::process::exit(2);
        }
        if loaded.band_cents != cli.band_cents {
            eprintln!(
                "warning: --v2-params-file {} specifies band_cents={}, differing from --band-cents={}; using --band-cents={}",
                path.display(),
                loaded.band_cents,
                cli.band_cents,
                cli.band_cents
            );
        }
        let leaked = loaded.leak_static();
        println!("v2_params_file={} {}", path.display(), leaked.one_line_summary());
        cfg.simulation.v2_params_override = Some(leaked);
    }
    print_report(&cli, &cfg, &ticker, &ticks);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_override_applies_to_all_windows_and_sigma_buckets() {
        let mut cli = CliArgs::default();
        apply_arg(&mut cli, "--distance-factor-6", "1.2").unwrap();
        let mut cfg = stock_grid_config_for_ticker("NVDA", 1, Vec::new());
        apply_overrides(&mut cfg, &cli).unwrap();
        for window_index in 0..6 {
            for sigma_bucket in 0..3 {
                let key = CalibrationKey {
                    window_index,
                    dist_bucket: 6,
                    sigma_bucket,
                };
                assert_eq!(cfg.calibration.factors.get(&key), Some(&1.2));
            }
        }
    }

    #[test]
    fn distance_override_rejects_nonpositive_and_nonfinite_factors() {
        let mut cli = CliArgs::default();
        for value in ["0", "-1", "NaN", "inf"] {
            assert!(apply_arg(&mut cli, "--distance-factor-1", value).is_err());
        }
    }

    #[test]
    fn horizon_flags_configure_windows_step_and_clear_factors() {
        let mut cli = CliArgs::default();
        apply_arg(&mut cli, "--windows", "24").unwrap();
        apply_arg(&mut cli, "--quote-step-secs", "30").unwrap();
        apply_arg(&mut cli, "--clear-factors", "true").unwrap();
        apply_arg(&mut cli, "--dump-counts", "true").unwrap();
        let mut cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        assert!(!cfg.calibration.factors.is_empty());
        apply_overrides(&mut cfg, &cli).unwrap();
        assert_eq!(cfg.simulation.quote_window_count, 24);
        assert!(cfg.calibration.factors.is_empty());
        assert_eq!(cli.quote_step_secs, Some(30));
        assert!(cli.dump_counts);
        assert!(apply_arg(&mut cli, "--windows", "0").is_err());
        assert!(apply_arg(&mut cli, "--quote-step-secs", "0").is_err());
    }

    #[test]
    fn source_flag_defaults_to_hyperliquid_and_accepts_deribit() {
        let mut cli = CliArgs::default();
        assert_eq!(cli.source, PriceSource::Hyperliquid);
        apply_arg(&mut cli, "--source", "deribit").unwrap();
        assert_eq!(cli.source, PriceSource::Deribit);
        assert!(apply_arg(&mut cli, "--source", "unknown").is_err());
        let mut cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        apply_overrides(&mut cfg, &cli).unwrap();
        assert_eq!(cfg.simulation.price_source, PriceSource::Deribit);

        apply_arg(&mut cli, "--model", "v2").unwrap();
        apply_arg(&mut cli, "--v2-params-file", "/tmp/nvda.json").unwrap();
        apply_arg(&mut cli, "--dump-v2-params", "true").unwrap();
        apply_overrides(&mut cfg, &cli).unwrap();
        assert_eq!(cfg.simulation.pricing_model, PricingModel::V2);
        assert_eq!(cli.v2_params_file, Some(PathBuf::from("/tmp/nvda.json")));
        assert!(cli.dump_v2_params);
        assert!(apply_arg(&mut cli, "--model", "v9").is_err());
    }
}
