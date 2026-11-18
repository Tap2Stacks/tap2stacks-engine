use std::env;
use std::path::PathBuf;

use st_engine::calibrate::{
    CalibrateOptions, CalibrationSample, brier_score, implied_house_edge, reliability_table,
    run_calibration, tail_reliability_table,
};
use st_engine::grid::{GridConfig, stock_grid_config_for_ticker};
use st_engine::price::{PriceTick, read_hl_jsonl};

struct CliArgs {
    files: Vec<PathBuf>,
    coin: String,
    band_cents: u32,
    sigma_scale: Option<f64>,
    sigma_ref: Option<f64>,
    ewma_half_life: Option<f64>,
    p_raw_gamma: Option<f64>,
    p_time_boost: Option<f64>,
    current_shift: Option<f64>,
    adjacent_shift: Option<f64>,
    zone_vol_scale: Option<f64>,
    m_base: Option<f64>,
    m_risk_max: Option<f64>,
    a_sigma: Option<f64>,
    jump_enabled: Option<bool>,
    bridge_epsilon: Option<f64>,
    variance_floor: Option<f64>,
    mc_n: Option<usize>,
    seed: Option<u64>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            coin: String::new(),
            band_cents: 0,
            sigma_scale: None,
            sigma_ref: None,
            ewma_half_life: None,
            p_raw_gamma: None,
            p_time_boost: None,
            current_shift: None,
            adjacent_shift: None,
            zone_vol_scale: None,
            m_base: None,
            m_risk_max: None,
            a_sigma: None,
            jump_enabled: None,
            bridge_epsilon: None,
            variance_floor: None,
            mc_n: None,
            seed: Some(42),
        }
    }
}

fn apply_arg(out: &mut CliArgs, key: &str, val: &str) -> Result<(), String> {
    match key {
        "--file" => out.files.push(PathBuf::from(val)),
        "--coin" => out.coin = val.to_string(),
        "--band-cents" => out.band_cents = val.parse().map_err(|_| "bad --band-cents")?,
        "--sigma-scale" => out.sigma_scale = Some(val.parse().map_err(|_| "bad --sigma-scale")?),
        "--sigma-ref" => out.sigma_ref = Some(val.parse().map_err(|_| "bad --sigma-ref")?),
        "--ewma-half-life" => out.ewma_half_life = Some(val.parse().map_err(|_| "bad --ewma-half-life")?),
        "--p-raw-gamma" => out.p_raw_gamma = Some(val.parse().map_err(|_| "bad --p-raw-gamma")?),
        "--p-time-boost" => out.p_time_boost = Some(val.parse().map_err(|_| "bad --p-time-boost")?),
        "--current-shift" => out.current_shift = Some(val.parse().map_err(|_| "bad --current-shift")?),
        "--adjacent-shift" => out.adjacent_shift = Some(val.parse().map_err(|_| "bad --adjacent-shift")?),
        "--zone-vol-scale" => out.zone_vol_scale = Some(val.parse().map_err(|_| "bad --zone-vol-scale")?),
        "--m-base" => out.m_base = Some(val.parse().map_err(|_| "bad --m-base")?),
        "--m-risk-max" => out.m_risk_max = Some(val.parse().map_err(|_| "bad --m-risk-max")?),
        "--a-sigma" => out.a_sigma = Some(val.parse().map_err(|_| "bad --a-sigma")?),
        "--jump-enabled" => out.jump_enabled = Some(val.parse().map_err(|_| "bad --jump-enabled")?),
        "--bridge-epsilon" => out.bridge_epsilon = Some(val.parse().map_err(|_| "bad --bridge-epsilon")?),
        "--variance-floor" => out.variance_floor = Some(val.parse().map_err(|_| "bad --variance-floor")?),
        "--mc-n" => out.mc_n = Some(val.parse().map_err(|_| "bad --mc-n")?),
        "--seed" => out.seed = Some(val.parse().map_err(|_| "bad --seed")?),
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
        let val = args.get(i + 1).ok_or_else(|| format!("missing value for {key}"))?;
        apply_arg(&mut out, key, val)?;
        i += 2;
    }
    if out.files.is_empty() || out.coin.is_empty() || out.band_cents == 0 {
        return Err("usage: calibrate --file <jsonl> --coin <coin> --band-cents <n>".into());
    }
    Ok(out)
}

fn apply_overrides(cfg: &mut GridConfig, cli: &CliArgs) {
    if let Some(v) = cli.sigma_scale { cfg.volatility.sigma_scale = v; }
    if let Some(v) = cli.sigma_ref { cfg.volatility.sigma_ref = v; }
    if let Some(v) = cli.ewma_half_life { cfg.volatility.ewma_half_life_seconds = v; }
    if let Some(v) = cli.p_raw_gamma { cfg.pricing.p_raw_gamma = v; }
    if let Some(v) = cli.p_time_boost { cfg.pricing.p_time_boost = v; }
    if let Some(v) = cli.current_shift { cfg.pricing.current_row_logit_shift = v; }
    if let Some(v) = cli.adjacent_shift { cfg.pricing.adjacent_row_logit_shift = v; }
    if let Some(v) = cli.zone_vol_scale { cfg.pricing.zone_vol_shift_scale = v; }
    if let Some(v) = cli.m_base { cfg.pricing.m_base = v; }
    if let Some(v) = cli.m_risk_max { cfg.pricing.m_risk_max = v; }
    if let Some(v) = cli.a_sigma { cfg.pricing.a_sigma = v; }
    if let Some(v) = cli.jump_enabled { cfg.volatility.jump.enabled = v; }
    if let Some(v) = cli.bridge_epsilon { cfg.simulation.bridge_epsilon = v; }
    if let Some(v) = cli.variance_floor { cfg.volatility.variance_floor = v; }
    if let Some(v) = cli.mc_n {
        cfg.simulation.mc_n_min = v;
        cfg.simulation.mc_n_max = v;
    }
    cfg.simulation.seed = cli.seed;
}

fn load_ticks(cli: &CliArgs) -> Vec<PriceTick> {
    let mut ticks = Vec::new();
    for path in &cli.files {
        let mut part = read_hl_jsonl(path, &cli.coin).unwrap_or_else(|e| {
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
    let edge_tuples: Vec<(f64, bool, f64)> =
        subset.iter().map(|s| (s.predicted, s.hit, s.multiplier)).collect();
    let unclamped: Vec<(f64, bool, f64)> = subset
        .iter()
        .filter(|s| s.multiplier < max_multiplier - 1e-6)
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

    println!(
        "\n=== {label} (samples={}, unclamped={}, brier={brier:.6}, edge={:.2}%, unclamped_edge={:.2}%, worst_gap_200={:.2}pp) ===",
        subset.len(),
        unclamped.len(),
        edge * 100.0,
        unclamped_edge * 100.0,
        worst_gap * 100.0
    );
    println!("bucket       |   count | mean_pred |  hit_rate |   diff_pp");
    for b in &table {
        if b.count == 0 { continue; }
        println!(
            "[{:.1}, {:.1}{}| {:7} | {:9.4} | {:9.4} | {:+8.2}",
            b.lo, b.hi, if (b.hi - 1.0).abs() < 1e-9 { "]" } else { ")" },
            b.count, b.mean_predicted, b.hit_rate, (b.mean_predicted - b.hit_rate) * 100.0
        );
    }

    let tail = tail_reliability_table(&edge_tuples);
    println!("tail p<0.1     |   count | mean_pred |  hit_rate | hit/pred |   edge_%");
    for t in &tail {
        if t.count == 0 { continue; }
        println!(
            "[{:.3}, {:.3}) | {:7} | {:9.5} | {:9.5} | {:8.3} | {:+7.2}%",
            t.lo, t.hi, t.count, t.mean_predicted, t.hit_rate, t.hit_to_pred_ratio, t.implied_edge * 100.0
        );
    }
}

fn print_report(cli: &CliArgs, cfg: &GridConfig, ticker: &str, ticks: &[PriceTick]) {
    println!(
        "coin={} ticker={} band_cents={} ticks={} sigma_scale={} sigma_ref={} ewma_hl={} p_raw_gamma={} p_time_boost={} cur_shift={} adj_shift={} zone_vol={} m_base={} m_risk_max={} a_sigma={} jump={} bridge_eps={:e} var_floor={:e}",
        cli.coin, ticker, cli.band_cents, ticks.len(), cfg.volatility.sigma_scale,
        cfg.volatility.sigma_ref, cfg.volatility.ewma_half_life_seconds, cfg.pricing.p_raw_gamma,
        cfg.pricing.p_time_boost, cfg.pricing.current_row_logit_shift,
        cfg.pricing.adjacent_row_logit_shift, cfg.pricing.zone_vol_shift_scale,
        cfg.pricing.m_base, cfg.pricing.m_risk_max, cfg.pricing.a_sigma,
        cfg.volatility.jump.enabled, cfg.simulation.bridge_epsilon, cfg.volatility.variance_floor,
    );
    let report = run_calibration(
        ticks,
        &CalibrateOptions {
            ticker: ticker.to_string(),
            band_cents: cli.band_cents,
            base_config: cfg.clone(),
            warmup_ms: 300_000,
            restart_gap_ms: 30_000,
        },
    );
    println!(
        "evaluated_windows={} voided_windows={} gaps_over_3s={}",
        report.evaluated_windows, report.voided_windows, report.gaps_over_3s.len()
    );
    for (from, to) in report.gaps_over_3s.iter().filter(|(f, t)| t - f > 10_000).take(20) {
        println!("  major_gap: {from} -> {to} ({:.1}s)", (to - from) as f64 / 1000.0);
    }
    let us_steady: Vec<_> = report.samples.iter().filter(|s| s.is_us_session && !s.is_warmup).collect();
    let us_warmup: Vec<_> = report.samples.iter().filter(|s| s.is_us_session && s.is_warmup).collect();
    let off_steady: Vec<_> = report.samples.iter().filter(|s| !s.is_us_session && !s.is_warmup).collect();
    let off_warmup: Vec<_> = report.samples.iter().filter(|s| !s.is_us_session && s.is_warmup).collect();
    let all_samples: Vec<_> = report.samples.iter().collect();

    let max_m = cfg.pricing.max_multiplier;
    print_section("US REGULAR (13:30-20:00 UTC) STEADY (>5m)", &us_steady, max_m);
    print_section("US REGULAR (13:30-20:00 UTC) WARMUP (0-5m)", &us_warmup, max_m);
    print_section("OFF-HOURS STEADY (>5m)", &off_steady, max_m);
    print_section("OFF-HOURS WARMUP (0-5m)", &off_warmup, max_m);
    print_section("ALL COMBINED", &all_samples, max_m);
}

fn main() {
    let cli = parse_args().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    let ticks = load_ticks(&cli);
    let ticker = cli.coin.split(':').next_back().unwrap_or(&cli.coin).to_string();
    let mut cfg = stock_grid_config_for_ticker(&ticker, cli.band_cents, Vec::new());
    apply_overrides(&mut cfg, &cli);
    print_report(&cli, &cfg, &ticker, &ticks);
}
