use std::env;
use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use st_engine::calibrate::{CalibrateOptions, run_calibration};
use st_engine::grid::{GridConfig, stock_grid_config_for_ticker};
use st_engine::price::{PriceTick, read_hl_jsonl};

const POLICIES: [&str; 6] = [
    "closest",
    "three_rows",
    "row_sweep",
    "long_bias",
    "short_bias",
    "random_near",
];

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
    };
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let key = raw[i].as_str();
        let val = raw.get(i + 1).unwrap_or_else(|| panic!("missing value for {key}"));
        match key {
            "--file" => args.files.push(PathBuf::from(val)),
            "--coin" => args.coin = val.to_string(),
            "--band-cents" => args.band_cents = val.parse().expect("bad --band-cents"),
            "--m-base" => args.m_base = Some(val.parse().expect("bad --m-base")),
            "--a-sigma" => args.a_sigma = Some(val.parse().expect("bad --a-sigma")),
            "--ewma-half-life" => args.ewma_half_life = Some(val.parse().expect("bad --ewma-half-life")),
            "--p-raw-gamma" => args.p_raw_gamma = Some(val.parse().expect("bad --p-raw-gamma")),
            "--current-shift" => args.current_shift = Some(val.parse().expect("bad --current-shift")),
            "--adjacent-shift" => args.adjacent_shift = Some(val.parse().expect("bad --adjacent-shift")),
            "--sessions" => args.sessions = val.parse().expect("bad --sessions"),
            "--seed" => args.seed = val.parse().expect("bad --seed"),
            _ => panic!("unknown arg {key}"),
        }
        i += 2;
    }
    if args.files.is_empty() || args.coin.is_empty() || args.band_cents == 0 {
        panic!("usage: policy_backtest --file <jsonl> --coin <coin> --band-cents <n>");
    }
    args
}

fn apply(cfg: &mut GridConfig, args: &Args) {
    if let Some(v) = args.m_base { cfg.pricing.m_base = v; }
    if let Some(v) = args.a_sigma { cfg.pricing.a_sigma = v; }
    if let Some(v) = args.ewma_half_life { cfg.volatility.ewma_half_life_seconds = v; }
    if let Some(v) = args.p_raw_gamma { cfg.pricing.p_raw_gamma = v; }
    if let Some(v) = args.current_shift { cfg.pricing.current_row_logit_shift = v; }
    if let Some(v) = args.adjacent_shift { cfg.pricing.adjacent_row_logit_shift = v; }
    cfg.simulation.mc_n_min = 1000;
    cfg.simulation.mc_n_max = 1000;
    cfg.simulation.seed = Some(args.seed);
}

fn load_ticks(args: &Args) -> Vec<PriceTick> {
    let mut ticks = Vec::new();
    for path in &args.files {
        let mut part = read_hl_jsonl(path, &args.coin).unwrap_or_else(|e| panic!("{e}"));
        ticks.append(&mut part);
    }
    ticks.sort_by_key(|t| t.ts_ms);
    ticks
}

#[derive(Clone, Copy)]
struct Tap {
    start_ms: u64,
    signed_dist: i64,
    dist: i64,
    multiplier: f64,
    hit: bool,
    us: bool,
}

fn choose(policy: &str, taps: &[Tap], count: usize, rng: &mut StdRng) -> Vec<Tap> {
    let mut pool = taps.to_vec();
    match policy {
        "closest" => pool.sort_by_key(|t| (t.dist, t.start_ms)),
        "three_rows" => {
            let nearest = pool.iter().map(|t| t.start_ms).min().unwrap_or(0);
            pool.retain(|t| t.start_ms == nearest);
            pool.sort_by_key(|t| t.dist);
        }
        "row_sweep" => {
            let best = pool.iter().map(|t| t.dist).min().unwrap_or(0);
            pool.retain(|t| t.dist == best);
            pool.sort_by_key(|t| t.start_ms);
        }
        "long_bias" => {
            pool.retain(|t| t.signed_dist > 0);
            pool.sort_by_key(|t| (t.dist, t.start_ms));
        }
        "short_bias" => {
            pool.retain(|t| t.signed_dist < 0);
            pool.sort_by_key(|t| (t.dist, t.start_ms));
        }
        "random_near" => {
            let mut picked = Vec::new();
            while picked.len() < count && !pool.is_empty() {
                let weights: Vec<f64> = pool.iter().map(|t| 1.0 / (1.0 + t.dist as f64)).collect();
                let total: f64 = weights.iter().sum();
                let mut draw = rng.random::<f64>() * total;
                let mut idx = pool.len() - 1;
                for (i, w) in weights.iter().enumerate() {
                    draw -= w;
                    if draw <= 0.0 { idx = i; break; }
                }
                picked.push(pool.remove(idx));
            }
            return picked;
        }
        _ => {}
    }
    pool.truncate(count);
    pool
}

fn main() {
    let args = parse_args();
    let ticker = args.coin.split(':').next_back().unwrap_or(&args.coin).to_string();
    let mut cfg = stock_grid_config_for_ticker(&ticker, args.band_cents, Vec::new());
    apply(&mut cfg, &args);
    let cur_shift = cfg.pricing.current_row_logit_shift;
    let adj_shift = cfg.pricing.adjacent_row_logit_shift;
    let ticks = load_ticks(&args);
    let report = run_calibration(&ticks, &CalibrateOptions {
        ticker: ticker.clone(),
        band_cents: args.band_cents,
        base_config: cfg,
        warmup_ms: 300_000,
        restart_gap_ms: 30_000,
    });
    let mut taps = Vec::new();
    for sample in report.samples {
        if sample.is_warmup { continue; }
        taps.push(Tap {
            start_ms: sample.window_start_ms,
            signed_dist: sample.signed_distance,
            dist: sample.center_distance as i64,
            multiplier: sample.multiplier,
            hit: sample.hit,
            us: sample.is_us_session,
        });
    }
    taps.sort_by_key(|t| t.start_ms);
    if taps.is_empty() { println!("no taps"); return; }
    let first = taps[0].start_ms;
    let last = taps[taps.len() - 1].start_ms;
    println!(
        "ticker={ticker} band={} cur_shift={} adj_shift={} samples={}",
        args.band_cents,
        cur_shift,
        adj_shift,
        taps.len()
    );
    for (policy_index, policy) in POLICIES.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(args.seed + policy_index as u64);
        let mut volume = [0.0; 2];
        let mut payout = [0.0; 2];
        let mut count = [0usize; 2];
        let mut multiplier_sum = 0.0;
        let mut by_dist = vec![(0.0f64, 0.0f64, 0usize); 11];
        let mut start = first + 120_000;
        let mut sessions = 0;
        while sessions < args.sessions && start + 30_000 < last {
            let end = start + 30_000;
            let window: Vec<Tap> = taps.iter().copied().filter(|t| (start..end).contains(&t.start_ms)).collect();
            for tap in choose(policy, &window, 18, &mut rng) {
                let side = usize::from(!tap.us);
                volume[side] += 10.0;
                count[side] += 1;
                multiplier_sum += tap.multiplier;
                let bucket = (tap.dist as usize).min(10);
                by_dist[bucket].0 += 10.0;
                by_dist[bucket].2 += 1;
                if tap.hit {
                    payout[side] += 10.0 * tap.multiplier;
                    by_dist[bucket].1 += 10.0 * tap.multiplier;
                }
            }
            sessions += 1;
            start += rng.random_range(60_000..300_000);
        }
        for (side, name) in ["US", "OFF"].iter().enumerate() {
            let edge = if volume[side] > 0.0 { (volume[side] - payout[side]) / volume[side] * 100.0 } else { 0.0 };
            println!("{policy} {name} sessions={sessions} taps={} volume={:.0} payout={:.0} edge={edge:.2}%", count[side], volume[side], payout[side]);
        }
        let taps_total = count[0] + count[1];
        let avg_multiplier = if taps_total > 0 { multiplier_sum / taps_total as f64 } else { 0.0 };
        println!("{policy} avg_multiplier={avg_multiplier:.2}");
        if *policy == "random_near" {
            for (dist, (vol, pay, n)) in by_dist.iter().enumerate() {
                if *n == 0 { continue; }
                let edge = (vol - pay) / vol * 100.0;
                println!("random_near dist={dist} taps={n} volume={vol:.0} edge={edge:.2}%");
            }
        }
    }
    println!("note: quotes cached every 5s; production reprices every 100ms. exposure skew and safety factor are off, so this slightly understates house edge.");
}
