use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use st_engine::grid::stock_grid_config_for_ticker;
use st_engine::horizon::{DEFAULT_TRIGGER_SIGMA, FAR_REFRESH_MS, stagger_phase_ms};
use st_engine::price::{PriceCache, PriceSource, PriceTick, read_ticks_for_source};
use st_engine::quote::QuoteEngine;
use st_engine::stress::parse_utc;
use st_engine::v2::PricingModel;

const MAX_COMPLETED_BARS: usize = 600;
const GRID_LOOKBACK_BARS: usize = 180;
const WARMUP_LOOKBACK_MS: u64 = 600_000;

struct TickerDef {
    ticker: String,
    coin: String,
    band_cents: u32,
    fallback_spot: f64,
}

struct Args {
    file: PathBuf,
    from_ms: u64,
    to_ms: u64,
    windows: usize,
    mc_n: usize,
    trigger_sigma: f64,
    far_refresh_ms: u64,
    stagger: bool,
    seed: u64,
    model: PricingModel,
    max_multiplier: Option<f64>,
    source: PriceSource,
}

fn parse_args() -> Args {
    let mut file = PathBuf::new();
    let mut from_ms = 0u64;
    let mut to_ms = 0u64;
    let mut windows = 24usize;
    let mut mc_n = 5_000usize;
    let mut trigger_sigma = DEFAULT_TRIGGER_SIGMA;
    let mut far_refresh_ms = FAR_REFRESH_MS;
    let mut stagger = true;
    let mut seed = 42u64;
    let mut model = PricingModel::V1;
    let mut max_multiplier: Option<f64> = None;
    let mut source = PriceSource::Hyperliquid;

    let raw: Vec<String> = env::args().skip(1).collect();
    let mut i = 0usize;
    while i < raw.len() {
        let key = raw[i].as_str();
        let val = raw
            .get(i + 1)
            .unwrap_or_else(|| panic!("missing value for {key}"));
        match key {
            "--file" => file = PathBuf::from(val),
            "--from-utc" => from_ms = parse_utc(val).unwrap_or_else(|err| panic!("{err}")),
            "--to-utc" => to_ms = parse_utc(val).unwrap_or_else(|err| panic!("{err}")),
            "--windows" => windows = val.parse().expect("bad --windows"),
            "--mc-n" => mc_n = val.parse().expect("bad --mc-n"),
            "--trigger-sigma" => trigger_sigma = val.parse().expect("bad --trigger-sigma"),
            "--far-refresh-ms" => far_refresh_ms = val.parse().expect("bad --far-refresh-ms"),
            "--stagger" => stagger = val.parse().expect("bad --stagger"),
            "--seed" => seed = val.parse().expect("bad --seed"),
            "--model" => {
                model = PricingModel::parse(val).expect("bad --model (expected v1, v2, or m4)")
            }
            "--max-multiplier" => {
                max_multiplier = Some(val.parse().expect("bad --max-multiplier"))
            }
            "--source" => {
                source =
                    PriceSource::parse(val).expect("bad --source (expected hyperliquid or deribit)")
            }
            _ => panic!("unknown arg {key}"),
        }
        i += 2;
    }
    if file.as_os_str().is_empty() || from_ms == 0 || to_ms <= from_ms {
        panic!(
            "usage: bench_horizon --file <jsonl> --from-utc <ts> --to-utc <ts> [--source hyperliquid|deribit] [--windows 24] [--mc-n 5000] [--trigger-sigma 2.0] [--far-refresh-ms 5000] [--stagger true] [--seed 42] [--model v1|v2|m4] [--max-multiplier 50]"
        );
    }
    Args {
        file,
        from_ms,
        to_ms,
        windows,
        mc_n,
        trigger_sigma,
        far_refresh_ms,
        stagger,
        seed,
        model,
        max_multiplier,
        source,
    }
}

/// Reads Linux process CPU time (user + system) in seconds from `/proc/self/stat`.
fn process_cpu_secs() -> Option<f64> {
    let text = fs::read_to_string("/proc/self/stat").ok()?;
    let rparen = text.rfind(')')?;
    let fields: Vec<&str> = text[rparen + 2..].split_whitespace().collect();
    // After ") ", field 0 is state (field 3 overall); utime is index 11 (field 14), stime is index 12 (field 15).
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    // Linux USER_HZ is 100 ticks/sec.
    Some((utime + stime) as f64 / 100.0)
}

fn percentile_us(sorted_us: &[u64], pct: f64) -> u64 {
    if sorted_us.is_empty() {
        return 0;
    }
    let idx = ((sorted_us.len() as f64 - 1.0) * pct).floor() as usize;
    sorted_us[idx.min(sorted_us.len() - 1)]
}

fn synthetic_ticks(from_ms: u64, to_ms: u64, base_price: f64, seed_offset: u64) -> Vec<PriceTick> {
    let mut ticks = Vec::new();
    let mut price = base_price;
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ seed_offset;
    let mut ts = from_ms;
    while ts <= to_ms {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let unit = ((state >> 33) as f64 / u32::MAX as f64) - 0.5;
        price *= (unit * 0.0004).exp();
        ticks.push(PriceTick::new(ts, price));
        ts = ts.saturating_add(1_000);
    }
    ticks
}

struct TickerState {
    ticker: String,
    ticks: Vec<PriceTick>,
    tick_idx: usize,
    real_ticks: usize,
    cache: PriceCache,
    engine: QuoteEngine,
    latencies_us: Vec<u64>,
}

fn main() {
    let args = parse_args();
    let defs = [
        TickerDef {
            ticker: "NVDA".to_string(),
            coin: "xyz:NVDA".to_string(),
            band_cents: 3,
            fallback_spot: 140.0,
        },
        TickerDef {
            ticker: "TSLA".to_string(),
            coin: "xyz:TSLA".to_string(),
            band_cents: 8,
            fallback_spot: 340.0,
        },
        TickerDef {
            ticker: "AAPL".to_string(),
            coin: "xyz:AAPL".to_string(),
            band_cents: 8,
            fallback_spot: 228.0,
        },
        TickerDef {
            ticker: "SP500".to_string(),
            coin: "xyz:SP500".to_string(),
            band_cents: 150,
            fallback_spot: 5900.0,
        },
    ];

    let warmup_start = args.from_ms.saturating_sub(WARMUP_LOOKBACK_MS);
    let mut states: Vec<TickerState> = Vec::with_capacity(defs.len());
    for (idx, def) in defs.iter().enumerate() {
        let query_coin = match args.source {
            PriceSource::Hyperliquid => def.coin.as_str(),
            PriceSource::Deribit => def.ticker.as_str(),
        };
        let mut ticks = read_ticks_for_source(&args.file, query_coin, args.source)
            .unwrap_or_else(|err| panic!("failed reading {}: {err}", args.file.display()));
        ticks.sort_by_key(|t| t.ts_ms);
        ticks.retain(|t| t.ts_ms >= warmup_start && t.ts_ms <= args.to_ms);
        let real_ticks = ticks.len();
        if real_ticks < 20 {
            ticks = synthetic_ticks(warmup_start, args.to_ms, def.fallback_spot, idx as u64 + 1);
        }

        let mut cfg = stock_grid_config_for_ticker(&def.ticker, def.band_cents, Vec::new());
        cfg.simulation.quote_window_count = args.windows;
        cfg.simulation.far_refresh_ms = args.far_refresh_ms;
        cfg.simulation.far_trigger_sigma = args.trigger_sigma;
        cfg.simulation.far_phase_ms = if args.stagger {
            stagger_phase_ms(idx, defs.len(), args.far_refresh_ms)
        } else {
            0
        };
        cfg.simulation.mc_n_min = args.mc_n;
        cfg.simulation.mc_n_max = args.mc_n;
        cfg.simulation.seed = Some(args.seed + idx as u64);
        cfg.simulation.pricing_model = args.model;
        cfg.simulation.price_source = args.source;
        if let Some(cap) = args.max_multiplier {
            cfg.pricing.max_multiplier = cap;
        }

        let mut engine = QuoteEngine::with_config(def.ticker.clone(), def.band_cents, cfg);
        let mut cache = PriceCache::new(MAX_COMPLETED_BARS);
        let mut tick_idx = 0usize;
        while tick_idx < ticks.len() && ticks[tick_idx].ts_ms < args.from_ms {
            let _ = cache.apply_tick(ticks[tick_idx]);
            engine.observe_tick(ticks[tick_idx]);
            tick_idx += 1;
        }

        states.push(TickerState {
            ticker: def.ticker.clone(),
            ticks,
            tick_idx,
            real_ticks,
            cache,
            engine,
            latencies_us: Vec::new(),
        });
    }

    let cpu_start = process_cpu_secs();
    let wall_start = Instant::now();
    let mut step_wall_us: Vec<u64> = Vec::new();
    let mut overruns_1s = 0usize;
    let mut sim_secs = 0u64;

    let mut sec_ms = args.from_ms;
    while sec_ms < args.to_ms {
        let next_ms = sec_ms.saturating_add(1_000);
        let mut jobs = Vec::with_capacity(states.len());
        for (idx, state) in states.iter_mut().enumerate() {
            let mut latest_tick: Option<PriceTick> = None;
            while state.tick_idx < state.ticks.len() && state.ticks[state.tick_idx].ts_ms < next_ms {
                let tick = state.ticks[state.tick_idx];
                let _ = state.cache.apply_tick(tick);
                state.engine.observe_tick(tick);
                latest_tick = Some(tick);
                state.tick_idx += 1;
            }
            if let Some(tick) = latest_tick {
                if let Some(grid_snap) = state.cache.snapshot().bars_for_grid(GRID_LOOKBACK_BARS) {
                    if let Some(job) = state.engine.prepare_sample_job(tick, grid_snap.history) {
                        jobs.push((idx, job));
                    }
                }
            }
        }

        if !jobs.is_empty() {
            let tick_started = Instant::now();
            let results: Vec<(usize, u64, Option<st_engine::quote::SampleResult>)> =
                std::thread::scope(|scope| {
                    let handles: Vec<_> = jobs
                        .into_iter()
                        .map(|(idx, job)| {
                            scope.spawn(move || {
                                let t0 = Instant::now();
                                let res = job.run();
                                let elapsed_us = t0.elapsed().as_micros() as u64;
                                (idx, elapsed_us, res)
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|handle| handle.join().expect("bench worker panicked"))
                        .collect()
                });
            let elapsed_us = tick_started.elapsed().as_micros() as u64;
            step_wall_us.push(elapsed_us);
            if elapsed_us > 1_000_000 {
                overruns_1s += 1;
            }
            for (idx, lat_us, res_opt) in results {
                if let Some(res) = res_opt {
                    states[idx].engine.apply_sample_result(res);
                    states[idx].latencies_us.push(lat_us);
                }
            }
        }
        sim_secs += 1;
        sec_ms = next_ms;
    }

    let wall_secs = wall_start.elapsed().as_secs_f64();
    let cpu_secs = match (cpu_start, process_cpu_secs()) {
        (Some(start), Some(end)) => (end - start).max(0.0),
        _ => 0.0,
    };
    let cpu_per_sim_sec = if sim_secs > 0 {
        cpu_secs / sim_secs as f64
    } else {
        0.0
    };
    let cpu_share_4vcpu_pct = cpu_per_sim_sec / 4.0 * 100.0;
    let cpu_share_1vcpu_pct = cpu_per_sim_sec * 100.0;

    step_wall_us.sort_unstable();
    let mean_wall_us = if step_wall_us.is_empty() {
        0.0
    } else {
        step_wall_us.iter().map(|&v| v as f64).sum::<f64>() / step_wall_us.len() as f64
    };
    let p50_wall_ms = percentile_us(&step_wall_us, 0.50) as f64 / 1_000.0;
    let p95_wall_ms = percentile_us(&step_wall_us, 0.95) as f64 / 1_000.0;
    let p99_wall_ms = percentile_us(&step_wall_us, 0.99) as f64 / 1_000.0;
    let max_wall_ms = step_wall_us.last().copied().unwrap_or(0) as f64 / 1_000.0;

    println!(
        "MULTI_BENCH windows={} mc_n={} trigger_sigma={:.2} far_refresh_ms={} stagger={} sim_secs={} active_secs={} wall_secs={:.2} cpu_secs={:.2} cpu_per_sim_sec={:.3} cpu_share_4vcpu_pct={:.2} cpu_share_1vcpu_pct={:.2} tick_wall_mean_ms={:.2} tick_wall_p50_ms={:.2} tick_wall_p95_ms={:.2} tick_wall_p99_ms={:.2} tick_wall_max_ms={:.2} overruns_1s={}",
        args.windows,
        args.mc_n,
        args.trigger_sigma,
        args.far_refresh_ms,
        args.stagger,
        sim_secs,
        step_wall_us.len(),
        wall_secs,
        cpu_secs,
        cpu_per_sim_sec,
        cpu_share_4vcpu_pct,
        cpu_share_1vcpu_pct,
        mean_wall_us / 1_000.0,
        p50_wall_ms,
        p95_wall_ms,
        p99_wall_ms,
        max_wall_ms,
        overruns_1s,
    );

    for state in &mut states {
        state.latencies_us.sort_unstable();
        let stats = state.engine.reprice_stats();
        let p50_ms = percentile_us(&state.latencies_us, 0.50) as f64 / 1_000.0;
        let p95_ms = percentile_us(&state.latencies_us, 0.95) as f64 / 1_000.0;
        println!(
            "TICKER_BENCH ticker={} real_ticks={} samples={} full={} near={} trigger={} cadence={} mean_ms={:.2} p50_ms={:.2} p95_ms={:.2} full_mean_ms={:.2} near_mean_ms={:.2}",
            state.ticker,
            state.real_ticks,
            stats.sample_count,
            stats.full_reprices,
            stats.near_reprices,
            stats.trigger_reprices,
            stats.cadence_reprices,
            stats.mean_us() as f64 / 1_000.0,
            p50_ms,
            p95_ms,
            stats.full_mean_us() as f64 / 1_000.0,
            stats.near_mean_us() as f64 / 1_000.0,
        );
    }
}
