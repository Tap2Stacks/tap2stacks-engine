use std::hint::black_box;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use st_engine::grid::{
    CellQuote, GridConfig, GridEngine, GridSnapshot, OhlcBar, PriceBand,
    compute_k_safety, compute_quote_probability, estimate_runtime_state,
    price_multiplier, quote_grid, quote_grid_with_liabilities, simulate_summary,
    smooth_used_liability,
};

#[derive(Debug, Clone, Copy)]
struct BenchCase {
    history_bars: usize,
    mc_paths: usize,
    adaptive_mc: bool,
    iterations: usize,
    warmup_iterations: usize,
}

impl Default for BenchCase {
    fn default() -> Self {
        Self {
            history_bars: 180,
            mc_paths: 5_000,
            adaptive_mc: false,
            iterations: 50,
            warmup_iterations: 5,
        }
    }
}

#[derive(Debug)]
struct BenchResult {
    name: &'static str,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn main() {
    let case = bench_case_from_env();
    let (spot, history, liabilities, config) = typical_case(case);
    let worker_sweep = std::env::var("BENCH_WORKERS")
        .ok()
        .map(parse_worker_sweep)
        .filter(|workers| !workers.is_empty())
        .unwrap_or_default();

    println!("Mode 10 Grid Benchmark");
    println!("spot={spot}");
    println!("history_bars={}", history.len());
    println!("cells={}", config.spec.rows * config.spec.windows.len());
    println!("mc_paths={}", config.simulation.mc_n_min);
    println!("adaptive_mc={}", config.simulation.adaptive_mc);
    println!("iterations={}", case.iterations);
    println!("warmup_iterations={}", case.warmup_iterations);
    println!("lock_offset={}", config.simulation.lock_offset);
    println!("use_antithetic={}", config.simulation.use_antithetic);
    println!("k_safety_ttl_ms={}", config.pricing.k_safety_ttl_ms);
    println!();

    let stateless = measure("stateless", case, || {
        let snapshot = quote_grid_with_liabilities(
            black_box(spot),
            black_box(&history),
            black_box(&config),
            Some(black_box(&liabilities)),
        )
        .expect("stateless quote should succeed");
        black_box(snapshot.cells[0].multiplier);
    });

    let plain = measure("plain", case, || {
        let snapshot =
            quote_grid(black_box(spot), black_box(&history), black_box(&config))
                .expect("plain quote should succeed");
        black_box(snapshot.cells[0].multiplier);
    });

    let mut engine = GridEngine::new(config.clone());
    for (cell_index, liability) in liabilities.iter().copied().enumerate() {
        if liability > 0.0 {
            engine.set_liability(cell_index, liability);
        }
    }
    let stateful = measure("stateful", case, || {
        let snapshot = engine
            .quote_grid(black_box(spot), black_box(&history))
            .expect("stateful quote should succeed");
        black_box(snapshot.cells[0].multiplier);
    });

    print_result(plain);
    print_result(stateless);
    print_result(stateful);
    println!();

    let anchor = aligned_anchor(spot, config.spec.band_width);
    let bands = build_bands(
        anchor,
        config.spec.rows,
        config.spec.center_row,
        config.spec.band_width,
    );
    let horizon = max_horizon(&config);
    let baseline_runtime_state =
        estimate_runtime_state(spot, &history, &config).expect("runtime state should succeed");
    let baseline_simulation =
        simulate_summary(
            spot,
            &bands,
            &config,
            &baseline_runtime_state,
            horizon,
            Some(&liabilities),
        );

    let phase_state = measure("phase_state", case, || {
        let runtime_state = estimate_runtime_state(
            black_box(spot),
            black_box(&history),
            black_box(&config),
        )
        .expect("runtime state should succeed");
        black_box(runtime_state.volatility.sigma);
    });

    let phase_simulation = measure("phase_simulate", case, || {
        let simulation = simulate_summary(
            black_box(spot),
            black_box(&bands),
            black_box(&config),
            black_box(&baseline_runtime_state),
            black_box(horizon),
            None,
        );
        black_box(simulation.mean_probabilities[0][0]);
    });

    let phase_materialize = measure("phase_materialize", case, || {
        let snapshot = materialize_snapshot(
            spot,
            anchor,
            &bands,
            &baseline_simulation,
            &baseline_runtime_state,
            &config,
            &liabilities,
            None,
        );
        black_box(snapshot.cells[0].multiplier);
    });

    let phase_total = measure("phase_total", case, || {
        let anchor = aligned_anchor(spot, config.spec.band_width);
        let bands = build_bands(
            anchor,
            config.spec.rows,
            config.spec.center_row,
            config.spec.band_width,
        );
        let runtime_state = estimate_runtime_state(spot, &history, &config)
            .expect("runtime state should succeed");
        let simulation =
            simulate_summary(spot, &bands, &config, &runtime_state, horizon, Some(&liabilities));
        let snapshot = materialize_snapshot(
            spot,
            anchor,
            &bands,
            &simulation,
            &runtime_state,
            &config,
            &liabilities,
            None,
        );
        black_box(snapshot.cells[0].multiplier);
    });

    print_result(phase_state);
    print_result(phase_simulation);
    print_result(phase_materialize);
    print_result(phase_total);

    if !worker_sweep.is_empty() {
        println!();
        println!("Worker Sweep");
        for workers in worker_sweep {
            let mut sweep_config = config.clone();
            sweep_config.simulation.parallel_workers = Some(workers);
            let total_result = measure("parallel_total", case, || {
                let snapshot = quote_grid_with_liabilities(
                    black_box(spot),
                    black_box(&history),
                    black_box(&sweep_config),
                    Some(black_box(&liabilities)),
                )
                .expect("parallel sweep quote should succeed");
                black_box(snapshot.cells[0].multiplier);
            });
            let runtime_state = estimate_runtime_state(spot, &history, &sweep_config)
                .expect("runtime state should succeed");
            let simulation_result = measure("parallel_simulate", case, || {
                let simulation = simulate_summary(
                    black_box(spot),
                    black_box(&bands),
                    black_box(&sweep_config),
                    black_box(&runtime_state),
                    black_box(horizon),
                    Some(black_box(&liabilities)),
                );
                black_box(simulation.mean_probabilities[0][0]);
            });
            println!(
                "workers={:<3} total_mean={:>8.3} ms total_p95={:>8.3} ms sim_mean={:>8.3} ms sim_p95={:>8.3} ms",
                workers,
                total_result.mean_ms,
                total_result.p95_ms,
                simulation_result.mean_ms,
                simulation_result.p95_ms,
            );
        }
    }
}

fn typical_case(case: BenchCase) -> (f64, Vec<OhlcBar>, Vec<f64>, GridConfig) {
    let mut config = GridConfig::default();
    config.simulation.seed = Some(12345);
    config.simulation.mc_n_min = case.mc_paths;
    config.simulation.mc_n_max = case.mc_paths;
    config.simulation.adaptive_mc = case.adaptive_mc;
    if let Ok(value) = std::env::var("BENCH_K_SAFETY_TTL_MS") {
        if let Ok(ttl_ms) = value.parse::<u64>() {
            config.pricing.k_safety_ttl_ms = ttl_ms;
        }
    }

    let history = generate_history(case.history_bars, 50_000.0, 42);
    let spot = history.last().map(|bar| bar.close).unwrap_or(50_000.0);
    let liabilities = generate_liabilities(config.spec.rows * config.spec.windows.len());

    (spot, history, liabilities, config)
}

fn bench_case_from_env() -> BenchCase {
    let mut case = BenchCase::default();
    if let Ok(value) = std::env::var("BENCH_HISTORY_BARS") {
        if let Ok(history_bars) = value.parse::<usize>() {
            case.history_bars = history_bars.max(1);
        }
    }
    if let Ok(value) = std::env::var("BENCH_MC_PATHS") {
        if let Ok(mc_paths) = value.parse::<usize>() {
            case.mc_paths = mc_paths.max(1);
        }
    }
    if let Ok(value) = std::env::var("BENCH_ITERATIONS") {
        if let Ok(iterations) = value.parse::<usize>() {
            case.iterations = iterations.max(1);
        }
    }
    if let Ok(value) = std::env::var("BENCH_WARMUP_ITERATIONS") {
        if let Ok(warmup_iterations) = value.parse::<usize>() {
            case.warmup_iterations = warmup_iterations;
        }
    }
    if let Ok(value) = std::env::var("BENCH_ADAPTIVE_MC") {
        if let Ok(enabled) = parse_bool_flag(&value) {
            case.adaptive_mc = enabled;
        }
    }
    case
}

fn generate_history(count: usize, start_price: f64, seed: u64) -> Vec<OhlcBar> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = StandardNormal;
    let mut current = start_price;
    let mut bars = Vec::with_capacity(count);

    for step in 0..count {
        let drift = (step as f64 / 17.0).sin() * 0.00003;
        let close_noise: f64 = normal.sample(&mut rng);
        let up_noise: f64 = normal.sample(&mut rng);
        let down_noise: f64 = normal.sample(&mut rng);
        let close_return: f64 = drift + 0.0008 * close_noise;
        let intrabar_up: f64 = 0.0005 * up_noise.abs();
        let intrabar_down: f64 = 0.0005 * down_noise.abs();

        let open = current;
        let close = (current * close_return.exp()).max(1.0);
        let high = open.max(close) * (1.0 + intrabar_up);
        let low = open.min(close) * (1.0 - intrabar_down).max(1e-6);

        bars.push(OhlcBar {
            open,
            high,
            low,
            close,
        });
        current = close;
    }

    bars
}

fn generate_liabilities(cell_count: usize) -> Vec<f64> {
    let mut values = vec![0.0; cell_count];
    for (index, value) in values.iter_mut().enumerate() {
        let hotspot = match index % 5 {
            0 => 180.0,
            1 => 120.0,
            2 => 80.0,
            3 => 40.0,
            _ => 20.0,
        };
        *value = hotspot + ((index / 5) as f64 * 3.5);
    }
    values
}

fn max_horizon(config: &GridConfig) -> usize {
    config
        .spec
        .windows
        .iter()
        .map(|window| window.end_secs)
        .max()
        .unwrap_or(0)
        .saturating_add(config.simulation.lock_offset) as usize
}

fn aligned_anchor(spot: f64, band_width: f64) -> f64 {
    (spot / band_width).floor() * band_width
}

fn build_bands(anchor: f64, rows: usize, center_row: usize, band_width: f64) -> Vec<PriceBand> {
    (0..rows)
        .map(|row_index| {
            let center = anchor + ((row_index as f64) - (center_row as f64)) * band_width;
            PriceBand {
                row_index,
                lower: center - band_width / 2.0,
                upper: center + band_width / 2.0,
            }
        })
        .collect()
}

fn materialize_snapshot(
    spot: f64,
    anchor: f64,
    bands: &[PriceBand],
    simulation: &st_engine::grid::SimulationSummary,
    runtime_state: &st_engine::grid::RuntimeState,
    config: &GridConfig,
    liabilities: &[f64],
    liability_used: Option<&mut std::collections::HashMap<usize, f64>>,
) -> GridSnapshot {
    let volatility = runtime_state.volatility;
    let k_safety = compute_k_safety(&simulation.scenario_payouts, config);
    let server_ts_ms = 0;
    let current_window0_row_index = bands
        .iter()
        .min_by(|left, right| {
            let left_center = (left.lower + left.upper) / 2.0;
            let right_center = (right.lower + right.upper) / 2.0;
            (left_center - spot)
                .abs()
                .partial_cmp(&(right_center - spot).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|band| band.row_index)
        .unwrap_or(config.spec.center_row);
    let mut cells = Vec::with_capacity(bands.len() * config.spec.windows.len());
    let mut liability_used = liability_used;

    for (row_index, band) in bands.iter().enumerate() {
        for (window_index, window) in config.spec.windows.iter().enumerate() {
            let cell_index = row_index * config.spec.windows.len() + window_index;
            let raw_hit_probability = simulation.mean_probabilities[row_index][window_index];
            let hit_probability = compute_quote_probability(
                raw_hit_probability,
                row_index,
                window_index,
                *window,
                volatility,
                current_window0_row_index,
                config,
            );
            let liability = if let Some(ref mut used) = liability_used {
                let instant = liabilities[cell_index];
                let previous = used.get(&cell_index).copied().unwrap_or(0.0);
                let smoothed = smooth_used_liability(previous, instant, config);
                used.insert(cell_index, smoothed);
                smoothed
            } else {
                liabilities[cell_index]
            };
            let multiplier =
                price_multiplier(hit_probability, volatility, liability, k_safety, config);
            cells.push(CellQuote {
                server_ts_ms,
                band: *band,
                window: *window,
                hit_probability,
                multiplier,
            });
        }
    }

    GridSnapshot {
        server_ts_ms,
        spot,
        anchor,
        volatility,
        cells,
    }
}

fn measure(name: &'static str, case: BenchCase, mut f: impl FnMut()) -> BenchResult {
    let mut samples = Vec::with_capacity(case.iterations);

    for _ in 0..case.warmup_iterations {
        f();
    }

    for _ in 0..case.iterations {
        let started = Instant::now();
        f();
        samples.push(started.elapsed());
    }

    summarise(name, &samples)
}

fn summarise(name: &'static str, samples: &[Duration]) -> BenchResult {
    let mut millis = samples
        .iter()
        .map(|sample| sample.as_secs_f64() * 1_000.0)
        .collect::<Vec<_>>();
    millis.sort_by(|left, right| left.partial_cmp(right).unwrap());

    let mean_ms = millis.iter().sum::<f64>() / millis.len() as f64;
    let p50_ms = percentile(&millis, 0.50);
    let p95_ms = percentile(&millis, 0.95);
    let min_ms = *millis.first().unwrap_or(&0.0);
    let max_ms = *millis.last().unwrap_or(&0.0);

    BenchResult {
        name,
        mean_ms,
        p50_ms,
        p95_ms,
        min_ms,
        max_ms,
    }
}

fn percentile(sorted_samples: &[f64], q: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.0;
    }
    let index = ((sorted_samples.len() - 1) as f64 * q).round() as usize;
    sorted_samples[index.min(sorted_samples.len() - 1)]
}

fn print_result(result: BenchResult) {
    println!(
        "{:<18} mean={:>8.3} ms p50={:>8.3} ms p95={:>8.3} ms min={:>8.3} ms max={:>8.3} ms",
        result.name,
        result.mean_ms,
        result.p50_ms,
        result.p95_ms,
        result.min_ms,
        result.max_ms,
    );
}

fn parse_worker_sweep(value: String) -> Vec<usize> {
    value
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .filter(|workers| *workers > 0)
        .collect()
}

fn parse_bool_flag(value: &str) -> Result<bool, ()> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(()),
    }
}
