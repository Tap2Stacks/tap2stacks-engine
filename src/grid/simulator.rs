use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use super::touch::{
    BRIDGE_PRUNE_EXPONENT, window_hit_discrete_steps, window_hit_probability_log_steps,
    window_hit_probability_steps,
};
use super::types::{GridConfig, PriceBand, RuntimeState, TouchMode};

#[derive(Debug, Clone, PartialEq)]
pub struct SimulationSummary {
    pub mean_probabilities: Vec<Vec<f64>>,
    pub scenario_payouts: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DebugPathTrace {
    pub paths: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Copy)]
struct WindowEval {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy)]
struct CellEval {
    cell_index: usize,
    lower: f64,
    upper: f64,
    log_lower: f64,
    log_upper: f64,
    window_index: usize,
}

pub fn debug_reconstruct_paths_from_shocks(
    spot: f64,
    mu: f64,
    eps_matrix: &[Vec<f64>],
    jump_matrix: &[Vec<f64>],
    sigma: f64,
) -> DebugPathTrace {
    let path_count = eps_matrix.len();
    let horizon_secs = eps_matrix.first().map(|row| row.len()).unwrap_or(0);
    let mut paths = Vec::with_capacity(path_count);

    for path_index in 0..path_count {
        let mut path = Vec::with_capacity(horizon_secs + 1);
        let mut log_price = (spot as f32).ln();
        let mu = mu as f32;
        let sigma = sigma as f32;
        path.push(spot);
        for step in 0..horizon_secs {
            let eps = eps_matrix[path_index][step] as f32;
            let jump = jump_matrix[path_index][step] as f32;
            log_price += mu + sigma * eps + jump;
            path.push(log_price.exp() as f64);
        }
        paths.push(path);
    }

    DebugPathTrace { paths }
}

pub fn debug_reconstruct_paths_from_returns(spot: f64, returns_matrix: &[Vec<f64>]) -> DebugPathTrace {
    let path_count = returns_matrix.len();
    let horizon_secs = returns_matrix.first().map(|row| row.len()).unwrap_or(0);
    let mut paths = Vec::with_capacity(path_count);

    for returns in returns_matrix {
        let mut path = Vec::with_capacity(horizon_secs + 1);
        let mut log_price = (spot as f32).ln();
        path.push(spot);
        for value in returns {
            log_price += *value as f32;
            path.push(log_price.exp() as f64);
        }
        paths.push(path);
    }

    DebugPathTrace { paths }
}

pub fn simulate_summary(
    spot: f64,
    bands: &[PriceBand],
    config: &GridConfig,
    runtime_state: &RuntimeState,
    max_horizon: usize,
    liabilities: Option<&[f64]>,
) -> SimulationSummary {
    let windows = precompute_windows(config, max_horizon);
    let cells = precompute_cells(bands, spot, config);
    if let Some(seed) = config.simulation.seed {
        let mut rng = StdRng::seed_from_u64(seed);
        simulate_summary_with_rng(
            spot,
            &cells,
            &windows,
            config,
            runtime_state,
            max_horizon,
            liabilities,
            &mut rng,
        )
    } else {
        let mut rng = rand::rng();
        simulate_summary_with_rng(
            spot,
            &cells,
            &windows,
            config,
            runtime_state,
            max_horizon,
            liabilities,
            &mut rng,
        )
    }
}

fn simulate_summary_with_rng(
    spot: f64,
    cells: &[CellEval],
    windows: &[WindowEval],
    config: &GridConfig,
    runtime_state: &RuntimeState,
    max_horizon: usize,
    liabilities: Option<&[f64]>,
    rng: &mut impl Rng,
) -> SimulationSummary {
    let window_count = config.spec.windows.len();
    let band_count = config.spec.rows;
    let cell_count = cells.len();
    let mut total_sum = vec![0.0; cell_count];
    let mut total_sq_sum = vec![0.0; cell_count];
    let track_payouts = liabilities
        .map(|values| values.iter().any(|value| *value > 0.0))
        .unwrap_or(false);
    let mut scenario_payouts = Vec::new();
    let batch_size = config.simulation.mc_n_min.max(1);
    let max_paths = config.simulation.mc_n_max.max(batch_size);
    let adaptive = config.simulation.adaptive_mc;
    let mut total_paths = 0usize;

    while total_paths < max_paths {
        let paths_this_batch = if adaptive {
            (max_paths - total_paths).min(batch_size)
        } else if total_paths == 0 {
            batch_size
        } else {
            break;
        };

        let batch = simulate_hit_batch(
            spot,
            cells,
            windows,
            config,
            runtime_state,
            max_horizon,
            paths_this_batch,
            liabilities.filter(|_| track_payouts),
            rng,
        );

        for cell_index in 0..cell_count {
            total_sum[cell_index] += batch.sum[cell_index];
            total_sq_sum[cell_index] += batch.squared_sum[cell_index];
        }
        scenario_payouts.extend(batch.scenario_payouts);

        total_paths += paths_this_batch;

        if adaptive && thresholds_met(&total_sum, &total_sq_sum, total_paths, config) {
            break;
        }
    }

    let mut mean_probabilities = vec![vec![0.0; window_count]; band_count];
    for row_index in 0..band_count {
        let row_offset = row_index * window_count;
        for window_index in 0..window_count {
            mean_probabilities[row_index][window_index] =
                total_sum[row_offset + window_index] / total_paths as f64;
        }
    }

    SimulationSummary {
        mean_probabilities,
        scenario_payouts,
    }
}

pub fn debug_simulate_paths(
    spot: f64,
    config: &GridConfig,
    runtime_state: &RuntimeState,
    horizon_secs: usize,
    path_count: usize,
    seed: u64,
) -> DebugPathTrace {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut paths = Vec::with_capacity(path_count);

    if config.simulation.use_antithetic && path_count > 1 {
        let base_paths = path_count.div_ceil(2);
        let anti_paths = path_count / 2;
        for base_index in 0..base_paths {
            let include_antithetic = base_index < anti_paths;
            let mut base_path = Vec::with_capacity(horizon_secs + 1);
            let mut antithetic_path = Vec::with_capacity(horizon_secs + 1);
            fill_path_pair(
                &mut base_path,
                Some(&mut antithetic_path),
                spot,
                horizon_secs,
                runtime_state,
                config,
                include_antithetic,
                &mut rng,
            );
            paths.push(base_path);
            if include_antithetic {
                paths.push(antithetic_path);
            }
        }
    } else {
        for _ in 0..path_count {
            let mut path = Vec::with_capacity(horizon_secs + 1);
            fill_path_pair(
                &mut path,
                None,
                spot,
                horizon_secs,
                runtime_state,
                config,
                false,
                &mut rng,
            );
            paths.push(path);
        }
    }

    paths.truncate(path_count);
    DebugPathTrace { paths }
}

struct BatchStats {
    sum: Vec<f64>,
    squared_sum: Vec<f64>,
    scenario_payouts: Vec<f64>,
}

fn simulate_hit_batch(
    spot: f64,
    cells: &[CellEval],
    windows: &[WindowEval],
    config: &GridConfig,
    runtime_state: &RuntimeState,
    max_horizon: usize,
    paths: usize,
    liabilities: Option<&[f64]>,
    rng: &mut impl Rng,
) -> BatchStats {
    let cell_count = cells.len();
    let parallel_workers = if config.simulation.use_parallel {
        config
            .simulation
            .parallel_workers
            .unwrap_or_else(rayon::current_num_threads)
            .min(paths.max(1))
    } else {
        1
    };

    if parallel_workers <= 1 || paths <= 1 {
        return simulate_hit_batch_chunk(
            spot,
            cells,
            windows,
            config,
            runtime_state,
            max_horizon,
            paths,
            liabilities,
            rng,
        );
    }

    let base_seed = config.simulation.seed.unwrap_or_else(rand::random::<u64>);
    let chunk_size = paths.div_ceil(parallel_workers);
    let task_count = paths.div_ceil(chunk_size);
    let worker_stats = (0..task_count)
        .into_par_iter()
        .map(|task_index| {
            let start = task_index * chunk_size;
            let count = (paths - start).min(chunk_size);
            let worker_seed = mix_seed(base_seed, task_index as u64);
            let mut worker_rng = StdRng::seed_from_u64(worker_seed);
            simulate_hit_batch_chunk(
                spot,
                cells,
                windows,
                config,
                runtime_state,
                max_horizon,
                count,
                liabilities,
                &mut worker_rng,
            )
        })
        .collect::<Vec<_>>();

    let mut sum = vec![0.0; cell_count];
    let mut squared_sum = vec![0.0; cell_count];
    let mut scenario_payouts = if liabilities.is_some() {
        Vec::with_capacity(paths)
    } else {
        Vec::new()
    };
    for stats in worker_stats {
        for cell_index in 0..cell_count {
            sum[cell_index] += stats.sum[cell_index];
            squared_sum[cell_index] += stats.squared_sum[cell_index];
        }
        scenario_payouts.extend(stats.scenario_payouts);
    }

    BatchStats { sum, squared_sum, scenario_payouts }
}

fn simulate_hit_batch_chunk(
    spot: f64,
    cells: &[CellEval],
    windows: &[WindowEval],
    config: &GridConfig,
    runtime_state: &RuntimeState,
    max_horizon: usize,
    paths: usize,
    liabilities: Option<&[f64]>,
    rng: &mut impl Rng,
) -> BatchStats {
    let cell_count = cells.len();
    let mut sum = vec![0.0; cell_count];
    let mut squared_sum = vec![0.0; cell_count];
    let mut scenario_payouts = if liabilities.is_some() {
        Vec::with_capacity(paths)
    } else {
        Vec::new()
    };
    let sigma = runtime_state
        .volatility
        .sigma
        .max(config.volatility.variance_floor.sqrt());

    if config.simulation.touch_mode == TouchMode::Continuous {
        let sigma_sq = sigma.powi(2);
        let neg_two_inv_denom = -2.0 / (sigma_sq + config.simulation.bridge_epsilon);
        let mut w_bounds = vec![(0.0_f64, 0.0_f64); windows.len()];
        if config.simulation.use_antithetic && paths > 1 {
            let base_paths = paths.div_ceil(2);
            let anti_paths = paths / 2;
            let mut base_log_path = Vec::with_capacity(max_horizon + 1);
            let mut anti_log_path = Vec::with_capacity(max_horizon + 1);

            for base_index in 0..base_paths {
                let include_antithetic = base_index < anti_paths;
                fill_log_path_pair(
                    &mut base_log_path,
                    if include_antithetic {
                        Some(&mut anti_log_path)
                    } else {
                        None
                    },
                    spot,
                    max_horizon,
                    runtime_state,
                    config,
                    rng,
                );
                accumulate_log_path_stats(
                    &base_log_path,
                    cells,
                    windows,
                    &mut w_bounds,
                    neg_two_inv_denom,
                    &mut sum,
                    &mut squared_sum,
                    &mut scenario_payouts,
                    liabilities,
                    rng,
                );
                if include_antithetic {
                    accumulate_log_path_stats(
                        &anti_log_path,
                        cells,
                        windows,
                        &mut w_bounds,
                        neg_two_inv_denom,
                        &mut sum,
                        &mut squared_sum,
                        &mut scenario_payouts,
                        liabilities,
                        rng,
                    );
                }
            }
        } else {
            let mut log_path = Vec::with_capacity(max_horizon + 1);
            for _ in 0..paths {
                fill_log_path_pair(
                    &mut log_path,
                    None,
                    spot,
                    max_horizon,
                    runtime_state,
                    config,
                    rng,
                );
                accumulate_log_path_stats(
                    &log_path,
                    cells,
                    windows,
                    &mut w_bounds,
                    neg_two_inv_denom,
                    &mut sum,
                    &mut squared_sum,
                    &mut scenario_payouts,
                    liabilities,
                    rng,
                );
            }
        }
        return BatchStats {
            sum,
            squared_sum,
            scenario_payouts,
        };
    }

    if config.simulation.use_antithetic && paths > 1 {
        let base_paths = paths.div_ceil(2);
        let anti_paths = paths / 2;
        let mut base_path = Vec::with_capacity(max_horizon + 1);
        let mut antithetic_path = Vec::with_capacity(max_horizon + 1);

        for base_index in 0..base_paths {
            let include_antithetic = base_index < anti_paths;
            fill_path_pair(
                &mut base_path,
                Some(&mut antithetic_path),
                spot,
                max_horizon,
                runtime_state,
                config,
                include_antithetic,
                rng,
            );

            accumulate_path_stats(
                &base_path,
                cells,
                windows,
                sigma,
                config.volatility.variance_floor,
                config.simulation.bridge_epsilon,
                config.simulation.touch_mode,
                &mut sum,
                &mut squared_sum,
                &mut scenario_payouts,
                liabilities,
                rng,
            );

            if include_antithetic {
                accumulate_path_stats(
                    &antithetic_path,
                    cells,
                    windows,
                    sigma,
                    config.volatility.variance_floor,
                    config.simulation.bridge_epsilon,
                    config.simulation.touch_mode,
                    &mut sum,
                    &mut squared_sum,
                    &mut scenario_payouts,
                    liabilities,
                    rng,
                );
            }
        }
    } else {
        let mut path = Vec::with_capacity(max_horizon + 1);
        for _ in 0..paths {
            fill_path_pair(
                &mut path,
                None,
                spot,
                max_horizon,
                runtime_state,
                config,
                false,
                rng,
            );
            accumulate_path_stats(
                &path,
                cells,
                windows,
                sigma,
                config.volatility.variance_floor,
                config.simulation.bridge_epsilon,
                config.simulation.touch_mode,
                &mut sum,
                &mut squared_sum,
                &mut scenario_payouts,
                liabilities,
                rng,
            );
        }
    }

    BatchStats { sum, squared_sum, scenario_payouts }
}

fn accumulate_path_stats(
    path: &[f64],
    cells: &[CellEval],
    windows: &[WindowEval],
    sigma: f64,
    variance_floor: f64,
    bridge_epsilon: f64,
    touch_mode: TouchMode,
    sum: &mut [f64],
    squared_sum: &mut [f64],
    scenario_payouts: &mut Vec<f64>,
    liabilities: Option<&[f64]>,
    rng: &mut impl Rng,
) {
    let mut scenario_payout = 0.0;
    let sigma_sq = sigma.max(variance_floor.sqrt()).powi(2);

    for cell in cells {
        let window = windows[cell.window_index];
        let p = match touch_mode {
            TouchMode::Continuous => window_hit_probability_steps(
                path,
                cell.lower,
                cell.upper,
                window.start,
                window.end,
                sigma_sq,
                bridge_epsilon,
            ),
            TouchMode::Discrete => window_hit_discrete_steps(
                path,
                cell.lower,
                cell.upper,
                window.start,
                window.end,
            ),
        };
        sum[cell.cell_index] += p;
        squared_sum[cell.cell_index] += p * p;
        if let Some(liabilities) = liabilities {
            if liabilities[cell.cell_index] > 0.0
                && rng.random_bool(p.clamp(0.0, 1.0))
            {
                scenario_payout += liabilities[cell.cell_index];
            }
        }
    }

    if liabilities.is_some() {
        scenario_payouts.push(scenario_payout);
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_log_path_stats(
    log_path: &[f64],
    cells: &[CellEval],
    windows: &[WindowEval],
    w_bounds: &mut [(f64, f64)],
    neg_two_inv_denom: f64,
    sum: &mut [f64],
    squared_sum: &mut [f64],
    scenario_payouts: &mut Vec<f64>,
    liabilities: Option<&[f64]>,
    rng: &mut impl Rng,
) {
    let last_idx = log_path.len().saturating_sub(1);
    for (w_idx, window) in windows.iter().enumerate() {
        let s_idx = window.start.min(last_idx);
        let e_idx = window.end.min(last_idx);
        let mut lo = log_path[s_idx];
        let mut hi = lo;
        for &val in &log_path[s_idx..=e_idx] {
            if val < lo {
                lo = val;
            }
            if val > hi {
                hi = val;
            }
        }
        w_bounds[w_idx] = (lo, hi);
    }

    let mut scenario_payout = 0.0;
    for cell in cells {
        let (w_min, w_max) = w_bounds[cell.window_index];
        let p = if w_max >= cell.log_lower && w_min <= cell.log_upper {
            1.0
        } else if w_max < cell.log_lower {
            let d = cell.log_lower - w_max;
            if neg_two_inv_denom * d * d <= BRIDGE_PRUNE_EXPONENT {
                0.0
            } else {
                let window = windows[cell.window_index];
                window_hit_probability_log_steps(
                    log_path,
                    cell.log_lower,
                    cell.log_upper,
                    window.start,
                    window.end,
                    neg_two_inv_denom,
                )
            }
        } else {
            let d = w_min - cell.log_upper;
            if neg_two_inv_denom * d * d <= BRIDGE_PRUNE_EXPONENT {
                0.0
            } else {
                let window = windows[cell.window_index];
                window_hit_probability_log_steps(
                    log_path,
                    cell.log_lower,
                    cell.log_upper,
                    window.start,
                    window.end,
                    neg_two_inv_denom,
                )
            }
        };
        if p > 0.0 {
            sum[cell.cell_index] += p;
            squared_sum[cell.cell_index] += p * p;
            if let Some(liabilities) = liabilities {
                if liabilities[cell.cell_index] > 0.0 && rng.random_bool(p.clamp(0.0, 1.0)) {
                    scenario_payout += liabilities[cell.cell_index];
                }
            }
        }
    }

    if liabilities.is_some() {
        scenario_payouts.push(scenario_payout);
    }
}

fn precompute_windows(config: &GridConfig, max_horizon: usize) -> Vec<WindowEval> {
    let n_steps = max_horizon;
    config
        .spec
        .windows
        .iter()
        .map(|window| WindowEval {
            start: (window.start_secs.saturating_add(config.simulation.lock_offset) as usize)
                .min(n_steps),
            end: (window.end_secs.saturating_add(config.simulation.lock_offset) as usize)
                .min(n_steps),
        })
        .collect()
}

fn precompute_cells(bands: &[PriceBand], spot: f64, config: &GridConfig) -> Vec<CellEval> {
    let window_count = config.spec.windows.len();
    let base_spot = spot.max(config.volatility.variance_floor);
    let mut cells = Vec::with_capacity(bands.len() * window_count);
    for band in bands {
        let row_offset = band.row_index * window_count;
        let log_lower = (band.lower.max(1e-9) / base_spot).ln();
        let log_upper = (band.upper.max(1e-9) / base_spot).ln();
        for window_index in 0..window_count {
            cells.push(CellEval {
                cell_index: row_offset + window_index,
                lower: band.lower,
                upper: band.upper,
                log_lower,
                log_upper,
                window_index,
            });
        }
    }
    cells
}

fn mix_seed(base_seed: u64, stream: u64) -> u64 {
    let mixed = base_seed
        ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ 0xBF58_476D_1CE4_E5B9;
    mixed ^ (mixed >> 30).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

fn thresholds_met(
    total_sum: &[f64],
    total_sq_sum: &[f64],
    total_paths: usize,
    config: &GridConfig,
) -> bool {
    let n = total_paths as f64;
    for cell_index in 0..total_sum.len() {
        let mean = total_sum[cell_index] / n;
        let variance = (total_sq_sum[cell_index] / n) - (mean * mean);
        let se = variance.max(0.0).sqrt() / n.sqrt();
        let rel = se / mean.max(config.simulation.p_floor);
        if se > config.simulation.se_abs || rel > config.simulation.se_rel {
            return false;
        }
    }
    true
}

#[cfg(test)]
fn simulate_path_pair(
    spot: f64,
    horizon_secs: usize,
    runtime_state: &RuntimeState,
    config: &GridConfig,
    include_antithetic: bool,
    rng: &mut impl Rng,
) -> (Vec<f64>, Option<Vec<f64>>) {
    let mut path = Vec::with_capacity(horizon_secs + 1);
    let mut antithetic_path = include_antithetic.then(|| Vec::with_capacity(horizon_secs + 1));
    fill_path_pair(
        &mut path,
        antithetic_path.as_mut(),
        spot,
        horizon_secs,
        runtime_state,
        config,
        include_antithetic,
        rng,
    );

    (path, antithetic_path)
}

fn fill_log_path_pair(
    log_path: &mut Vec<f64>,
    mut antithetic_log_path: Option<&mut Vec<f64>>,
    spot: f64,
    horizon_secs: usize,
    runtime_state: &RuntimeState,
    config: &GridConfig,
    rng: &mut impl Rng,
) {
    log_path.clear();
    if let Some(anti_path) = antithetic_log_path.as_mut() {
        anti_path.clear();
    }

    let base_spot = spot.max(config.volatility.variance_floor);
    let rel_floor = (1e-9_f64 / base_spot).ln();
    let mut log_price = 0.0_f64;
    let mut antithetic_log_price = 0.0_f64;
    let normal = StandardNormal;
    let sigma = runtime_state
        .volatility
        .sigma
        .max(config.volatility.variance_floor.sqrt());
    let lambda_intensity = runtime_state
        .volatility
        .jump_intensity
        .max(config.volatility.jump.hawkes_mu0);
    let p_jump = if config.volatility.jump.enabled && lambda_intensity > 0.0 {
        1.0 - (-lambda_intensity).exp()
    } else {
        0.0
    };

    log_path.push(log_price);
    if let Some(anti_path) = antithetic_log_path.as_mut() {
        anti_path.push(antithetic_log_price);
    }

    for _ in 0..horizon_secs {
        let z: f64 = normal.sample(rng);
        let diffusion = config.simulation.mu + sigma * z;
        let jump = sample_jump(p_jump, runtime_state, config, rng);
        log_price += diffusion + jump;
        log_path.push(log_price.max(rel_floor));

        if let Some(anti_path) = antithetic_log_path.as_mut() {
            let antithetic_diffusion = config.simulation.mu - sigma * z;
            antithetic_log_price += antithetic_diffusion + jump;
            anti_path.push(antithetic_log_price.max(rel_floor));
        }
    }
}

fn fill_path_pair(
    path: &mut Vec<f64>,
    mut antithetic_path: Option<&mut Vec<f64>>,
    spot: f64,
    horizon_secs: usize,
    runtime_state: &RuntimeState,
    config: &GridConfig,
    _include_antithetic: bool,
    rng: &mut impl Rng,
) {
    path.clear();
    if let Some(anti_path) = antithetic_path.as_mut() {
        anti_path.clear();
    }

    let mut log_price = spot.max(config.volatility.variance_floor).ln();
    let mut antithetic_log_price = log_price;
    let normal = StandardNormal;
    let sigma = runtime_state
        .volatility
        .sigma
        .max(config.volatility.variance_floor.sqrt());
    let lambda_intensity = runtime_state
        .volatility
        .jump_intensity
        .max(config.volatility.jump.hawkes_mu0);
    let p_jump = if config.volatility.jump.enabled && lambda_intensity > 0.0 {
        1.0 - (-lambda_intensity).exp()
    } else {
        0.0
    };

    path.push(spot);
    let mut antithetic_path = antithetic_path;
    if let Some(anti_path) = &mut antithetic_path {
        anti_path.push(spot);
    }

    for _ in 0..horizon_secs {
        let z: f64 = normal.sample(rng);
        let diffusion = config.simulation.mu + sigma * z;
        let jump = sample_jump(p_jump, runtime_state, config, rng);
        let log_return = diffusion + jump;

        log_price += log_return;
        path.push(log_price.exp().max(1e-9));

        if let Some(anti_path) = &mut antithetic_path {
            let antithetic_diffusion = config.simulation.mu - sigma * z;
            antithetic_log_price += antithetic_diffusion + jump;
            anti_path.push(antithetic_log_price.exp().max(1e-9));
        }
    }
}

fn sample_jump(
    jump_probability: f64,
    runtime_state: &RuntimeState,
    config: &GridConfig,
    rng: &mut impl Rng,
) -> f64 {
    if !config.volatility.jump.enabled || jump_probability <= 0.0 {
        return 0.0;
    }

    if !rng.random_bool(jump_probability.clamp(0.0, 1.0)) {
        return 0.0;
    }

    runtime_state
        .jump_sampler
        .sample(1, rng, runtime_state.volatility.sigma_raw)
        .into_iter()
        .next()
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::grid::types::{JumpConfig, SimulationConfig, VolatilityConfig};

    fn test_runtime_state(config: &GridConfig) -> RuntimeState {
        RuntimeState {
            volatility: super::super::types::VolatilitySnapshot {
                variance: 0.0,
                sigma_raw: config.volatility.sigma_ref,
                sigma: config.volatility.sigma_ref,
                jump_intensity: 0.0,
                jump_flag: false,
            },
            ..RuntimeState::default()
        }
    }

    #[test]
    fn antithetic_pair_uses_opposite_diffusion_shocks() {
        let config = GridConfig {
            simulation: SimulationConfig {
                mu: 0.001,
                use_antithetic: true,
                ..SimulationConfig::default()
            },
            volatility: VolatilityConfig {
                jump: JumpConfig {
                    enabled: false,
                    ..JumpConfig::default()
                },
                ..VolatilityConfig::default()
            },
            ..GridConfig::default()
        };
        let runtime_state = test_runtime_state(&config);
        let mut rng = StdRng::seed_from_u64(7);

        let (base, anti) =
            simulate_path_pair(50_000.0, 8, &runtime_state, &config, true, &mut rng);
        let anti = anti.unwrap();

        for step in 1..base.len() {
            let base_return = (base[step] / base[step - 1]).ln();
            let anti_return = (anti[step] / anti[step - 1]).ln();
            let expected_sum = 2.0 * config.simulation.mu;

            assert!((base_return + anti_return - expected_sum).abs() < 1e-12);
        }
    }

    #[test]
    fn fast_log_bridge_pruning_matches_reference_touch_across_24_windows() {
        // Pruning bound check: below 1.0_f64, ULP spacing is 2^-53 ≈ 1.11e-16 (half of f64::EPSILON),
        // so rounding 1.0 - x to 1.0 requires x < 2^-54 = 0.25 * f64::EPSILON ≈ 5.55e-17.
        // exp(-38.0) ≈ 3.14e-17 < 2^-54, so 1.0 - exp(BRIDGE_PRUNE_EXPONENT) == 1.0 bit-for-bit (0 ULP cut).
        let half_ulp_below_one = 0.25 * f64::EPSILON;
        assert!(BRIDGE_PRUNE_EXPONENT.exp() < half_ulp_below_one);
        assert_eq!(1.0_f64 - BRIDGE_PRUNE_EXPONENT.exp(), 1.0_f64);

        let mut cfg = crate::grid::stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        cfg.spec.windows = (0..24u32)
            .map(|i| crate::grid::types::TimeWindow {
                start_secs: 5 + i * 5,
                end_secs: 10 + i * 5,
            })
            .collect();
        let runtime_state = test_runtime_state(&cfg);
        let spot = 140.00_f64;
        let bands: Vec<PriceBand> = (0..20usize)
            .map(|row_index| {
                let lower = 140.00 + (row_index as f64 - 9.0) * 0.03;
                PriceBand {
                    row_index,
                    lower,
                    upper: lower + 0.03,
                }
            })
            .collect();
        let max_horizon = 125usize;
        let windows = precompute_windows(&cfg, max_horizon);
        let cells = precompute_cells(&bands, spot, &cfg);
        let sigma = runtime_state.volatility.sigma;
        let neg_two_inv_denom = -2.0 / (sigma * sigma + cfg.simulation.bridge_epsilon);

        let mut rng_ref = StdRng::seed_from_u64(99);
        let mut rng_fast = StdRng::seed_from_u64(99);
        let mut path = Vec::with_capacity(max_horizon + 1);
        let mut log_path = Vec::with_capacity(max_horizon + 1);
        let mut sum_ref = vec![0.0_f64; cells.len()];
        let mut sq_ref = vec![0.0_f64; cells.len()];
        let mut sum_fast = vec![0.0_f64; cells.len()];
        let mut sq_fast = vec![0.0_f64; cells.len()];
        let mut pay_ref = Vec::new();
        let mut pay_fast = Vec::new();
        let mut w_bounds = vec![(0.0_f64, 0.0_f64); windows.len()];

        for _ in 0..40 {
            fill_path_pair(&mut path, None, spot, max_horizon, &runtime_state, &cfg, false, &mut rng_ref);
            fill_log_path_pair(&mut log_path, None, spot, max_horizon, &runtime_state, &cfg, &mut rng_fast);
            accumulate_path_stats(
                &path,
                &cells,
                &windows,
                sigma,
                cfg.volatility.variance_floor,
                cfg.simulation.bridge_epsilon,
                TouchMode::Continuous,
                &mut sum_ref,
                &mut sq_ref,
                &mut pay_ref,
                None,
                &mut rng_ref,
            );
            accumulate_log_path_stats(
                &log_path,
                &cells,
                &windows,
                &mut w_bounds,
                neg_two_inv_denom,
                &mut sum_fast,
                &mut sq_fast,
                &mut pay_fast,
                None,
                &mut rng_fast,
            );
        }
        // Tolerance derivation:
        // The reference path accumulates ~2e-4 returns onto ln(spot) ≈ 4.94 (ULP ≈ 8.9e-16),
        // then rounds through exp() and (L / P).ln(), introducing Δd ≈ 1e-15 in log-distance.
        // In p_hit = exp(-2*d0*d1/sigma^2), max |Δp_hit| <= 1.72 * (Δd / sigma) ≈ 8e-12 per step
        // (for NVDA sigma = 2.3e-4), giving ~2e-12..1e-11 relative drift across 5 steps.
        // We assert <= 1e-10 relative error and <= 1e-11 absolute error for cells near 0.
        for i in 0..cells.len() {
            let diff = (sum_ref[i] - sum_fast[i]).abs();
            let tol = 1e-11_f64.max(1e-10 * sum_ref[i].abs());
            assert!(
                diff <= tol,
                "cell {i} mismatch: ref={} fast={} diff={diff:e} tol={tol:e}",
                sum_ref[i],
                sum_fast[i]
            );
        }
    }

    #[test]
    fn default_six_window_nvda_quote_matches_pre_e13_reference() {
        let windows: Vec<crate::grid::types::TimeWindow> = (0..6u32)
            .map(|i| crate::grid::types::TimeWindow {
                start_secs: 5 + i * 5,
                end_secs: 10 + i * 5,
            })
            .collect();
        let mut cfg = crate::grid::stock_grid_config_for_ticker("NVDA", 3, windows);
        cfg.simulation.mc_n_min = 500;
        cfg.simulation.mc_n_max = 500;
        cfg.simulation.seed = Some(42);
        cfg.simulation.use_parallel = false;
        assert_eq!(cfg.simulation.quote_window_count, 6);

        let spot = 157.53_f64;
        let runtime_state = test_runtime_state(&cfg);
        let bands: Vec<PriceBand> = (0..cfg.spec.rows)
            .map(|row_index| {
                let lower = 157.50 - 0.01 + (row_index as f64 - cfg.spec.center_row as f64) * 0.03;
                PriceBand {
                    row_index,
                    lower,
                    upper: lower + 0.03,
                }
            })
            .collect();
        let max_horizon = 35usize;

        // Production fast path.
        let fast_summary = simulate_summary(spot, &bands, &cfg, &runtime_state, max_horizon, None);

        // Pre-E13 reference path (antithetic fill_path_pair + accumulate_path_stats, same seed).
        let w_eval = precompute_windows(&cfg, max_horizon);
        let c_eval = precompute_cells(&bands, spot, &cfg);
        let sigma = runtime_state
            .volatility
            .sigma
            .max(cfg.volatility.variance_floor.sqrt());
        let mut rng_ref = StdRng::seed_from_u64(42);
        let mut sum_ref = vec![0.0_f64; c_eval.len()];
        let mut sq_ref = vec![0.0_f64; c_eval.len()];
        let mut pay_ref = Vec::new();
        let mut base_path = Vec::with_capacity(max_horizon + 1);
        let mut anti_path = Vec::with_capacity(max_horizon + 1);
        let base_paths = 500usize.div_ceil(2);
        let anti_paths = 500usize / 2;
        for base_index in 0..base_paths {
            let include_anti = base_index < anti_paths;
            fill_path_pair(
                &mut base_path,
                Some(&mut anti_path),
                spot,
                max_horizon,
                &runtime_state,
                &cfg,
                include_anti,
                &mut rng_ref,
            );
            accumulate_path_stats(
                &base_path,
                &c_eval,
                &w_eval,
                sigma,
                cfg.volatility.variance_floor,
                cfg.simulation.bridge_epsilon,
                TouchMode::Continuous,
                &mut sum_ref,
                &mut sq_ref,
                &mut pay_ref,
                None,
                &mut rng_ref,
            );
            if include_anti {
                accumulate_path_stats(
                    &anti_path,
                    &c_eval,
                    &w_eval,
                    sigma,
                    cfg.volatility.variance_floor,
                    cfg.simulation.bridge_epsilon,
                    TouchMode::Continuous,
                    &mut sum_ref,
                    &mut sq_ref,
                    &mut pay_ref,
                    None,
                    &mut rng_ref,
                );
            }
        }

        let vol = runtime_state.volatility;
        let window_count = cfg.spec.windows.len();
        for row_index in 0..cfg.spec.rows {
            for window_index in 0..window_count {
                let ref_raw = sum_ref[row_index * window_count + window_index] / 500.0;
                let fast_raw = fast_summary.mean_probabilities[row_index][window_index];
                let raw_diff = (ref_raw - fast_raw).abs();
                let raw_tol = 1e-11_f64.max(1e-10 * ref_raw.abs());
                assert!(
                    raw_diff <= raw_tol,
                    "row={row_index} win={window_index} raw ref={ref_raw} fast={fast_raw}"
                );

                let ref_p = crate::grid::compute_quote_probability(
                    ref_raw,
                    row_index,
                    window_index,
                    cfg.spec.windows[window_index],
                    vol,
                    cfg.spec.center_row,
                    &cfg,
                );
                let fast_p = crate::grid::compute_quote_probability(
                    fast_raw,
                    row_index,
                    window_index,
                    cfg.spec.windows[window_index],
                    vol,
                    cfg.spec.center_row,
                    &cfg,
                );
                for m_base in [cfg.pricing.m_base, cfg.pricing.off_hours_m_base.unwrap()] {
                    let ref_m = crate::grid::price_multiplier_with_margin(
                        ref_p, vol, 0.0, 1.0, m_base, &cfg,
                    );
                    let fast_m = crate::grid::price_multiplier_with_margin(
                        fast_p, vol, 0.0, 1.0, m_base, &cfg,
                    );
                    let m_diff = (ref_m - fast_m).abs();
                    let m_tol = 1e-11_f64.max(1e-10 * ref_m.abs());
                    assert!(
                        m_diff <= m_tol,
                        "row={row_index} win={window_index} m_base={m_base} mult ref={ref_m} fast={fast_m} diff={m_diff:e}"
                    );
                }
            }
        }
    }
}
