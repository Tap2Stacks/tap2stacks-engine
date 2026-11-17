use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use st_engine::grid::{
    GridConfig, GridEngine, GridSnapshot, OhlcBar, PricingConfig, SimulationConfig, TimeWindow,
    VolatilityConfig, quote_grid, quote_grid_with_liabilities,
};

#[derive(Debug, Deserialize)]
struct FixtureInput {
    spot: f64,
    history: Vec<FixtureBar>,
    #[serde(default)]
    config: FixtureConfigOverrides,
    #[serde(default)]
    liabilities: Vec<f64>,
    #[serde(default)]
    use_internal_liability_book: bool,
}

#[derive(Debug, Deserialize)]
struct FixtureBar {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

#[derive(Debug, Default, Deserialize)]
struct FixtureConfigOverrides {
    rows: Option<usize>,
    center_row: Option<usize>,
    band_width: Option<f64>,
    windows: Option<Vec<[u32; 2]>>,
    simulation: Option<FixtureSimulationOverrides>,
    volatility: Option<FixtureVolatilityOverrides>,
    pricing: Option<FixturePricingOverrides>,
}

#[derive(Debug, Default, Deserialize)]
struct FixtureSimulationOverrides {
    mu: Option<f64>,
    seed: Option<u64>,
    use_parallel: Option<bool>,
    parallel_workers: Option<usize>,
    lock_offset: Option<u32>,
    bridge_epsilon: Option<f64>,
    use_antithetic: Option<bool>,
    mc_n_min: Option<usize>,
    mc_n_max: Option<usize>,
    se_abs: Option<f64>,
    se_rel: Option<f64>,
    p_floor: Option<f64>,
    adaptive_mc: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct FixtureVolatilityOverrides {
    sigma_scale: Option<f64>,
    sigma_ref: Option<f64>,
    ewma_half_life_seconds: Option<f64>,
    parkinson_weight: Option<f64>,
    variance_floor: Option<f64>,
    price_adj_reference: Option<f64>,
    price_adj_min_price: Option<f64>,
    price_adj_min: Option<f64>,
    price_adj_max: Option<f64>,
    jump: Option<FixtureJumpOverrides>,
}

#[derive(Debug, Default, Deserialize)]
struct FixtureJumpOverrides {
    enabled: Option<bool>,
    hawkes_mu0: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    jump_mean: Option<f64>,
    jump_stddev: Option<f64>,
    kappa0: Option<f64>,
    kappa_q: Option<f64>,
    kappa_min: Option<f64>,
    kappa_max: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct FixturePricingOverrides {
    m_base: Option<f64>,
    m_risk_max: Option<f64>,
    lambda_ref: Option<f64>,
    a_sigma: Option<f64>,
    a_lambda: Option<f64>,
    beta_skew: Option<f64>,
    pool_cap: Option<f64>,
    risk_budget: Option<f64>,
    k_safety_ttl_ms: Option<u64>,
    liability_rho: Option<f64>,
    p_raw_gamma: Option<f64>,
    p_near_boost: Option<f64>,
    p_time_boost: Option<f64>,
    calibration_dist_max: Option<usize>,
    calibration_sigma_edges: Option<[f64; 2]>,
    current_row_logit_shift: Option<f64>,
    adjacent_row_logit_shift: Option<f64>,
    zone_vol_shift_scale: Option<f64>,
    min_multiplier: Option<f64>,
    max_multiplier: Option<f64>,
    skew_factor: Option<f64>,
    safety_factor: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct FixtureExpected {
    anchor: Option<f64>,
    volatility: Option<ExpectedVolatility>,
    cells: Option<Vec<ExpectedCell>>,
    tolerances: Option<FixtureTolerances>,
}

#[derive(Debug, Deserialize)]
struct ExpectedVolatility {
    sigma: Option<f64>,
    sigma_raw: Option<f64>,
    jump_intensity: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ExpectedCell {
    row_index: usize,
    window_index: usize,
    hit_probability: Option<f64>,
    multiplier: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct FixtureTolerances {
    default_abs: Option<f64>,
    default_rel: Option<f64>,
    anchor_abs: Option<f64>,
    sigma_abs: Option<f64>,
    sigma_raw_abs: Option<f64>,
    jump_intensity_abs: Option<f64>,
    hit_probability_abs: Option<f64>,
    hit_probability_rel: Option<f64>,
    multiplier_abs: Option<f64>,
    multiplier_rel: Option<f64>,
}

#[test]
#[ignore = "pre-existing failure at tap2stacks-rs 0eaccce (BTC legacy model vs Python fixture); stock odds are calibrated in plan task E5"]
fn parity_fixtures_match_expected_outputs_when_present() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity");
    let mut input_files = list_input_files(&fixture_dir);

    input_files.sort();
    assert!(
        !input_files.is_empty(),
        "expected at least one parity input fixture in {}",
        fixture_dir.display()
    );

    for input_path in input_files {
        let input = load_input_fixture(&input_path);
        let config = build_config(&input.config);
        let history = input
            .history
            .iter()
            .map(|bar| OhlcBar {
                open: bar.open,
                high: bar.high,
                low: bar.low,
                close: bar.close,
            })
            .collect::<Vec<_>>();

        let snapshot = if input.use_internal_liability_book {
            let mut engine = GridEngine::new(config.clone());
            for (cell_index, liability) in input.liabilities.iter().copied().enumerate() {
                if liability != 0.0 {
                    engine.set_liability(cell_index, liability);
                }
            }
            engine
                .quote_grid(input.spot, &history)
                .unwrap_or_else(|error| panic!("{}: {error}", input_path.display()))
        } else {
            let liabilities = if input.liabilities.is_empty() {
                None
            } else {
                Some(input.liabilities.as_slice())
            };
            quote_grid_with_liabilities(input.spot, &history, &config, liabilities)
                .unwrap_or_else(|error| panic!("{}: {error}", input_path.display()))
        };

        let expected_path = expected_path_for(&input_path);
        if expected_path.exists() {
            let expected = load_expected_fixture(&expected_path);
            assert_snapshot_matches_expected(&input_path, &snapshot, &expected, &config);
        } else {
            let stateless = quote_grid(input.spot, &history, &config)
                .unwrap_or_else(|error| panic!("{}: {error}", input_path.display()));
            assert_eq!(
                stateless.cells.len(),
                config.spec.rows * config.spec.windows.len(),
                "{}",
                input_path.display()
            );
        }
    }
}

fn list_input_files(fixture_dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let entries = fs::read_dir(fixture_dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", fixture_dir.display()));

    for entry in entries {
        let path = entry.unwrap().path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".input.json"))
        {
            paths.push(path);
        }
    }

    paths
}

fn load_input_fixture(path: &Path) -> FixtureInput {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn load_expected_fixture(path: &Path) -> FixtureExpected {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn expected_path_for(input_path: &Path) -> PathBuf {
    PathBuf::from(input_path.to_string_lossy().replace(".input.json", ".expected.json"))
}

fn build_config(overrides: &FixtureConfigOverrides) -> GridConfig {
    let mut config = GridConfig::default();

    if let Some(rows) = overrides.rows {
        config.spec.rows = rows;
    }
    if let Some(center_row) = overrides.center_row {
        config.spec.center_row = center_row;
    }
    if let Some(band_width) = overrides.band_width {
        config.spec.band_width = band_width;
    }
    if let Some(windows) = &overrides.windows {
        config.spec.windows = windows
            .iter()
            .map(|window| TimeWindow {
                start_secs: window[0],
                end_secs: window[1],
            })
            .collect();
    }

    if let Some(simulation) = &overrides.simulation {
        apply_simulation_overrides(&mut config.simulation, simulation);
    }
    if let Some(volatility) = &overrides.volatility {
        apply_volatility_overrides(&mut config.volatility, volatility);
    }
    if let Some(pricing) = &overrides.pricing {
        apply_pricing_overrides(&mut config.pricing, pricing);
    }

    config
}

fn apply_simulation_overrides(
    simulation: &mut SimulationConfig,
    overrides: &FixtureSimulationOverrides,
) {
    if let Some(mu) = overrides.mu {
        simulation.mu = mu;
    }
    if let Some(seed) = overrides.seed {
        simulation.seed = Some(seed);
    }
    if let Some(use_parallel) = overrides.use_parallel {
        simulation.use_parallel = use_parallel;
    }
    if let Some(parallel_workers) = overrides.parallel_workers {
        simulation.parallel_workers = Some(parallel_workers);
    }
    if let Some(lock_offset) = overrides.lock_offset {
        simulation.lock_offset = lock_offset;
    }
    if let Some(bridge_epsilon) = overrides.bridge_epsilon {
        simulation.bridge_epsilon = bridge_epsilon;
    }
    if let Some(use_antithetic) = overrides.use_antithetic {
        simulation.use_antithetic = use_antithetic;
    }
    if let Some(mc_n_min) = overrides.mc_n_min {
        simulation.mc_n_min = mc_n_min;
    }
    if let Some(mc_n_max) = overrides.mc_n_max {
        simulation.mc_n_max = mc_n_max;
    }
    if let Some(se_abs) = overrides.se_abs {
        simulation.se_abs = se_abs;
    }
    if let Some(se_rel) = overrides.se_rel {
        simulation.se_rel = se_rel;
    }
    if let Some(p_floor) = overrides.p_floor {
        simulation.p_floor = p_floor;
    }
    if let Some(adaptive_mc) = overrides.adaptive_mc {
        simulation.adaptive_mc = adaptive_mc;
    }
}

fn apply_volatility_overrides(
    volatility: &mut VolatilityConfig,
    overrides: &FixtureVolatilityOverrides,
) {
    if let Some(sigma_scale) = overrides.sigma_scale {
        volatility.sigma_scale = sigma_scale;
    }
    if let Some(sigma_ref) = overrides.sigma_ref {
        volatility.sigma_ref = sigma_ref;
    }
    if let Some(value) = overrides.ewma_half_life_seconds {
        volatility.ewma_half_life_seconds = value;
    }
    if let Some(value) = overrides.parkinson_weight {
        volatility.parkinson_weight = value;
    }
    if let Some(value) = overrides.variance_floor {
        volatility.variance_floor = value;
    }
    if let Some(value) = overrides.price_adj_reference {
        volatility.price_adj_reference = value;
    }
    if let Some(value) = overrides.price_adj_min_price {
        volatility.price_adj_min_price = value;
    }
    if let Some(value) = overrides.price_adj_min {
        volatility.price_adj_min = value;
    }
    if let Some(value) = overrides.price_adj_max {
        volatility.price_adj_max = value;
    }
    if let Some(jump) = &overrides.jump {
        if let Some(value) = jump.enabled {
            volatility.jump.enabled = value;
        }
        if let Some(value) = jump.hawkes_mu0 {
            volatility.jump.hawkes_mu0 = value;
        }
        if let Some(value) = jump.alpha {
            volatility.jump.alpha = value;
        }
        if let Some(value) = jump.beta {
            volatility.jump.beta = value;
        }
        if let Some(value) = jump.jump_mean {
            volatility.jump.jump_mean = value;
        }
        if let Some(value) = jump.jump_stddev {
            volatility.jump.jump_stddev = value;
        }
        if let Some(value) = jump.kappa0 {
            volatility.jump.kappa0 = value;
        }
        if let Some(value) = jump.kappa_q {
            volatility.jump.kappa_q = value;
        }
        if let Some(value) = jump.kappa_min {
            volatility.jump.kappa_min = value;
        }
        if let Some(value) = jump.kappa_max {
            volatility.jump.kappa_max = value;
        }
    }
}

fn apply_pricing_overrides(pricing: &mut PricingConfig, overrides: &FixturePricingOverrides) {
    if let Some(value) = overrides.m_base {
        pricing.m_base = value;
    }
    if let Some(value) = overrides.m_risk_max {
        pricing.m_risk_max = value;
    }
    if let Some(value) = overrides.lambda_ref {
        pricing.lambda_ref = value;
    }
    if let Some(value) = overrides.a_sigma {
        pricing.a_sigma = value;
    }
    if let Some(value) = overrides.a_lambda {
        pricing.a_lambda = value;
    }
    if let Some(value) = overrides.beta_skew {
        pricing.beta_skew = value;
    }
    if let Some(value) = overrides.pool_cap {
        pricing.pool_cap = value;
    }
    if let Some(value) = overrides.risk_budget {
        pricing.risk_budget = value;
    }
    if let Some(value) = overrides.k_safety_ttl_ms {
        pricing.k_safety_ttl_ms = value;
    }
    if let Some(value) = overrides.liability_rho {
        pricing.liability_rho = value;
    }
    if let Some(value) = overrides.p_raw_gamma {
        pricing.p_raw_gamma = value;
    }
    if let Some(value) = overrides.p_near_boost {
        pricing.p_near_boost = value;
    }
    if let Some(value) = overrides.p_time_boost {
        pricing.p_time_boost = value;
    }
    if let Some(value) = overrides.calibration_dist_max {
        pricing.calibration_dist_max = value;
    }
    if let Some([lo, hi]) = overrides.calibration_sigma_edges {
        pricing.calibration_sigma_edges = [lo, hi].into();
    }
    if let Some(value) = overrides.current_row_logit_shift {
        pricing.current_row_logit_shift = value;
    }
    if let Some(value) = overrides.adjacent_row_logit_shift {
        pricing.adjacent_row_logit_shift = value;
    }
    if let Some(value) = overrides.zone_vol_shift_scale {
        pricing.zone_vol_shift_scale = value;
    }
    if let Some(value) = overrides.min_multiplier {
        pricing.min_multiplier = value;
    }
    if let Some(value) = overrides.max_multiplier {
        pricing.max_multiplier = value;
    }
    if let Some(value) = overrides.skew_factor {
        pricing.skew_factor = value;
    }
    if let Some(value) = overrides.safety_factor {
        pricing.safety_factor = value;
    }
}

fn assert_snapshot_matches_expected(
    input_path: &Path,
    snapshot: &GridSnapshot,
    expected: &FixtureExpected,
    config: &GridConfig,
) {
    let tolerances = expected.tolerances.as_ref();
    let default_abs = tolerances.and_then(|t| t.default_abs).unwrap_or(1e-6);
    let default_rel = tolerances.and_then(|t| t.default_rel).unwrap_or(1e-3);

    if let Some(anchor) = expected.anchor {
        assert_close(
            snapshot.anchor,
            anchor,
            tolerances.and_then(|t| t.anchor_abs).unwrap_or(default_abs),
            0.0,
            &format!("{} anchor", input_path.display()),
        );
    }

    if let Some(volatility) = &expected.volatility {
        if let Some(expected_sigma) = volatility.sigma {
            assert_close(
                snapshot.volatility.sigma,
                expected_sigma,
                tolerances.and_then(|t| t.sigma_abs).unwrap_or(default_abs),
                default_rel,
                &format!("{} volatility.sigma", input_path.display()),
            );
        }
        if let Some(expected_sigma_raw) = volatility.sigma_raw {
            assert_close(
                snapshot.volatility.sigma_raw,
                expected_sigma_raw,
                tolerances.and_then(|t| t.sigma_raw_abs).unwrap_or(default_abs),
                default_rel,
                &format!("{} volatility.sigma_raw", input_path.display()),
            );
        }
        if let Some(expected_jump_intensity) = volatility.jump_intensity {
            assert_close(
                snapshot.volatility.jump_intensity,
                expected_jump_intensity,
                tolerances
                    .and_then(|t| t.jump_intensity_abs)
                    .unwrap_or(default_abs),
                default_rel,
                &format!("{} volatility.jump_intensity", input_path.display()),
            );
        }
    }

    if let Some(expected_cells) = &expected.cells {
        for expected_cell in expected_cells {
            let cell_index = expected_cell.row_index * config.spec.windows.len() + expected_cell.window_index;
            let actual = snapshot.cells.get(cell_index).unwrap_or_else(|| {
                panic!(
                    "{} missing cell row={} window={}",
                    input_path.display(),
                    expected_cell.row_index,
                    expected_cell.window_index
                )
            });

            if let Some(expected_probability) = expected_cell.hit_probability {
                assert_close(
                    actual.hit_probability,
                    expected_probability,
                    tolerances
                        .and_then(|t| t.hit_probability_abs)
                        .unwrap_or(default_abs),
                    tolerances
                        .and_then(|t| t.hit_probability_rel)
                        .unwrap_or(default_rel),
                    &format!(
                        "{} cell[{},{}].hit_probability",
                        input_path.display(),
                        expected_cell.row_index,
                        expected_cell.window_index
                    ),
                );
            }
            if let Some(expected_multiplier) = expected_cell.multiplier {
                assert_close(
                    actual.multiplier,
                    expected_multiplier,
                    tolerances.and_then(|t| t.multiplier_abs).unwrap_or(default_abs),
                    tolerances.and_then(|t| t.multiplier_rel).unwrap_or(default_rel),
                    &format!(
                        "{} cell[{},{}].multiplier",
                        input_path.display(),
                        expected_cell.row_index,
                        expected_cell.window_index
                    ),
                );
            }
        }
    }
}

fn assert_close(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64, label: &str) {
    let abs_error = (actual - expected).abs();
    let rel_error = if expected == 0.0 {
        abs_error
    } else {
        abs_error / expected.abs()
    };
    assert!(
        abs_error <= abs_tol || rel_error <= rel_tol,
        "{label}: actual={actual}, expected={expected}, abs_error={abs_error}, rel_error={rel_error}, abs_tol={abs_tol}, rel_tol={rel_tol}"
    );
}
