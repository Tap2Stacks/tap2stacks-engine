use serde::Deserialize;
use std::fs;
use std::path::Path;

use st_engine::grid::{
    DebugPathTrace, GridConfig, OhlcBar, PriceBand, TimeWindow, debug_reconstruct_paths_from_returns,
    debug_reconstruct_paths_from_shocks, debug_simulate_paths, estimate_runtime_state,
    interval_hit_probabilities, quote_grid,
};

#[derive(Debug, Deserialize)]
struct FixtureInput {
    spot: f64,
    history: Vec<FixtureBar>,
    #[serde(default)]
    config: FixtureConfigOverrides,
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

#[derive(Debug, Deserialize)]
struct ForensicDump {
    sigma: f64,
    row_index: usize,
    window_index: usize,
    band: ForensicBand,
    paths: Vec<ForensicPath>,
}

#[derive(Debug, Deserialize)]
struct ForensicBand {
    lower: f64,
    upper: f64,
}

#[derive(Debug, Deserialize)]
struct ForensicPath {
    path: Vec<f64>,
    interval_p: Vec<f64>,
    pwin: f64,
}

#[derive(Debug, Deserialize)]
struct ShockDump {
    spot: f64,
    mu: f64,
    sigma: f64,
    eps: Vec<Vec<f64>>,
    jumps: Vec<Vec<f64>>,
    returns: Vec<Vec<f64>>,
    paths: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct ExpectedFixture {
    cells: Vec<ExpectedCell>,
}

#[derive(Debug, Deserialize)]
struct ExpectedCell {
    row_index: usize,
    window_index: usize,
    hit_probability: Option<f64>,
    multiplier: Option<f64>,
}

#[test]
#[ignore]
fn debug_mode10_smoke_fixture() {
    let (fixture, config, history) = load_fixture();

    let snapshot = quote_grid(fixture.spot, &history, &config).unwrap();
    println!("anchor={}", snapshot.anchor);
    println!(
        "sigma={} sigma_raw={} lambda={}",
        snapshot.volatility.sigma,
        snapshot.volatility.sigma_raw,
        snapshot.volatility.jump_intensity
    );
    for cell in snapshot.cells.iter().take(10) {
        println!(
            "row={} window=({},{}) p={} m={}",
            cell.band.row_index,
            cell.window.start_secs,
            cell.window.end_secs,
            cell.hit_probability,
            cell.multiplier
        );
    }
}

#[test]
#[ignore]
fn debug_mode10_smoke_cell_forensics() {
    let (fixture, config, history) = load_fixture();
    let runtime_state = estimate_runtime_state(fixture.spot, &history, &config).unwrap();
    let horizon = config
        .spec
        .windows
        .iter()
        .map(|window| window.end_secs)
        .max()
        .unwrap_or(0)
        .saturating_add(config.simulation.lock_offset) as usize;
    let traces = debug_simulate_paths(
        fixture.spot,
        &config,
        &runtime_state,
        horizon,
        5,
        7,
    );
    let band = band_for_row(fixture.spot, &config, 0);
    let window = config.spec.windows[0];

    println!(
        "forensics row=0 window=({},{}) band=[{},{}] sigma={} lock_offset={}",
        window.start_secs,
        window.end_secs,
        band.lower,
        band.upper,
        runtime_state.volatility.sigma,
        config.simulation.lock_offset
    );

    for (path_index, path) in traces.paths.iter().enumerate() {
        let interval_probabilities = interval_hit_probabilities(
            path,
            band,
            window,
            runtime_state.volatility.sigma,
            config.simulation.lock_offset,
            config.simulation.bridge_epsilon,
        );
        let pwin = 1.0
            - interval_probabilities
                .iter()
                .fold(1.0, |p_not_touch, p_hit| p_not_touch * (1.0 - p_hit));
        println!("path[{path_index}]={path:?}");
        println!("path[{path_index}].interval_p={interval_probabilities:?}");
        println!("path[{path_index}].pwin={pwin}");
    }
}

#[test]
#[ignore]
fn debug_mode10_smoke_endpoint_distribution() {
    let (fixture, config, history) = load_fixture();
    let runtime_state = estimate_runtime_state(fixture.spot, &history, &config).unwrap();
    let horizon = config
        .spec
        .windows
        .iter()
        .map(|window| window.end_secs)
        .max()
        .unwrap_or(0)
        .saturating_add(config.simulation.lock_offset) as usize;
    let traces = debug_simulate_paths(
        fixture.spot,
        &config,
        &runtime_state,
        horizon,
        256,
        12345,
    );

    for step in [6usize, 7, 8, 9, 10] {
        let mut values = traces
            .paths
            .iter()
            .map(|path| path[step])
            .collect::<Vec<_>>();
        values.sort_by(|left, right| left.partial_cmp(right).unwrap());
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let p10 = values[((n - 1) as f64 * 0.10).floor() as usize];
        let p50 = values[((n - 1) as f64 * 0.50).floor() as usize];
        let p90 = values[((n - 1) as f64 * 0.90).floor() as usize];
        println!(
            "RS {step} mean {mean} p10 {p10} p50 {p50} p90 {p90}"
        );
    }
}

#[test]
#[ignore]
fn compare_python_forensic_dump_for_row0_window0() {
    let (fixture, config, _history) = load_fixture();
    let dump_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/parity/mode10_smoke.row0_window0.forensic.json");
    let raw = fs::read_to_string(&dump_path).unwrap();
    let dump: ForensicDump = serde_json::from_str(&raw).unwrap();
    let band = PriceBand {
        row_index: dump.row_index,
        lower: dump.band.lower,
        upper: dump.band.upper,
    };
    let window = config.spec.windows[dump.window_index];

    for (index, forensic_path) in dump.paths.iter().enumerate() {
        let actual_interval = interval_hit_probabilities(
            &forensic_path.path,
            band,
            window,
            dump.sigma,
            config.simulation.lock_offset,
            config.simulation.bridge_epsilon,
        );
        assert_eq!(
            actual_interval.len(),
            forensic_path.interval_p.len(),
            "path[{index}] interval length mismatch"
        );
        for (step, (actual, expected)) in actual_interval
            .iter()
            .zip(forensic_path.interval_p.iter())
            .enumerate()
        {
            let abs_error = (actual - expected).abs();
            assert!(
                abs_error <= 1e-4,
                "path[{index}] interval[{step}] mismatch: actual={actual} expected={expected} abs_error={abs_error}"
            );
        }

        let actual_pwin = 1.0
            - actual_interval
                .iter()
                .fold(1.0, |p_not_touch, p_hit| p_not_touch * (1.0 - p_hit));
        let abs_error = (actual_pwin - forensic_path.pwin).abs();
        assert!(
            abs_error <= 1e-4,
            "path[{index}] pwin mismatch: actual={actual_pwin} expected={} abs_error={abs_error}",
            forensic_path.pwin
        );
    }

    println!(
        "compared {} dumped paths for row {} window {} on shared path matrix",
        dump.paths.len(),
        dump.row_index,
        dump.window_index
    );
    let _ = fixture;
}

#[test]
#[ignore]
fn reconstruct_python_paths_from_dumped_shocks() {
    let dump_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/mode10_smoke.shocks.json");
    let raw = fs::read_to_string(&dump_path).unwrap();
    let dump: ShockDump = serde_json::from_str(&raw).unwrap();

    let reconstructed = debug_reconstruct_paths_from_shocks(
        dump.spot,
        dump.mu,
        &dump.eps,
        &dump.jumps,
        dump.sigma,
    );

    assert_eq!(reconstructed.paths.len(), dump.paths.len());
    for (path_index, (actual_path, expected_path)) in reconstructed
        .paths
        .iter()
        .zip(dump.paths.iter())
        .enumerate()
    {
        assert_eq!(actual_path.len(), expected_path.len());
        for (step, (actual, expected)) in actual_path.iter().zip(expected_path.iter()).enumerate() {
            let abs_error = (actual - expected).abs();
            assert!(
                abs_error <= 0.2,
                "path[{path_index}] step[{step}] mismatch: actual={actual} expected={expected} abs_error={abs_error}"
            );
        }
    }

    let _trace: DebugPathTrace = reconstructed;
}

#[test]
#[ignore]
fn reconstruct_python_paths_from_dumped_returns() {
    let dump_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/mode10_smoke.shocks.json");
    let raw = fs::read_to_string(&dump_path).unwrap();
    let dump: ShockDump = serde_json::from_str(&raw).unwrap();

    let reconstructed = debug_reconstruct_paths_from_returns(dump.spot, &dump.returns);

    assert_eq!(reconstructed.paths.len(), dump.paths.len());
    for (path_index, (actual_path, expected_path)) in reconstructed
        .paths
        .iter()
        .zip(dump.paths.iter())
        .enumerate()
    {
        for (step, (actual, expected)) in actual_path.iter().zip(expected_path.iter()).enumerate() {
            let abs_error = (actual - expected).abs();
            assert!(
                abs_error <= 0.2,
                "returns path[{path_index}] step[{step}] mismatch: actual={actual} expected={expected} abs_error={abs_error}"
            );
        }
    }

    let _trace: DebugPathTrace = reconstructed;
}

#[test]
#[ignore]
fn debug_mode10_smoke_representative_cells() {
    let (fixture, config, history) = load_fixture();
    let expected_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/mode10_smoke.expected.json");
    let expected_raw = fs::read_to_string(&expected_path).unwrap();
    let expected: ExpectedFixture = serde_json::from_str(&expected_raw).unwrap();
    let snapshot = st_engine::grid::quote_grid_with_liabilities(
        fixture.spot,
        &history,
        &config,
        None,
    )
    .unwrap();

    for cell in expected.cells {
        let index = cell.row_index * config.spec.windows.len() + cell.window_index;
        let actual = &snapshot.cells[index];
        println!(
            "row={} window={} rust_p={} py_p={} rust_m={} py_m={}",
            cell.row_index,
            cell.window_index,
            actual.hit_probability,
            cell.hit_probability.unwrap_or_default(),
            actual.multiplier,
            cell.multiplier.unwrap_or_default(),
        );
    }
}

fn load_fixture() -> (FixtureInput, GridConfig, Vec<OhlcBar>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/mode10_smoke.input.json");
    let raw = fs::read_to_string(&path).unwrap();
    let fixture: FixtureInput = serde_json::from_str(&raw).unwrap();

    let mut config = GridConfig::default();
    if let Some(rows) = fixture.config.rows {
        config.spec.rows = rows;
    }
    if let Some(center_row) = fixture.config.center_row {
        config.spec.center_row = center_row;
    }
    if let Some(band_width) = fixture.config.band_width {
        config.spec.band_width = band_width;
    }
    if let Some(windows) = &fixture.config.windows {
        config.spec.windows = windows
            .iter()
            .map(|window| TimeWindow {
                start_secs: window[0],
                end_secs: window[1],
            })
            .collect();
    }
    if let Some(simulation) = &fixture.config.simulation {
        if let Some(mu) = simulation.mu {
            config.simulation.mu = mu;
        }
        if let Some(seed) = simulation.seed {
            config.simulation.seed = Some(seed);
        }
        if let Some(use_parallel) = simulation.use_parallel {
            config.simulation.use_parallel = use_parallel;
        }
        if let Some(parallel_workers) = simulation.parallel_workers {
            config.simulation.parallel_workers = Some(parallel_workers);
        }
        if let Some(lock_offset) = simulation.lock_offset {
            config.simulation.lock_offset = lock_offset;
        }
        if let Some(bridge_epsilon) = simulation.bridge_epsilon {
            config.simulation.bridge_epsilon = bridge_epsilon;
        }
        if let Some(use_antithetic) = simulation.use_antithetic {
            config.simulation.use_antithetic = use_antithetic;
        }
        if let Some(mc_n_min) = simulation.mc_n_min {
            config.simulation.mc_n_min = mc_n_min;
        }
        if let Some(mc_n_max) = simulation.mc_n_max {
            config.simulation.mc_n_max = mc_n_max;
        }
        if let Some(se_abs) = simulation.se_abs {
            config.simulation.se_abs = se_abs;
        }
        if let Some(se_rel) = simulation.se_rel {
            config.simulation.se_rel = se_rel;
        }
        if let Some(p_floor) = simulation.p_floor {
            config.simulation.p_floor = p_floor;
        }
        if let Some(adaptive_mc) = simulation.adaptive_mc {
            config.simulation.adaptive_mc = adaptive_mc;
        }
    }

    let history = fixture
        .history
        .iter()
        .map(|bar| OhlcBar {
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
        })
        .collect::<Vec<_>>();

    (fixture, config, history)
}

fn band_for_row(spot: f64, config: &GridConfig, row_index: usize) -> PriceBand {
    let anchor = (spot / config.spec.band_width).floor() * config.spec.band_width;
    let center = anchor + ((row_index as f64) - (config.spec.center_row as f64)) * config.spec.band_width;
    PriceBand {
        row_index,
        lower: center - config.spec.band_width / 2.0,
        upper: center + config.spec.band_width / 2.0,
    }
}
