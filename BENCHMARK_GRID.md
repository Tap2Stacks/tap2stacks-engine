# Grid Engine Benchmark

## Goal

Measure end-to-end latency for producing a full `20 x 5` grid with final multipliers for every cell.

Target:

- `p95 < 20ms`

## Typical Production Case

This benchmark currently models the most typical single-mode production case for `mode 10`.

- Grid: `20` rows x `5` windows = `100` cells
- Band width: `$10`
- Windows: `(5,10) (10,15) (15,20) (20,25) (25,30)`
- History bars: `180` bars of `1s` OHLC
- Spot: around `$50,000`
- Monte Carlo: `5,000` paths
- Adaptive MC: disabled for the baseline latency test
- Antithetic variates: enabled
- Lock offset: `1`
- Brownian Bridge epsilon: `1e-6`

## Maintained Runtime State

For the benchmark to reflect production shape, the engine should maintain:

- `180 x 1s` OHLC bars for state estimation
- current `spot`
- runtime volatility state:
  - `variance`
  - `sigma_raw`
  - `sigma`
  - `jump_intensity`
  - `last_price`
- jump sampler memory from recent detected jumps
- liability book for all `100` cells
- smoothed liability-used state for skew pricing

The current benchmark covers:

- full history-based state estimation
- full Monte Carlo quote generation
- optional external liabilities
- stateful `GridEngine` path with internal liability book

## Benchmark Modes

The benchmark binary measures three paths:

- `plain`
  - `quote_grid(...)`
  - no liabilities
- `stateless`
  - `quote_grid_with_liabilities(...)`
  - liabilities passed in externally
- `stateful`
  - `GridEngine::quote_grid(...)`
  - internal liability book + smoothing

## Run

```bash
cargo run --release --bin bench_grid
```

## Notes

- Use `--release`; debug results are not meaningful.
- The current synthetic history is deterministic so runs are reproducible.
- If latency is above target, the next optimizations should focus on:
  - avoiding full history recomputation on every quote
  - reducing allocations in simulator/pricing flow
  - reusing precomputed grid metadata
  - batching/vectorizing Monte Carlo work

## Current Result

Representative benchmark result on the current implementation:

- Initial baseline:
  - `phase_state`: about `0.014 ms`
  - `phase_simulate`: about `84 ms`
  - `phase_materialize`: about `0.84 ms`
  - `phase_total`: about `84.5 ms`

- After optimization pass 1:
  - removed interval-probability temporary allocation in `touch`
  - flattened simulator accumulators from `Vec<Vec<f64>>` to flat buffers
  - result:
    - `phase_simulate`: about `64.3 ms`
    - `phase_total`: about `65.2 ms`

- After optimization pass 2:
  - removed full `win_matrix` storage from the hot path
  - replaced it with direct `scenario_payouts` accumulation for `k_safety`
  - result:
    - `plain`: about `57.2 ms`
    - `stateless`: about `67.1 ms`
    - `stateful`: about `66.1 ms`
    - `phase_materialize`: about `0.09 ms`
    - `phase_total`: about `66.4 ms`

- After optimization pass 3:
  - reused path buffers inside the simulator batch loop
  - precomputed window bounds and flattened cell metadata outside the hot loop
  - result:
    - `phase_simulate`: about `56.0 ms`
    - `phase_total`: about `64.5 ms`
    - `plain`: about `58.1 ms`
    - `stateless`: about `65.1 ms`

Note:

- the `stateful` benchmark can show extra variance because `GridEngine::quote_grid(...)`
  mutates the internal smoothed liability-used state across benchmark iterations

- After optimization pass 4:
  - parallelized Monte Carlo along the path axis with `rayon`
  - each worker uses its own:
    - RNG
    - path buffers
    - local accumulators
  - worker results are reduced after the batch completes
  - result on the current machine:
    - `plain`: about `16.6 ms`
    - `stateless`: about `19.4 ms`
    - `stateful`: about `18.9 ms`
    - `phase_simulate`: about `20.1 ms`
    - `phase_total`: about `19.5 ms`

Current interpretation:

- the target is now within reach for mean and median latency
- `p95` is still the metric to watch carefully, especially for the liability-aware paths

- Longer-run validation on the target machine:
  - run config:
    - `BENCH_ITERATIONS=200`
    - `BENCH_WARMUP_ITERATIONS=20`
    - `mc_paths=5000`
    - `parallel_workers=8`
    - `k_safety_ttl_ms=100`
  - result:
    - `plain`: mean `15.842 ms`, p95 `19.346 ms`
    - `stateless`: mean `17.516 ms`, p95 `19.612 ms`
    - `stateful`: mean `15.549 ms`, p95 `18.018 ms`

- Stateful-focused run on the target machine with slower `k_safety` refresh:
  - run config:
    - `BENCH_ITERATIONS=200`
    - `BENCH_WARMUP_ITERATIONS=20`
    - `mc_paths=5000`
    - `parallel_workers=8`
    - `k_safety_ttl_ms=1000`
  - result:
    - `stateful`: mean `15.690 ms`, p95 `17.607 ms`

Interpretation:

- the `stateful` production path is currently within the `< 20 ms p95` target
- the `stateless` path is also within target on this longer run
- isolated phase benchmarks still show runtime jitter and are useful mainly for hotspot analysis, not as the final product KPI

## Current Parallelism Decision

The implementation keeps:

- parallel Monte Carlo on the path axis
- configurable `parallel_workers`

The current default execution mode uses the global Rayon pool.

Reason:

- dedicated per-simulator pools were tested
- they did not improve p95 on the measured machine
- keeping the global pool currently gives simpler behavior and better practical benchmark results

So the remaining tuning knob is:

- `parallel_workers`

This should still be A/B tested on the target machine.

## Runtime Default

Current code default:

- `simulation.parallel_workers = Some(8)`
- `pricing.k_safety_ttl_ms = 1000`

Recommended production policy:

- keep `8` as the library default
- allow the app/service layer to override it from environment configuration

Suggested runtime override pattern:

- if `GRID_PARALLEL_WORKERS` is set, use that value
- if `GRID_K_SAFETY_TTL_MS` is set, use that value
- otherwise fall back to the library default

Reason:

- benchmark results are machine-dependent
- the target machine used here showed `8` as the best current candidate
- production deployments should still be able to tune worker count without code changes
- `k_safety` is intentionally allowed to lag slightly behind quote ticks

## Current Production Candidate

Based on the current benchmark data, the production candidate configuration is:

- engine mode: `stateful`
- `simulation.parallel_workers = 8`
- `pricing.k_safety_ttl_ms = 1000`
- `simulation.mc_n_min = 5000`
- `simulation.mc_n_max = 5000`
- `simulation.use_antithetic = true`

Current position:

- this candidate meets the latency target on the target machine in the longer benchmark run
- further work should now prioritize quote quality, calibration, and production integration over additional micro-optimizations

## Benchmark Overrides

The benchmark runner supports:

- `BENCH_WORKERS=2,4,6,8`
- `BENCH_K_SAFETY_TTL_MS=100`

Example:

```bash
BENCH_WORKERS=2,4,6,8 BENCH_K_SAFETY_TTL_MS=250 cargo run --release --bin bench_grid
```

Conclusion:

- the bottleneck is overwhelmingly in `simulate_summary(...)`
- state estimation and pricing/materialization are already negligible

## Optimization Plan

### Goal

Reach `p95 < 20 ms` for a full `100`-cell grid quote.

### Principle

The main problem is not orchestration overhead. The main problem is Monte Carlo throughput.

So the plan should prioritize reducing simulation cost per path before relying on lower path counts.

### Step 1: Benchmark tradeoff configs

Run a small sweep to measure latency sensitivity to:

- `mc_n_min`
- `mc_n_max`
- `adaptive_mc`

Purpose:

- quantify the latency/quality curve
- estimate how much path count reduction alone can buy

Note:

- this is useful for measurement
- this is not the primary long-term fix if production still requires a large `mc_n_min`

### Step 2: Reduce simulator allocations

Focus on `simulate_summary(...)`:

- replace nested `Vec<Vec<f64>>` with flat buffers
- reduce temporary per-path allocations
- avoid rebuilding large intermediate structures inside the hot loop
- minimize `sampled_wins` memory traffic

Purpose:

- improve cost per path without changing model behavior

Status:

- partially completed
- flat accumulators are done
- full `win_matrix` storage has been removed from the hot path
- next remaining target is path-buffer reuse inside the simulator

### Step 3: Make `k_safety` cheaper

Current implementation stores scenario-level sampled wins for all paths.

Investigate:

- whether `k_safety` must be updated on every quote
- whether scenario payout aggregation can be accumulated directly without storing full `win` matrices
- whether `k_safety` can run on a lower-frequency path than the main quote path

Purpose:

- reduce overhead tied to tail-risk estimation

### Step 4: Reuse persistent quote state

Keep more runtime state hot between quotes:

- incremental runtime-state updates instead of replaying history every time
- reuse grid metadata that does not change per quote
- reuse engine-side buffers where possible

Purpose:

- remove repeated work around the simulator

### Step 5: Parallelize simulator

If single-thread optimization is still insufficient:

- parallelize Monte Carlo batches across cores
- keep output aggregation deterministic enough for benchmarking and validation

Purpose:

- unlock the remaining large speedup once single-thread waste has been removed

## Important Product Decision

`adaptive_mc` is useful, but it does not by itself solve the core performance problem if production still requires a high `mc_n_min`.

Why:

- if `mc_n_min` remains large, the engine still pays that minimum simulation cost on every quote
- adaptive stopping only helps after the minimum work has already been done

So:

- `adaptive_mc` should be treated as a measurement and tuning tool first
- the primary engineering objective should be improving simulator throughput and reducing mandatory work per quote
