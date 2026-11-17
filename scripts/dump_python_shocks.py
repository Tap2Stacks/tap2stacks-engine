#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON_MODEL_ROOT = ROOT / "tap2stacks-model"
ENGINE_PATH = PYTHON_MODEL_ROOT / "fortress" / "engine.py"
ENGINE_SPEC = importlib.util.spec_from_file_location("tap2stacks_engine", ENGINE_PATH)
if ENGINE_SPEC is None or ENGINE_SPEC.loader is None:
    raise RuntimeError(f"failed to load Python engine from {ENGINE_PATH}")
ENGINE_MODULE = importlib.util.module_from_spec(ENGINE_SPEC)
sys.modules[ENGINE_SPEC.name] = ENGINE_MODULE
ENGINE_SPEC.loader.exec_module(ENGINE_MODULE)

FortressEngine = ENGINE_MODULE.FortressEngine
GlobalConfig = ENGINE_MODULE.GlobalConfig
default_modes = ENGINE_MODULE.default_modes


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump Python eps/jump/path matrices for parity debugging.")
    parser.add_argument("fixture", help="Path to *.input.json fixture")
    parser.add_argument("--mode", default="10")
    parser.add_argument("--paths", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fixture = load_json(Path(args.fixture))
    engine = build_engine(fixture.get("config", {}), args.mode)
    replay_history(engine, fixture["history"], run_pricing=False)
    spot = float(fixture["spot"])
    reprice_state_for_spot(engine, spot, args.mode)

    mode = engine.modes[args.mode]
    state = engine._mode_state[args.mode]
    horizon = max(t_end for (_, t_end) in mode.windows) + max(0, engine.config.lock_offset)
    n = int(args.paths)
    mu = float(engine.config.mu)
    sigma = float(state.sigma)
    lambda_intensity = float(state.lambda_intensity)

    rng = engine._rng
    if not engine.config.use_antithetic or n == 1:
        eps = rng.standard_normal((n, horizon)).astype(np.float32)
    else:
        base_n = (n + 1) // 2
        base_eps = rng.standard_normal((base_n, horizon)).astype(np.float32)
        anti_eps = -base_eps[: n // 2]
        eps = np.vstack([base_eps, anti_eps])

    returns = mu + sigma * eps
    jumps = np.zeros_like(returns, dtype=np.float32)

    if lambda_intensity > 0.0:
        p_jump = 1.0 - math.exp(-lambda_intensity)
        if not engine.config.use_antithetic or n == 1:
            jump_mask = rng.random((n, horizon)) < p_jump
            jump_count = int(jump_mask.sum())
            if jump_count > 0:
                sampled = state.jump_sampler.sample(jump_count, rng, sigma).astype(np.float32)
                jumps[jump_mask] = sampled
        else:
            base_n = (n + 1) // 2
            base_mask = rng.random((base_n, horizon)) < p_jump
            jump_count = int(base_mask.sum())
            jumps_base = np.zeros((base_n, horizon), dtype=np.float32)
            if jump_count > 0:
                sampled = state.jump_sampler.sample(jump_count, rng, sigma).astype(np.float32)
                jumps_base[base_mask] = sampled
            jumps = np.vstack([jumps_base, jumps_base[: n // 2]])

    returns = returns + jumps
    log_paths = np.cumsum(returns, axis=1, dtype=np.float32) + math.log(max(spot, engine.config.epsilon))
    price_paths = np.exp(log_paths).astype(np.float32)
    start_col = np.full((n, 1), float(spot), dtype=np.float32)
    full_paths = np.hstack([start_col, price_paths])

    output = {
        "spot": spot,
        "mu": mu,
        "sigma": sigma,
        "lambda_intensity": lambda_intensity,
        "horizon": int(horizon),
        "eps": eps.tolist(),
        "jumps": jumps.tolist(),
        "returns": returns.tolist(),
        "paths": full_paths.tolist(),
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_engine(overrides: dict[str, Any], mode_id: str) -> FortressEngine:
    global_config = GlobalConfig()
    modes = default_modes()
    mode = deepcopy(modes[mode_id])
    apply_overrides(global_config, mode, overrides)
    return FortressEngine(modes={mode_id: mode}, config=global_config)


def apply_overrides(global_config: GlobalConfig, mode: Any, overrides: dict[str, Any]) -> None:
    simulation = overrides.get("simulation", {})
    volatility = overrides.get("volatility", {})
    if "mu" in simulation:
        global_config.mu = float(simulation["mu"])
    if "seed" in simulation:
        global_config.seed = int(simulation["seed"])
    if "lock_offset" in simulation:
        global_config.lock_offset = int(simulation["lock_offset"])
    if "use_antithetic" in simulation:
        global_config.use_antithetic = bool(simulation["use_antithetic"])
    if "mc_n_min" in simulation:
        mode.mc_n_min = int(simulation["mc_n_min"])
    if "mc_n_max" in simulation:
        mode.mc_n_max = int(simulation["mc_n_max"])
    if "adaptive_mc" in simulation:
        mode.adaptive_mc = bool(simulation["adaptive_mc"])
    if "sigma_scale" in volatility:
        mode.sigma_scale = float(volatility["sigma_scale"])
    if "sigma_ref" in volatility:
        global_config.sigma_ref = float(volatility["sigma_ref"])
    if "ewma_half_life_seconds" in volatility:
        mode.hl_seconds = float(volatility["ewma_half_life_seconds"])
    if "parkinson_weight" in volatility:
        global_config.parkinson_weight = float(volatility["parkinson_weight"])
    jump = volatility.get("jump", {})
    if "enabled" in jump and not bool(jump["enabled"]):
        global_config.hawkes_mu0 = 0.0
        global_config.hawkes_alpha = 0.0
        global_config.hawkes_beta = 1.0


def replay_history(engine: FortressEngine, history: list[dict[str, Any]], run_pricing: bool) -> None:
    for second, bar in enumerate(history, start=1):
        engine.update_oracle(
            float(bar["close"]),
            oracle_second=second,
            run_pricing=run_pricing,
            high=float(bar["high"]),
            low=float(bar["low"]),
        )


def reprice_state_for_spot(engine: FortressEngine, spot: f64, mode_id: str) -> None:
    state = engine._mode_state[mode_id]
    mode = engine.modes[mode_id]
    if state.sigma_raw <= 0.0:
        return
    price_adj = math.sqrt(max(spot, 10000.0) / 100000.0)
    price_adj = max(0.7, min(1.3, price_adj))
    state.sigma = state.sigma_raw * mode.sigma_scale * price_adj


if __name__ == "__main__":
    main()
