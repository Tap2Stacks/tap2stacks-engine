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
    parser = argparse.ArgumentParser(description="Dump Python forensic details for one parity cell.")
    parser.add_argument("fixture", help="Path to *.input.json fixture")
    parser.add_argument("--mode", default="10")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    fixture = load_json(Path(args.fixture))
    engine = build_engine(fixture.get("config", {}), args.mode)
    replay_history(engine, fixture["history"], run_pricing=False)
    spot = float(fixture["spot"])
    reprice_state_for_spot(engine, spot, args.mode)
    engine._oracle_price = spot
    engine._build_cells(spot)

    mode = engine.modes[args.mode]
    horizon = max(t_end for (_, t_end) in mode.windows) + max(0, engine.config.lock_offset)
    state = engine._mode_state[args.mode]
    paths = engine._simulate_paths(
        spot,
        args.paths,
        horizon,
        state.sigma,
        state.lambda_intensity,
        state.jump_sampler,
    )

    target_cell = next(
        cell
        for cell in engine.get_cells(args.mode)
        if cell.row == args.row and cell.window_index == args.window
    )
    seg_start = paths[:, :-1]
    seg_end = paths[:, 1:]
    p_hit = engine._brownian_bridge_hit(
        seg_start,
        seg_end,
        target_cell.L,
        target_cell.R,
        max(state.sigma, engine.config.epsilon),
        1.0,
    )
    t_start = target_cell.T_start + max(0, engine.config.lock_offset)
    t_end = target_cell.T_end + max(0, engine.config.lock_offset)

    report = {
        "anchor": math.floor(spot / mode.band_width) * mode.band_width,
        "sigma": float(state.sigma),
        "sigma_raw": float(state.sigma_raw),
        "jump_intensity": float(state.lambda_intensity),
        "row_index": int(args.row),
        "window_index": int(args.window),
        "band": {"lower": float(target_cell.L), "upper": float(target_cell.R)},
        "shifted_window": {"start": int(t_start), "end": int(t_end)},
        "paths": [],
    }

    print(f"anchor={report['anchor']}")
    print(
        "sigma={} sigma_raw={} lambda={}".format(
            state.sigma, state.sigma_raw, state.lambda_intensity
        )
    )
    print(
        "cell row={} window_index={} band=[{},{}] shifted_window=[{},{}]".format(
            args.row, args.window, target_cell.L, target_cell.R, t_start, t_end
        )
    )
    for index in range(min(args.paths, paths.shape[0])):
        interval = p_hit[index, t_start:t_end].tolist()
        pwin = 1.0 - float(np.prod(1.0 - np.array(interval), dtype=np.float32))
        report["paths"].append(
            {
                "index": index,
                "path": paths[index].tolist(),
                "interval_p": interval,
                "pwin": pwin,
            }
        )
        print(f"path[{index}]={paths[index].tolist()}")
        print(f"path[{index}].interval_p={interval}")
        print(f"path[{index}].pwin={pwin}")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
    pricing = overrides.get("pricing", {})

    if "rows" in overrides:
        mode.row_count = int(overrides["rows"])
    if "center_row" in overrides:
        mode.center_row = int(overrides["center_row"])
    if "band_width" in overrides:
        mode.band_width = float(overrides["band_width"])
    if "windows" in overrides:
        mode.windows = [tuple(map(int, window)) for window in overrides["windows"]]

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

    if "m_base" in pricing:
        mode.m_base = float(pricing["m_base"])


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
