#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import importlib.util
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON_MODEL_ROOT = Path(
    os.environ.get("TAP2STACKS_PYTHON_MODEL_ROOT", ROOT / "tap2stacks-model")
)
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
    parser = argparse.ArgumentParser(
        description="Generate Rust parity expected outputs from the Python mode-10 engine."
    )
    parser.add_argument(
        "fixtures",
        nargs="*",
        help="Input fixture paths (*.input.json). Defaults to fixtures/parity/*.input.json.",
    )
    parser.add_argument(
        "--mode",
        default="10",
        help="Mode id to export from the Python engine. Default: 10",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *.expected.json files.",
    )
    parser.add_argument(
        "--cells",
        default="all",
        help="Comma-separated row:window selectors, or 'all'. Example: 9:0,9:4,0:0",
    )
    args = parser.parse_args()

    input_paths = resolve_input_paths(args.fixtures)
    selected_cells = parse_selected_cells(args.cells)

    for input_path in input_paths:
        expected_path = input_path.with_name(
            input_path.name.replace(".input.json", ".expected.json")
        )
        if expected_path.exists() and not args.overwrite:
            raise SystemExit(
                f"{expected_path} already exists. Pass --overwrite to replace it."
            )

        fixture = load_json(input_path)
        expected = build_expected(
            fixture, mode_id=args.mode, selected_cells=selected_cells
        )
        expected_path.write_text(
            json.dumps(expected, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {expected_path}")


def resolve_input_paths(cli_paths: list[str]) -> list[Path]:
    if cli_paths:
        return [Path(path).resolve() for path in cli_paths]
    fixture_dir = ROOT / "fixtures" / "parity"
    return sorted(fixture_dir.glob("*.input.json"))


def parse_selected_cells(raw: str) -> list[tuple[int, int]] | None:
    if raw == "all":
        return None
    selected = []
    for token in raw.split(","):
        row_raw, window_raw = token.split(":")
        selected.append((int(row_raw), int(window_raw)))
    return selected


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_expected(
    fixture: dict[str, Any],
    mode_id: str,
    selected_cells: list[tuple[int, int]] | None,
) -> dict[str, Any]:
    spot = float(fixture["spot"])
    history = fixture["history"]
    overrides = fixture.get("config", {})
    liabilities = fixture.get("liabilities", [])
    use_internal_liability_book = bool(
        fixture.get("use_internal_liability_book", False)
    )

    engine = build_engine(overrides, mode_id)
    replay_history(engine, history, run_pricing=False)
    reprice_state_for_spot(engine, spot, mode_id)
    engine._oracle_price = float(spot)
    engine._build_cells(float(spot))
    engine._run_mc(float(spot))

    mode_cells = engine.get_cells(mode_id)
    cell_map = {(cell.row, cell.window_index): cell for cell in mode_cells}

    if liabilities:
        if use_internal_liability_book:
            for index, value in enumerate(liabilities):
                if value:
                    engine.set_liability(mode_cells[index].cell_id, float(value))
            quotes = engine.compute_quotes()[mode_id]
        else:
            quotes = engine.compute_quotes(
                {
                    mode_cells[index].cell_id: float(value)
                    for index, value in enumerate(liabilities)
                    if value
                }
            )[mode_id]
    else:
        quotes = engine.compute_quotes()[mode_id]

    if selected_cells is None:
        selected_quotes = quotes
    else:
        selectors = set(selected_cells)
        selected_quotes = [
            quote
            for quote in quotes
            if (int(quote["row"]), int(quote["window_index"])) in selectors
        ]

    anchor = None
    center_row = engine.modes[mode_id].center_row
    for cell in mode_cells:
        if cell.row == center_row and cell.window_index == 0:
            anchor = (cell.L + cell.R) / 2.0
            break

    state = engine._mode_state[mode_id]
    expected = {
        "anchor": anchor,
        "volatility": {
            "sigma": float(state.sigma),
            "sigma_raw": float(state.sigma_raw),
            "jump_intensity": float(state.lambda_intensity),
        },
        "cells": [
            {
                "row_index": int(quote["row"]),
                "window_index": int(quote["window_index"]),
                "hit_probability": float(quote["P_raw"]),
                "multiplier": float(quote["M_final"]),
            }
            for quote in selected_quotes
        ],
        "tolerances": {
            "default_abs": 1e-6,
            "default_rel": 0.05,
            "hit_probability_abs": 0.03,
            "hit_probability_rel": 0.10,
            "multiplier_abs": 0.05,
            "multiplier_rel": 0.20,
        },
    }

    # Keep deterministic ordering by row, then window.
    expected["cells"].sort(key=lambda item: (item["row_index"], item["window_index"]))
    return expected


def build_engine(overrides: dict[str, Any], mode_id: str) -> FortressEngine:
    global_config = GlobalConfig()
    modes = default_modes()
    if mode_id not in modes:
        raise ValueError(f"unknown mode_id {mode_id}")

    mode = deepcopy(modes[mode_id])
    apply_overrides(global_config, mode, overrides)
    return FortressEngine(modes={mode_id: mode}, config=global_config)


def apply_overrides(
    global_config: GlobalConfig, mode: Any, overrides: dict[str, Any]
) -> None:
    if "rows" in overrides:
        mode.row_count = int(overrides["rows"])
    if "center_row" in overrides:
        mode.center_row = int(overrides["center_row"])
    if "band_width" in overrides:
        mode.band_width = float(overrides["band_width"])
    if "windows" in overrides:
        mode.windows = [tuple(map(int, window)) for window in overrides["windows"]]

    simulation = overrides.get("simulation", {})
    volatility = overrides.get("volatility", {})
    pricing = overrides.get("pricing", {})

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
    if "se_abs" in simulation:
        mode.se_abs = float(simulation["se_abs"])
    if "se_rel" in simulation:
        mode.se_rel = float(simulation["se_rel"])
    if "p_floor" in simulation:
        mode.p_floor = float(simulation["p_floor"])
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
    if "hawkes_mu0" in jump:
        global_config.hawkes_mu0 = float(jump["hawkes_mu0"])
    if "alpha" in jump:
        global_config.hawkes_alpha = float(jump["alpha"])
    if "beta" in jump:
        global_config.hawkes_beta = float(jump["beta"])
    if "kappa0" in jump:
        mode.kappa0 = float(jump["kappa0"])
    if "kappa_q" in jump:
        mode.kappa_q = float(jump["kappa_q"])
    if "kappa_min" in jump:
        mode.kappa_min = float(jump["kappa_min"])
    if "kappa_max" in jump:
        mode.kappa_max = float(jump["kappa_max"])

    if "m_base" in pricing:
        mode.m_base = float(pricing["m_base"])
    if "m_risk_max" in pricing:
        mode.m_risk_max = float(pricing["m_risk_max"])
    if "lambda_ref" in pricing:
        global_config.lambda_ref = float(pricing["lambda_ref"])
    if "a_sigma" in pricing:
        global_config.a_sigma = float(pricing["a_sigma"])
    if "a_lambda" in pricing:
        global_config.a_lambda = float(pricing["a_lambda"])
    if "beta_skew" in pricing:
        mode.beta_skew = float(pricing["beta_skew"])
    if "pool_cap" in pricing:
        mode.pool_cap = float(pricing["pool_cap"])
    if "risk_budget" in pricing:
        mode.risk_budget = float(pricing["risk_budget"])
    if "liability_rho" in pricing:
        global_config.liability_rho = float(pricing["liability_rho"])
    if "p_raw_gamma" in pricing:
        mode.p_raw_gamma = float(pricing["p_raw_gamma"])
    if "p_near_boost" in pricing:
        mode.p_near_boost = float(pricing["p_near_boost"])
    if "p_time_boost" in pricing:
        mode.p_time_boost = float(pricing["p_time_boost"])
    if "calibration_dist_max" in pricing:
        global_config.p_calibration_dist_max = int(pricing["calibration_dist_max"])
    if "calibration_sigma_edges" in pricing:
        lo, hi = pricing["calibration_sigma_edges"]
        global_config.p_calibration_sigma_edges = (float(lo), float(hi))
    if "current_row_logit_shift" in pricing:
        mode.current_row_logit_shift = float(pricing["current_row_logit_shift"])
    if "adjacent_row_logit_shift" in pricing:
        mode.adjacent_row_logit_shift = float(pricing["adjacent_row_logit_shift"])
    if "zone_vol_shift_scale" in pricing:
        mode.zone_vol_shift_scale = float(pricing["zone_vol_shift_scale"])
    if "min_multiplier" in pricing:
        mode.m_min = float(pricing["min_multiplier"])
    if "max_multiplier" in pricing:
        mode.m_max = float(pricing["max_multiplier"])


def replay_history(
    engine: FortressEngine, history: list[dict[str, Any]], run_pricing: bool
) -> None:
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
