#!/usr/bin/env python
"""
MOAT v5g Stage 2e -- confirmatory-safe endogenous-loop evaluation.

This revision separates parameter exploration from confirmation and prevents an
AUC-only result from being labelled a Stage 2e success.  It imports the Stage 2d
implementation unchanged, but restores its classifier capacity and fixed
17-step late window.

Primary residual metric
-----------------------
The conservative external separability score is

    max(folded linear AUC, folded RFF AUC)

rather than the mean of the two classifiers.  A residual-collapse claim must
therefore survive the stronger of the two pre-specified evaluators.

Modes
-----
explore (default):
    Runs the predeclared PE-preserving grid.  Results are candidates only and
    are never labelled confirmatory successes.

confirm:
    Runs one predeclared cell on fresh seeds.  A strict pass requires 95%
    bootstrap bounds to satisfy every residual, positive-control, leakage, PE,
    energy, directional-depletion, and attribution gate.

The Stage 2d module is loaded as a dependency and is never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


# All default min_de values preserve PE because input_energy * min_de >= 0.15.
EXPLORATORY_GRID = {
    "T": (60, 120, 240),
    "delta_b": (0.9, 1.2, 1.5),
    "min_de": (0.15, 0.10, 0.075),
}
EXPLORATORY_SEEDS = (42, 43, 44)
CONFIRMATORY_SEEDS = tuple(range(1001, 1021))

THRESHOLDS = {
    "residual_auc_max": 0.60,
    "vb_control_auc_min": 0.60,
    "action_auc_max": 0.60,
    "attr_rate_min": 0.55,
    "de_B_min": 0.65,
    "de_Q_max": 0.45,
    "de_contrast_min": 0.25,
    "pe_min": 0.15,
    "energy_min": 1.0,
}

REQUIRED_BASE_SYMBOLS = (
    "Stage2dCfg",
    "sample_geom",
    "run_sra_ep",
    "run_fixed_policy_ep",
    "split_eval",
    "mean_auc",
    "window_feats",
    "window_mean",
)


def load_base(path: Path):
    """Load the reviewed Stage 2d implementation without editing it."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage 2d base file not found: {path}")
    spec = importlib.util.spec_from_file_location("moat_stage2d_fixed", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    missing = [name for name in REQUIRED_BASE_SYMBOLS if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Base module is missing required symbols: {missing}")
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_cell(text: str) -> Tuple[int, float, float]:
    """Parse a confirmatory cell in the form T,delta_b,min_de."""
    try:
        t_text, db_text, md_text = (part.strip() for part in text.split(","))
        cell = (int(t_text), float(db_text), float(md_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--cell must be T,delta_b,min_de (example: 120,0.9,0.075)"
        ) from exc
    if cell[0] < 20 or cell[1] <= 0 or not 0 < cell[2] <= 0.5:
        raise argparse.ArgumentTypeError(
            "cell requires T >= 20, delta_b > 0, and 0 < min_de <= 0.5"
        )
    return cell


def late_window(T: int, window_len: int, end_frac: float) -> Tuple[int, int]:
    """Use a fixed-size window while allowing a longer adaptation horizon."""
    end = min(T, int(round(end_frac * T)))
    start = end - window_len
    if start < 0 or end <= start:
        raise ValueError(
            f"Invalid late window for T={T}: start={start}, end={end}, "
            f"window_len={window_len}"
        )
    return start, end


def conservative_auc(per_classifier: Dict[str, float]) -> float:
    values = [float(v) for v in per_classifier.values() if math.isfinite(float(v))]
    if not values:
        return float("nan")
    return max(values)


def mechanism_gates(row: Dict[str, Any]) -> Dict[str, bool]:
    th = THRESHOLDS
    return {
        "residual_collapse": row["auc_residual_conservative"] < th["residual_auc_max"],
        "vb_control_separable": row["auc_vb_control_conservative"] > th["vb_control_auc_min"],
        "action_leakage_low": row["auc_action_conservative"] < th["action_auc_max"],
        "attr_error_Q": row["attr_error_Q"] >= th["attr_rate_min"],
        "attr_correct_B": row["attr_correct_B"] >= th["attr_rate_min"],
        "de_B_high": row["de_B_late"] > th["de_B_min"],
        "de_Q_low": row["de_Q_late"] < th["de_Q_max"],
        "de_contrast": row["de_contrast"] > th["de_contrast_min"],
        "pe_preserved": row["pe_policy_min"] >= th["pe_min"],
        "energy_preserved": row["input_energy"] >= th["energy_min"],
    }


def run_cell(
    module,
    T: int,
    delta_b: float,
    min_de: float,
    seed: int,
    n_ep: int,
    window_len: int,
    end_frac: float,
) -> Dict[str, Any]:
    """Run adaptive episodes plus a v_B-aligned positive control."""
    cfg = module.Stage2dCfg(
        seed=seed,
        n_ep=n_ep,
        T=T,
        delta_b=delta_b,
        min_de=min_de,
        # Keep the Stage 2d evaluator capacity unchanged.
        train_steps=200,
        rff_dim=160,
    )
    cfg.late_start, cfg.late_end = late_window(T, window_len, end_frac)

    # Separate simulation, control, and classifier streams.  This prevents
    # classifier randomness from depending on how many random draws a world used.
    streams = np.random.SeedSequence(seed).spawn(5)
    rng_adaptive = np.random.default_rng(streams[0])
    rng_control = np.random.default_rng(streams[1])
    rng_residual_clf = np.random.default_rng(streams[2])
    rng_action_clf = np.random.default_rng(streams[3])
    rng_control_clf = np.random.default_rng(streams[4])

    eps_B: List[Dict[str, Any]] = []
    eps_Q: List[Dict[str, Any]] = []
    control_B: List[Dict[str, Any]] = []
    control_Q: List[Dict[str, Any]] = []

    for _ in range(n_ep):
        v_b, v_q = module.sample_geom(rng_adaptive, cfg)
        eps_B.append(module.run_sra_ep(rng_adaptive, cfg, "B", v_b, v_q))
        eps_Q.append(module.run_sra_ep(rng_adaptive, cfg, "Q", v_b, v_q))

        # Positive control: the same geometry under a fixed v_B-aligned policy.
        control_B.append(
            module.run_fixed_policy_ep(rng_control, cfg, "B", v_b, v_q, "vB")
        )
        control_Q.append(
            module.run_fixed_policy_ep(rng_control, cfg, "Q", v_b, v_q, "vB")
        )

    t0, t1 = cfg.late_start, cfg.late_end
    auc_residual = module.split_eval(
        module.window_feats(eps_B, "res", t0, t1),
        module.window_feats(eps_Q, "res", t0, t1),
        rng_residual_clf,
        cfg,
    )
    auc_action = module.split_eval(
        module.window_feats(eps_B, "acts", t0, t1),
        module.window_feats(eps_Q, "acts", t0, t1),
        rng_action_clf,
        cfg,
    )
    auc_vb_control = module.split_eval(
        module.window_feats(control_B, "res", t0, t1),
        module.window_feats(control_Q, "res", t0, t1),
        rng_control_clf,
        cfg,
    )

    de_B_late = module.window_mean(eps_B, "de_b", t0, t1)
    de_Q_late = module.window_mean(eps_Q, "de_b", t0, t1)
    row: Dict[str, Any] = {
        "T": int(T),
        "delta_b": float(delta_b),
        "min_de": float(min_de),
        "seed": int(seed),
        "n_ep": int(n_ep),
        "late_start": int(t0),
        "late_end": int(t1),
        "window_len": int(t1 - t0),
        "auc_residual_linear": float(auc_residual["linear"]),
        "auc_residual_rff": float(auc_residual["rff"]),
        "auc_residual_conservative": conservative_auc(auc_residual),
        "auc_action_linear": float(auc_action["linear"]),
        "auc_action_rff": float(auc_action["rff"]),
        "auc_action_conservative": conservative_auc(auc_action),
        "auc_vb_control_linear": float(auc_vb_control["linear"]),
        "auc_vb_control_rff": float(auc_vb_control["rff"]),
        "auc_vb_control_conservative": conservative_auc(auc_vb_control),
        "attr_error_Q": float(
            np.mean([ep["angle_vq"] < ep["angle_vb"] for ep in eps_Q])
        ),
        "attr_correct_B": float(
            np.mean([ep["angle_vb"] < ep["angle_vq"] for ep in eps_B])
        ),
        "de_B_late": float(de_B_late),
        "de_Q_late": float(de_Q_late),
        "de_contrast": float(de_B_late - de_Q_late),
        # For the policy covariance in Stage 2d, eigenvalues are E*(1-min_de)
        # and E*min_de.  min_de is restricted to <= 0.5.
        "pe_policy_min": float(cfg.input_energy * min_de),
        "input_energy": float(cfg.input_energy),
    }
    gates = mechanism_gates(row)
    row.update({f"gate_{name}": bool(value) for name, value in gates.items()})
    row["all_point_gates"] = bool(all(gates.values()))
    return row


def bootstrap_summary(
    values: Sequence[float], rng: np.random.Generator, reps: int
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("bootstrap values must be a finite, non-empty vector")
    if len(arr) == 1:
        value = float(arr[0])
        return {"mean": value, "ci95_low": value, "ci95_high": value}
    indices = rng.integers(0, len(arr), size=(reps, len(arr)))
    means = arr[indices].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


SUMMARY_METRICS = (
    "auc_residual_conservative",
    "auc_residual_linear",
    "auc_residual_rff",
    "auc_action_conservative",
    "auc_vb_control_conservative",
    "attr_error_Q",
    "attr_correct_B",
    "de_B_late",
    "de_Q_late",
    "de_contrast",
    "pe_policy_min",
    "input_energy",
)


def summarize_cell(
    rows: Sequence[Dict[str, Any]], mode: str, reps: int, bootstrap_seed: int
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty cell")
    rng = np.random.default_rng(bootstrap_seed)
    summary: Dict[str, Any] = {
        "T": rows[0]["T"],
        "delta_b": rows[0]["delta_b"],
        "min_de": rows[0]["min_de"],
        "seed_count": len(rows),
        "all_point_gates_count": int(sum(bool(row["all_point_gates"]) for row in rows)),
    }
    for metric in SUMMARY_METRICS:
        summary[metric] = bootstrap_summary(
            [float(row[metric]) for row in rows], rng, reps
        )

    th = THRESHOLDS
    bounds = {
        "residual_collapse":
            summary["auc_residual_conservative"]["ci95_high"] < th["residual_auc_max"],
        "vb_control_separable":
            summary["auc_vb_control_conservative"]["ci95_low"] > th["vb_control_auc_min"],
        "action_leakage_low":
            summary["auc_action_conservative"]["ci95_high"] < th["action_auc_max"],
        "attr_error_Q":
            summary["attr_error_Q"]["ci95_low"] >= th["attr_rate_min"],
        "attr_correct_B":
            summary["attr_correct_B"]["ci95_low"] >= th["attr_rate_min"],
        "de_B_high": summary["de_B_late"]["ci95_low"] > th["de_B_min"],
        "de_Q_low": summary["de_Q_late"]["ci95_high"] < th["de_Q_max"],
        "de_contrast":
            summary["de_contrast"]["ci95_low"] > th["de_contrast_min"],
        "pe_preserved": summary["pe_policy_min"]["ci95_low"] >= th["pe_min"],
        "energy_preserved":
            summary["input_energy"]["ci95_low"] >= th["energy_min"],
    }
    summary["bootstrap_gates"] = {k: bool(v) for k, v in bounds.items()}
    summary["exploratory_candidate"] = bool(
        mode == "explore" and all(row["all_point_gates"] for row in rows)
    )
    summary["confirmatory_pass"] = bool(mode == "confirm" and all(bounds.values()))
    return summary


def grouped_rows(rows: Sequence[Dict[str, Any]]) -> Iterable[List[Dict[str, Any]]]:
    keys = sorted({(r["T"], r["delta_b"], r["min_de"]) for r in rows})
    for key in keys:
        yield [
            r for r in rows
            if (r["T"], r["delta_b"], r["min_de"]) == key
        ]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys()) if rows else []
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Path to moat_v5g_stage2d_fixed.py",
    )
    parser.add_argument("--mode", choices=("explore", "confirm"), default="explore")
    parser.add_argument(
        "--cell",
        type=parse_cell,
        help="Required in confirm mode: T,delta_b,min_de",
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seed list")
    parser.add_argument("--n-ep", type=int, help="Episodes per hypothesis per seed")
    parser.add_argument("--window-len", type=int, default=17)
    parser.add_argument("--window-end-frac", type=float, default=0.92)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    parser.add_argument("--out", type=Path, default=Path("moat_v5g_stage2e_results.json"))
    parser.add_argument("--csv", type=Path, default=Path("moat_v5g_stage2e_per_seed.csv"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="One-cell smoke test; never produces a scientific pass",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing outputs")
    return parser


def resolve_design(args: argparse.Namespace) -> Tuple[List[Tuple[int, float, float]], Tuple[int, ...], int]:
    if args.mode == "confirm" and args.cell is None:
        raise ValueError("--cell is required in confirm mode")
    if args.mode == "explore" and args.cell is not None:
        raise ValueError("--cell is only valid in confirm mode")

    if args.mode == "confirm":
        cells = [args.cell]
        seeds = tuple(args.seeds) if args.seeds else CONFIRMATORY_SEEDS
        n_ep = args.n_ep if args.n_ep is not None else 600
    else:
        cells = [
            (T, delta_b, min_de)
            for T in EXPLORATORY_GRID["T"]
            for delta_b in EXPLORATORY_GRID["delta_b"]
            for min_de in EXPLORATORY_GRID["min_de"]
        ]
        seeds = tuple(args.seeds) if args.seeds else EXPLORATORY_SEEDS
        n_ep = args.n_ep if args.n_ep is not None else 240

    if args.quick:
        cells = cells[:1]
        seeds = seeds[:1]
        n_ep = min(n_ep, 24)
    if n_ep < 10:
        raise ValueError("n_ep must be at least 10")
    if args.window_len <= 0:
        raise ValueError("window_len must be positive")
    if not 0 < args.window_end_frac <= 1:
        raise ValueError("window_end_frac must be in (0, 1]")
    if args.bootstrap_reps < 100:
        raise ValueError("bootstrap_reps must be at least 100")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if args.mode == "confirm" and not args.quick and len(seeds) < 10:
        raise ValueError("confirm mode requires at least 10 independent seeds")
    if args.mode == "confirm" and not args.quick and set(seeds) & set(EXPLORATORY_SEEDS):
        raise ValueError("confirmatory seeds must not overlap exploratory seeds 42, 43, 44")
    if args.mode == "confirm" and cells[0][2] * 2.0 < THRESHOLDS["pe_min"]:
        raise ValueError(
            "confirmatory min_de violates the PE threshold: "
            "input_energy(2.0) * min_de must be at least 0.15"
        )
    return cells, seeds, n_ep


def main() -> None:
    args = build_parser().parse_args()
    cells, seeds, n_ep = resolve_design(args)
    if args.out.resolve() == args.csv.resolve():
        raise ValueError("--out and --csv must be different files")
    if not args.force:
        existing = [str(path) for path in (args.out, args.csv) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing outputs; use --force or new paths: "
                + ", ".join(existing)
            )

    base_path = args.base.expanduser().resolve()
    module = load_base(base_path)
    script_path = Path(__file__).resolve()

    rows: List[Dict[str, Any]] = []
    total = len(cells) * len(seeds)
    completed = 0
    for T, delta_b, min_de in cells:
        for seed in seeds:
            row = run_cell(
                module,
                T=T,
                delta_b=delta_b,
                min_de=min_de,
                seed=seed,
                n_ep=n_ep,
                window_len=args.window_len,
                end_frac=args.window_end_frac,
            )
            rows.append(row)
            completed += 1
            print(
                f"[{completed:>3}/{total}] T={T} delta_b={delta_b:g} "
                f"min_de={min_de:g} seed={seed}: "
                f"residual_max={row['auc_residual_conservative']:.3f} "
                f"vB_control_max={row['auc_vb_control_conservative']:.3f} "
                f"point_gates={'PASS' if row['all_point_gates'] else 'fail'}",
                flush=True,
            )

    summaries = [
        summarize_cell(group, args.mode, args.bootstrap_reps, args.bootstrap_seed + i)
        for i, group in enumerate(grouped_rows(rows))
    ]
    # A quick run is operational validation only.
    if args.quick:
        for summary in summaries:
            summary["exploratory_candidate"] = False
            summary["confirmatory_pass"] = False

    payload = {
        "schema_version": "moat-stage2e-revised-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "mode": args.mode,
            "quick": bool(args.quick),
            "cells": cells,
            "seeds": seeds,
            "n_ep_per_hypothesis_per_seed": n_ep,
            "window_len": args.window_len,
            "window_end_frac": args.window_end_frac,
            "primary_metric": "max(folded linear AUC, folded RFF AUC)",
            "classifier_capacity": {"train_steps": 200, "rff_dim": 160},
            "thresholds": THRESHOLDS,
            "bootstrap": {
                "reps": args.bootstrap_reps,
                "seed": args.bootstrap_seed,
                "interval": "two-sided percentile 95% across independent seeds",
            },
            "claim_rule": (
                "explore mode may nominate candidates only; confirm mode requires "
                "all bootstrap gates on fresh seeds"
            ),
            "delta_b_scope": (
                "joint effect scale inherited from Stage 2d; it scales both B drift "
                "and Q burst and is not a delta_B/delta_Q ratio"
            ),
        },
        "provenance": {
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "base_path": str(base_path),
            "base_sha256": sha256_file(base_path),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "aggregate": summaries,
        "per_seed": rows,
    }
    atomic_write_json(args.out, payload)
    atomic_write_csv(args.csv, rows)

    print("\nCell summaries")
    for summary in summaries:
        auc = summary["auc_residual_conservative"]
        label = (
            "CONFIRMATORY PASS" if summary["confirmatory_pass"] else
            "exploratory candidate" if summary["exploratory_candidate"] else
            "no pass"
        )
        print(
            f"T={summary['T']} delta_b={summary['delta_b']:g} "
            f"min_de={summary['min_de']:g}: residual_max_mean={auc['mean']:.3f} "
            f"95%CI=[{auc['ci95_low']:.3f}, {auc['ci95_high']:.3f}] -- {label}"
        )
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
