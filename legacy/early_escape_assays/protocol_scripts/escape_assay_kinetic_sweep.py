#!/usr/bin/env python3
"""Sweep paired association/dissociation rates at constant KD on one bowl."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from utils.escape_assay_sweeps import (
    condition_token,
    execute_tasks,
    load_json,
    load_params_json,
    replicate_seed_pairs,
    validate_assay_settings,
    validate_bowl_geometry,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--params-json", type=Path, required=True)
    value.add_argument("--geometry-json", type=Path, required=True)
    value.add_argument("--sweep-json", type=Path, required=True)
    value.add_argument("--n-replicates", type=int, required=True)
    value.add_argument("--n-workers", type=int, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def validate_conditions(sweep: dict) -> tuple[list[dict], float]:
    if sweep.get("parameter") != "kinetic_scale":
        raise ValueError("Kinetic sweep parameter must be 'kinetic_scale'.")
    target_kd = float(sweep.get("target_KD_M", np.nan))
    if not np.isfinite(target_kd) or target_kd <= 0:
        raise ValueError("target_KD_M must be finite and positive.")
    raw_conditions = sweep.get("conditions", [])
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("sweep.conditions must be a nonempty list.")

    conditions = []
    for raw in raw_conditions:
        condition = dict(raw)
        required = {"label", "kinetic_scale", "k_on_M_inv_s", "k_off_s"}
        missing = required.difference(condition)
        if missing:
            raise ValueError(f"Kinetic condition is missing: {sorted(missing)}")
        condition["label"] = str(condition["label"])
        for key in ("kinetic_scale", "k_on_M_inv_s", "k_off_s"):
            condition[key] = float(condition[key])
            if not np.isfinite(condition[key]) or condition[key] <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        condition["KD_M"] = condition["k_off_s"] / condition["k_on_M_inv_s"]
        if not np.isclose(condition["KD_M"], target_kd, rtol=1e-10, atol=0.0):
            raise ValueError(
                f"Condition {condition['label']!r} has KD={condition['KD_M']:.6g} M; "
                f"expected {target_kd:.6g} M."
            )
        conditions.append(condition)

    for field in ("label", "kinetic_scale"):
        values = [condition[field] for condition in conditions]
        if len(set(values)) != len(values):
            raise ValueError(f"Kinetic condition {field}s must be unique.")
    return conditions, target_kd


def main() -> None:
    args = parser().parse_args()
    if args.n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")
    base_params, params_config = load_params_json(args.params_json)
    geometry_config = load_json(args.geometry_json)
    if geometry_config.get("geometry_type") != "spherical_bowl":
        raise ValueError("geometry_type must be 'spherical_bowl'.")
    geometry = validate_bowl_geometry(
        geometry_config.get("geometry", {}), base_params
    )
    sweep_config = load_json(args.sweep_json)
    sweep = sweep_config.get("sweep", sweep_config)
    if not isinstance(sweep, dict):
        raise ValueError("Kinetic sweep configuration requires a sweep object.")
    conditions, target_kd = validate_conditions(sweep)
    assay = validate_assay_settings(sweep_config.get("assay", {}))

    output_root = args.output_root.expanduser().resolve()
    seed_pairs = replicate_seed_pairs(base_params.seed, args.n_replicates)
    tasks = []
    for condition in conditions:
        params = replace(
            base_params,
            k_on_M_inv_s=condition["k_on_M_inv_s"],
            k_off_s=condition["k_off_s"],
        )
        condition_directory = (
            f"{condition_token('kinetic_scale', condition['kinetic_scale'])}_"
            f"kon_{condition['k_on_M_inv_s']:.8g}_koff_{condition['k_off_s']:.8g}"
        )
        for replicate, (receptor_seed, trajectory_seed) in enumerate(seed_pairs, 1):
            tasks.append(
                {
                    "protocol": "escape_assay_kinetic_sweep",
                    "geometry_type": "spherical_bowl",
                    "sweep_parameter": "kinetic_scale",
                    "sweep_value": condition["kinetic_scale"],
                    "replicate": replicate,
                    "receptor_seed": receptor_seed,
                    "trajectory_seed": trajectory_seed,
                    "params": asdict(params),
                    "geometry": geometry,
                    "assay": assay,
                    "condition_metadata": condition,
                    "run_directory": str(
                        output_root / condition_directory / f"replicate_{replicate:03d}"
                    ),
                    "overwrite": bool(args.overwrite),
                }
            )

    print("=" * 72)
    print("Spherical-bowl constant-KD escape-assay kinetic sweep")
    print(f"Target KD  : {target_kd:.6g} M")
    for condition in conditions:
        print(
            f"  {condition['kinetic_scale']:g}x: "
            f"kon={condition['k_on_M_inv_s']:.6g} M^-1 s^-1, "
            f"koff={condition['k_off_s']:.6g} s^-1"
        )
    print(f"Periodic   : {base_params.periodic_axes}")
    print(f"Replicates : {args.n_replicates}")
    print(f"Tasks      : {len(tasks)}")
    print(f"Workers    : {args.n_workers}")
    print(f"Output     : {output_root}")
    print("=" * 72)
    execute_tasks(
        tasks=tasks,
        n_workers=args.n_workers,
        output_root=output_root,
        summary_metadata={
            "protocol": "escape_assay_kinetic_sweep",
            "geometry_type": "spherical_bowl",
            "sweep_parameter": "kinetic_scale",
            "sweep_values": [c["kinetic_scale"] for c in conditions],
            "target_KD_M": target_kd,
            "kinetic_conditions": conditions,
            "n_replicates": args.n_replicates,
            "params_config": params_config,
            "geometry_config": geometry_config,
            "sweep_config": sweep_config,
        },
    )


if __name__ == "__main__":
    main()
