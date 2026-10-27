#!/usr/bin/env python3
"""Run independent escape-assay replicates on one sinusoidal height field."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from utils.escape_assay_sweeps import (
    execute_tasks,
    load_json,
    load_params_json,
    replicate_seed_pairs,
    validate_assay_settings,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--params-json", type=Path, required=True)
    value.add_argument("--geometry-json", type=Path, required=True)
    value.add_argument("--n-replicates", type=int, required=True)
    value.add_argument("--n-workers", type=int, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")
    params, params_config = load_params_json(args.params_json)
    config = load_json(args.geometry_json)
    if config.get("geometry_type") != "sinusoidal_height_field":
        raise ValueError("geometry_type must be 'sinusoidal_height_field'.")
    geometry = dict(config.get("geometry", {}))
    required = {"amplitude_m", "wavelength_x_m", "wavelength_y_m", "mean_z_m"}
    missing = required.difference(geometry)
    if missing:
        raise ValueError(f"Sinusoidal geometry is missing: {sorted(missing)}")
    if any(float(geometry[key]) <= 0 for key in required):
        raise ValueError("Amplitude, wavelengths, and mean height must be positive.")
    if float(geometry["mean_z_m"]) - float(geometry["amplitude_m"]) < 0:
        raise ValueError("The sinusoidal surface cannot extend below z=0.")
    if float(geometry["mean_z_m"]) + float(geometry["amplitude_m"]) > params.H_m:
        raise ValueError("The sinusoidal surface cannot exceed H_m.")

    assay = validate_assay_settings(config.get("assay", {}))
    output_root = args.output_root.expanduser().resolve()
    seed_pairs = replicate_seed_pairs(params.seed, args.n_replicates)
    tasks = []
    for replicate, (receptor_seed, trajectory_seed) in enumerate(seed_pairs, 1):
        tasks.append(
            {
                "protocol": "escape_assay_sinusoidal_field_replicates",
                "geometry_type": "sinusoidal_height_field",
                "sweep_parameter": "geometry_replicate",
                "sweep_value": 0.0,
                "replicate": replicate,
                "receptor_seed": receptor_seed,
                "trajectory_seed": trajectory_seed,
                "params": asdict(params),
                "geometry": geometry,
                "assay": assay,
                "run_directory": str(output_root / f"replicate_{replicate:03d}"),
                "overwrite": bool(args.overwrite),
            }
        )

    print("=" * 72)
    print("Fixed sinusoidal-field escape-assay replicates")
    print(f"Geometry   : {geometry}")
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
            "protocol": "escape_assay_sinusoidal_field_replicates",
            "geometry_type": "sinusoidal_height_field",
            "sweep_parameter": None,
            "sweep_values": [],
            "n_replicates": args.n_replicates,
            "curvature_columns": [
                "mean_curvature_m_inv",
                "abs_mean_curvature_m_inv",
                "gaussian_curvature_m_inv2",
                "principal_curvature_1_m_inv",
                "principal_curvature_2_m_inv"
            ],
            "params_config": params_config,
            "geometry_config": config,
        },
    )


if __name__ == "__main__":
    main()
