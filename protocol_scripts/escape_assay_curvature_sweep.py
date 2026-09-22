#!/usr/bin/env python3
"""Run spherical-bowl curvature escape assays with IPyParallel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

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
    if config.get("geometry_type") != "spherical_bowl":
        raise ValueError("geometry_type must be 'spherical_bowl'.")
    geometry_base = config.get("geometry")
    sweep = config.get("sweep")
    if not isinstance(geometry_base, dict) or not isinstance(sweep, dict):
        raise ValueError("Geometry config requires geometry and sweep objects.")
    if sweep.get("parameter") != "radius_m":
        raise ValueError("Curvature sweep parameter must be 'radius_m'.")
    radii = [float(value) for value in sweep.get("values", [])]
    if not radii or len(set(radii)) != len(radii):
        raise ValueError("Radius values must be a nonempty unique list.")
    assay = validate_assay_settings(config.get("assay", {}))
    output_root = args.output_root.expanduser().resolve()
    seed_pairs = replicate_seed_pairs(params.seed, args.n_replicates)
    tasks = []
    for radius_m in radii:
        geometry_values = dict(geometry_base)
        geometry_values["radius_m"] = radius_m
        geometry_values["name"] = f"spherical_bowl_radius_{radius_m:.8g}m"
        geometry_values = validate_bowl_geometry(geometry_values, params)
        for replicate, (receptor_seed, trajectory_seed) in enumerate(seed_pairs, start=1):
            tasks.append(
                {
                    "protocol": "escape_assay_curvature_sweep",
                    "sweep_parameter": "radius_m",
                    "sweep_value": radius_m,
                    "replicate": replicate,
                    "receptor_seed": receptor_seed,
                    "trajectory_seed": trajectory_seed,
                    "params": asdict(params),
                    "geometry": geometry_values,
                    "assay": assay,
                    "run_directory": str(
                        output_root
                        / condition_token("radius_m", radius_m)
                        / f"replicate_{replicate:03d}"
                    ),
                    "overwrite": bool(args.overwrite),
                }
            )

    print("=" * 72)
    print("Spherical-bowl escape-assay curvature sweep")
    print(f"Radii (m)  : {radii}")
    print(f"Curvatures : {[1.0 / value for value in radii]}")
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
            "protocol": "escape_assay_curvature_sweep",
            "sweep_parameter": "radius_m",
            "sweep_values": radii,
            "curvature_values_m_inv": [1.0 / value for value in radii],
            "n_replicates": args.n_replicates,
            "params_config": params_config,
            "geometry_config": config,
        },
    )


if __name__ == "__main__":
    main()
