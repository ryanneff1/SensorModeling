#!/usr/bin/env python3
"""Run receptor-density escape assays on one spherical bowl with IPyParallel."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
    value.add_argument("--sweep-json", type=Path, required=True)
    value.add_argument("--n-replicates", type=int, required=True)
    value.add_argument("--n-workers", type=int, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")
    base_params, params_config = load_params_json(args.params_json)
    config = load_json(args.geometry_json)
    if config.get("geometry_type") != "spherical_bowl":
        raise ValueError("geometry_type must be 'spherical_bowl'.")
    geometry_values = validate_bowl_geometry(config.get("geometry", {}), base_params)
    sweep_config = load_json(args.sweep_json)
    sweep = sweep_config.get("sweep", sweep_config)
    if not isinstance(sweep, dict) or sweep.get("parameter") != "receptor_density_m2":
        raise ValueError("Density sweep parameter must be 'receptor_density_m2'.")
    densities = [float(value) for value in sweep.get("values", [])]
    if not densities or len(set(densities)) != len(densities):
        raise ValueError("Density values must be a nonempty unique list.")
    if any(value <= 0 for value in densities):
        raise ValueError("Densities must be positive for receptor-release assays.")
    assay = validate_assay_settings(sweep_config.get("assay", {}))
    output_root = args.output_root.expanduser().resolve()
    seed_pairs = replicate_seed_pairs(base_params.seed, args.n_replicates)
    tasks = []
    for density in densities:
        params = replace(
            base_params,
            receptor_density_m2=density,
            receptor_count_override=None,
        )
        for replicate, (receptor_seed, trajectory_seed) in enumerate(seed_pairs, start=1):
            tasks.append(
                {
                    "protocol": "escape_assay_receptor_density_sweep",
                    "sweep_parameter": "receptor_density_m2",
                    "sweep_value": density,
                    "replicate": replicate,
                    "receptor_seed": receptor_seed,
                    "trajectory_seed": trajectory_seed,
                    "params": asdict(params),
                    "geometry": geometry_values,
                    "assay": assay,
                    "run_directory": str(
                        output_root
                        / condition_token("receptor_density_m2", density)
                        / f"replicate_{replicate:03d}"
                    ),
                    "overwrite": bool(args.overwrite),
                }
            )

    print("=" * 72)
    print("Spherical-bowl escape-assay receptor-density sweep")
    print(f"Densities  : {densities}")
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
            "protocol": "escape_assay_receptor_density_sweep",
            "sweep_parameter": "receptor_density_m2",
            "sweep_values": densities,
            "n_replicates": args.n_replicates,
            "params_config": params_config,
            "geometry_config": config,
            "sweep_config": sweep_config,
        },
    )


if __name__ == "__main__":
    main()
