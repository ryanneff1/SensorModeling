#!/usr/bin/env python3
"""Sweep a cylindrical sheet from planar to a semicircular trough."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from utils.escape_assay_sweeps import (
    condition_token,
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
    if config.get("geometry_type") != "cylindrically_curved_sheet":
        raise ValueError("geometry_type must be 'cylindrically_curved_sheet'.")

    geometry_base = config.get("geometry", {})
    sweep = config.get("sweep", {})
    if sweep.get("parameter") != "radius_m":
        raise ValueError("Sheet sweep parameter must be 'radius_m'.")
    raw_radii = sweep.get("values", [])
    if not raw_radii:
        raise ValueError("Radius values must be a nonempty list.")
    radii = [np.inf if value is None else float(value) for value in raw_radii]
    if sum(np.isinf(radii)) != 1 or any(value <= 0 for value in radii):
        raise ValueError("Use exactly one null planar radius and positive finite radii.")
    if len(set(radii)) != len(radii):
        raise ValueError("Radius values must be unique.")

    axis = str(geometry_base.get("axis", "y")).lower()
    if axis not in {"x", "y"}:
        raise ValueError("geometry.axis must be 'x' or 'y'.")
    perpendicular_size_m = params.Ly_m if axis == "x" else params.Lx_m
    perpendicular_count = int(round(perpendicular_size_m / params.a_m))
    half_span_m = 0.5 * (perpendicular_count - 1) * params.a_m
    finite_radii = np.asarray([value for value in radii if np.isfinite(value)])
    if np.any(finite_radii < half_span_m - 1e-15):
        raise ValueError(
            f"Every finite radius must be >= the lattice half-span {half_span_m:.6e} m."
        )

    edge_z_m = float(geometry_base.get("edge_z_m", half_span_m))
    assay = validate_assay_settings(config.get("assay", {}))
    output_root = args.output_root.expanduser().resolve()
    seed_pairs = replicate_seed_pairs(params.seed, args.n_replicates)
    tasks = []
    for radius_m in radii:
        if np.isinf(radius_m):
            sagitta_m = 0.0
            label = "planar"
        else:
            root_m = np.sqrt(max(radius_m**2 - half_span_m**2, 0.0))
            sagitta_m = half_span_m**2 / (radius_m + root_m)
            label = condition_token("radius_m", radius_m)
        vertex_z_m = edge_z_m - sagitta_m
        if vertex_z_m < -1e-15:
            raise ValueError("edge_z_m is too low for the requested sheet curvature.")
        geometry_values = {
            "radius_m": radius_m,
            "axis": axis,
            "center_xy_m": geometry_base.get("center_xy_m"),
            "vertex_z_m": max(vertex_z_m, 0.0),
            "name": f"cylindrically_curved_sheet_{label}",
        }
        for replicate, (receptor_seed, trajectory_seed) in enumerate(seed_pairs, 1):
            tasks.append(
                {
                    "protocol": "escape_assay_cylindrical_sheet_curvature_sweep",
                    "geometry_type": "cylindrically_curved_sheet",
                    "sweep_parameter": "radius_m",
                    "sweep_value": None if np.isinf(radius_m) else radius_m,
                    "replicate": replicate,
                    "receptor_seed": receptor_seed,
                    "trajectory_seed": trajectory_seed,
                    "params": asdict(params),
                    "geometry": geometry_values,
                    "assay": assay,
                    "run_directory": str(
                        output_root / label / f"replicate_{replicate:03d}"
                    ),
                    "overwrite": bool(args.overwrite),
                }
            )

    curvature_values = [0.0 if np.isinf(value) else 1.0 / value for value in radii]
    print("=" * 72)
    print("Cylindrically curved-sheet escape-assay sweep")
    print(f"Radii (m)  : {radii}")
    print(f"Curvatures : {curvature_values}")
    print(f"Half-span  : {half_span_m:.6e} m (semicircle radius)")
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
            "protocol": "escape_assay_cylindrical_sheet_curvature_sweep",
            "geometry_type": "cylindrically_curved_sheet",
            "sweep_parameter": "radius_m",
            "sweep_values": [None if np.isinf(value) else value for value in radii],
            "curvature_values_m_inv": curvature_values,
            "semicircle_radius_m": half_span_m,
            "n_replicates": args.n_replicates,
            "params_config": params_config,
            "geometry_config": config,
        },
    )


if __name__ == "__main__":
    main()
