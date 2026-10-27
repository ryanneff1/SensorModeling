#!/usr/bin/env python3
"""Run configuration-defined 2D manuscript escape-assay sweeps."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from utils.escape_assay_2d import EscapeAssay2DParams
from utils.escape_assay_2d_sweeps import execute_escape_assay_2d_tasks
from utils.escape_assay_sweeps import replicate_seed_pairs


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--params-json", type=Path, required=True)
    value.add_argument("--experiment-json", type=Path, required=True)
    value.add_argument("--n-workers", type=int, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def load_json(path: Path) -> dict:
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def load_params(path: Path) -> tuple[EscapeAssay2DParams, dict]:
    raw = load_json(path)
    values = dict(raw.get("parameters", raw))
    factor = float(values.pop("max_time_factor_over_k_off", 1e6))
    if factor <= 0:
        raise ValueError("max_time_factor_over_k_off must be positive.")
    k_off_s = float(values.get("k_off_s", EscapeAssay2DParams.k_off_s))
    if k_off_s <= 0:
        raise ValueError("k_off_s must be positive for factor-based censoring.")
    values["max_time_s"] = factor / k_off_s
    valid = {field.name for field in fields(EscapeAssay2DParams)}
    unknown = set(values).difference(valid)
    if unknown:
        raise ValueError(f"Unknown EscapeAssay2DParams fields: {sorted(unknown)}")
    return EscapeAssay2DParams(**values), raw


def radius_token(radius_nm: float | None) -> str:
    return "planar" if radius_nm is None else f"R{int(radius_nm):03d}nm"


def density_token(density_m_inv: float) -> str:
    return f"rho_{density_m_inv:.8g}".replace("+", "")


def condition(
    radius_nm: float | None,
    density_m_inv: float,
    *,
    design_memberships: set[str],
) -> dict:
    is_planar = radius_nm is None
    geometry = {
        "kind": "planar" if is_planar else "curved",
        "name": radius_token(radius_nm),
    }
    radius_m = None if is_planar else float(radius_nm) * 1e-9
    if radius_m is not None:
        geometry["radius_m"] = radius_m
    label = f"{radius_token(radius_nm)}__{density_token(density_m_inv)}"
    return {
        "label": label,
        "radius_nm": radius_nm,
        "radius_m": radius_m,
        "curvature_m_inv": 0.0 if is_planar else 1.0 / radius_m,
        "density_m_inv": float(density_m_inv),
        "geometry": geometry,
        "design_memberships": sorted(design_memberships),
    }


def expand_full_union(config: dict) -> list[dict]:
    design = config["design"]
    reference_length_m = float(design["planar_reference_length_m"])
    radii = [
        None if value is None else float(value)
        for value in design["curvature_radii_nm"]
    ]
    counts = [int(value) for value in design["planar_receptor_counts"]]
    if len(radii) != len(counts):
        raise ValueError(
            "curvature_radii_nm and planar_receptor_counts must have equal lengths."
        )
    densities = [value / reference_length_m for value in counts]
    curvature_density = float(design["curvature_sweep_density_m_inv"])
    density_radius = float(design["density_sweep_radius_nm"])
    factorial_radii = [
        None if value is None else float(value)
        for value in design["factorial_radii_nm"]
    ]
    factorial_counts = [int(value) for value in design["factorial_planar_counts"]]
    factorial_densities = [value / reference_length_m for value in factorial_counts]

    memberships: dict[tuple[float | None, float], set[str]] = {}
    for radius in radii:
        memberships.setdefault((radius, curvature_density), set()).add(
            "curvature_sweep"
        )
    for density in densities:
        memberships.setdefault((density_radius, density), set()).add(
            "density_sweep"
        )
    for radius in factorial_radii:
        for density in factorial_densities:
            memberships.setdefault((radius, density), set()).add("factorial_5x5")

    conditions = [
        condition(radius, density, design_memberships=value)
        for (radius, density), value in memberships.items()
    ]
    return sorted(
        conditions,
        key=lambda value: (
            -1.0 if value["radius_nm"] is None else value["radius_nm"],
            value["density_m_inv"],
        ),
        reverse=True,
    )


def expand_corner_controls(config: dict) -> list[dict]:
    design = config["design"]
    reference_length_m = float(design["planar_reference_length_m"])
    radii = [
        None if value is None else float(value)
        for value in design["corner_radii_nm"]
    ]
    densities = [
        int(value) / reference_length_m
        for value in design["corner_planar_counts"]
    ]
    return [
        condition(radius, density, design_memberships={"no_rebinding_control"})
        for radius in radii
        for density in densities
    ]


def main() -> None:
    args = parser().parse_args()
    base_params, raw_params = load_params(args.params_json)
    config = load_json(args.experiment_json)
    mode = str(config["design_type"])
    if mode == "full_rebinding_union":
        conditions = expand_full_union(config)
    elif mode == "corner_controls":
        conditions = expand_corner_controls(config)
    else:
        raise ValueError(f"Unsupported design_type: {mode!r}")

    n_replicates = int(config.get("n_replicates", 8))
    seed_pairs = replicate_seed_pairs(base_params.seed, n_replicates)
    p_bind_contact = float(
        config.get("p_bind_contact", base_params.p_bind_contact)
    )
    output_root = args.output_root.expanduser().resolve()
    tasks = []
    condition_manifest = []
    for condition_index, value in enumerate(conditions):
        params = replace(
            base_params,
            receptor_line_density_m_inv=value["density_m_inv"],
            p_bind_contact=p_bind_contact,
        )
        metadata = {
            "radius_nm": value["radius_nm"],
            "radius_m": value["radius_m"],
            "curvature_m_inv": value["curvature_m_inv"],
            "receptor_line_density_m_inv": value["density_m_inv"],
            "expected_planar_receptor_count": (
                value["density_m_inv"]
                * float(config["design"]["planar_reference_length_m"])
            ),
            "design_memberships": value["design_memberships"],
            "rebinding_enabled": p_bind_contact > 0,
        }
        condition_manifest.append(
            {
                "label": value["label"],
                "geometry": value["geometry"],
                "params": asdict(params),
                "metadata": metadata,
            }
        )
        for replicate, (receptor_seed, trajectory_seed) in enumerate(
            seed_pairs, start=1
        ):
            tasks.append(
                {
                    "protocol": str(config["protocol"]),
                    "condition_label": value["label"],
                    "condition_index": condition_index,
                    "replicate": replicate,
                    "receptor_seed": receptor_seed,
                    "trajectory_seed": trajectory_seed,
                    "params": asdict(params),
                    "geometry": value["geometry"],
                    "metadata": metadata,
                    "run_directory": str(
                        output_root
                        / value["label"]
                        / f"replicate_{replicate:03d}"
                    ),
                    "overwrite": bool(args.overwrite),
                }
            )

    print("=" * 72)
    print(config["protocol"])
    print(f"Conditions : {len(conditions)}")
    print(f"Replicates : {n_replicates}")
    print(f"Tasks      : {len(tasks)}")
    print(f"Workers    : {args.n_workers}")
    print(f"Output     : {output_root}")
    print("=" * 72)
    execute_escape_assay_2d_tasks(
        tasks=tasks,
        n_workers=args.n_workers,
        output_root=output_root,
        summary_metadata={
            "protocol": config["protocol"],
            "design_type": mode,
            "n_replicates": n_replicates,
            "params_config": raw_params,
            "experiment_config": config,
            "conditions": condition_manifest,
        },
    )


if __name__ == "__main__":
    main()
