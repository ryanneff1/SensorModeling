#!/usr/bin/env python3
"""Run a configuration-driven escape-assay manuscript experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from utils.escape_assay_sweeps import (
    execute_tasks,
    load_json,
    load_params_json,
    replicate_seed_pairs,
    validate_assay_settings,
)


DEFAULT_RESPONSE_OCCUPANCY_GRID = {
    "spacing": "logit",
    "minimum": 1e-3,
    "maximum": 1.0 - 1e-3,
    "count": 51,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--params-json", type=Path, required=True)
    value.add_argument("--experiment-json", type=Path, required=True)
    value.add_argument("--n-workers", type=int, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def _merged(base: dict, override: dict | None) -> dict:
    result = dict(base)
    result.update({} if override is None else override)
    return result


def _normalize_geometry(condition: dict) -> dict:
    geometry = dict(condition.get("geometry", {}))
    if condition.get("geometry_type") == "cylindrically_curved_sheet":
        # JSON null is the readable representation of the planar limit.
        if geometry.get("radius_m", "missing") is None:
            geometry["radius_m"] = np.inf
    return geometry


def _kinetic_parameter_conditions(config: dict) -> list[dict[str, Any]] | None:
    """Expand compact, readable logarithmic kinetic-sweep definitions."""

    sweeps = config.get("kinetic_sweeps")
    if sweeps is None:
        return None
    if not isinstance(sweeps, list) or not sweeps:
        raise ValueError("kinetic_sweeps must be a nonempty list.")

    conditions: list[dict[str, Any]] = []
    for sweep in sweeps:
        parameter = str(sweep["parameter"])
        if parameter not in {"k_on_M_inv_s", "k_off_s"}:
            raise ValueError(
                "Kinetic sweep parameter must be k_on_M_inv_s or k_off_s."
            )
        values = np.logspace(
            float(sweep["log10_start"]),
            float(sweep["log10_stop"]),
            int(sweep["count"]),
        )
        fixed = {
            key: float(value)
            for key, value in dict(sweep.get("fixed_params_overrides", {})).items()
        }
        prefix = str(sweep.get("label", parameter))
        for index, raw_value in enumerate(values):
            value = float(raw_value)
            params_overrides = dict(fixed)
            params_overrides[parameter] = value
            if "k_on_M_inv_s" not in params_overrides or "k_off_s" not in params_overrides:
                raise ValueError(
                    f"{prefix}: each generated condition must define both kinetic rates."
                )
            k_on = float(params_overrides["k_on_M_inv_s"])
            k_off = float(params_overrides["k_off_s"])
            assay_overrides = dict(sweep.get("assay_overrides", {}))
            if "max_time_factor_over_k_off" in sweep:
                factor = float(sweep["max_time_factor_over_k_off"])
                if factor <= 0 or k_off <= 0:
                    raise ValueError(
                        "max_time_factor_over_k_off and k_off_s must be positive."
                    )
                assay_overrides["max_time_s"] = factor / k_off
            conditions.append(
                {
                    "label": f"{prefix}_{index + 1:02d}",
                    "params_overrides": params_overrides,
                    "assay_overrides": assay_overrides,
                    "metadata": {
                        "kinetic_sweep_axis": parameter,
                        "kinetic_sweep_index": index,
                        "kinetic_sweep_value": value,
                        "k_on_M_inv_s": k_on,
                        "k_off_s": k_off,
                        "KD_M": k_off / k_on,
                        "max_time_s": float(assay_overrides.get("max_time_s", np.nan)),
                    },
                }
            )
    return conditions


def _background_occupancy_values(config: dict) -> list[float] | None:
    """Return explicit occupancies or construct the standard response grid."""

    explicit = config.get("background_occupancy_values")
    grid = config.get("background_occupancy_grid")
    if explicit is not None and grid is not None:
        raise ValueError(
            "Use either background_occupancy_values or "
            "background_occupancy_grid, not both."
        )
    if explicit is not None:
        return [float(value) for value in explicit]
    if grid is None:
        if config.get("sweep_parameter") != "background_occupancy_fraction":
            return None
        grid = DEFAULT_RESPONSE_OCCUPANCY_GRID
    if grid == "default":
        grid = DEFAULT_RESPONSE_OCCUPANCY_GRID
    if not isinstance(grid, dict):
        raise ValueError("background_occupancy_grid must be an object or 'default'.")

    spacing = str(grid.get("spacing", "logit")).lower()
    minimum = float(grid.get("minimum", 1e-3))
    maximum = float(grid.get("maximum", 1.0 - 1e-3))
    count = int(grid.get("count", 51))
    if not 0 < minimum < maximum < 1:
        raise ValueError("Generated occupancy limits must satisfy 0 < min < max < 1.")
    if count < 2:
        raise ValueError("Generated occupancy grids require at least two values.")
    if spacing == "logit":
        endpoints = np.log(
            np.asarray([minimum, maximum])
            / (1.0 - np.asarray([minimum, maximum]))
        )
        logits = np.linspace(endpoints[0], endpoints[1], count)
        values = 1.0 / (1.0 + np.exp(-logits))
    elif spacing == "linear":
        values = np.linspace(minimum, maximum, count)
    elif spacing == "log":
        values = np.geomspace(minimum, maximum, count)
    else:
        raise ValueError("Occupancy-grid spacing must be logit, linear, or log.")
    return [float(value) for value in values]


def _expand_conditions(config: dict) -> list[dict[str, Any]]:
    explicit = config.get("conditions")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("conditions must be a nonempty list.")
        conditions = [dict(value) for value in explicit]
    else:
        geometries = config.get("geometry_conditions", [])
        generated_parameters = _kinetic_parameter_conditions(config)
        if generated_parameters is not None and "parameter_conditions" in config:
            raise ValueError(
                "Use either kinetic_sweeps or parameter_conditions, not both."
            )
        parameters = (
            generated_parameters
            if generated_parameters is not None
            else config.get("parameter_conditions", [{}])
        )
        if not geometries or not isinstance(geometries, list):
            raise ValueError("geometry_conditions must be a nonempty list.")
        if not parameters or not isinstance(parameters, list):
            raise ValueError("parameter_conditions must be a nonempty list.")
        conditions = []
        for geometry_condition in geometries:
            for parameter_condition in parameters:
                geometry_label = str(geometry_condition["label"])
                parameter_label = str(parameter_condition.get("label", "baseline"))
                conditions.append(
                    {
                        "label": f"{geometry_label}__{parameter_label}",
                        "geometry_type": geometry_condition["geometry_type"],
                        "geometry": geometry_condition["geometry"],
                        "params_overrides": _merged(
                            geometry_condition.get("params_overrides", {}),
                            parameter_condition.get("params_overrides", {}),
                        ),
                        "assay_overrides": _merged(
                            geometry_condition.get("assay_overrides", {}),
                            parameter_condition.get("assay_overrides", {}),
                        ),
                        "metadata": _merged(
                            geometry_condition.get("metadata", {}),
                            parameter_condition.get("metadata", {}),
                        ),
                    }
                )

    occupancy_values = _background_occupancy_values(config)
    if occupancy_values is not None:
        values = occupancy_values
        if not values or len(set(values)) != len(values):
            raise ValueError(
                "background_occupancy_values must be a nonempty unique list."
            )
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("Background occupancies must lie in [0, 1].")
        # Retain the historical two-decimal directory names whenever they are
        # unique (and therefore preserve paths used by existing manuscript
        # sweeps). Log-spaced dilute grids need more precision: for example,
        # 1e-6 and 1e-5 would otherwise both become ``occupancy_0p00``.
        tokens = [f"{value:.2f}".replace(".", "p") for value in values]
        if len(set(tokens)) != len(tokens):
            tokens = [
                f"{value:.12g}"
                .replace(".", "p")
                .replace("+", "")
                .replace("-", "m")
                for value in values
            ]
        if len(set(tokens)) != len(tokens):
            raise ValueError(
                "Background occupancies could not be assigned unique labels."
            )
        expanded = []
        for condition in conditions:
            for occupancy, token in zip(values, tokens):
                value = dict(condition)
                value["label"] = f"{condition['label']}__occupancy_{token}"
                value["assay_overrides"] = _merged(
                    condition.get("assay_overrides", {}),
                    {"background_occupancy_fraction": occupancy},
                )
                value["metadata"] = _merged(
                    condition.get("metadata", {}),
                    {"background_occupancy_fraction": occupancy},
                )
                expanded.append(value)
        conditions = expanded
    return conditions


def main() -> None:
    args = parser().parse_args()
    base_params, params_config = load_params_json(args.params_json)
    config = load_json(args.experiment_json)
    protocol = str(config.get("protocol", args.experiment_json.stem))
    n_replicates = int(config.get("n_replicates", 1))
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least one.")
    base_assay = dict(config.get("assay", {}))
    conditions = _expand_conditions(config)
    export_settings = dict(config.get("export", {}))
    valid_params = {field.name for field in fields(type(base_params))}
    labels = [str(condition["label"]) for condition in conditions]
    if len(set(labels)) != len(labels):
        raise ValueError("Condition labels must be unique.")

    seed_pairs = replicate_seed_pairs(base_params.seed, n_replicates)
    output_root = args.output_root.expanduser().resolve()
    tasks = []
    condition_manifest = []
    for condition_index, condition in enumerate(conditions):
        label = str(condition["label"])
        overrides = dict(condition.get("params_overrides", {}))
        unknown = set(overrides).difference(valid_params)
        if unknown:
            raise ValueError(f"{label}: unknown Params fields {sorted(unknown)}")
        for tuple_field in ("open_boundaries", "periodic_axes"):
            if tuple_field in overrides:
                overrides[tuple_field] = tuple(overrides[tuple_field])
        params = replace(base_params, **overrides)
        assay = validate_assay_settings(
            _merged(base_assay, condition.get("assay_overrides", {}))
        )
        geometry_type = str(condition["geometry_type"])
        geometry = _normalize_geometry(condition)
        metadata = dict(condition.get("metadata", {}))
        metadata.update(
            {
                "condition_label": label,
                "condition_index": condition_index,
                "params_overrides": overrides,
                "assay_overrides": condition.get("assay_overrides", {}),
            }
        )
        condition_manifest.append(
            {
                "label": label,
                "geometry_type": geometry_type,
                "geometry": geometry,
                "params": asdict(params),
                "assay": assay,
                "metadata": metadata,
            }
        )
        for replicate, (receptor_seed, trajectory_seed) in enumerate(seed_pairs, 1):
            tasks.append(
                {
                    "protocol": protocol,
                    "geometry_type": geometry_type,
                    "sweep_parameter": str(config.get("sweep_parameter", "condition")),
                    "sweep_value": float(condition_index),
                    "replicate": replicate,
                    "receptor_seed": receptor_seed,
                    "trajectory_seed": trajectory_seed,
                    "params": asdict(params),
                    "geometry": geometry,
                    "assay": assay,
                    "condition_metadata": metadata,
                    "export": export_settings,
                    "run_directory": str(
                        output_root / label / f"replicate_{replicate:03d}"
                    ),
                    "overwrite": bool(args.overwrite),
                }
            )

    print("=" * 72)
    print(protocol)
    print(f"Conditions : {len(conditions)}")
    print(f"Replicates : {n_replicates}")
    print(f"Tasks      : {len(tasks)}")
    print(f"Workers    : {args.n_workers}")
    print(f"Output     : {output_root}")
    print("=" * 72)
    execute_tasks(
        tasks=tasks,
        n_workers=args.n_workers,
        output_root=output_root,
        summary_metadata={
            "protocol": protocol,
            "sweep_parameter": config.get("sweep_parameter", "condition"),
            "n_replicates": n_replicates,
            "params_config": params_config,
            "experiment_config": config,
            "conditions": condition_manifest,
        },
    )


if __name__ == "__main__":
    main()
