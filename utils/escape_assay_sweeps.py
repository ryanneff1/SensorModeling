"""Shared IPyParallel machinery for escape-assay sweeps."""

from __future__ import annotations

import json
import math
import traceback
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from utils.biosensor_mc import Params


def load_params_json(path: str | Path) -> tuple[Params, dict]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    values = raw.get("parameters", raw)
    if not isinstance(values, dict):
        raise ValueError("Parameter JSON must contain an object.")
    valid = {field.name for field in fields(Params)}
    unknown = set(values).difference(valid)
    if unknown:
        raise ValueError(f"Unknown Params fields: {sorted(unknown)}")
    values = dict(values)
    for tuple_field in ("open_boundaries", "periodic_axes"):
        if tuple_field in values:
            values[tuple_field] = tuple(values[tuple_field])
    return Params(**values), raw


def load_json(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def replicate_seed_pairs(base_seed: int, n_replicates: int) -> List[Tuple[int, int]]:
    replicate_sequences = np.random.SeedSequence(int(base_seed)).spawn(n_replicates)
    pairs = []
    for sequence in replicate_sequences:
        receptor_sequence, trajectory_sequence = sequence.spawn(2)
        pairs.append(
            (
                int(receptor_sequence.generate_state(1, dtype=np.uint32)[0]),
                int(trajectory_sequence.generate_state(1, dtype=np.uint32)[0]),
            )
        )
    return pairs


def condition_token(parameter: str, value: float) -> str:
    return f"{parameter}_{float(value):.8g}".replace("+", "")


def validate_assay_settings(settings: dict) -> dict:
    required = {"n_trials_per_location", "max_time_s", "escape_mode"}
    missing = required.difference(settings)
    if missing:
        raise ValueError(f"Assay configuration is missing: {sorted(missing)}")
    normalized = dict(settings)
    normalized["n_trials_per_location"] = int(normalized["n_trials_per_location"])
    normalized["max_time_s"] = float(normalized["max_time_s"])
    if normalized["n_trials_per_location"] < 1:
        raise ValueError("n_trials_per_location must be at least 1.")
    if normalized["max_time_s"] <= 0:
        raise ValueError("max_time_s must be positive.")
    if normalized["escape_mode"] not in {"z_plane", "surface_distance", "domain_exit"}:
        raise ValueError("Unsupported escape_mode.")
    if normalized["escape_mode"] == "z_plane":
        if normalized.get("escape_z_m") is None:
            raise ValueError("z_plane assays require escape_z_m.")
        normalized["escape_z_m"] = float(normalized["escape_z_m"])
    if normalized.get("escape_distance_m") is not None:
        normalized["escape_distance_m"] = float(normalized["escape_distance_m"])
    return normalized


def validate_bowl_geometry(geometry: dict, params: Params) -> dict:
    allowed = {"radius_m", "depth_m", "center_xy_m", "rim_z_m", "name"}
    unknown = set(geometry).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown spherical-bowl arguments: {sorted(unknown)}")
    value = dict(geometry)
    if "center_xy_m" in value:
        value["center_xy_m"] = tuple(map(float, value["center_xy_m"]))
    for key in ("radius_m", "depth_m", "rim_z_m"):
        if key in value:
            value[key] = float(value[key])
    if value["radius_m"] <= 0 or value["depth_m"] <= 0:
        raise ValueError("Bowl radius and depth must be positive.")
    if value["depth_m"] > value["radius_m"]:
        raise ValueError("Bowl depth cannot exceed its sphere radius.")
    if value.get("rim_z_m", value["depth_m"]) > params.H_m:
        raise ValueError("Bowl rim_z_m cannot exceed H_m.")
    return value


def run_escape_sweep_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run one condition/replicate and export a self-contained run bundle."""

    from pathlib import Path
    import numpy as np

    from utils.biosensor_mc import Params
    from utils.escape_assay import (
        run_escape_assay,
        sample_receptor_faces,
        surface_release_locations,
    )
    from utils.escape_assay_viz import (
        export_escape_assay_run,
        summarize_receptor_trajectories,
    )
    from utils.generate_geometries import (
        make_cylindrically_curved_sheet_geometry,
        make_sinusoidal_height_field_geometry,
        make_spherical_bowl_geometry,
        sinusoidal_surface_curvatures,
    )

    run_directory = Path(task["run_directory"])
    manifest_path = run_directory / "manifest.json"
    raw_sweep_value = task.get("sweep_value")
    sweep_value = (
        None
        if raw_sweep_value is None or not np.isfinite(float(raw_sweep_value))
        else float(raw_sweep_value)
    )
    try:
        if manifest_path.exists() and not task["overwrite"]:
            return {
                "status": "skipped",
                "sweep_parameter": task["sweep_parameter"],
                "sweep_value": sweep_value,
                "replicate": task["replicate"],
                "run_directory": str(run_directory),
            }

        params_data = dict(task["params"])
        for tuple_field in ("open_boundaries", "periodic_axes"):
            if tuple_field in params_data:
                params_data[tuple_field] = tuple(params_data[tuple_field])
        params = Params(**params_data)
        geometry_type = task.get("geometry_type", "spherical_bowl")
        geometry_builders = {
            "spherical_bowl": make_spherical_bowl_geometry,
            "cylindrically_curved_sheet": make_cylindrically_curved_sheet_geometry,
            "sinusoidal_height_field": make_sinusoidal_height_field_geometry,
        }
        if geometry_type not in geometry_builders:
            raise ValueError(f"Unsupported geometry_type: {geometry_type!r}")
        geometry = geometry_builders[geometry_type](params, **task["geometry"])
        receptor_faces = sample_receptor_faces(
            params,
            geometry,
            seed=int(task["receptor_seed"]),
        )
        if receptor_faces.size == 0:
            raise ValueError(
                "This condition sampled zero receptors, so no receptor-release "
                "locations exist. Increase receptor density."
            )
        releases = surface_release_locations(
            geometry,
            receptor_faces,
            prefix="receptor",
        )
        assay = task["assay"]
        result = run_escape_assay(
            params,
            geometry,
            release_locations=releases,
            n_trials_per_location=int(assay["n_trials_per_location"]),
            max_time_s=float(assay["max_time_s"]),
            escape_mode=str(assay["escape_mode"]),
            escape_distance_m=assay.get("escape_distance_m"),
            escape_z_m=assay.get("escape_z_m"),
            receptor_face_ids=receptor_faces,
            seed=int(task["trajectory_seed"]),
        )
        receptor_summary = summarize_receptor_trajectories(result)
        if geometry_type == "sinusoidal_height_field":
            curvature_arguments = {
                key: task["geometry"][key]
                for key in (
                    "amplitude_m",
                    "wavelength_x_m",
                    "wavelength_y_m",
                    "phase_x_rad",
                    "phase_y_rad",
                )
                if key in task["geometry"]
            }
            if curvature_arguments.get("wavelength_y_m") is None:
                raise ValueError(
                    "Curvature annotation currently requires a two-dimensional "
                    "sinusoidal field with wavelength_y_m set."
                )
            curvature = sinusoidal_surface_curvatures(
                receptor_summary["surface_x_m"].to_numpy(float),
                receptor_summary["surface_y_m"].to_numpy(float),
                **curvature_arguments,
            )
            for column, values in curvature.items():
                receptor_summary[column] = values
            receptor_summary["abs_mean_curvature_m_inv"] = np.abs(
                receptor_summary["mean_curvature_m_inv"]
            )
        metadata = {
            "protocol": task["protocol"],
            "geometry_type": geometry_type,
            "sweep_parameter": task["sweep_parameter"],
            "sweep_value": sweep_value,
            "replicate": int(task["replicate"]),
            "receptor_seed": int(task["receptor_seed"]),
            "trajectory_seed": int(task["trajectory_seed"]),
            "geometry": task["geometry"],
            "assay": assay,
        }
        if task.get("condition_metadata") is not None:
            metadata["condition"] = task["condition_metadata"]
        export_escape_assay_run(
            result,
            run_directory,
            receptor_summary=receptor_summary,
            run_metadata=metadata,
        )
        return {
            "status": "completed",
            "sweep_parameter": task["sweep_parameter"],
            "sweep_value": sweep_value,
            "replicate": int(task["replicate"]),
            "run_directory": str(run_directory),
            "n_receptors": int(receptor_faces.size),
            "n_trajectories": int(len(result.trajectories)),
            "escape_probability": float(result.trajectories["escaped"].mean()),
            "median_observation_time_s": float(
                result.trajectories["observation_time_s"].median()
            ),
            "median_rebindings": float(result.trajectories["n_rebindings"].median()),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "sweep_parameter": task.get("sweep_parameter"),
            "sweep_value": task.get("sweep_value"),
            "replicate": task.get("replicate"),
            "run_directory": str(run_directory),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def execute_tasks(
    *,
    tasks: list[dict],
    n_workers: int,
    output_root: Path,
    summary_metadata: dict,
) -> list[dict]:
    import ipyparallel as ipp

    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")
    output_root.mkdir(parents=True, exist_ok=True)
    with ipp.Cluster(n=int(n_workers)) as client:
        client.wait_for_engines(int(n_workers))
        client[:].use_cloudpickle()
        results = client.load_balanced_view().map_async(
            run_escape_sweep_task,
            tasks,
        ).get()

    summary = dict(summary_metadata)
    summary["n_workers"] = int(n_workers)
    summary["results"] = results
    summary_path = output_root / "task_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    failed = [result for result in results if result.get("status") == "failed"]
    print(f"Completed: {sum(r.get('status') == 'completed' for r in results)}")
    print(f"Skipped:   {sum(r.get('status') == 'skipped' for r in results)}")
    print(f"Failed:    {len(failed)}")
    print(f"Summary:   {summary_path}")
    if failed:
        for result in failed:
            print(
                f"FAILED {result.get('sweep_parameter')}={result.get('sweep_value')}, "
                f"replicate={result.get('replicate')}: {result.get('error')}"
            )
        raise RuntimeError(f"{len(failed)} task(s) failed; see task_summary.json.")
    return results


__all__ = [
    "condition_token",
    "execute_tasks",
    "load_json",
    "load_params_json",
    "replicate_seed_pairs",
    "validate_assay_settings",
    "validate_bowl_geometry",
]
