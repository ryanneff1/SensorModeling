"""HPC execution and export helpers for two-dimensional escape assays."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_escape_assay_2d_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run and export one geometry/density/replicate task."""

    import pandas as pd

    from utils.escape_assay_2d import (
        EscapeAssay2DParams,
        make_escape_geometry_2d,
        run_escape_assay_2d,
        summarize_geometry_comparison_2d,
    )

    run_directory = Path(task["run_directory"])
    manifest_path = run_directory / "manifest.json"
    try:
        if manifest_path.exists() and not task.get("overwrite", False):
            return {
                "status": "skipped",
                "condition_label": task["condition_label"],
                "replicate": int(task["replicate"]),
                "run_directory": str(run_directory),
            }

        params = EscapeAssay2DParams(**task["params"])
        geometry_arguments = dict(task["geometry"])
        geometry_kind = str(geometry_arguments.pop("kind"))
        geometry_name = str(geometry_arguments.pop("name"))
        geometry = make_escape_geometry_2d(
            params,
            geometry_kind,
            seed=int(task["receptor_seed"]),
            name=geometry_name,
            **geometry_arguments,
        )
        result = run_escape_assay_2d(
            params,
            geometry,
            seed=int(task["trajectory_seed"]),
        )
        summary = summarize_geometry_comparison_2d(result.trajectories).iloc[0]

        run_directory.mkdir(parents=True, exist_ok=True)
        result.trajectories.to_csv(
            run_directory / "trajectories.csv.gz", index=False
        )
        result.survival.to_csv(run_directory / "survival.csv.gz", index=False)
        result.receptors.to_csv(run_directory / "receptors.csv", index=False)
        pd.DataFrame(
            {
                "x_m": geometry.x_m,
                "height_m": geometry.height_m,
                "height_index": geometry.height_index,
            }
        ).to_csv(run_directory / "geometry_profile.csv", index=False)

        metadata = dict(task.get("metadata", {}))
        metadata.update(
            {
                "condition_label": task["condition_label"],
                "condition_index": int(task["condition_index"]),
                "replicate": int(task["replicate"]),
                "receptor_seed": int(task["receptor_seed"]),
                "trajectory_seed": int(task["trajectory_seed"]),
                "realized_receptor_count": int(geometry.n_receptors),
            }
        )
        # Store the exact profile length independently of receptor sampling.
        height_m = geometry.height_m
        slope = (
            np.roll(height_m, -1) - np.roll(height_m, 1)
        ) / (2.0 * params.lattice_spacing_m)
        metadata["surface_length_m"] = float(
            np.sum(params.lattice_spacing_m * np.sqrt(1.0 + slope**2))
        )

        manifest = {
            "format": "escape-assay-2d-run",
            "format_version": 1,
            "protocol": task["protocol"],
            "condition_label": task["condition_label"],
            "condition_index": int(task["condition_index"]),
            "replicate": int(task["replicate"]),
            "params": asdict(params),
            "geometry": {
                "kind": geometry_kind,
                "name": geometry_name,
                **geometry_arguments,
                "escape_z_m": float(geometry.escape_z_m),
            },
            "metadata": metadata,
            "summary": {key: _json_safe(value) for key, value in summary.items()},
            "files": {
                "trajectories": "trajectories.csv.gz",
                "survival": "survival.csv.gz",
                "receptors": "receptors.csv",
                "geometry_profile": "geometry_profile.csv",
            },
        }
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(manifest), stream, indent=2, sort_keys=True)

        return {
            "status": "completed",
            "condition_label": task["condition_label"],
            "replicate": int(task["replicate"]),
            "run_directory": str(run_directory),
            "n_receptors": int(geometry.n_receptors),
            "n_trajectories": int(len(result.trajectories)),
            "escape_probability": float(summary["escape_probability"]),
            "mean_bound_time_s": float(summary["mean_bound_time_s"]),
            "mean_rebindings": float(summary["mean_rebindings"]),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "condition_label": task.get("condition_label"),
            "replicate": task.get("replicate"),
            "run_directory": str(run_directory),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def execute_escape_assay_2d_tasks(
    *,
    tasks: list[dict[str, Any]],
    n_workers: int,
    output_root: Path,
    summary_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute tasks with IPyParallel and write a restart summary."""

    import ipyparallel as ipp

    if n_workers < 1:
        raise ValueError("n_workers must be at least one.")
    output_root.mkdir(parents=True, exist_ok=True)
    with ipp.Cluster(n=int(n_workers)) as client:
        client.wait_for_engines(int(n_workers))
        client[:].use_cloudpickle()
        results = client.load_balanced_view().map_async(
            run_escape_assay_2d_task, tasks
        ).get()

    summary = dict(summary_metadata)
    summary["n_workers"] = int(n_workers)
    summary["results"] = results
    summary_path = output_root / "task_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2)

    completed = sum(value.get("status") == "completed" for value in results)
    skipped = sum(value.get("status") == "skipped" for value in results)
    failed = [value for value in results if value.get("status") == "failed"]
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {len(failed)}")
    print(f"Summary:   {summary_path}")
    if failed:
        for value in failed:
            print(
                f"FAILED {value.get('condition_label')} replicate "
                f"{value.get('replicate')}: {value.get('error')}"
            )
        raise RuntimeError(f"{len(failed)} task(s) failed; see task_summary.json.")
    return results


__all__ = [
    "execute_escape_assay_2d_tasks",
    "run_escape_assay_2d_task",
]
