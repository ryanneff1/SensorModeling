"""Export, analysis, and visualization helpers for first-capture assays."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.biosensor_mc import Params, SensorGeometry
from utils.capture_assay import CaptureAssayResult
from utils.escape_assay_viz import plot_surface_metric_3d, plot_surface_metric_xy


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class CaptureAssayArchive:
    """A self-contained first-capture assay loaded from an export folder."""

    directory: Path
    params: Params
    geometry: SensorGeometry
    trajectories: pd.DataFrame
    receptors: pd.DataFrame
    condition_summary: pd.DataFrame
    receptor_summary: pd.DataFrame
    metadata: dict


def export_capture_assay_run(
    result: CaptureAssayResult,
    output_directory,
    *,
    run_metadata: Optional[dict] = None,
) -> Path:
    """Export all tables and exact geometry required to reproduce plots."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    result.trajectories.to_csv(directory / "trajectories.csv.gz", index=False)
    result.receptors.to_csv(directory / "receptors.csv", index=False)
    result.condition_summary.to_csv(directory / "condition_summary.csv", index=False)
    result.receptor_summary.to_csv(directory / "receptor_summary.csv", index=False)

    geometry = result.derived.geometry
    np.savez_compressed(
        directory / "geometry.npz",
        solid_mask=geometry.solid_mask,
        surface_solid_xyz=geometry.surface_solid_xyz,
        surface_fluid_xyz=geometry.surface_fluid_xyz,
        surface_normals=geometry.surface_normals,
        surface_centers_m=geometry.surface_centers_m,
        surface_area_m2=geometry.surface_area_m2,
        reactive_face_mask=geometry.reactive_face_mask,
    )
    metadata = {
        "format": "capture-assay-run",
        "format_version": 1,
        "geometry_name": geometry.name,
        "params": asdict(result.params),
        "n_trajectories": int(len(result.trajectories)),
        "n_receptors": int(len(result.receptors)),
        "capture_probability": float(
            result.condition_summary.loc[0, "capture_probability"]
        ),
        "files": {
            "geometry": "geometry.npz",
            "trajectories": "trajectories.csv.gz",
            "receptors": "receptors.csv",
            "condition_summary": "condition_summary.csv",
            "receptor_summary": "receptor_summary.csv",
        },
    }
    if run_metadata is not None:
        metadata["run_metadata"] = dict(run_metadata)
    with (directory / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(metadata), stream, indent=2, sort_keys=True, allow_nan=False)
    return directory


def load_capture_assay_run(input_directory) -> CaptureAssayArchive:
    """Load a folder produced by :func:`export_capture_assay_run`."""

    directory = Path(input_directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No capture-assay manifest found at {manifest_path}.")
    with manifest_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("format") != "capture-assay-run":
        raise ValueError(f"Unsupported run format in {manifest_path}.")
    if metadata.get("format_version") != 1:
        raise ValueError(
            f"Unsupported capture-assay format version: {metadata.get('format_version')}"
        )
    files = metadata["files"]
    with np.load(directory / files["geometry"], allow_pickle=False) as arrays:
        geometry = SensorGeometry(
            name=str(metadata["geometry_name"]),
            solid_mask=arrays["solid_mask"].copy(),
            surface_solid_xyz=arrays["surface_solid_xyz"].copy(),
            surface_fluid_xyz=arrays["surface_fluid_xyz"].copy(),
            surface_normals=arrays["surface_normals"].copy(),
            surface_centers_m=arrays["surface_centers_m"].copy(),
            surface_area_m2=arrays["surface_area_m2"].copy(),
            reactive_face_mask=arrays["reactive_face_mask"].copy(),
        )
    params_data = dict(metadata["params"])
    for tuple_field in ("open_boundaries", "periodic_axes"):
        if tuple_field in params_data:
            params_data[tuple_field] = tuple(params_data[tuple_field])
    return CaptureAssayArchive(
        directory=directory,
        params=Params(**params_data),
        geometry=geometry,
        trajectories=pd.read_csv(directory / files["trajectories"]),
        receptors=pd.read_csv(directory / files["receptors"]),
        condition_summary=pd.read_csv(directory / files["condition_summary"]),
        receptor_summary=pd.read_csv(directory / files["receptor_summary"]),
        metadata=metadata,
    )


def plot_capture_surface_3d(
    geometry: SensorGeometry,
    receptor_summary: pd.DataFrame,
    metric: str = "fraction_of_captures",
    **plot_kwargs,
):
    """Plot a nearest-receptor interpolation of capture outcomes in 3D."""

    plot_kwargs.setdefault("cmap", "magma")
    plot_kwargs.setdefault("colorbar_label", metric.replace("_", " "))
    return plot_surface_metric_3d(
        geometry, receptor_summary, metric, **plot_kwargs
    )


def plot_capture_surface_xy(
    geometry: SensorGeometry,
    receptor_summary: pd.DataFrame,
    metric: str = "fraction_of_captures",
    **plot_kwargs,
):
    """Plot an overhead nearest-receptor interpolation of capture outcomes."""

    plot_kwargs.setdefault("cmap", "magma")
    plot_kwargs.setdefault("colorbar_label", metric.replace("_", " "))
    return plot_surface_metric_xy(
        geometry, receptor_summary, metric, **plot_kwargs
    )


def plot_capture_time_distribution(
    trajectories: pd.DataFrame,
    *,
    bins: int = 50,
    time_scale: float = 1e3,
    time_unit: str = "ms",
    ax=None,
):
    """Plot the conditional distribution of first-capture times."""

    times = trajectories.loc[trajectories["captured"], "capture_time_s"].dropna()
    if times.empty:
        raise ValueError("No captured trajectories are available.")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(times * time_scale, bins=bins, density=True, alpha=0.75, color="tab:blue")
    ax.axvline(times.mean() * time_scale, color="black", linestyle="--", label="mean")
    ax.axvline(times.median() * time_scale, color="tab:red", linestyle=":", label="median")
    ax.set(
        xlabel=f"First-capture time ({time_unit})",
        ylabel="Conditional probability density",
        title="Capture-time distribution given capture",
    )
    ax.legend()
    return ax


def plot_cumulative_capture_and_escape(
    trajectories: pd.DataFrame,
    *,
    time_scale: float = 1e3,
    time_unit: str = "ms",
    ax=None,
):
    """Plot empirical cumulative capture and bulk-escape probabilities."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    n_total = len(trajectories)
    for event_column, time_column, label, color in (
        ("captured", "capture_time_s", "captured", "tab:blue"),
        ("escaped", "escape_time_s", "escaped to bulk", "tab:orange"),
    ):
        times = np.sort(
            trajectories.loc[trajectories[event_column], time_column]
            .dropna()
            .to_numpy(float)
        )
        x = np.concatenate([[0.0], times]) * time_scale
        y = np.arange(times.size + 1, dtype=float) / n_total
        ax.step(x, y, where="post", color=color, linewidth=2, label=label)
    ax.set(
        xlabel=f"Time ({time_unit})",
        ylabel="Fraction of launched trajectories",
        ylim=(-0.02, 1.02),
        title="Competing first-capture and bulk-escape outcomes",
    )
    ax.legend()
    return ax


def plot_initial_plane_capture_probability(
    trajectories: pd.DataFrame,
    *,
    bins: int | Sequence[int] = 20,
    coordinate_scale: float = 1e9,
    coordinate_unit: str = "nm",
    ax=None,
):
    """Plot capture probability as a function of initial lateral position."""

    x = trajectories["initial_x_m"].to_numpy(float) * coordinate_scale
    y = trajectories["initial_y_m"].to_numpy(float) * coordinate_scale
    captured = trajectories["captured"].to_numpy(float)
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    captures, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=captured)
    probability = np.divide(
        captures, counts, out=np.full_like(captures, np.nan), where=counts > 0
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    image = ax.pcolormesh(x_edges, y_edges, probability.T, cmap="viridis", vmin=0, vmax=1)
    ax.set(
        xlabel=f"Initial x ({coordinate_unit})",
        ylabel=f"Initial y ({coordinate_unit})",
        title="Capture probability by injection position",
        aspect="equal",
    )
    ax.figure.colorbar(image, ax=ax, label="Capture probability")
    return ax, probability, counts


def plot_capture_vs_surface_height(
    receptor_summary: pd.DataFrame,
    *,
    metric: str = "n_captures",
    coordinate_scale: float = 1e9,
    coordinate_unit: str = "nm",
    ax=None,
):
    """Inspect how receptor capture outcome varies with surface height."""

    if metric not in receptor_summary:
        raise ValueError(f"Unknown receptor-summary metric: {metric!r}")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    x = receptor_summary["surface_z_m"].to_numpy(float) * coordinate_scale
    y = receptor_summary[metric].to_numpy(float)
    ax.scatter(x, y, s=12, alpha=0.35, color="tab:blue")
    if np.unique(x).size > 1:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 200)
        ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=2)
    ax.set(
        xlabel=f"Receptor surface height ({coordinate_unit})",
        ylabel=metric.replace("_", " "),
        title="Capture outcome versus surface position",
    )
    return ax


def plot_capture_surface_from_folder(input_directory, metric="fraction_of_captures", **kwargs):
    archive = load_capture_assay_run(input_directory)
    kwargs.setdefault("a_m", archive.params.a_m)
    figure, axes, interpolated = plot_capture_surface_3d(
        archive.geometry, archive.receptor_summary, metric, **kwargs
    )
    return figure, axes, interpolated, archive


__all__ = [
    "CaptureAssayArchive",
    "export_capture_assay_run",
    "load_capture_assay_run",
    "plot_capture_surface_3d",
    "plot_capture_surface_xy",
    "plot_capture_surface_from_folder",
    "plot_capture_time_distribution",
    "plot_cumulative_capture_and_escape",
    "plot_initial_plane_capture_probability",
    "plot_capture_vs_surface_height",
]
