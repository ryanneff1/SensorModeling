"""Analysis and surface visualization helpers for single-molecule escape assays."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

from utils.biosensor_mc import Params, SensorGeometry
from utils.escape_assay import EscapeAssayResult

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


@dataclass
class EscapeAssayArchive:
    """A self-contained escape-assay run loaded from an export folder."""

    directory: Path
    params: Params
    geometry: SensorGeometry
    trajectories: pd.DataFrame
    survival: pd.DataFrame
    releases: pd.DataFrame
    receptors: pd.DataFrame
    receptor_summary: pd.DataFrame
    metadata: dict


def export_escape_assay_run(
    result: EscapeAssayResult,
    output_directory,
    *,
    receptor_summary: Optional[pd.DataFrame] = None,
    run_metadata: Optional[dict] = None,
) -> Path:
    """Export tables and exact geometry needed to reproduce assay plots.

    The resulting directory is portable within this project and can be loaded
    using :func:`load_escape_assay_run`. The compressed geometry archive stores
    the voxel mask and every surface-face array rather than relying on the
    original geometry-construction arguments.
    """

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if receptor_summary is None:
        receptor_summary = summarize_receptor_trajectories(result)

    result.trajectories.to_csv(directory / "trajectories.csv.gz", index=False)
    result.survival.to_csv(directory / "survival.csv.gz", index=False)
    result.releases.to_csv(directory / "release_locations.csv", index=False)
    result.receptors.to_csv(directory / "receptors.csv", index=False)
    receptor_summary.to_csv(directory / "receptor_summary.csv", index=False)

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
        "format": "escape-assay-run",
        "format_version": 1,
        "geometry_name": geometry.name,
        "params": asdict(result.params),
        "n_trajectories": int(len(result.trajectories)),
        "n_release_locations": int(len(result.releases)),
        "n_receptors": int(len(result.receptors)),
        "files": {
            "geometry": "geometry.npz",
            "trajectories": "trajectories.csv.gz",
            "survival": "survival.csv.gz",
            "releases": "release_locations.csv",
            "receptors": "receptors.csv",
            "receptor_summary": "receptor_summary.csv",
        },
    }
    if run_metadata is not None:
        metadata["run_metadata"] = dict(run_metadata)
    with (directory / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    return directory


def load_escape_assay_run(input_directory) -> EscapeAssayArchive:
    """Load a self-contained assay folder created by the export helper."""

    directory = Path(input_directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No escape-assay manifest found at {manifest_path}. "
            "Export the run with export_escape_assay_run()."
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("format") != "escape-assay-run":
        raise ValueError(f"Unsupported run format in {manifest_path}.")
    if metadata.get("format_version") != 1:
        raise ValueError(
            f"Unsupported escape-assay format version: {metadata.get('format_version')}"
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
    if "open_boundaries" in params_data:
        params_data["open_boundaries"] = tuple(params_data["open_boundaries"])
    params = Params(**params_data)
    return EscapeAssayArchive(
        directory=directory,
        params=params,
        geometry=geometry,
        trajectories=pd.read_csv(directory / files["trajectories"]),
        survival=pd.read_csv(directory / files["survival"]),
        releases=pd.read_csv(directory / files["releases"]),
        receptors=pd.read_csv(directory / files["receptors"]),
        receptor_summary=pd.read_csv(directory / files["receptor_summary"]),
        metadata=metadata,
    )


def summarize_receptor_trajectories(
    result: EscapeAssayResult,
) -> pd.DataFrame:
    """Return one spatially referenced summary row per release receptor."""

    summary = (
        result.trajectories.groupby(["location_id", "label"], sort=False)
        .agg(
            n_trials=("trial_id", "size"),
            n_escaped=("escaped", "sum"),
            escape_probability=("escaped", "mean"),
            median_observation_time_s=("observation_time_s", "median"),
            mean_observation_time_s=("observation_time_s", "mean"),
            median_escape_time_s=("escape_time_s", "median"),
            median_rebindings=("n_rebindings", "median"),
            mean_rebindings=("n_rebindings", "mean"),
            probability_any_rebinding=(
                "n_rebindings",
                lambda values: float(np.mean(values.to_numpy() > 0)),
            ),
            median_self_rebindings=("n_self_rebindings", "median"),
            median_cross_rebindings=("n_cross_rebindings", "median"),
            mean_bound_time_s=("total_bound_time_s", "mean"),
        )
        .reset_index()
    )

    releases = result.releases[
        [
            "location_id",
            "source_face_id",
            "release_x_m",
            "release_y_m",
            "release_z_m",
        ]
    ]
    summary = summary.merge(releases, on="location_id", how="left", validate="one_to_one")

    receptor_positions = result.receptors.rename(
        columns={
            "face_id": "source_face_id",
            "x_m": "surface_x_m",
            "y_m": "surface_y_m",
            "z_m": "surface_z_m",
        }
    )
    return summary.merge(
        receptor_positions,
        on="source_face_id",
        how="left",
        validate="one_to_one",
    )


def interpolate_receptor_metric_to_surface(
    geometry: SensorGeometry,
    receptor_summary: pd.DataFrame,
    metric: str,
    *,
    surface_face_ids: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Assign each surface face the value of its nearest release receptor.

    This is a nearest-neighbor (surface Voronoi) interpolation in 3D Euclidean
    space. The returned table records the source receptor and interpolation
    distance so sparse regions can be identified explicitly.
    """

    required = {"source_face_id", "surface_x_m", "surface_y_m", "surface_z_m", metric}
    missing = required.difference(receptor_summary.columns)
    if missing:
        raise ValueError(f"receptor_summary is missing columns: {sorted(missing)}")

    usable = receptor_summary.dropna(
        subset=["surface_x_m", "surface_y_m", "surface_z_m", metric]
    ).copy()
    if usable.empty:
        raise ValueError(f"No finite receptor values are available for metric {metric!r}.")

    if surface_face_ids is None:
        face_ids = np.flatnonzero(geometry.reactive_face_mask).astype(np.int64)
    else:
        face_ids = np.asarray(surface_face_ids, dtype=np.int64)
    if face_ids.ndim != 1 or face_ids.size == 0:
        raise ValueError("surface_face_ids must identify at least one surface face.")
    if np.any(face_ids < 0) or np.any(face_ids >= geometry.n_surface_faces):
        raise ValueError("surface_face_ids contains an invalid face ID.")

    receptor_xyz = usable[["surface_x_m", "surface_y_m", "surface_z_m"]].to_numpy(float)
    surface_xyz = geometry.surface_centers_m[face_ids]

    if cKDTree is not None:
        distances, nearest = cKDTree(receptor_xyz).query(surface_xyz, k=1)
    else:  # pragma: no cover
        nearest = np.empty(face_ids.size, dtype=np.int64)
        distances = np.empty(face_ids.size, dtype=float)
        for start in range(0, face_ids.size, 1_000):
            stop = min(start + 1_000, face_ids.size)
            delta = surface_xyz[start:stop, None, :] - receptor_xyz[None, :, :]
            distance_squared = np.sum(delta * delta, axis=2)
            nearest[start:stop] = np.argmin(distance_squared, axis=1)
            distances[start:stop] = np.sqrt(
                distance_squared[np.arange(stop - start), nearest[start:stop]]
            )

    nearest_rows = usable.iloc[nearest]
    return pd.DataFrame(
        {
            "face_id": face_ids,
            "surface_x_m": surface_xyz[:, 0],
            "surface_y_m": surface_xyz[:, 1],
            "surface_z_m": surface_xyz[:, 2],
            "nearest_source_face_id": nearest_rows["source_face_id"].to_numpy(np.int64),
            "nearest_location_id": nearest_rows["location_id"].to_numpy(np.int64),
            "nearest_receptor_distance_m": distances,
            metric: nearest_rows[metric].to_numpy(float),
        }
    )


def _surface_face_vertices(
    geometry: SensorGeometry,
    face_ids: np.ndarray,
    a_m: float,
) -> np.ndarray:
    centers = geometry.surface_centers_m[face_ids]
    normals = geometry.surface_normals[face_ids]
    vertices = np.empty((face_ids.size, 4, 3), dtype=float)
    signs = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))

    for row, (center, normal) in enumerate(zip(centers, normals)):
        tangent_axes = np.flatnonzero(normal == 0)
        if tangent_axes.size != 2:
            raise ValueError("Surface normals must be Cartesian unit vectors.")
        vertices[row] = center
        for corner, (first_sign, second_sign) in enumerate(signs):
            vertices[row, corner, tangent_axes[0]] += first_sign * 0.5 * a_m
            vertices[row, corner, tangent_axes[1]] += second_sign * 0.5 * a_m
    return vertices


def plot_surface_metric_3d(
    geometry: SensorGeometry,
    receptor_summary: pd.DataFrame,
    metric: str,
    *,
    a_m: float,
    surface_face_ids: Optional[Sequence[int]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: Optional[str] = None,
    coordinate_scale: float = 1e9,
    coordinate_unit: str = "nm",
    show_receptors: bool = True,
    receptor_size: float = 5.0,
    elev: float = 28.0,
    azim: float = -55.0,
    figsize: tuple[float, float] = (9.0, 7.0),
    ax=None,
):
    """Plot nearest-receptor interpolation as colored 3D surface faces."""

    interpolated = interpolate_receptor_metric_to_surface(
        geometry,
        receptor_summary,
        metric,
        surface_face_ids=surface_face_ids,
    )
    face_ids = interpolated["face_id"].to_numpy(np.int64)
    values = interpolated[metric].to_numpy(float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"Metric {metric!r} has no finite interpolated values.")

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.01, 1e-12)
        vmin -= padding
        vmax += padding

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    face_colors = colormap(norm(values))
    face_colors[~finite] = (0.7, 0.7, 0.7, 1.0)
    vertices = _surface_face_vertices(geometry, face_ids, a_m) * coordinate_scale

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    collection = Poly3DCollection(
        vertices,
        facecolors=face_colors,
        edgecolors="none",
        linewidths=0,
        antialiased=False,
    )
    ax.add_collection3d(collection)

    centers = geometry.surface_centers_m[face_ids] * coordinate_scale
    xyz_min = np.min(centers, axis=0)
    xyz_max = np.max(centers, axis=0)
    padding = np.maximum((xyz_max - xyz_min) * 0.03, a_m * coordinate_scale)
    ax.set_xlim(xyz_min[0] - padding[0], xyz_max[0] + padding[0])
    ax.set_ylim(xyz_min[1] - padding[1], xyz_max[1] + padding[1])
    ax.set_zlim(xyz_min[2] - padding[2], xyz_max[2] + padding[2])
    ax.set_box_aspect(np.maximum(xyz_max - xyz_min, a_m * coordinate_scale))

    if show_receptors:
        receptor_xyz = receptor_summary[
            ["surface_x_m", "surface_y_m", "surface_z_m"]
        ].to_numpy(float) * coordinate_scale
        ax.scatter(
            receptor_xyz[:, 0], receptor_xyz[:, 1], receptor_xyz[:, 2],
            s=receptor_size, c="black", alpha=0.55, depthshade=False,
            label="release receptors",
        )

    ax.set(
        xlabel=f"x ({coordinate_unit})",
        ylabel=f"y ({coordinate_unit})",
        zlabel=f"z ({coordinate_unit})",
        title=metric.replace("_", " "),
    )
    ax.view_init(elev=elev, azim=azim)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    scalar_mappable.set_array(values)
    colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.68, pad=0.08)
    colorbar.set_label(colorbar_label or metric.replace("_", " "))
    return fig, ax, interpolated


def plot_kaplan_meier_curves(
    survival: pd.DataFrame,
    *,
    location_ids: Optional[Sequence[int]] = None,
    time_scale: float = 1e3,
    time_unit: str = "ms",
    max_curves: int = 20,
    ax=None,
):
    """Plot selected receptor-specific Kaplan--Meier survival curves."""

    data = survival
    if location_ids is not None:
        selected = set(map(int, location_ids))
        data = data[data["location_id"].isin(selected)]
    available = data["location_id"].drop_duplicates().to_numpy(np.int64)
    if available.size > max_curves:
        chosen = available[np.linspace(0, available.size - 1, max_curves, dtype=int)]
        data = data[data["location_id"].isin(chosen)]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for _, curve in data.groupby("location_id", sort=False):
        ax.step(
            curve["t_s"] * time_scale,
            curve["survival_probability"],
            where="post",
            alpha=0.65,
            linewidth=1.2,
        )
    ax.set(
        xlabel=f"Time after release ({time_unit})",
        ylabel="Survival probability",
        ylim=(-0.02, 1.02),
        title="Receptor-specific Kaplan–Meier curves",
    )
    return ax


def plot_surface_metric_from_folder(
    input_directory,
    metric: str,
    **plot_kwargs,
):
    """Load an exported run folder and plot one interpolated surface metric.

    Returns ``(figure, axes, interpolated_table, archive)`` so callers can
    reuse the loaded tables for additional plots without reading them again.
    """

    archive = load_escape_assay_run(input_directory)
    plot_kwargs.setdefault("a_m", archive.params.a_m)
    figure, axes, interpolated = plot_surface_metric_3d(
        archive.geometry,
        archive.receptor_summary,
        metric,
        **plot_kwargs,
    )
    return figure, axes, interpolated, archive


def plot_kaplan_meier_from_folder(
    input_directory,
    **plot_kwargs,
):
    """Load an exported run folder and plot its Kaplan--Meier curves."""

    archive = load_escape_assay_run(input_directory)
    axes = plot_kaplan_meier_curves(archive.survival, **plot_kwargs)
    return axes, archive


__all__ = [
    "EscapeAssayArchive",
    "export_escape_assay_run",
    "interpolate_receptor_metric_to_surface",
    "load_escape_assay_run",
    "plot_kaplan_meier_from_folder",
    "plot_kaplan_meier_curves",
    "plot_surface_metric_from_folder",
    "plot_surface_metric_3d",
    "summarize_receptor_trajectories",
]
