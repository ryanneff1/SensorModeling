"""Shared analysis helpers for the manuscript escape-assay datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _cloud_timeout_message(path: Path) -> str:
    return (
        f"Timed out while reading {path}. This commonly means the file is an "
        "unhydrated OneDrive/iCloud placeholder. Download the complete sweep "
        "folder with Finder's 'Download Now' or 'Always Keep on This Device', "
        "or copy it to a non-cloud local folder, before running the analysis. "
        "Do not skip the file: doing so would silently remove a replicate."
    )


def discover_runs(
    root: str | Path,
    *,
    cache_path: str | Path | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Index every completed escape-assay bundle below *root*.

    An optional CSV cache avoids rescanning large cloud-backed directory
    trees after the first successful discovery pass.
    """

    root = Path(root).expanduser().resolve()
    cache = None if cache_path is None else Path(cache_path).expanduser().resolve()
    if cache is not None and cache.exists() and not refresh_cache:
        cached = pd.read_csv(cache)
        cached["run_directory"] = cached["run_directory"].map(Path)
        return cached
    rows = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
        except TimeoutError as exc:
            raise TimeoutError(_cloud_timeout_message(manifest_path)) from exc
        if manifest.get("format") != "escape-assay-run":
            continue
        run = manifest.get("run_metadata", {})
        condition = run.get("condition", {}) or {}
        params = manifest.get("params", {})
        assay = run.get("assay", {}) or {}
        geometry = run.get("geometry", {}) or {}
        row = {
            "run_directory": manifest_path.parent,
            "relative_directory": str(manifest_path.parent.relative_to(root)),
            "protocol": run.get("protocol", "unknown"),
            "condition_label": condition.get(
                "condition_label", manifest_path.parent.parent.name
            ),
            "condition_index": condition.get("condition_index", np.nan),
            "replicate": int(run.get("replicate", 1)),
            "geometry_type": run.get("geometry_type", manifest.get("geometry_name")),
            "geometry_name": manifest.get("geometry_name"),
            "n_receptors": int(manifest.get("n_receptors", 0)),
            "n_trajectories": int(manifest.get("n_trajectories", 0)),
            "k_on_M_inv_s": float(params.get("k_on_M_inv_s", np.nan)),
            "k_off_s": float(params.get("k_off_s", np.nan)),
            "receptor_density_m2": float(params.get("receptor_density_m2", np.nan)),
            "receptor_count_override": params.get("receptor_count_override"),
            "a_m": float(params.get("a_m", np.nan)),
            "Lx_m": float(params.get("Lx_m", np.nan)),
            "Ly_m": float(params.get("Ly_m", np.nan)),
            "H_m": float(params.get("H_m", np.nan)),
            "escape_z_m": assay.get("escape_z_m"),
            "max_time_s": float(assay.get("max_time_s", np.nan)),
            "n_trials_per_location": int(assay.get("n_trials_per_location", 0)),
            "rebinding_multiplier": float(
                assay.get("rebinding_k_on_multiplier", 1.0)
            ),
            "background_occupancy_fraction": float(
                assay.get("background_occupancy_fraction", 0.0)
            ),
            "rebinding_mode": str(assay.get("rebinding_mode", "all")),
            "rebinding_classification": str(
                assay.get("rebinding_classification", "source_receptor")
            ),
        }
        row["KD_classical_M"] = (
            row["k_off_s"] / row["k_on_M_inv_s"]
            if row["k_on_M_inv_s"] > 0
            else np.nan
        )
        for source in (geometry, condition):
            for key, value in source.items():
                if key not in row and not isinstance(value, (dict, list)):
                    row[key] = value
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No escape-assay manifests found below {root}")
    result = pd.DataFrame(rows).sort_values(
        ["condition_index", "condition_label", "replicate"],
        na_position="last",
    ).reset_index(drop=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache, index=False)
    return result


def load_trajectories(run_directory: str | Path) -> pd.DataFrame:
    path = Path(run_directory) / "trajectories.csv.gz"
    try:
        frame = pd.read_csv(path)
    except TimeoutError as exc:
        raise TimeoutError(_cloud_timeout_message(path)) from exc
    frame["free_time_s"] = (
        frame["observation_time_s"] - frame["total_bound_time_s"]
    ).clip(lower=0)
    return frame


def summarize_replicates(
    run_index: pd.DataFrame,
    *,
    cache_path: str | Path | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Create one statistically independent summary row per replicate.

    When ``cache_path`` is supplied, the completed table is stored as a CSV
    and reused on later calls. This is especially useful for occupancy grids
    containing hundreds of compressed trajectory archives.
    """

    cache = None if cache_path is None else Path(cache_path).expanduser().resolve()
    if cache is not None and cache.exists() and not refresh_cache:
        return pd.read_csv(cache)

    rows = []
    for run in run_index.to_dict("records"):
        trajectory_path = Path(run["run_directory"]) / "trajectories.csv.gz"
        required_columns = [
            "escaped",
            "censored",
            "observation_time_s",
            "total_bound_time_s",
            "n_rebindings",
        ]
        optional_columns = ["n_self_rebindings", "n_cross_rebindings"]
        try:
            available_columns = pd.read_csv(trajectory_path, nrows=0).columns
            usecols = required_columns + [
                column for column in optional_columns if column in available_columns
            ]
            trajectories = pd.read_csv(trajectory_path, usecols=usecols)
        except TimeoutError as exc:
            raise TimeoutError(_cloud_timeout_message(trajectory_path)) from exc
        trajectories["free_time_s"] = (
            trajectories["observation_time_s"]
            - trajectories["total_bound_time_s"]
        ).clip(lower=0)
        mean_bound = float(trajectories["total_bound_time_s"].mean())
        summary = dict(run)
        summary.update(
            {
                "escape_probability": float(trajectories["escaped"].mean()),
                "censoring_fraction": float(trajectories["censored"].mean()),
                "rmst_s": float(trajectories["observation_time_s"].mean()),
                "median_observation_time_s": float(
                    trajectories["observation_time_s"].median()
                ),
                "mean_bound_time_s": mean_bound,
                "median_bound_time_s": float(
                    trajectories["total_bound_time_s"].median()
                ),
                "mean_rebindings": float(trajectories["n_rebindings"].mean()),
                "median_rebindings": float(trajectories["n_rebindings"].median()),
                "probability_any_rebinding": float(
                    (trajectories["n_rebindings"] > 0).mean()
                ),
                "mean_free_time_s": float(trajectories["free_time_s"].mean()),
            }
        )
        summary["KD_apparent_M"] = (
            1.0 / (summary["k_on_M_inv_s"] * mean_bound)
            if summary["k_on_M_inv_s"] > 0 and mean_bound > 0
            else np.nan
        )
        summary["affinity_shift_fold"] = (
            summary["KD_classical_M"] / summary["KD_apparent_M"]
            if summary["KD_apparent_M"] > 0
            else np.nan
        )
        if "n_self_rebindings" in trajectories:
            summary["mean_self_rebindings"] = float(
                trajectories["n_self_rebindings"].mean()
            )
        if "n_cross_rebindings" in trajectories:
            summary["mean_cross_rebindings"] = float(
                trajectories["n_cross_rebindings"].mean()
            )
        if {"n_self_rebindings", "n_cross_rebindings"}.issubset(trajectories):
            total = (
                trajectories["n_self_rebindings"]
                + trajectories["n_cross_rebindings"]
            ).sum()
            summary["cross_rebinding_fraction"] = (
                float(trajectories["n_cross_rebindings"].sum() / total)
                if total > 0
                else 0.0
            )
        rows.append(summary)
    result = pd.DataFrame(rows)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache, index=False)
    return result


def load_receptor_summaries(run_index: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for run in run_index.to_dict("records"):
        path = Path(run["run_directory"]) / "receptor_summary.csv"
        try:
            frame = pd.read_csv(path)
        except TimeoutError as exc:
            raise TimeoutError(_cloud_timeout_message(path)) from exc
        for key, value in run.items():
            if key != "run_directory" and key not in frame.columns:
                frame[key] = value
        frame["run_directory"] = str(run["run_directory"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def occupancy_from_bound_time(
    bound_time_s, k_on_M_inv_s, concentration_M
) -> np.ndarray:
    q = (
        np.asarray(bound_time_s, dtype=float)
        * np.asarray(k_on_M_inv_s, dtype=float)
        * np.asarray(concentration_M, dtype=float)
    )
    return np.divide(q, 1 + q, out=np.full(np.shape(q), np.nan), where=q >= 0)


def classical_occupancy(k_on_M_inv_s, k_off_s, concentration_M) -> np.ndarray:
    association = np.asarray(k_on_M_inv_s) * np.asarray(concentration_M)
    denominator = association + np.asarray(k_off_s)
    return np.divide(
        association,
        denominator,
        out=np.full(np.shape(denominator), np.nan),
        where=denominator > 0,
    )


def condition_statistics(
    replicate_summary: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    grouped = replicate_summary.groupby(list(group_columns), dropna=False, sort=False)
    pieces = []
    for metric in metrics:
        stats = grouped[metric].agg(["mean", "std", "sem", "count"]).reset_index()
        stats = stats.rename(
            columns={name: f"{metric}_{name}" for name in ("mean", "std", "sem", "count")}
        )
        pieces.append(stats)
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=list(group_columns), how="outer")
    return result


def kaplan_meier(times_s, events) -> tuple[np.ndarray, np.ndarray]:
    """Return right-continuous Kaplan–Meier coordinates including t=0."""

    times = np.asarray(times_s, dtype=float)
    events = np.asarray(events, dtype=bool)
    usable = np.isfinite(times)
    times = times[usable]
    events = events[usable]
    if times.size == 0:
        return np.array([0.0]), np.array([1.0])
    order = np.argsort(times)
    times, events = times[order], events[order]
    unique_times, first_indices = np.unique(times, return_index=True)
    event_counts = np.add.reduceat(events.astype(np.int64), first_indices)
    at_risk = times.size - first_indices
    has_events = event_counts > 0
    event_times = unique_times[has_events]
    conditional_survival = 1.0 - (
        event_counts[has_events] / at_risk[has_events]
    )
    survival = np.cumprod(conditional_survival)
    return (
        np.concatenate(([0.0], event_times.astype(float))),
        np.concatenate(([1.0], survival.astype(float))),
    )


def survival_on_grid(times_s, events, grid_s) -> np.ndarray:
    x, y = kaplan_meier(times_s, events)
    indices = np.searchsorted(x, np.asarray(grid_s), side="right") - 1
    return y[np.maximum(indices, 0)]


def replicate_survival_curves(
    run_index: pd.DataFrame,
    grid_s: Iterable[float],
) -> pd.DataFrame:
    frames = []
    grid = np.asarray(list(grid_s), dtype=float)
    for run in run_index.to_dict("records"):
        # Survival requires only two columns. Avoid parsing the much wider
        # trajectory table again after replicate-summary construction.
        path = Path(run["run_directory"]) / "trajectories.csv.gz"
        try:
            trajectories = pd.read_csv(
                path,
                usecols=["observation_time_s", "escaped"],
            )
        except TimeoutError as exc:
            raise TimeoutError(_cloud_timeout_message(path)) from exc
        frame = pd.DataFrame(
            {
                "time_s": grid,
                "survival": survival_on_grid(
                    trajectories["observation_time_s"],
                    trajectories["escaped"],
                    grid,
                ),
                "condition_label": run["condition_label"],
                "replicate": run["replicate"],
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_figure(
    fig, output_directory: str | Path, stem: str, dpi: int | None = None
) -> None:
    """Save PDF and PNG panels, honoring active rcParams by default."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    png_options = {"bbox_inches": "tight"}
    if dpi is not None:
        png_options["dpi"] = dpi
    fig.savefig(output / f"{stem}.png", **png_options)


__all__ = [
    "classical_occupancy",
    "condition_statistics",
    "discover_runs",
    "kaplan_meier",
    "load_receptor_summaries",
    "load_trajectories",
    "occupancy_from_bound_time",
    "replicate_survival_curves",
    "save_figure",
    "summarize_replicates",
    "survival_on_grid",
]
