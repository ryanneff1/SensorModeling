from __future__ import annotations

import gzip
import json
import os
import pickle
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


try:
    from utils.biosensor_mc import (
        MODEL_VERSION,
        state_from_checkpoint,
        state_to_checkpoint,
    )
except ImportError:
    MODEL_VERSION = "unknown"
    state_from_checkpoint = None
    state_to_checkpoint = None


PathLike = Union[str, Path]


def _json_safe(value: Any) -> Any:
    """
    Convert common NumPy and dataclass objects into JSON-compatible values.
    """
    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return None
        return str(value)

    return value


def _atomic_json_dump(
    data: Dict[str, Any],
    path: Path,
) -> None:
    """
    Write JSON through a temporary file to reduce the chance of leaving a
    partial result if a cluster job is interrupted.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(data),
            file,
            indent=2,
            sort_keys=True,
        )

    os.replace(temporary_path, path)


def _atomic_gzip_pickle_dump(
    value: Any,
    path: Path,
) -> None:
    """
    Write a gzip-compressed pickle atomically.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with gzip.open(temporary_path, "wb") as file:
        pickle.dump(
            value,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(temporary_path, path)


def _gzip_pickle_load(path: Path) -> Any:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def _save_dataframe(
    dataframe: pd.DataFrame,
    output_directory: Path,
    stem: str,
    table_format: str,
) -> Path:
    """
    Save a DataFrame as Parquet or compressed CSV.

    Parquet is preferred for speed, size, and dtype preservation. If the
    required Parquet dependency is unavailable, the function automatically
    falls back to gzip-compressed CSV.
    """
    table_format = table_format.lower()

    if table_format not in {"parquet", "csv"}:
        raise ValueError(
            "table_format must be 'parquet' or 'csv'."
        )

    if table_format == "parquet":
        final_path = output_directory / f"{stem}.parquet"
        temporary_path = output_directory / f"{stem}.parquet.tmp"

        try:
            dataframe.to_parquet(
                temporary_path,
                index=False,
            )
            os.replace(temporary_path, final_path)
            return final_path

        except (ImportError, ModuleNotFoundError):
            if temporary_path.exists():
                temporary_path.unlink()

            # Continue using compressed CSV.
            table_format = "csv"

    final_path = output_directory / f"{stem}.csv.gz"
    temporary_path = output_directory / f"{stem}.csv.gz.tmp"

    dataframe.to_csv(
        temporary_path,
        index=False,
        compression="gzip",
    )

    os.replace(temporary_path, final_path)
    return final_path


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.name.endswith(".csv.gz"):
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported table file: {path}"
    )


def save_simulation_results(
    output_directory: PathLike,
    *,
    P,
    history: pd.DataFrame,
    state=None,
    G=None,
    state_frames=None,
    rebinding_events: Optional[pd.DataFrame] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    table_format: str = "parquet",
    overwrite: bool = False,
) -> Dict[str, Path]:
    """
    Save all outputs from one biosensor Monte Carlo run.

    Parameters
    ----------
    output_directory
        Directory dedicated to this simulation run.

    P
        Params instance used for the most recent simulation phase.

    history
        History DataFrame returned by run_simulation.

    state
        Optional microscopic State returned when return_state=True. When the
        updated resumable simulation is available, this is converted to a
        portable checkpoint before saving.

    G
        Optional Derived object. The function saves G.geometry rather than the
        complete Derived object because Derived can be recalculated for a new
        assay phase.

    state_frames
        Optional list of state frames used for animation.

    rebinding_events
        Optional event-level rebinding DataFrame.

    run_metadata
        Optional user metadata, such as nanopore diameter, replicate number,
        assay name, cluster job ID, or protocol name.

    table_format
        "parquet" or "csv". Parquet automatically falls back to compressed CSV
        if pyarrow or fastparquet is unavailable.

    overwrite
        If True, remove an existing output directory before saving.

    Returns
    -------
    dict
        Mapping from result type to saved file path.
    """
    output_directory = Path(output_directory).expanduser().resolve()

    if output_directory.exists():
        has_contents = any(output_directory.iterdir())

        if has_contents and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_directory}. "
                "Use overwrite=True or choose a unique run directory."
            )

        if has_contents and overwrite:
            shutil.rmtree(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_paths: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    parameter_data = (
        asdict(P)
        if is_dataclass(P)
        else dict(vars(P))
    )

    parameters_path = output_directory / "parameters.json"

    _atomic_json_dump(
        parameter_data,
        parameters_path,
    )

    saved_paths["parameters"] = parameters_path

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    history_path = _save_dataframe(
        history,
        output_directory,
        stem="history",
        table_format=table_format,
    )

    saved_paths["history"] = history_path

    # ------------------------------------------------------------------
    # Rebinding event table
    # ------------------------------------------------------------------

    if rebinding_events is not None:
        events_path = _save_dataframe(
            rebinding_events,
            output_directory,
            stem="rebinding_events",
            table_format=table_format,
        )

        saved_paths["rebinding_events"] = events_path

    # ------------------------------------------------------------------
    # Exact microscopic state checkpoint
    # ------------------------------------------------------------------

    if state is not None:
        if state_to_checkpoint is not None:
            checkpoint = state_to_checkpoint(state)
            checkpoint_type = "state_checkpoint"
        else:
            # Fallback for older biosensor_mc versions.
            checkpoint = state
            checkpoint_type = "pickled_state"

        checkpoint_path = (
            output_directory
            / "state_checkpoint.pkl.gz"
        )

        _atomic_gzip_pickle_dump(
            checkpoint,
            checkpoint_path,
        )

        saved_paths["state_checkpoint"] = checkpoint_path
    else:
        checkpoint_type = None

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    if G is not None:
        geometry = G.geometry

        geometry_path = (
            output_directory
            / "geometry.pkl.gz"
        )

        _atomic_gzip_pickle_dump(
            geometry,
            geometry_path,
        )

        saved_paths["geometry"] = geometry_path

        geometry_summary = {
            "name": geometry.name,
            "shape": geometry.shape,
            "n_surface_faces": geometry.n_surface_faces,
            "n_reactive_faces": geometry.n_reactive_faces,
            "reactive_area_m2": float(
                np.sum(
                    geometry.surface_area_m2[
                        geometry.reactive_face_mask
                    ]
                )
            ),
        }

        # Include dynamically attached nanopore metadata when present.
        optional_geometry_fields = (
            "pore_diameter_m",
            "pore_depth_m",
            "pore_pitch_m",
            "rim_z_m",
            "layout",
            "pore_centers_xy_m",
        )

        for field in optional_geometry_fields:
            if hasattr(geometry, field):
                geometry_summary[field] = getattr(
                    geometry,
                    field,
                )

        geometry_summary_path = (
            output_directory
            / "geometry_summary.json"
        )

        _atomic_json_dump(
            geometry_summary,
            geometry_summary_path,
        )

        saved_paths["geometry_summary"] = (
            geometry_summary_path
        )

    # ------------------------------------------------------------------
    # Animation frames
    # ------------------------------------------------------------------

    if state_frames is not None:
        frames_path = (
            output_directory
            / "state_frames.pkl.gz"
        )

        _atomic_gzip_pickle_dump(
            state_frames,
            frames_path,
        )

        saved_paths["state_frames"] = frames_path

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest = {
        "saved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_version": MODEL_VERSION,
        "checkpoint_type": checkpoint_type,
        "n_history_rows": int(len(history)),
        "n_rebinding_event_rows": (
            int(len(rebinding_events))
            if rebinding_events is not None
            else 0
        ),
        "n_state_frames": (
            int(len(state_frames))
            if state_frames is not None
            else 0
        ),
        "files": {
            name: path.name
            for name, path in saved_paths.items()
        },
        "run_metadata": run_metadata or {},
    }

    manifest_path = output_directory / "manifest.json"

    _atomic_json_dump(
        manifest,
        manifest_path,
    )

    saved_paths["manifest"] = manifest_path

    return saved_paths


def load_simulation_results(
    output_directory: PathLike,
    *,
    restore_state: bool = True,
) -> Dict[str, Any]:
    """
    Load a simulation directory created by save_simulation_results.

    Parameters
    ----------
    output_directory
        Saved simulation directory.

    restore_state
        If True and state_from_checkpoint is available, reconstruct the State
        object. Otherwise return the raw checkpoint dictionary.

    Returns
    -------
    dict
        Loaded parameters, history, state/checkpoint, geometry, state frames,
        event table, and manifest.
    """
    output_directory = Path(
        output_directory
    ).expanduser().resolve()

    manifest_path = output_directory / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found in {output_directory}."
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    result: Dict[str, Any] = {
        "manifest": manifest,
    }

    parameters_path = output_directory / "parameters.json"

    with parameters_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        result["parameters"] = json.load(file)

    file_names = manifest.get("files", {})

    if "history" in file_names:
        result["history"] = _load_dataframe(
            output_directory / file_names["history"]
        )

    if "rebinding_events" in file_names:
        result["rebinding_events"] = _load_dataframe(
            output_directory
            / file_names["rebinding_events"]
        )

    if "geometry" in file_names:
        result["geometry"] = _gzip_pickle_load(
            output_directory / file_names["geometry"]
        )

    if "state_frames" in file_names:
        result["state_frames"] = _gzip_pickle_load(
            output_directory
            / file_names["state_frames"]
        )

    if "state_checkpoint" in file_names:
        checkpoint = _gzip_pickle_load(
            output_directory
            / file_names["state_checkpoint"]
        )

        result["checkpoint"] = checkpoint

        if (
            restore_state
            and state_from_checkpoint is not None
            and manifest.get("checkpoint_type")
            == "state_checkpoint"
        ):
            result["state"] = state_from_checkpoint(
                checkpoint
            )
        else:
            result["state"] = checkpoint

    return result