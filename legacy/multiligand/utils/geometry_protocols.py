"""
biosensor_mc_protocols.py

Protocol utilities for systematic geometry sweeps with the generalized
biosensor lattice Monte Carlo model.

UPDATED PROTOCOL VERSION: 2026-07-29-nanopore-diameter-sweep-v2

The main entry point is ``run_nanopore_diameter_sweep``. It creates a
nanopore-array geometry for every requested pore diameter, runs paired-seed
replicates, and returns:

- one summary row per simulation,
- replicate-aggregated diameter summaries,
- optional concatenated time histories,
- optional concatenated event-level rebinding records, and
- the generated geometry objects.

This module supports either of these project layouts:

1. Package layout::

       utils/
           __init__.py
           biosensor_mc.py
           generate_geometries.py
           biosensor_mc_protocols.py

2. Same-directory scripts::

       biosensor_mc.py
       generate_geometries.py
       biosensor_mc_protocols.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Support package, utils-package, and same-directory imports.
try:  # package-relative imports
    from .biosensor_mc import Params, SensorGeometry, derive, run_simulation
    from .generate_geometries import (
        make_flat_geometry,
        make_nanopore_array_geometry,
    )
except ImportError:
    try:  # user's current utils package layout
        from utils.biosensor_mc import Params, SensorGeometry, derive, run_simulation
        from utils.generate_geometries import (
            make_flat_geometry,
            make_nanopore_array_geometry,
        )
    except ImportError:  # same-directory scripts
        from biosensor_mc import Params, SensorGeometry, derive, run_simulation
        from generate_geometries import (
            make_flat_geometry,
            make_nanopore_array_geometry,
        )


PROTOCOL_VERSION = "2026-07-29-nanopore-diameter-sweep-v2"

SweepValue = Union[
    float,
    Sequence[float],
    np.ndarray,
    Callable[[float], float],
]


# -----------------------------------------------------------------------------
# Validation and small utilities
# -----------------------------------------------------------------------------


def _as_positive_1d_array(values: Sequence[float], name: str) -> np.ndarray:
    """Return a validated, finite, positive one-dimensional float array."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")

    if np.any(array <= 0.0):
        raise ValueError(f"All values in {name} must be positive.")

    return array


def _resolve_sweep_value(
    specification: SweepValue,
    diameter_m: float,
    diameter_index: int,
    n_diameters: int,
    name: str,
    *,
    allow_zero: bool = False,
) -> float:
    """
    Resolve a sweep parameter supplied as a scalar, per-diameter sequence,
    or callable of ``diameter_m``.
    """
    if callable(specification):
        value = specification(float(diameter_m))
    elif np.isscalar(specification):
        value = specification
    else:
        values = np.asarray(specification, dtype=float)

        if values.ndim != 1 or values.size != n_diameters:
            raise ValueError(
                f"{name} must be a scalar, a callable, or a sequence with "
                f"exactly {n_diameters} values."
            )

        value = values[diameter_index]

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"Resolved {name} is non-finite for diameter "
            f"{diameter_m:.6e} m."
        )

    lower_bound_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_bound_ok:
        comparison = "nonnegative" if allow_zero else "positive"
        raise ValueError(
            f"Resolved {name} must be {comparison} for diameter "
            f"{diameter_m:.6e} m; received {value:.6e}."
        )

    return value


def _resolve_replicate_seeds(
    base_seed: int,
    n_replicates: int,
    seeds: Optional[Sequence[int]],
) -> np.ndarray:
    """Create or validate the paired replicate seeds used for every diameter."""
    if int(n_replicates) != n_replicates or n_replicates < 1:
        raise ValueError("n_replicates must be a positive integer.")

    n_replicates = int(n_replicates)

    if seeds is None:
        return np.arange(
            int(base_seed),
            int(base_seed) + n_replicates,
            dtype=np.int64,
        )

    result = np.asarray(seeds, dtype=np.int64)

    if result.ndim != 1 or result.size != n_replicates:
        raise ValueError(
            "seeds must contain exactly one integer for each replicate."
        )

    if np.unique(result).size != result.size:
        raise ValueError("Replicate seeds must be unique.")

    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when denominator is zero."""
    if denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _time_weighted_mean(time_s: np.ndarray, values: np.ndarray) -> float:
    """Compute a trapezoidal time-weighted mean for a sampled trajectory."""
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)

    finite = np.isfinite(time_s) & np.isfinite(values)
    time_s = time_s[finite]
    values = values[finite]

    if time_s.size == 0:
        return np.nan
    if time_s.size == 1 or time_s[-1] <= time_s[0]:
        return float(values[-1])

    return float(np.trapz(values, time_s) / (time_s[-1] - time_s[0]))


def _attach_metadata(table: pd.DataFrame, metadata: Mapping[str, object]) -> pd.DataFrame:
    """Return a copy of a table with protocol metadata columns appended."""
    result = table.copy()
    for key, value in metadata.items():
        result[key] = value
    return result


# -----------------------------------------------------------------------------
# Event and run summaries
# -----------------------------------------------------------------------------


def summarize_rebinding_events(events: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Summarize event-level dissociation excursions.

    ``open`` events are unresolved when the simulation ends. They are reported
    but excluded from completed-excursion fractions.
    """
    empty_summary = {
        "n_dissociation_excursions_recorded": 0,
        "n_completed_excursions": 0,
        "n_open_excursions": 0,
        "n_self_rebindings_events": 0,
        "n_cross_rebindings_events": 0,
        "n_all_rebindings_events": 0,
        "n_local_escapes_events": 0,
        "n_bulk_losses_events": 0,
        "rebinding_fraction_completed": np.nan,
        "escape_fraction_completed": np.nan,
        "self_fraction_of_rebindings": np.nan,
        "cross_fraction_of_rebindings": np.nan,
        "mean_free_excursion_time_s": np.nan,
        "median_free_excursion_time_s": np.nan,
        "mean_rebinding_excursion_time_s": np.nan,
        "median_rebinding_excursion_time_s": np.nan,
        "mean_receptor_distance_m": np.nan,
        "median_receptor_distance_m": np.nan,
        "mean_max_surface_distance_m": np.nan,
    }

    if events is None or events.empty:
        return empty_summary

    required_columns = {
        "outcome",
        "is_rebinding",
        "is_self_rebinding",
        "is_cross_rebinding",
        "free_excursion_time_s",
        "receptor_distance_m",
        "max_surface_distance_m",
    }
    missing = required_columns.difference(events.columns)

    if missing:
        raise KeyError(
            "The event table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    completed = events.loc[events["outcome"] != "open"].copy()
    rebindings = completed.loc[completed["is_rebinding"].astype(bool)].copy()

    n_completed = int(completed.shape[0])
    n_open = int((events["outcome"] == "open").sum())
    n_self = int(completed["is_self_rebinding"].astype(bool).sum())
    n_cross = int(completed["is_cross_rebinding"].astype(bool).sum())
    n_rebindings = int(rebindings.shape[0])
    n_local_escape = int((completed["outcome"] == "local_escape").sum())
    n_bulk_loss = int((completed["outcome"] == "bulk_loss").sum())
    n_escape = n_local_escape + n_bulk_loss

    summary = {
        "n_dissociation_excursions_recorded": int(events.shape[0]),
        "n_completed_excursions": n_completed,
        "n_open_excursions": n_open,
        "n_self_rebindings_events": n_self,
        "n_cross_rebindings_events": n_cross,
        "n_all_rebindings_events": n_rebindings,
        "n_local_escapes_events": n_local_escape,
        "n_bulk_losses_events": n_bulk_loss,
        "rebinding_fraction_completed": _safe_ratio(n_rebindings, n_completed),
        "escape_fraction_completed": _safe_ratio(n_escape, n_completed),
        "self_fraction_of_rebindings": _safe_ratio(n_self, n_rebindings),
        "cross_fraction_of_rebindings": _safe_ratio(n_cross, n_rebindings),
        "mean_free_excursion_time_s": float(
            completed["free_excursion_time_s"].mean()
        ),
        "median_free_excursion_time_s": float(
            completed["free_excursion_time_s"].median()
        ),
        "mean_rebinding_excursion_time_s": float(
            rebindings["free_excursion_time_s"].mean()
        ),
        "median_rebinding_excursion_time_s": float(
            rebindings["free_excursion_time_s"].median()
        ),
        "mean_receptor_distance_m": float(
            rebindings["receptor_distance_m"].mean()
        ),
        "median_receptor_distance_m": float(
            rebindings["receptor_distance_m"].median()
        ),
        "mean_max_surface_distance_m": float(
            completed["max_surface_distance_m"].mean()
        ),
    }

    return summary


def summarize_simulation_run(
    history: pd.DataFrame,
    events: Optional[pd.DataFrame],
    geometry: SensorGeometry,
    P: Params,
    simulation_time_s: float,
) -> Dict[str, float]:
    """Create one scalar summary row for a completed simulation run."""
    if history is None or history.empty:
        raise ValueError("history must contain at least one row.")

    required_history_columns = {"t_s", "theta"}
    missing = required_history_columns.difference(history.columns)
    if missing:
        raise KeyError(
            "history is missing required columns: "
            + ", ".join(sorted(missing))
        )

    final = history.iloc[-1]
    G = derive(P, geometry=geometry)

    n_bindings = int(final.get("binding_events_total", 0))
    n_unbindings = int(final.get("unbinding_events_total", 0))
    n_rebindings = int(final.get("rebinding_events_total", 0))
    n_self = int(final.get("self_rebindings_total", 0))
    n_cross = int(final.get("cross_rebindings_total", 0))

    duration = float(final.get("t_s", simulation_time_s))
    if duration <= 0:
        duration = float(simulation_time_s)

    time_s = history["t_s"].to_numpy(dtype=float)
    occupancy = history["theta"].to_numpy(dtype=float)

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "geometry_name": geometry.name,
        "seed": int(P.seed),
        "simulation_time_s": duration,
        "dt_s": float(G.dt_s),
        "n_steps": int(final.get("step", 0)),
        "n_receptors": int(G.NR),
        "n_reactive_faces": int(G.reactive_face_ids.size),
        "reactive_area_m2": float(G.sensing_area_m2),
        "projected_area_m2": float(P.Lx_m * P.Ly_m),
        "surface_area_ratio": _safe_ratio(
            float(G.sensing_area_m2),
            float(P.Lx_m * P.Ly_m),
        ),
        "accessible_volume_m3": float(G.volume_m3),
        "final_bound_receptors": int(final.get("B", 0)),
        "final_occupancy": float(final.get("theta", np.nan)),
        "mean_occupancy_sampled": float(np.nanmean(occupancy)),
        "mean_occupancy_time_weighted": _time_weighted_mean(time_s, occupancy),
        "max_occupancy": float(np.nanmax(occupancy)),
        "binding_events_total": n_bindings,
        "unbinding_events_total": n_unbindings,
        "rebinding_events_total": n_rebindings,
        "self_rebindings_total": n_self,
        "cross_rebindings_total": n_cross,
        "local_rebinding_escapes_total": int(
            final.get("local_rebinding_escapes_total", 0)
        ),
        "rebind_watch_bulk_losses_total": int(
            final.get("rebind_watch_bulk_losses_total", 0)
        ),
        "open_rebinding_watches_final": int(
            final.get("N_open_rebinding_watches", 0)
        ),
        "binding_rate_s_inv": _safe_ratio(n_bindings, duration),
        "unbinding_rate_s_inv": _safe_ratio(n_unbindings, duration),
        "rebinding_rate_s_inv": _safe_ratio(n_rebindings, duration),
        "rebinding_fraction_per_unbinding": _safe_ratio(
            n_rebindings,
            n_unbindings,
        ),
        "self_fraction_per_unbinding": _safe_ratio(n_self, n_unbindings),
        "cross_fraction_per_unbinding": _safe_ratio(n_cross, n_unbindings),
        "self_fraction_of_history_rebindings": _safe_ratio(
            n_self,
            n_rebindings,
        ),
        "cross_fraction_of_history_rebindings": _safe_ratio(
            n_cross,
            n_rebindings,
        ),
    }
    summary.update(summarize_rebinding_events(events))
    return summary


# -----------------------------------------------------------------------------
# Nanopore-diameter protocol
# -----------------------------------------------------------------------------


def run_nanopore_diameter_sweep(
    P: Params,
    nanopore_diameters_m: Sequence[float],
    simulation_time_s: float,
    pore_depth_m: SweepValue,
    pitch_m: SweepValue,
    rim_z_m: SweepValue,
    *,
    n_replicates: int = 3,
    seeds: Optional[Sequence[int]] = None,
    layout: str = "square",
    edge_margin_m: Optional[SweepValue] = None,
    pore_centers_xy_m: Optional[np.ndarray] = None,
    include_flat_control: bool = False,
    flat_control_surface_z_m: Optional[float] = None,
    record_every_s: Optional[float] = None,
    show_simulation_progress: bool = True,
    verbose: bool = False,
    keep_histories: bool = True,
    keep_event_tables: bool = True,
    keep_geometries: bool = True,
    simulation_kwargs: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """
    Run replicate simulations across an array of nanopore diameters.

    Parameters
    ----------
    P
        Base simulation parameters. The object is not modified.
    nanopore_diameters_m
        Positive pore diameters in meters.
    simulation_time_s
        Duration of every simulation.
    pore_depth_m, pitch_m, rim_z_m
        Scalar, one value per diameter, or callable ``f(diameter_m)``.
    n_replicates
        Number of independent replicate seeds per geometry.
    seeds
        Optional explicit replicate seeds. The same seed set is reused for
        every diameter, creating paired-seed comparisons.
    layout
        ``"square"`` or ``"hexagonal"`` as accepted by the geometry builder.
    edge_margin_m
        Optional scalar, per-diameter sequence, or callable.
    pore_centers_xy_m
        Optional fixed array of shape ``(N_pores, 2)``. Supplying fixed centers
        is recommended when diameter should be the only changing feature.
    include_flat_control
        Run a flat control with the same seeds.
    flat_control_surface_z_m
        Flat-control surface height. If omitted, the first resolved rim height
        is used.
    record_every_s
        History sampling interval passed to ``run_simulation``.
    show_simulation_progress
        Show the internal progress bar for each simulation.
    verbose
        Print initialization details for each simulation.
    keep_histories
        Concatenate and return all time-course histories.
    keep_event_tables
        Concatenate and return all event-level rebinding tables.
    keep_geometries
        Return generated geometry objects keyed by a stable string label.
    simulation_kwargs
        Additional ``run_simulation`` keyword arguments, such as compaction
        settings. Return-mode and geometry arguments are reserved.

    Returns
    -------
    dict
        Keys are ``run_summary``, ``diameter_summary``, ``histories``,
        ``rebinding_events``, ``geometries``, and ``protocol_config``.
    """
    diameters = _as_positive_1d_array(
        nanopore_diameters_m,
        "nanopore_diameters_m",
    )

    if not np.isfinite(simulation_time_s) or simulation_time_s <= 0:
        raise ValueError("simulation_time_s must be finite and positive.")

    replicate_seeds = _resolve_replicate_seeds(
        base_seed=int(P.seed),
        n_replicates=n_replicates,
        seeds=seeds,
    )

    if pore_centers_xy_m is not None:
        pore_centers_xy_m = np.asarray(pore_centers_xy_m, dtype=float)
        if pore_centers_xy_m.ndim != 2 or pore_centers_xy_m.shape[1] != 2:
            raise ValueError("pore_centers_xy_m must have shape (N_pores, 2).")

    extra_run_kwargs = dict(simulation_kwargs or {})
    reserved_run_arguments = {
        "P",
        "seconds",
        "record_every_s",
        "return_state",
        "show_progress",
        "verbose",
        "save_state_frames",
        "geometry",
        "return_rebinding_events",
    }
    conflicts = reserved_run_arguments.intersection(extra_run_kwargs)
    if conflicts:
        raise ValueError(
            "simulation_kwargs may not contain reserved arguments: "
            + ", ".join(sorted(conflicts))
        )

    run_rows: List[Dict[str, object]] = []
    history_tables: List[pd.DataFrame] = []
    event_tables: List[pd.DataFrame] = []
    geometries: Dict[str, SensorGeometry] = {}
    geometry_parameter_rows: List[Dict[str, object]] = []

    n_diameters = int(diameters.size)

    for diameter_index, diameter_m in enumerate(diameters):
        depth_value = _resolve_sweep_value(
            pore_depth_m,
            float(diameter_m),
            diameter_index,
            n_diameters,
            "pore_depth_m",
        )
        pitch_value = _resolve_sweep_value(
            pitch_m,
            float(diameter_m),
            diameter_index,
            n_diameters,
            "pitch_m",
        )
        rim_value = _resolve_sweep_value(
            rim_z_m,
            float(diameter_m),
            diameter_index,
            n_diameters,
            "rim_z_m",
            allow_zero=True,
        )

        if edge_margin_m is None:
            edge_value = None
        else:
            edge_value = _resolve_sweep_value(
                edge_margin_m,
                float(diameter_m),
                diameter_index,
                n_diameters,
                "edge_margin_m",
                allow_zero=True,
            )

        geometry_label = f"nanopore_{diameter_m * 1e9:.6g}_nm"
        geometry = make_nanopore_array_geometry(
            P,
            pore_diameter_m=float(diameter_m),
            pore_depth_m=depth_value,
            pitch_m=pitch_value,
            rim_z_m=rim_value,
            layout=layout,
            edge_margin_m=edge_value,
            pore_centers_xy_m=pore_centers_xy_m,
            name=geometry_label,
        )

        if keep_geometries:
            geometries[geometry_label] = geometry

        centers = np.asarray(
            getattr(geometry, "pore_centers_xy_m", np.empty((0, 2))),
            dtype=float,
        )
        n_pores = int(centers.shape[0])
        projected_porosity = float(
            n_pores * np.pi * (0.5 * diameter_m) ** 2 / (P.Lx_m * P.Ly_m)
        )

        geometry_metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "geometry_type": "nanopore_array",
            "geometry_label": geometry_label,
            "diameter_index": int(diameter_index),
            "nanopore_diameter_m": float(diameter_m),
            "nanopore_diameter_nm": float(diameter_m * 1e9),
            "pore_depth_m": float(depth_value),
            "pore_depth_nm": float(depth_value * 1e9),
            "pitch_m": float(pitch_value),
            "pitch_nm": float(pitch_value * 1e9),
            "rim_z_m": float(rim_value),
            "rim_z_nm": float(rim_value * 1e9),
            "edge_margin_m": np.nan if edge_value is None else float(edge_value),
            "edge_margin_nm": (
                np.nan if edge_value is None else float(edge_value * 1e9)
            ),
            "layout": str(layout),
            "n_pores": n_pores,
            "projected_porosity": projected_porosity,
        }
        geometry_parameter_rows.append(geometry_metadata.copy())

        for replicate_index, seed in enumerate(replicate_seeds, start=1):
            P_run = replace(P, seed=int(seed))

            history, events = run_simulation(
                P_run,
                seconds=float(simulation_time_s),
                record_every_s=record_every_s,
                return_state=False,
                show_progress=show_simulation_progress,
                verbose=verbose,
                save_state_frames=False,
                geometry=geometry,
                return_rebinding_events=True,
                **extra_run_kwargs,
            )

            run_metadata = {
                **geometry_metadata,
                "replicate": int(replicate_index),
                "seed": int(seed),
            }

            run_summary = summarize_simulation_run(
                history=history,
                events=events,
                geometry=geometry,
                P=P_run,
                simulation_time_s=float(simulation_time_s),
            )
            run_summary.update(run_metadata)
            run_rows.append(run_summary)

            if keep_histories:
                history_tables.append(_attach_metadata(history, run_metadata))

            if keep_event_tables:
                event_tables.append(_attach_metadata(events, run_metadata))

    if include_flat_control:
        if flat_control_surface_z_m is None:
            flat_surface_z_m = _resolve_sweep_value(
                rim_z_m,
                float(diameters[0]),
                0,
                n_diameters,
                "rim_z_m",
                allow_zero=True,
            )
        else:
            flat_surface_z_m = float(flat_control_surface_z_m)
            if not np.isfinite(flat_surface_z_m) or flat_surface_z_m < 0:
                raise ValueError(
                    "flat_control_surface_z_m must be finite and nonnegative."
                )

        geometry_label = "flat_control"
        flat_geometry = make_flat_geometry(
            P,
            surface_z_m=flat_surface_z_m,
            name=geometry_label,
        )

        if keep_geometries:
            geometries[geometry_label] = flat_geometry

        flat_metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "geometry_type": "flat_control",
            "geometry_label": geometry_label,
            "diameter_index": -1,
            "nanopore_diameter_m": np.nan,
            "nanopore_diameter_nm": np.nan,
            "pore_depth_m": np.nan,
            "pore_depth_nm": np.nan,
            "pitch_m": np.nan,
            "pitch_nm": np.nan,
            "rim_z_m": float(flat_surface_z_m),
            "rim_z_nm": float(flat_surface_z_m * 1e9),
            "edge_margin_m": np.nan,
            "edge_margin_nm": np.nan,
            "layout": "flat",
            "n_pores": 0,
            "projected_porosity": 0.0,
        }
        geometry_parameter_rows.append(flat_metadata.copy())

        for replicate_index, seed in enumerate(replicate_seeds, start=1):
            P_run = replace(P, seed=int(seed))

            history, events = run_simulation(
                P_run,
                seconds=float(simulation_time_s),
                record_every_s=record_every_s,
                return_state=False,
                show_progress=show_simulation_progress,
                verbose=verbose,
                save_state_frames=False,
                geometry=flat_geometry,
                return_rebinding_events=True,
                **extra_run_kwargs,
            )

            run_metadata = {
                **flat_metadata,
                "replicate": int(replicate_index),
                "seed": int(seed),
            }

            run_summary = summarize_simulation_run(
                history=history,
                events=events,
                geometry=flat_geometry,
                P=P_run,
                simulation_time_s=float(simulation_time_s),
            )
            run_summary.update(run_metadata)
            run_rows.append(run_summary)

            if keep_histories:
                history_tables.append(_attach_metadata(history, run_metadata))

            if keep_event_tables:
                event_tables.append(_attach_metadata(events, run_metadata))

    run_summary = pd.DataFrame(run_rows)
    diameter_summary = aggregate_nanopore_diameter_sweep(run_summary)

    histories = (
        pd.concat(history_tables, ignore_index=True)
        if keep_histories and history_tables
        else None
    )
    rebinding_events = (
        pd.concat(event_tables, ignore_index=True)
        if keep_event_tables and event_tables
        else None
    )
    geometry_parameters = pd.DataFrame(geometry_parameter_rows).drop_duplicates(
        subset=["geometry_label"]
    )

    return {
        "run_summary": run_summary,
        "diameter_summary": diameter_summary,
        "histories": histories,
        "rebinding_events": rebinding_events,
        "geometry_parameters": geometry_parameters,
        "geometries": geometries if keep_geometries else None,
        "protocol_config": {
            "protocol_version": PROTOCOL_VERSION,
            "simulation_time_s": float(simulation_time_s),
            "n_replicates": int(len(replicate_seeds)),
            "seeds": replicate_seeds.copy(),
            "record_every_s": record_every_s,
            "include_flat_control": bool(include_flat_control),
        },
    }


# -----------------------------------------------------------------------------
# Replicate aggregation
# -----------------------------------------------------------------------------


def aggregate_nanopore_diameter_sweep(
    run_summary: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate replicate results by geometry and nanopore diameter.

    Output columns are flattened as ``<metric>_mean``, ``<metric>_std``,
    ``<metric>_sem``, and ``<metric>_n``.
    """
    required = {
        "geometry_type",
        "geometry_label",
        "nanopore_diameter_m",
        "nanopore_diameter_nm",
    }
    missing = required.difference(run_summary.columns)
    if missing:
        raise KeyError(
            "run_summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if metrics is None:
        preferred_metrics = [
            "final_occupancy",
            "mean_occupancy_time_weighted",
            "max_occupancy",
            "binding_events_total",
            "unbinding_events_total",
            "rebinding_events_total",
            "self_rebindings_total",
            "cross_rebindings_total",
            "binding_rate_s_inv",
            "unbinding_rate_s_inv",
            "rebinding_rate_s_inv",
            "rebinding_fraction_per_unbinding",
            "rebinding_fraction_completed",
            "escape_fraction_completed",
            "self_fraction_of_rebindings",
            "cross_fraction_of_rebindings",
            "mean_rebinding_excursion_time_s",
            "median_rebinding_excursion_time_s",
            "mean_receptor_distance_m",
            "mean_max_surface_distance_m",
            "n_receptors",
            "n_reactive_faces",
            "reactive_area_m2",
            "surface_area_ratio",
            "accessible_volume_m3",
            "n_pores",
            "projected_porosity",
        ]
        metrics = [
            metric for metric in preferred_metrics if metric in run_summary.columns
        ]
    else:
        metrics = list(metrics)
        unknown = set(metrics).difference(run_summary.columns)
        if unknown:
            raise KeyError("Unknown metric columns: " + ", ".join(sorted(unknown)))

    group_columns = [
        "protocol_version",
        "geometry_type",
        "geometry_label",
        "nanopore_diameter_m",
        "nanopore_diameter_nm",
        "pore_depth_m",
        "pore_depth_nm",
        "pitch_m",
        "pitch_nm",
        "rim_z_m",
        "rim_z_nm",
        "edge_margin_m",
        "edge_margin_nm",
        "layout",
    ]
    group_columns = [column for column in group_columns if column in run_summary.columns]

    grouped = run_summary.groupby(group_columns, dropna=False, sort=True)
    aggregated = grouped[list(metrics)].agg(["mean", "std", "sem", "count"])
    aggregated.columns = [
        f"{metric}_{'n' if statistic == 'count' else statistic}"
        for metric, statistic in aggregated.columns
    ]

    return aggregated.reset_index()


# -----------------------------------------------------------------------------
# Convenience plotting and saving
# -----------------------------------------------------------------------------


def plot_nanopore_diameter_metric(
    sweep_results: Union[Mapping[str, object], pd.DataFrame],
    metric: str = "rebinding_fraction_completed",
    *,
    error: Optional[str] = "sem",
    ax=None,
    include_flat_control: bool = True,
    xlabel: str = "Nanopore diameter (nm)",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    marker: str = "o",
    capsize: float = 4.0,
):
    """
    Plot an aggregated simulation metric against nanopore diameter.

    ``sweep_results`` may be the dictionary returned by the sweep or its
    ``diameter_summary`` DataFrame. Legends are placed outside when a flat
    control is shown.
    """
    import matplotlib.pyplot as plt

    if isinstance(sweep_results, pd.DataFrame):
        summary = sweep_results
    else:
        summary = sweep_results.get("diameter_summary")

    if not isinstance(summary, pd.DataFrame):
        raise TypeError(
            "sweep_results must be a diameter-summary DataFrame or the "
            "dictionary returned by run_nanopore_diameter_sweep."
        )

    mean_column = f"{metric}_mean"
    if mean_column not in summary.columns:
        raise KeyError(f"The summary does not contain {mean_column!r}.")

    if error is None:
        error_column = None
    else:
        if error not in {"std", "sem"}:
            raise ValueError("error must be None, 'std', or 'sem'.")
        error_column = f"{metric}_{error}"
        if error_column not in summary.columns:
            raise KeyError(f"The summary does not contain {error_column!r}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    nanopore_rows = summary.loc[summary["geometry_type"] == "nanopore_array"].copy()
    nanopore_rows = nanopore_rows.sort_values("nanopore_diameter_nm")

    yerr = None if error_column is None else nanopore_rows[error_column].to_numpy()
    ax.errorbar(
        nanopore_rows["nanopore_diameter_nm"],
        nanopore_rows[mean_column],
        yerr=yerr,
        marker=marker,
        capsize=capsize,
        label="Nanopore array",
    )

    if include_flat_control:
        flat_rows = summary.loc[summary["geometry_type"] == "flat_control"]
        if not flat_rows.empty:
            flat_mean = float(flat_rows[mean_column].iloc[0])
            ax.axhline(flat_mean, linestyle="--", label="Flat control")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric.replace("_", " ").capitalize())
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.25)

    if include_flat_control and "flat_control" in set(summary["geometry_type"]):
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        fig.subplots_adjust(right=0.76)
    else:
        fig.tight_layout()

    return fig, ax


def save_nanopore_diameter_sweep(
    results: Mapping[str, object],
    output_directory: Union[str, Path],
    prefix: str = "nanopore_diameter_sweep",
) -> Dict[str, Path]:
    """Save all returned DataFrames as CSV files and return their paths."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    table_keys = (
        "run_summary",
        "diameter_summary",
        "histories",
        "rebinding_events",
        "geometry_parameters",
    )

    for key in table_keys:
        table = results.get(key)
        if isinstance(table, pd.DataFrame):
            path = output_directory / f"{prefix}_{key}.csv"
            table.to_csv(path, index=False)
            paths[key] = path

    return paths


__all__ = [
    "PROTOCOL_VERSION",
    "summarize_rebinding_events",
    "summarize_simulation_run",
    "run_nanopore_diameter_sweep",
    "aggregate_nanopore_diameter_sweep",
    "plot_nanopore_diameter_metric",
    "save_nanopore_diameter_sweep",
]
