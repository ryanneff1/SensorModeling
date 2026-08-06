"""
Analysis utilities for saved biosensor Monte Carlo simulations.

The module is designed for run directories produced by save_simulation_results,
but every reader also accepts a direct file path.

Primary features
----------------
1. Read saved history tables and fit every assay phase to a single-exponential
   relaxation model.
2. Read saved state frames and calculate receptor/ligand height distributions.
3. Plot time-height heatmaps for bound-receptor occupancy or free-ligand
   distributions.
4. Calculate scalar penetration metrics such as mean bound height, mean pore
   depth, and the fraction of bound receptors below a selected depth.

A fitted phase uses

    y(t) = y_inf + (y0 - y_inf) * exp(-k_eff * (t - t_start))

where k_eff is an empirical relaxation rate. For reversible binding this is an
observed/effective rate, not automatically the microscopic k_on or k_off.
"""

from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
except ImportError as exc:  # pragma: no cover
    curve_fit = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


PathLike = Union[str, Path]


@dataclass
class HeightProfileResult:
    """Container returned by calculate_height_profile_matrix."""

    matrix: np.ndarray
    times_s: np.ndarray
    bin_edges_m: np.ndarray
    bin_centers_m: np.ndarray
    phase_labels: np.ndarray
    metric: str
    coordinate: str
    reference_z_m: Optional[float]


# -----------------------------------------------------------------------------
# File readers
# -----------------------------------------------------------------------------


def _resolve_saved_file(
    source: PathLike,
    candidates: Sequence[str],
) -> Path:
    """Resolve either a direct file path or a file inside a saved run directory."""
    source_path = Path(source).expanduser().resolve()

    if source_path.is_file():
        return source_path

    if not source_path.exists():
        raise FileNotFoundError(f"Saved-data path does not exist: {source_path}")

    if not source_path.is_dir():
        raise ValueError(f"Expected a file or directory: {source_path}")

    manifest_path = source_path / "manifest.json"

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        files = manifest.get("files", {})

        for candidate in candidates:
            manifest_name = files.get(candidate)

            if manifest_name:
                path = source_path / manifest_name

                if path.exists():
                    return path

    for candidate in candidates:
        direct = source_path / candidate

        if direct.exists():
            return direct

    candidate_text = ", ".join(candidates)
    raise FileNotFoundError(
        f"Could not locate any of [{candidate_text}] in {source_path}."
    )


def _read_dataframe(path: PathLike) -> pd.DataFrame:
    """Read Parquet or compressed/uncompressed CSV without Arrow extensions."""
    path = Path(path).expanduser().resolve()

    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Reading Parquet requires pyarrow. Install it with "
                "`python -m pip install pyarrow`."
            ) from exc

        try:
            table = pq.read_table(
                path,
                arrow_extensions_enabled=False,
            )
        except TypeError:
            # Compatibility with older PyArrow versions that do not expose
            # arrow_extensions_enabled on read_table.
            table = pq.read_table(path)

        return table.to_pandas(ignore_metadata=True)

    if path.name.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table format: {path}")


def load_history(source: PathLike) -> pd.DataFrame:
    """
    Load a saved history table.

    Parameters
    ----------
    source
        Direct history file path or a run directory containing manifest.json.
    """
    path = _resolve_saved_file(
        source,
        candidates=("history", "history.parquet", "history.csv.gz", "history.csv"),
    )
    history = _read_dataframe(path)

    if "t_s" not in history.columns:
        raise KeyError("History table must contain a 't_s' column.")

    return history


def load_rebinding_events(source: PathLike) -> pd.DataFrame:
    """Load a saved event-level rebinding table."""
    path = _resolve_saved_file(
        source,
        candidates=(
            "rebinding_events",
            "rebinding_events.parquet",
            "rebinding_events.csv.gz",
            "rebinding_events.csv",
        ),
    )
    return _read_dataframe(path)


def load_state_frames(source: PathLike) -> List[Dict[str, Any]]:
    """
    Load saved visualization state frames from a gzip-compressed pickle.

    Parameters
    ----------
    source
        Direct state-frame file path or a saved run directory.
    """
    path = _resolve_saved_file(
        source,
        candidates=("state_frames", "state_frames.pkl.gz", "state_frames.pkl"),
    )

    if path.name.endswith(".pkl.gz"):
        with gzip.open(path, "rb") as file:
            frames = pickle.load(file)
    elif path.suffix == ".pkl":
        with path.open("rb") as file:
            frames = pickle.load(file)
    else:
        raise ValueError(f"Unsupported state-frame format: {path}")

    if not isinstance(frames, list):
        raise TypeError("Saved state frames must be a list of frame dictionaries.")

    if frames and not isinstance(frames[0], Mapping):
        raise TypeError("Each state frame must be a dictionary-like object.")

    return frames


# -----------------------------------------------------------------------------
# Exponential phase fitting
# -----------------------------------------------------------------------------


def _single_exponential_fixed_initial(
    time_since_phase_start_s: np.ndarray,
    y_inf: float,
    k_eff_s_inv: float,
    y0: float,
) -> np.ndarray:
    return y_inf + (y0 - y_inf) * np.exp(
        -k_eff_s_inv * time_since_phase_start_s
    )


def _single_exponential_free_initial(
    time_since_phase_start_s: np.ndarray,
    y0: float,
    y_inf: float,
    k_eff_s_inv: float,
) -> np.ndarray:
    return y_inf + (y0 - y_inf) * np.exp(
        -k_eff_s_inv * time_since_phase_start_s
    )


def _estimate_initial_rate(
    tau_s: np.ndarray,
    values: np.ndarray,
    y0: float,
    y_inf_guess: float,
) -> float:
    span_s = float(np.max(tau_s) - np.min(tau_s))

    if span_s <= 0:
        return 1.0

    target = y_inf_guess + (y0 - y_inf_guess) / np.e
    nearest = int(np.argmin(np.abs(values - target)))
    tau_guess = float(tau_s[nearest])

    if tau_guess <= 0:
        tau_guess = span_s / 3.0

    return max(1.0 / tau_guess, np.finfo(float).eps)


def fit_history_phases(
    history_or_source: Union[pd.DataFrame, PathLike],
    value_column: str = "theta",
    phase_column: str = "phase_label",
    time_column: str = "t_s",
    phase_labels: Optional[Sequence[str]] = None,
    min_points: int = 5,
    fit_initial_value: bool = False,
    value_bounds: Optional[Tuple[float, float]] = None,
    drop_duplicate_times: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit every history phase to a single-exponential relaxation.

    The fitted model is

        y(t) = y_inf + (y0 - y_inf) exp[-k_eff (t - t_start)].

    Parameters
    ----------
    history_or_source
        History DataFrame, history file, or saved run directory.
    value_column
        Observable to fit, normally ``theta`` or ``B``.
    phase_column
        Column identifying assay phases. If absent, the full history is
        treated as one phase named ``all``.
    time_column
        Absolute simulation-time column.
    phase_labels
        Optional ordered subset of phase labels to fit.
    min_points
        Minimum number of finite data points required for a fit.
    fit_initial_value
        If False, y0 is fixed to the first value in the phase. If True, y0 is
        fitted as an additional parameter.
    value_bounds
        Optional lower and upper bounds for y0/y_inf. For occupancy use
        ``(0, 1)`` when strict physical bounds are desired.
    drop_duplicate_times
        Remove duplicate times within each phase, keeping the final row.

    Returns
    -------
    fits
        One row per phase with k_eff, time constant, half-time, asymptote,
        goodness-of-fit metrics, and uncertainty estimates.
    fitted_history
        Point-level table containing observed values, fitted values, and
        residuals for plotting.

    Notes
    -----
    ``k_eff_s_inv`` is an empirical relaxation rate. In a reversible binding
    phase it generally combines association, dissociation, transport, and
    geometry effects and should not automatically be interpreted as k_on or
    k_off.
    """
    if curve_fit is None:  # pragma: no cover
        raise ImportError(
            "fit_history_phases requires SciPy. Install it with "
            "`python -m pip install scipy`."
        ) from _SCIPY_IMPORT_ERROR

    if isinstance(history_or_source, pd.DataFrame):
        history = history_or_source.copy()
    else:
        history = load_history(history_or_source)

    required = {time_column, value_column}
    missing = required.difference(history.columns)

    if missing:
        raise KeyError(
            "History is missing required columns: " + ", ".join(sorted(missing))
        )

    if phase_column not in history.columns:
        history[phase_column] = "all"

    if phase_labels is None:
        phase_labels = list(pd.unique(history[phase_column].astype(str)))
    else:
        phase_labels = [str(label) for label in phase_labels]

    if min_points < 3:
        raise ValueError("min_points must be at least 3.")

    fit_rows: List[Dict[str, Any]] = []
    point_tables: List[pd.DataFrame] = []

    for phase_index, phase_label in enumerate(phase_labels):
        phase = history.loc[
            history[phase_column].astype(str) == phase_label,
            [time_column, value_column],
        ].copy()

        phase[time_column] = pd.to_numeric(phase[time_column], errors="coerce")
        phase[value_column] = pd.to_numeric(phase[value_column], errors="coerce")
        phase = phase.dropna().sort_values(time_column)

        if drop_duplicate_times:
            phase = phase.drop_duplicates(subset=time_column, keep="last")

        n_points = int(len(phase))
        base_row: Dict[str, Any] = {
            "phase_index": phase_index,
            "phase_label": phase_label,
            "value_column": value_column,
            "n_points": n_points,
            "fit_success": False,
            "fit_message": "",
        }

        if n_points < min_points:
            base_row["fit_message"] = (
                f"Only {n_points} finite points; at least {min_points} are required."
            )
            fit_rows.append(base_row)
            continue

        absolute_time_s = phase[time_column].to_numpy(dtype=float)
        observed = phase[value_column].to_numpy(dtype=float)
        phase_start_s = float(absolute_time_s[0])
        tau_s = absolute_time_s - phase_start_s

        if np.ptp(tau_s) <= 0:
            base_row["fit_message"] = "Phase contains no positive time span."
            fit_rows.append(base_row)
            continue

        y0_observed = float(observed[0])
        tail_count = max(3, int(np.ceil(0.2 * n_points)))
        y_inf_guess = float(np.nanmedian(observed[-tail_count:]))
        k_guess = _estimate_initial_rate(tau_s, observed, y0_observed, y_inf_guess)

        if value_bounds is None:
            finite_min = float(np.min(observed))
            finite_max = float(np.max(observed))
            data_span = max(finite_max - finite_min, abs(finite_max), 1.0) * 0.25
            lower_value = finite_min - data_span
            upper_value = finite_max + data_span
        else:
            lower_value, upper_value = map(float, value_bounds)

            if not lower_value < upper_value:
                raise ValueError("value_bounds must satisfy lower < upper.")

        try:
            if fit_initial_value:
                parameters, covariance = curve_fit(
                    _single_exponential_free_initial,
                    tau_s,
                    observed,
                    p0=(y0_observed, y_inf_guess, k_guess),
                    bounds=(
                        (lower_value, lower_value, 0.0),
                        (upper_value, upper_value, np.inf),
                    ),
                    maxfev=50_000,
                )
                fitted_y0, fitted_y_inf, fitted_k = map(float, parameters)
                fitted_values = _single_exponential_free_initial(
                    tau_s,
                    fitted_y0,
                    fitted_y_inf,
                    fitted_k,
                )
                parameter_errors = np.sqrt(np.diag(covariance))
                y0_se, y_inf_se, k_se = map(float, parameter_errors)
            else:
                model = lambda t, y_inf, k: _single_exponential_fixed_initial(
                    t,
                    y_inf,
                    k,
                    y0_observed,
                )
                parameters, covariance = curve_fit(
                    model,
                    tau_s,
                    observed,
                    p0=(y_inf_guess, k_guess),
                    bounds=(
                        (lower_value, 0.0),
                        (upper_value, np.inf),
                    ),
                    maxfev=50_000,
                )
                fitted_y0 = y0_observed
                fitted_y_inf, fitted_k = map(float, parameters)
                fitted_values = model(tau_s, fitted_y_inf, fitted_k)
                parameter_errors = np.sqrt(np.diag(covariance))
                y0_se = 0.0
                y_inf_se, k_se = map(float, parameter_errors)

            residuals = observed - fitted_values
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = float(np.sqrt(np.mean(residuals**2)))

            if fitted_k > 0:
                tau_eff_s = 1.0 / fitted_k
                half_time_s = np.log(2.0) / fitted_k
                tau_eff_se_s = k_se / fitted_k**2 if np.isfinite(k_se) else np.nan
                half_time_se_s = (
                    np.log(2.0) * k_se / fitted_k**2
                    if np.isfinite(k_se)
                    else np.nan
                )
            else:
                tau_eff_s = np.inf
                half_time_s = np.inf
                tau_eff_se_s = np.nan
                half_time_se_s = np.nan

            direction = (
                "increasing"
                if fitted_y_inf > fitted_y0
                else "decreasing"
                if fitted_y_inf < fitted_y0
                else "flat"
            )

            fit_rows.append(
                {
                    **base_row,
                    "fit_success": True,
                    "fit_message": "ok",
                    "phase_start_t_s": phase_start_s,
                    "phase_end_t_s": float(absolute_time_s[-1]),
                    "phase_duration_s": float(tau_s[-1]),
                    "direction": direction,
                    "y0": fitted_y0,
                    "y0_se": y0_se,
                    "y_inf": fitted_y_inf,
                    "y_inf_se": y_inf_se,
                    "amplitude": fitted_y0 - fitted_y_inf,
                    "k_eff_s_inv": fitted_k,
                    "k_eff_se_s_inv": k_se,
                    "tau_eff_s": tau_eff_s,
                    "tau_eff_se_s": tau_eff_se_s,
                    "half_time_s": half_time_s,
                    "half_time_se_s": half_time_se_s,
                    "r_squared": r_squared,
                    "rmse": rmse,
                }
            )

            point_tables.append(
                pd.DataFrame(
                    {
                        "phase_index": phase_index,
                        "phase_label": phase_label,
                        "t_s": absolute_time_s,
                        "phase_elapsed_s": tau_s,
                        "observed": observed,
                        "fitted": fitted_values,
                        "residual": residuals,
                    }
                )
            )

        except Exception as exc:
            base_row["fit_message"] = f"{type(exc).__name__}: {exc}"
            fit_rows.append(base_row)

    fits = pd.DataFrame(fit_rows)

    if point_tables:
        fitted_history = pd.concat(point_tables, ignore_index=True)
    else:
        fitted_history = pd.DataFrame(
            columns=[
                "phase_index",
                "phase_label",
                "t_s",
                "phase_elapsed_s",
                "observed",
                "fitted",
                "residual",
            ]
        )

    return fits, fitted_history


def plot_phase_exponential_fits(
    history_or_source: Union[pd.DataFrame, PathLike],
    fits: Optional[pd.DataFrame] = None,
    fitted_history: Optional[pd.DataFrame] = None,
    value_column: str = "theta",
    phase_column: str = "phase_label",
    time_column: str = "t_s",
    figsize: Tuple[float, float] = (8, 5),
    ax=None,
):
    """Plot observed phase trajectories and fitted exponential relaxations."""
    import matplotlib.pyplot as plt

    if isinstance(history_or_source, pd.DataFrame):
        history = history_or_source.copy()
    else:
        history = load_history(history_or_source)

    if fits is None or fitted_history is None:
        fits, fitted_history = fit_history_phases(
            history,
            value_column=value_column,
            phase_column=phase_column,
            time_column=time_column,
        )

    if phase_column not in history.columns:
        history[phase_column] = "all"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for phase_label, phase_data in history.groupby(phase_column, sort=False):
        phase_data = phase_data.sort_values(time_column)
        ax.plot(
            phase_data[time_column],
            phase_data[value_column],
            marker="o",
            markersize=3,
            linewidth=1,
            alpha=0.65,
            label=f"{phase_label}: observed",
        )

        fitted_phase = fitted_history.loc[
            fitted_history["phase_label"].astype(str) == str(phase_label)
        ]

        if not fitted_phase.empty:
            fit_row = fits.loc[
                (fits["phase_label"].astype(str) == str(phase_label))
                & fits["fit_success"]
            ]
            fit_label = f"{phase_label}: fit"

            if not fit_row.empty:
                k_eff = float(fit_row.iloc[0]["k_eff_s_inv"])
                fit_label += f" (k={k_eff:.3g} s$^{{-1}}$)"

            ax.plot(
                fitted_phase["t_s"],
                fitted_phase["fitted"],
                linewidth=2,
                label=fit_label,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(value_column)
    ax.set_title("Single-exponential fits by assay phase")
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# State-frame height analysis
# -----------------------------------------------------------------------------


def _validated_frames(
    frames_or_source: Union[Sequence[Mapping[str, Any]], PathLike],
    deduplicate_times: bool = True,
) -> List[Dict[str, Any]]:
    if isinstance(frames_or_source, (str, Path)):
        frames = load_state_frames(frames_or_source)
    else:
        frames = [dict(frame) for frame in frames_or_source]

    if not frames:
        raise ValueError("No state frames were supplied.")

    for index, frame in enumerate(frames):
        if "t_s" not in frame:
            raise KeyError(f"State frame {index} does not contain 't_s'.")

    frames = sorted(frames, key=lambda frame: float(frame["t_s"]))

    if deduplicate_times:
        by_time: Dict[float, Dict[str, Any]] = {}

        for frame in frames:
            by_time[float(frame["t_s"])] = frame

        frames = [by_time[key] for key in sorted(by_time)]

    return frames


def _frame_receptor_positions(frame: Mapping[str, Any]) -> np.ndarray:
    if "receptor_surface_center_m" in frame:
        xyz = np.asarray(frame["receptor_surface_center_m"], dtype=float)
    elif "receptor_xyz_m" in frame:
        xyz = np.asarray(frame["receptor_xyz_m"], dtype=float)
    elif "receptor_xy_m" in frame:
        xy = np.asarray(frame["receptor_xy_m"], dtype=float)
        xyz = np.column_stack([xy, np.zeros(xy.shape[0], dtype=float)])
    else:
        raise KeyError(
            "State frame must contain 'receptor_surface_center_m', "
            "'receptor_xyz_m', or 'receptor_xy_m'."
        )

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Receptor coordinates must have shape (N, 3).")

    return xyz


def _frame_vertical_values(
    frame: Mapping[str, Any],
    metric: str,
    coordinate: str,
    reference_z_m: Optional[float],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    metric = metric.lower()
    coordinate = coordinate.lower()

    if coordinate not in {"z", "depth"}:
        raise ValueError("coordinate must be 'z' or 'depth'.")

    if coordinate == "depth" and reference_z_m is None:
        raise ValueError("reference_z_m is required when coordinate='depth'.")

    receptor_xyz = None
    receptor_bound = None

    if metric in {
        "bound_receptor_count",
        "bound_receptor_fraction",
        "all_receptor_count",
    }:
        receptor_xyz = _frame_receptor_positions(frame)
        receptor_bound = np.asarray(frame["receptor_bound"], dtype=bool)

        if receptor_bound.shape != (receptor_xyz.shape[0],):
            raise ValueError(
                "receptor_bound length does not match receptor coordinate count."
            )

    if metric == "bound_receptor_count":
        z_values = receptor_xyz[receptor_bound, 2]
        denominator_values = None
    elif metric == "bound_receptor_fraction":
        z_values = receptor_xyz[receptor_bound, 2]
        denominator_values = receptor_xyz[:, 2]
    elif metric == "all_receptor_count":
        z_values = receptor_xyz[:, 2]
        denominator_values = None
    elif metric in {"free_ligand_count", "free_ligand_probability"}:
        ligand_xyz = np.asarray(frame.get("ligand_xyz_m", np.empty((0, 3))), dtype=float)

        if ligand_xyz.size == 0:
            ligand_xyz = np.empty((0, 3), dtype=float)

        if ligand_xyz.ndim != 2 or ligand_xyz.shape[1] != 3:
            raise ValueError("ligand_xyz_m must have shape (N, 3).")

        z_values = ligand_xyz[:, 2]
        denominator_values = None
    else:
        raise ValueError(
            "metric must be one of 'bound_receptor_count', "
            "'bound_receptor_fraction', 'all_receptor_count', "
            "'free_ligand_count', or 'free_ligand_probability'."
        )

    if coordinate == "depth":
        z_values = float(reference_z_m) - z_values

        if denominator_values is not None:
            denominator_values = float(reference_z_m) - denominator_values

    return z_values, denominator_values


def _default_vertical_range(
    frames: Sequence[Mapping[str, Any]],
    metric: str,
    coordinate: str,
    reference_z_m: Optional[float],
) -> Tuple[float, float]:
    values: List[np.ndarray] = []

    for frame in frames:
        numerator, denominator = _frame_vertical_values(
            frame,
            metric,
            coordinate,
            reference_z_m,
        )

        if numerator.size:
            values.append(numerator[np.isfinite(numerator)])

        if denominator is not None and denominator.size:
            values.append(denominator[np.isfinite(denominator)])

    if not values or sum(value.size for value in values) == 0:
        raise ValueError("No finite vertical coordinates are available.")

    combined = np.concatenate([value for value in values if value.size])
    lower = float(np.min(combined))
    upper = float(np.max(combined))

    if np.isclose(lower, upper):
        padding = max(abs(lower), 1e-9) * 0.01
        lower -= padding
        upper += padding

    return lower, upper


def calculate_height_profile_matrix(
    frames_or_source: Union[Sequence[Mapping[str, Any]], PathLike],
    metric: str = "bound_receptor_fraction",
    bin_edges_m: Optional[Sequence[float]] = None,
    bin_width_m: Optional[float] = None,
    n_bins: int = 20,
    coordinate: str = "z",
    reference_z_m: Optional[float] = None,
    deduplicate_times: bool = True,
) -> HeightProfileResult:
    """
    Calculate a time-by-height matrix from saved state frames.

    Parameters
    ----------
    metric
        ``bound_receptor_count``
            Number of bound receptors in each height bin.
        ``bound_receptor_fraction``
            Bound receptors divided by all receptors in the same height bin.
            This is usually the best metric for nanopores because different
            height bins may contain different numbers of receptor-bearing
            surface faces.
        ``all_receptor_count``
            Static receptor-height distribution.
        ``free_ligand_count``
            Number of free ligands in each height bin.
        ``free_ligand_probability``
            Free-ligand histogram normalized to sum to one within each frame.
    coordinate
        ``z`` for absolute height or ``depth`` for reference_z_m - z.
    reference_z_m
        Rim/top-surface height used when coordinate='depth'.
    bin_edges_m
        Explicit vertical bin edges in meters.
    bin_width_m
        Desired bin width. Ignored if bin_edges_m is supplied.
    n_bins
        Number of bins used when neither bin_edges_m nor bin_width_m is given.
    """
    frames = _validated_frames(frames_or_source, deduplicate_times)
    metric = metric.lower()
    coordinate = coordinate.lower()

    if bin_edges_m is None:
        lower, upper = _default_vertical_range(
            frames,
            metric,
            coordinate,
            reference_z_m,
        )

        if bin_width_m is not None:
            bin_width_m = float(bin_width_m)

            if bin_width_m <= 0:
                raise ValueError("bin_width_m must be positive.")

            start = np.floor(lower / bin_width_m) * bin_width_m
            stop = np.ceil(upper / bin_width_m) * bin_width_m

            if np.isclose(start, stop):
                stop = start + bin_width_m

            edges = np.arange(start, stop + 0.5 * bin_width_m, bin_width_m)
        else:
            if n_bins < 1:
                raise ValueError("n_bins must be at least 1.")

            edges = np.linspace(lower, upper, int(n_bins) + 1)
    else:
        edges = np.asarray(bin_edges_m, dtype=float)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges_m must contain at least two values.")

    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges_m must be finite and strictly increasing.")

    matrix = np.zeros((len(frames), edges.size - 1), dtype=float)
    times_s = np.empty(len(frames), dtype=float)
    phase_labels = np.empty(len(frames), dtype=object)

    for frame_index, frame in enumerate(frames):
        numerator_values, denominator_values = _frame_vertical_values(
            frame,
            metric,
            coordinate,
            reference_z_m,
        )
        numerator_counts, _ = np.histogram(numerator_values, bins=edges)
        values = numerator_counts.astype(float)

        if metric == "bound_receptor_fraction":
            denominator_counts, _ = np.histogram(denominator_values, bins=edges)
            values = np.divide(
                numerator_counts,
                denominator_counts,
                out=np.full_like(values, np.nan, dtype=float),
                where=denominator_counts > 0,
            )
        elif metric == "free_ligand_probability":
            total = float(np.sum(values))

            if total > 0:
                values /= total
            else:
                values[:] = np.nan

        matrix[frame_index] = values
        times_s[frame_index] = float(frame["t_s"])
        phase_labels[frame_index] = str(frame.get("phase_label", "all"))

    return HeightProfileResult(
        matrix=matrix,
        times_s=times_s,
        bin_edges_m=edges,
        bin_centers_m=0.5 * (edges[:-1] + edges[1:]),
        phase_labels=phase_labels,
        metric=metric,
        coordinate=coordinate,
        reference_z_m=reference_z_m,
    )


def _coordinate_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.size == 1:
        width = max(abs(values[0]), 1.0) * 0.01
        return np.array([values[0] - width, values[0] + width])

    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], midpoints, [last]])


def plot_height_profile_heatmap(
    profile_or_frames: Union[
        HeightProfileResult,
        Sequence[Mapping[str, Any]],
        PathLike,
    ],
    metric: str = "bound_receptor_fraction",
    bin_edges_m: Optional[Sequence[float]] = None,
    bin_width_m: Optional[float] = None,
    n_bins: int = 20,
    coordinate: str = "z",
    reference_z_m: Optional[float] = None,
    vertical_units: str = "nm",
    show_phase_boundaries: bool = True,
    figsize: Tuple[float, float] = (8, 5),
    ax=None,
    cmap=None,
):
    """Plot a time-height heatmap from state-frame receptor or ligand data."""
    import matplotlib.pyplot as plt

    if isinstance(profile_or_frames, HeightProfileResult):
        profile = profile_or_frames
    else:
        profile = calculate_height_profile_matrix(
            profile_or_frames,
            metric=metric,
            bin_edges_m=bin_edges_m,
            bin_width_m=bin_width_m,
            n_bins=n_bins,
            coordinate=coordinate,
            reference_z_m=reference_z_m,
        )

    unit_scales = {
        "m": (1.0, "m"),
        "um": (1e6, "µm"),
        "µm": (1e6, "µm"),
        "nm": (1e9, "nm"),
    }

    if vertical_units not in unit_scales:
        raise ValueError("vertical_units must be 'm', 'um', 'µm', or 'nm'.")

    scale, unit_label = unit_scales[vertical_units]
    time_edges = _coordinate_edges(profile.times_s)
    vertical_edges = profile.bin_edges_m * scale

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        time_edges,
        vertical_edges,
        profile.matrix.T,
        shading="auto",
        cmap=cmap,
    )

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar_labels = {
        "bound_receptor_count": "Bound receptor count",
        "bound_receptor_fraction": "Bound fraction within height bin",
        "all_receptor_count": "Receptor count",
        "free_ligand_count": "Free ligand count",
        "free_ligand_probability": "Free ligand probability",
    }
    colorbar.set_label(colorbar_labels.get(profile.metric, profile.metric))

    coordinate_label = "Height z" if profile.coordinate == "z" else "Depth below reference"
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{coordinate_label} ({unit_label})")
    ax.set_title(
        colorbar_labels.get(profile.metric, profile.metric)
        + " versus time and height"
    )

    if show_phase_boundaries and profile.phase_labels.size > 1:
        changes = np.flatnonzero(profile.phase_labels[1:] != profile.phase_labels[:-1]) + 1

        for index in changes:
            boundary_t = 0.5 * (
                profile.times_s[index - 1] + profile.times_s[index]
            )
            ax.axvline(boundary_t, linestyle="--", linewidth=1, alpha=0.7)

    fig.tight_layout()
    return fig, ax, profile


def calculate_bound_height_metrics(
    frames_or_source: Union[Sequence[Mapping[str, Any]], PathLike],
    reference_z_m: Optional[float] = None,
    deep_region_depth_m: Optional[float] = None,
    deduplicate_times: bool = True,
) -> pd.DataFrame:
    """
    Calculate one row of bound-receptor penetration metrics per state frame.

    When reference_z_m is supplied, depth is defined as reference_z_m - z.
    Positive depth therefore indicates binding below the selected rim/top
    surface. ``deep_region_depth_m`` reports the fraction of bound receptors at
    or deeper than that depth.
    """
    frames = _validated_frames(frames_or_source, deduplicate_times)
    rows: List[Dict[str, Any]] = []

    for frame_index, frame in enumerate(frames):
        receptor_xyz = _frame_receptor_positions(frame)
        bound_mask = np.asarray(frame["receptor_bound"], dtype=bool)

        if bound_mask.shape != (receptor_xyz.shape[0],):
            raise ValueError(
                "receptor_bound length does not match receptor coordinate count."
            )

        bound_z = receptor_xyz[bound_mask, 2]
        n_bound = int(bound_z.size)
        n_receptors = int(receptor_xyz.shape[0])

        row: Dict[str, Any] = {
            "frame_index": frame_index,
            "t_s": float(frame["t_s"]),
            "phase_label": str(frame.get("phase_label", "all")),
            "n_bound": n_bound,
            "n_receptors": n_receptors,
            "theta": n_bound / n_receptors if n_receptors > 0 else np.nan,
        }

        if n_bound > 0:
            row.update(
                {
                    "mean_bound_z_m": float(np.mean(bound_z)),
                    "median_bound_z_m": float(np.median(bound_z)),
                    "std_bound_z_m": float(np.std(bound_z)),
                    "min_bound_z_m": float(np.min(bound_z)),
                    "max_bound_z_m": float(np.max(bound_z)),
                    "bound_z_q10_m": float(np.quantile(bound_z, 0.10)),
                    "bound_z_q90_m": float(np.quantile(bound_z, 0.90)),
                }
            )
        else:
            row.update(
                {
                    "mean_bound_z_m": np.nan,
                    "median_bound_z_m": np.nan,
                    "std_bound_z_m": np.nan,
                    "min_bound_z_m": np.nan,
                    "max_bound_z_m": np.nan,
                    "bound_z_q10_m": np.nan,
                    "bound_z_q90_m": np.nan,
                }
            )

        if reference_z_m is not None:
            bound_depth = float(reference_z_m) - bound_z

            if n_bound > 0:
                row.update(
                    {
                        "mean_bound_depth_m": float(np.mean(bound_depth)),
                        "median_bound_depth_m": float(np.median(bound_depth)),
                        "max_bound_depth_m": float(np.max(bound_depth)),
                        "bound_depth_q90_m": float(np.quantile(bound_depth, 0.90)),
                        "fraction_bound_below_reference": float(
                            np.mean(bound_depth > 0)
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "mean_bound_depth_m": np.nan,
                        "median_bound_depth_m": np.nan,
                        "max_bound_depth_m": np.nan,
                        "bound_depth_q90_m": np.nan,
                        "fraction_bound_below_reference": np.nan,
                    }
                )

            if deep_region_depth_m is not None:
                row["fraction_bound_in_deep_region"] = (
                    float(np.mean(bound_depth >= deep_region_depth_m))
                    if n_bound > 0
                    else np.nan
                )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_bound_height_metrics(
    metrics_or_frames: Union[pd.DataFrame, Sequence[Mapping[str, Any]], PathLike],
    columns: Optional[Sequence[str]] = None,
    reference_z_m: Optional[float] = None,
    deep_region_depth_m: Optional[float] = None,
    distance_units: str = "nm",
    figsize: Tuple[float, float] = (8, 5),
    ax=None,
):
    """Plot selected scalar penetration metrics versus simulation time."""
    import matplotlib.pyplot as plt

    if isinstance(metrics_or_frames, pd.DataFrame):
        metrics = metrics_or_frames.copy()
    else:
        metrics = calculate_bound_height_metrics(
            metrics_or_frames,
            reference_z_m=reference_z_m,
            deep_region_depth_m=deep_region_depth_m,
        )

    if columns is None:
        if "mean_bound_depth_m" in metrics.columns:
            columns = ["mean_bound_depth_m", "bound_depth_q90_m"]
        else:
            columns = ["mean_bound_z_m", "bound_z_q10_m", "bound_z_q90_m"]

    scales = {
        "m": (1.0, "m"),
        "um": (1e6, "µm"),
        "µm": (1e6, "µm"),
        "nm": (1e9, "nm"),
    }

    if distance_units not in scales:
        raise ValueError("distance_units must be 'm', 'um', 'µm', or 'nm'.")

    scale, unit_label = scales[distance_units]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for column in columns:
        if column not in metrics.columns:
            raise KeyError(f"Metrics table does not contain {column!r}.")

        values = metrics[column].to_numpy(dtype=float)
        use_distance_scale = column.endswith("_m")
        plotted = values * scale if use_distance_scale else values
        label = column.replace("_m", "").replace("_", " ")

        ax.plot(
            metrics["t_s"],
            plotted,
            marker="o",
            markersize=3,
            label=label,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Distance ({unit_label})")
    ax.set_title("Bound-target penetration metrics")
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Convenience analysis of one saved run
# -----------------------------------------------------------------------------


def analyze_saved_run(
    run_directory: PathLike,
    fit_value_column: str = "theta",
    height_metric: str = "bound_receptor_fraction",
    bin_width_m: Optional[float] = None,
    n_height_bins: int = 20,
    coordinate: str = "z",
    reference_z_m: Optional[float] = None,
    deep_region_depth_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Load one saved run and calculate the primary kinetic/spatial results."""
    history = load_history(run_directory)
    fits, fitted_history = fit_history_phases(
        history,
        value_column=fit_value_column,
    )
    frames = load_state_frames(run_directory)
    height_profile = calculate_height_profile_matrix(
        frames,
        metric=height_metric,
        bin_width_m=bin_width_m,
        n_bins=n_height_bins,
        coordinate=coordinate,
        reference_z_m=reference_z_m,
    )
    bound_height_metrics = calculate_bound_height_metrics(
        frames,
        reference_z_m=reference_z_m,
        deep_region_depth_m=deep_region_depth_m,
    )

    return {
        "history": history,
        "phase_fits": fits,
        "fitted_history": fitted_history,
        "state_frames": frames,
        "height_profile": height_profile,
        "bound_height_metrics": bound_height_metrics,
    }


__all__ = [
    "HeightProfileResult",
    "load_history",
    "load_rebinding_events",
    "load_state_frames",
    "fit_history_phases",
    "plot_phase_exponential_fits",
    "calculate_height_profile_matrix",
    "plot_height_profile_heatmap",
    "calculate_bound_height_metrics",
    "plot_bound_height_metrics",
    "analyze_saved_run",
]
