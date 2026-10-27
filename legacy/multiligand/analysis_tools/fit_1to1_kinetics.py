#!/usr/bin/env python3
"""
Fit the Sartorius 1:1 binding model to saved biosensor association/dissociation
simulations.

For each replicate, this script loads history.parquet, identifies the
association and dissociation phases, and jointly fits one shared pair of
kinetic parameters:

    k_on  [M^-1 s^-1]
    k_off [s^-1]

For normalized receptor occupancy theta, Rmax = 1.

The Sartorius 1:1 model is:

Association:
    theta_eq = (k_on * C) / (k_on * C + k_off)

    theta(t) =
        theta_eq
        + (theta_initial - theta_eq)
          * exp[-(k_on*C + k_off) * t]

If theta_initial = 0, this reduces exactly to the usual Sartorius 1:1
association expression:

    theta(t) =
        [1 / (1 + k_off/(k_on*C))]
        * [1 - exp(-(k_on*C + k_off)t)]

Dissociation:
    theta(t) = theta_at_transition * exp(-k_off * t)

where theta_at_transition is predicted by the association fit at the
association/dissociation phase boundary.

The association and dissociation phases are therefore constrained by the same
k_on and k_off values rather than being fit as unrelated exponentials.

Affinity:
    K_D = k_off / k_on

Outputs one CSV row per replicate and can optionally save long-form fitted
curves.

Example
-------
python fit_1to1_kinetics.py \
    --data-root /home/rnt2664/results/260806_nanopore_geometry_sweep \
    --output-csv /home/rnt2664/results/260806_nanopore_geometry_sweep/one_to_one_kinetics.csv \
    --fit-curves-csv /home/rnt2664/results/260806_nanopore_geometry_sweep/one_to_one_fit_curves.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


LOG10_LOWER = np.array([0.0, -8.0], dtype=float)
LOG10_UPPER = np.array([12.0, 3.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly fit association and dissociation histories to the "
            "Sartorius 1:1 binding model."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing replicate folders with history.parquet.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV containing one row per replicate.",
    )
    parser.add_argument(
        "--fit-curves-csv",
        type=Path,
        default=None,
        help=(
            "Optional long-form CSV containing observed and fitted theta values."
        ),
    )
    parser.add_argument(
        "--association-label",
        default="association",
        help="phase_label identifying the association phase.",
    )
    parser.add_argument(
        "--dissociation-label",
        default="dissociation",
        help="phase_label identifying the dissociation phase.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum finite points required in each phase.",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=5000,
        help="Maximum least-squares function evaluations per starting point.",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=12,
        help="Number of multistart optimization guesses per replicate.",
    )

    return parser.parse_args()


def read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_finite_value(
    frame: pd.DataFrame,
    candidate_columns: Iterable[str],
) -> Optional[float]:
    for column in candidate_columns:
        if column not in frame.columns:
            continue

        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values)]

        if len(values):
            return float(values.iloc[0])

    return None


def parse_replicate_number(run_directory: Path) -> Optional[int]:
    match = re.search(r"replicate[_-]?(\d+)", run_directory.name, flags=re.I)
    return int(match.group(1)) if match else None


SWEEP_PARAMETER_INFO = {
    "pore_diameter_m": {"label": "Pore diameter", "unit": "m"},
    "pore_depth_m": {"label": "Pore height/depth", "unit": "m"},
    "pore_height_m": {"label": "Pore height", "unit": "m"},
    "association_concentration_M": {"label": "Association concentration", "unit": "M"},
    "ligand_conc_M": {"label": "Ligand concentration", "unit": "M"},
    "receptor_density_m2": {"label": "Receptor density", "unit": "m^-2"},
    "spike_density_m2": {"label": "Primary spike density", "unit": "m^-2"},
    "branch_levels": {"label": "Branch levels", "unit": ""},
    "branches_per_segment_mean": {"label": "Branches per segment", "unit": ""},
}


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def sweep_parameter_label(parameter: str) -> str:
    info = SWEEP_PARAMETER_INFO.get(parameter)
    if info is not None:
        return str(info["label"])
    return parameter.replace("_", " ").strip().capitalize()


def sweep_parameter_unit(parameter: str) -> str:
    info = SWEEP_PARAMETER_INFO.get(parameter)
    return str(info["unit"]) if info is not None else ""


def _run_metadata(run_directory: Path) -> dict:
    manifest = read_json_if_present(run_directory / "manifest.json")
    value = manifest.get("run_metadata", {})
    return value if isinstance(value, dict) else {}


def _sweep_identity_from_metadata(run_directory: Path):
    run_metadata = _run_metadata(run_directory)

    parameter = run_metadata.get("sweep_parameter")
    if parameter is not None and "sweep_value" in run_metadata:
        return str(parameter), run_metadata["sweep_value"]

    parameter = run_metadata.get("sweep_variable")
    if parameter is not None:
        parameter = str(parameter)
        if parameter in run_metadata:
            return parameter, run_metadata[parameter]

        aliases = {
            "pore_depth_m": ("pore_depth_m", "pore_height_m", "pore_height_nm"),
            "pore_diameter_m": ("pore_diameter_m", "diameter_m", "pore_diameter_nm", "diameter_nm"),
            "association_concentration_M": ("association_concentration_M", "ligand_conc_M"),
            "receptor_density_m2": ("receptor_density_m2",),
        }

        for candidate in aliases.get(parameter, ()):
            if candidate not in run_metadata:
                continue
            value = run_metadata[candidate]
            if candidate.endswith("_nm") and parameter.endswith("_m"):
                number = _finite_float(value)
                if number is not None:
                    value = number * 1e-9
            return parameter, value

    return None


def _sweep_identity_from_directory(run_directory: Path):
    names = [run_directory.parent.name]
    names.extend(parent.name for parent in list(run_directory.parents)[1:3])

    patterns = (
        (r"^diameter_([0-9eE.+-]+)m$", "pore_diameter_m", lambda x: float(x)),
        (r"^height_([0-9eE.+-]+)nm$", "pore_depth_m", lambda x: float(x) * 1e-9),
        (r"^pore[_-]?height_([0-9eE.+-]+)nm$", "pore_depth_m", lambda x: float(x) * 1e-9),
        (r"^concentration_([0-9eE.+-]+)M$", "association_concentration_M", lambda x: float(x)),
        (r"^density_([0-9eE.+-]+)_per_m2$", "receptor_density_m2", lambda x: float(x)),
        (r"^receptor_density_m2_([0-9eE.+-]+)$", "receptor_density_m2", lambda x: float(x)),
        (r"^spike_density_m2_([0-9eE.+-]+)$", "spike_density_m2", lambda x: float(x)),
        (r"^branch_levels_([0-9eE.+-]+)$", "branch_levels", lambda x: int(round(float(x)))),
        (r"^branches_per_segment_mean_([0-9eE.+-]+)$", "branches_per_segment_mean", lambda x: float(x)),
    )

    for name in names:
        for pattern, parameter, converter in patterns:
            match = re.fullmatch(pattern, name, flags=re.I)
            if match:
                return parameter, converter(match.group(1))
    return None


def geometry_metadata(run_directory: Path) -> Dict[str, object]:
    """Collect geometry metadata and identify the swept parameter generically."""
    geometry = read_json_if_present(run_directory / "geometry_summary.json")
    run_metadata = _run_metadata(run_directory)

    metadata: Dict[str, object] = {}

    # Preserve useful geometry/protocol metadata when present. This intentionally
    # includes the new nanospike fields written by the well-mixed sweep scripts.
    keys = (
        "geometry_name", "name", "geometry_type", "layout",
        "pore_diameter_m", "pore_diameter_nm", "diameter_m", "diameter_nm",
        "pore_depth_m", "pore_depth_nm", "pore_height_nm",
        "pore_pitch_m", "pore_pitch_nm", "pitch_m", "rim_z_m", "rim_z_nm",
        "receptor_density_m2", "actual_receptor_density_m2", "receptor_count",
        "spike_density_m2", "branch_levels", "branches_per_segment_mean",
        "geometry_seed", "nanospike_primary_count", "nanospike_segment_count",
        "nanospike_canopy_z_m", "true_exposed_area_m2", "projected_area_m2",
        "roughness_factor", "reservoir_interface_z_m", "reservoir_offset_layers",
        "use_well_mixed_reservoir",
    )

    for source in (geometry, run_metadata):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source and key not in metadata:
                metadata[key] = source[key]

    # New protocol metadata has highest priority.
    identity = _sweep_identity_from_metadata(run_directory)
    if identity is None:
        identity = _sweep_identity_from_directory(run_directory)

    # Legacy fallback for old nanopore diameter sweeps.
    if identity is None and "pore_diameter_m" in metadata:
        identity = ("pore_diameter_m", metadata["pore_diameter_m"])

    if identity is not None:
        parameter, value = identity
        metadata["sweep_parameter"] = str(parameter)
        metadata["sweep_value"] = value
        metadata["sweep_parameter_label"] = sweep_parameter_label(str(parameter))
        metadata["sweep_parameter_unit"] = sweep_parameter_unit(str(parameter))

        numeric = _finite_float(value)
        if numeric is not None:
            metadata["sweep_value_numeric"] = numeric

    # Add convenient converted legacy columns where applicable.
    if "pore_diameter_m" in metadata:
        number = _finite_float(metadata["pore_diameter_m"])
        if number is not None:
            metadata["pore_diameter_m"] = number
            metadata.setdefault("pore_diameter_nm", number * 1e9)

    if "pore_depth_m" in metadata:
        number = _finite_float(metadata["pore_depth_m"])
        if number is not None:
            metadata["pore_depth_m"] = number
            metadata.setdefault("pore_depth_nm", number * 1e9)

    return metadata

def clean_phase(phase: pd.DataFrame) -> pd.DataFrame:
    data = phase.copy()

    data["t_s"] = pd.to_numeric(data["t_s"], errors="coerce")
    data["theta"] = pd.to_numeric(data["theta"], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["t_s", "theta"])
    data = data.sort_values("t_s")
    data = data.drop_duplicates(subset="t_s", keep="last").reset_index(drop=True)

    return data


def get_phase(history: pd.DataFrame, label: str) -> pd.DataFrame:
    if "phase_label" not in history.columns:
        raise KeyError(
            "history.parquet does not contain 'phase_label'; "
            "association/dissociation phases cannot be identified."
        )

    phase = history.loc[
        history["phase_label"].astype(str).str.lower() == label.lower()
    ].copy()

    if phase.empty:
        raise ValueError(f"No rows found with phase_label={label!r}.")

    return clean_phase(phase)


def association_concentration_M(
    association: pd.DataFrame,
    run_directory: Path,
) -> Optional[float]:
    concentration = first_finite_value(
        association,
        ["phase_ligand_conc_M", "ligand_conc_M"],
    )

    if concentration is not None:
        return concentration

    manifest = read_json_if_present(run_directory / "manifest.json")
    run_metadata = manifest.get("run_metadata", {})

    if isinstance(run_metadata, dict):
        association_params = run_metadata.get("association_params", {})
        if isinstance(association_params, dict):
            value = association_params.get("ligand_conc_M")
            if value is not None:
                return float(value)

    parameters = read_json_if_present(run_directory / "parameters.json")
    value = parameters.get("ligand_conc_M")

    return None if value is None else float(value)


def saved_kinetic_guesses(
    association: pd.DataFrame,
    run_directory: Path,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Use saved microscopic parameters only as starting guesses.
    """
    kon = first_finite_value(
        association,
        ["phase_k_on_M_inv_s", "k_on_M_inv_s"],
    )
    koff = first_finite_value(
        association,
        ["phase_k_off_s", "k_off_s"],
    )

    if kon is not None and koff is not None:
        return kon, koff

    parameters = read_json_if_present(run_directory / "parameters.json")

    if kon is None and parameters.get("k_on_M_inv_s") is not None:
        kon = float(parameters["k_on_M_inv_s"])

    if koff is None and parameters.get("k_off_s") is not None:
        koff = float(parameters["k_off_s"])

    return kon, koff


def phase_start_time(phase: pd.DataFrame) -> float:
    """
    Prefer saved exact phase_start_t_s when available.
    """
    start = first_finite_value(phase, ["phase_start_t_s"])

    if start is not None:
        return float(start)

    return float(phase["t_s"].iloc[0])


def one_to_one_association(
    elapsed_s: np.ndarray,
    *,
    theta_initial: float,
    concentration_M: float,
    kon_M_inv_s: float,
    koff_s_inv: float,
) -> np.ndarray:
    """
    General 1:1 association solution with Rmax = 1.

    Reduces to the Sartorius equation when theta_initial = 0.
    """
    elapsed_s = np.asarray(elapsed_s, dtype=float)

    kobs = kon_M_inv_s * concentration_M + koff_s_inv

    if kobs <= 0:
        return np.full_like(elapsed_s, theta_initial, dtype=float)

    theta_eq = (
        kon_M_inv_s * concentration_M / kobs
    )

    return theta_eq + (
        theta_initial - theta_eq
    ) * np.exp(-kobs * elapsed_s)


def one_to_one_dissociation(
    elapsed_s: np.ndarray,
    *,
    theta_at_transition: float,
    koff_s_inv: float,
) -> np.ndarray:
    elapsed_s = np.asarray(elapsed_s, dtype=float)

    return theta_at_transition * np.exp(
        -koff_s_inv * elapsed_s
    )


def r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    fitted = np.asarray(fitted, dtype=float)

    ss_res = float(np.sum((observed - fitted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))

    if ss_tot <= 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def simple_koff_guess(dissociation: pd.DataFrame) -> float:
    t = dissociation["t_s"].to_numpy(dtype=float)
    y = dissociation["theta"].to_numpy(dtype=float)
    tau = t - t[0]

    mask = (
        np.isfinite(tau)
        & np.isfinite(y)
        & (y > 1e-10)
    )

    if np.count_nonzero(mask) >= 3:
        slope, _ = np.polyfit(
            tau[mask],
            np.log(y[mask]),
            deg=1,
        )

        if np.isfinite(slope) and slope < 0:
            return max(-float(slope), 1e-8)

    duration = max(float(tau[-1]), 1e-6)
    return 1.0 / duration


def simple_kon_guess(
    association: pd.DataFrame,
    concentration_M: float,
    koff_guess: float,
) -> float:
    t = association["t_s"].to_numpy(dtype=float)
    y = association["theta"].to_numpy(dtype=float)
    tau = t - t[0]

    duration = max(float(tau[-1]), 1e-6)

    theta_initial = float(y[0])
    theta_final = float(y[-1])

    if theta_final > theta_initial:
        target = theta_initial + 0.6321205588 * (
            theta_final - theta_initial
        )

        indices = np.flatnonzero(y >= target)

        if len(indices):
            t63 = max(
                float(tau[indices[0]]),
                duration / 1000.0,
            )
            kobs_guess = 1.0 / t63
        else:
            kobs_guess = 1.0 / duration
    else:
        kobs_guess = 1.0 / duration

    kinetic_increment = max(
        kobs_guess - koff_guess,
        0.1 * kobs_guess,
        1e-8,
    )

    return kinetic_increment / concentration_M


def initial_guesses(
    association: pd.DataFrame,
    dissociation: pd.DataFrame,
    concentration_M: float,
    run_directory: Path,
    n_starts: int,
) -> List[np.ndarray]:
    saved_kon, saved_koff = saved_kinetic_guesses(
        association,
        run_directory,
    )

    koff_guess = (
        float(saved_koff)
        if saved_koff is not None and saved_koff > 0
        else simple_koff_guess(dissociation)
    )

    kon_guess = (
        float(saved_kon)
        if saved_kon is not None and saved_kon > 0
        else simple_kon_guess(
            association,
            concentration_M,
            koff_guess,
        )
    )

    kon_guess = float(
        np.clip(
            kon_guess,
            10 ** LOG10_LOWER[0],
            10 ** LOG10_UPPER[0],
        )
    )
    koff_guess = float(
        np.clip(
            koff_guess,
            10 ** LOG10_LOWER[1],
            10 ** LOG10_UPPER[1],
        )
    )

    kon_scales = np.array(
        [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    )
    koff_scales = np.array(
        [0.1, 0.3, 1.0, 3.0, 10.0]
    )

    candidates: List[np.ndarray] = []

    for kon_scale in kon_scales:
        for koff_scale in koff_scales:
            kon = np.clip(
                kon_guess * kon_scale,
                10 ** LOG10_LOWER[0],
                10 ** LOG10_UPPER[0],
            )
            koff = np.clip(
                koff_guess * koff_scale,
                10 ** LOG10_LOWER[1],
                10 ** LOG10_UPPER[1],
            )

            candidates.append(
                np.log10([kon, koff])
            )

    unique = []
    seen = set()

    for candidate in candidates:
        key = tuple(np.round(candidate, decimals=8))

        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    if n_starts >= len(unique):
        return unique

    indices = np.linspace(
        0,
        len(unique) - 1,
        n_starts,
        dtype=int,
    )

    return [unique[i] for i in indices]


def parameter_standard_errors(
    fit,
    n_residuals: int,
    n_parameters: int,
) -> np.ndarray:
    """
    Approximate standard errors in log10-parameter space from the Jacobian.
    """
    if fit.jac is None or n_residuals <= n_parameters:
        return np.full(n_parameters, np.nan)

    try:
        jac = np.asarray(fit.jac, dtype=float)
        jtj = jac.T @ jac

        covariance = np.linalg.pinv(jtj)

        residual_variance = (
            2.0 * float(fit.cost)
            / (n_residuals - n_parameters)
        )

        covariance *= residual_variance

        return np.sqrt(np.diag(covariance))

    except Exception:
        return np.full(n_parameters, np.nan)


def fit_replicate(
    history_path: Path,
    *,
    association_label: str,
    dissociation_label: str,
    min_points: int,
    max_nfev: int,
    n_starts: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    run_directory = history_path.parent
    history = pd.read_parquet(history_path)

    for column in ("t_s", "theta"):
        if column not in history.columns:
            raise KeyError(
                f"{history_path} is missing required column {column!r}."
            )

    association = get_phase(history, association_label)
    dissociation = get_phase(history, dissociation_label)

    if len(association) < min_points:
        raise ValueError(
            f"Association has {len(association)} points; "
            f"need at least {min_points}."
        )

    if len(dissociation) < min_points:
        raise ValueError(
            f"Dissociation has {len(dissociation)} points; "
            f"need at least {min_points}."
        )

    concentration_M = association_concentration_M(
        association,
        run_directory,
    )

    if concentration_M is None or not np.isfinite(concentration_M):
        raise ValueError(
            "Could not determine the association analyte concentration."
        )

    if concentration_M <= 0:
        raise ValueError(
            f"Association concentration must be > 0; "
            f"found {concentration_M}."
        )

    association_start_t_s = phase_start_time(association)
    dissociation_start_t_s = phase_start_time(dissociation)

    t_assoc_abs = association["t_s"].to_numpy(dtype=float)
    y_assoc = association["theta"].to_numpy(dtype=float)

    t_diss_abs = dissociation["t_s"].to_numpy(dtype=float)
    y_diss = dissociation["theta"].to_numpy(dtype=float)

    t_assoc = t_assoc_abs - association_start_t_s
    t_diss = t_diss_abs - dissociation_start_t_s

    if np.any(t_assoc < -1e-12):
        # Fallback when phase_start_t_s is unavailable/inconsistent.
        association_start_t_s = float(t_assoc_abs[0])
        t_assoc = t_assoc_abs - association_start_t_s

    if np.any(t_diss < -1e-12):
        dissociation_start_t_s = float(t_diss_abs[0])
        t_diss = t_diss_abs - dissociation_start_t_s

    association_duration_s = (
        dissociation_start_t_s - association_start_t_s
    )

    if association_duration_s <= 0:
        association_duration_s = float(t_assoc[-1])

    theta_initial = float(y_assoc[0])

    def predictions(log10_parameters: np.ndarray):
        kon, koff = 10.0 ** np.asarray(
            log10_parameters,
            dtype=float,
        )

        pred_assoc = one_to_one_association(
            t_assoc,
            theta_initial=theta_initial,
            concentration_M=concentration_M,
            kon_M_inv_s=kon,
            koff_s_inv=koff,
        )

        theta_transition = float(
            one_to_one_association(
                np.array([association_duration_s]),
                theta_initial=theta_initial,
                concentration_M=concentration_M,
                kon_M_inv_s=kon,
                koff_s_inv=koff,
            )[0]
        )

        pred_diss = one_to_one_dissociation(
            t_diss,
            theta_at_transition=theta_transition,
            koff_s_inv=koff,
        )

        return pred_assoc, pred_diss, theta_transition

    # The first association point is fixed by theta_initial, so omit that
    # guaranteed-zero residual. Dissociation is retained in full because its
    # first point tests continuity across the phase boundary.
    def residuals(log10_parameters: np.ndarray) -> np.ndarray:
        pred_assoc, pred_diss, _ = predictions(
            log10_parameters
        )

        residual = np.concatenate(
            [
                pred_assoc[1:] - y_assoc[1:],
                pred_diss - y_diss,
            ]
        )

        if not np.all(np.isfinite(residual)):
            return np.full_like(residual, 1e6)

        return residual

    starts = initial_guesses(
        association,
        dissociation,
        concentration_M,
        run_directory,
        n_starts=n_starts,
    )

    best_fit = None

    for start in starts:
        fit = least_squares(
            residuals,
            x0=np.clip(
                start,
                LOG10_LOWER,
                LOG10_UPPER,
            ),
            bounds=(
                LOG10_LOWER,
                LOG10_UPPER,
            ),
            method="trf",
            max_nfev=max_nfev,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )

        if best_fit is None or fit.cost < best_fit.cost:
            best_fit = fit

    if best_fit is None:
        raise RuntimeError(
            "No optimization attempt was completed."
        )

    kon, koff = 10.0 ** best_fit.x

    pred_assoc, pred_diss, theta_transition = predictions(
        best_fit.x
    )

    assoc_r2 = r_squared(
        y_assoc,
        pred_assoc,
    )
    diss_r2 = r_squared(
        y_diss,
        pred_diss,
    )

    observed_joint = np.concatenate(
        [y_assoc, y_diss]
    )
    fitted_joint = np.concatenate(
        [pred_assoc, pred_diss]
    )

    joint_r2 = r_squared(
        observed_joint,
        fitted_joint,
    )

    joint_rmse = float(
        np.sqrt(
            np.mean(
                (observed_joint - fitted_joint) ** 2
            )
        )
    )

    assoc_rmse = float(
        np.sqrt(
            np.mean(
                (y_assoc - pred_assoc) ** 2
            )
        )
    )

    diss_rmse = float(
        np.sqrt(
            np.mean(
                (y_diss - pred_diss) ** 2
            )
        )
    )

    log_se = parameter_standard_errors(
        best_fit,
        n_residuals=(len(y_assoc) - 1) + len(y_diss),
        n_parameters=2,
    )

    parameter_values = np.array(
        [kon, koff],
        dtype=float,
    )

    # Delta method for p = 10^x:
    # sigma_p ~= ln(10) * p * sigma_x.
    parameter_se = (
        np.log(10.0)
        * parameter_values
        * log_se
    )

    kd_M = koff / kon
    kd_nM = kd_M * 1e9

    kobs_assoc = (
        kon * concentration_M + koff
    )

    theta_eq = (
        kon * concentration_M / kobs_assoc
    )

    result: Dict[str, object] = {
        "replicate": parse_replicate_number(
            run_directory
        ),
        "run_directory": str(run_directory),
        "history_path": str(history_path),
        "model": "Sartorius 1:1 binding",
        "association_concentration_M": concentration_M,
        "association_concentration_nM": (
            concentration_M * 1e9
        ),
        "k_on_eff_M_inv_s": kon,
        "k_on_eff_se_M_inv_s": parameter_se[0],
        "k_off_eff_s_inv": koff,
        "k_off_eff_se_s_inv": parameter_se[1],
        "K_D_eff_M": kd_M,
        "K_D_eff_nM": kd_nM,
        "k_obs_assoc_s_inv": kobs_assoc,
        "theta_eq_predicted": theta_eq,
        "theta_transition_predicted": theta_transition,
        "association_r_squared": assoc_r2,
        "dissociation_r_squared": diss_r2,
        "joint_r_squared": joint_r2,
        "association_rmse": assoc_rmse,
        "dissociation_rmse": diss_rmse,
        "joint_rmse": joint_rmse,
        "association_n_points": len(association),
        "dissociation_n_points": len(dissociation),
        "association_start_t_s": association_start_t_s,
        "dissociation_start_t_s": dissociation_start_t_s,
        "association_duration_s": association_duration_s,
        "association_theta_initial": theta_initial,
        "association_theta_final_observed": float(
            y_assoc[-1]
        ),
        "dissociation_theta_initial_observed": float(
            y_diss[0]
        ),
        "dissociation_theta_final_observed": float(
            y_diss[-1]
        ),
        "optimizer_success": bool(best_fit.success),
        "optimizer_status": int(best_fit.status),
        "optimizer_message": str(best_fit.message),
        "optimizer_cost": float(best_fit.cost),
        "optimizer_nfev": int(best_fit.nfev),
        "fit_at_lower_bound": bool(
            np.any(
                np.isclose(
                    best_fit.x,
                    LOG10_LOWER,
                    atol=1e-4,
                )
            )
        ),
        "fit_at_upper_bound": bool(
            np.any(
                np.isclose(
                    best_fit.x,
                    LOG10_UPPER,
                    atol=1e-4,
                )
            )
        ),
    }

    result.update(
        geometry_metadata(run_directory)
    )

    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "replicate": result["replicate"],
                    "phase_label": association_label,
                    "t_s": t_assoc_abs,
                    "phase_elapsed_s": t_assoc,
                    "theta_observed": y_assoc,
                    "theta_fitted": pred_assoc,
                }
            ),
            pd.DataFrame(
                {
                    "replicate": result["replicate"],
                    "phase_label": dissociation_label,
                    "t_s": t_diss_abs,
                    "phase_elapsed_s": t_diss,
                    "theta_observed": y_diss,
                    "theta_fitted": pred_diss,
                }
            ),
        ],
        ignore_index=True,
    )

    for key in (
        "sweep_parameter",
        "sweep_parameter_label",
        "sweep_parameter_unit",
        "sweep_value",
        "sweep_value_numeric",
        "pore_diameter_m",
        "pore_diameter_nm",
        "pore_depth_m",
        "pore_depth_nm",
        "receptor_density_m2",
        "spike_density_m2",
        "branch_levels",
        "branches_per_segment_mean",
        "roughness_factor",
        "layout",
    ):
        if key in result:
            curves[key] = result[key]

    curves["run_directory"] = str(
        run_directory
    )

    return result, curves


def failure_row(
    history_path: Path,
    error: Exception,
) -> Dict[str, object]:
    run_directory = history_path.parent

    row: Dict[str, object] = {
        "replicate": parse_replicate_number(
            run_directory
        ),
        "run_directory": str(run_directory),
        "history_path": str(history_path),
        "model": "Sartorius 1:1 binding",
        "optimizer_success": False,
        "analysis_error": (
            f"{type(error).__name__}: {error}"
        ),
    }

    row.update(
        geometry_metadata(run_directory)
    )

    return row


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root does not exist: {data_root}"
        )

    history_paths = sorted(
        data_root.rglob("history.parquet")
    )

    if not history_paths:
        raise FileNotFoundError(
            f"No history.parquet files found under "
            f"{data_root}"
        )

    print(
        f"Found {len(history_paths)} replicate "
        f"history files."
    )

    rows: List[Dict[str, object]] = []
    curve_frames: List[pd.DataFrame] = []

    for index, history_path in enumerate(
        history_paths,
        start=1,
    ):
        print(
            f"[{index}/{len(history_paths)}] "
            f"{history_path.parent}"
        )

        try:
            result, curves = fit_replicate(
                history_path,
                association_label=args.association_label,
                dissociation_label=args.dissociation_label,
                min_points=args.min_points,
                max_nfev=args.max_nfev,
                n_starts=args.n_starts,
            )

            rows.append(result)

            if args.fit_curves_csv is not None:
                curve_frames.append(curves)

            print(
                "    "
                f"kon={result['k_on_eff_M_inv_s']:.4g} "
                f"M^-1 s^-1, "
                f"koff={result['k_off_eff_s_inv']:.4g} "
                f"s^-1, "
                f"KD={result['K_D_eff_nM']:.4g} nM, "
                f"R2_assoc="
                f"{result['association_r_squared']:.4f}, "
                f"R2_diss="
                f"{result['dissociation_r_squared']:.4f}"
            )

        except Exception as exc:
            rows.append(
                failure_row(
                    history_path,
                    exc,
                )
            )

            print(
                f"    FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

    results = pd.DataFrame(rows)

    sort_columns = [
        column
        for column in (
            "sweep_parameter",
            "sweep_value_numeric",
            "sweep_value",
            "replicate",
        )
        if column in results.columns
    ]

    if sort_columns:
        results = results.sort_values(
            sort_columns
        ).reset_index(drop=True)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_csv,
        index=False,
    )

    print()
    print(
        f"Saved parameter table: {output_csv}"
    )

    if args.fit_curves_csv is not None:
        fit_curves_csv = (
            args.fit_curves_csv
            .expanduser()
            .resolve()
        )

        fit_curves_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if curve_frames:
            curves = pd.concat(
                curve_frames,
                ignore_index=True,
            )

            curve_sort = [
                column
                for column in (
                    "sweep_parameter",
                    "sweep_value_numeric",
                    "sweep_value",
                    "replicate",
                    "phase_label",
                    "t_s",
                )
                if column in curves.columns
            ]

            if curve_sort:
                curves = curves.sort_values(
                    curve_sort
                )

            curves.to_csv(
                fit_curves_csv,
                index=False,
            )

            print(
                f"Saved fitted curves:   "
                f"{fit_curves_csv}"
            )

        else:
            print(
                "No successful fits; "
                "fitted-curves CSV was not written."
            )

    if "optimizer_success" in results.columns:
        n_success = int(
            results["optimizer_success"]
            .fillna(False)
            .sum()
        )
    else:
        n_success = 0

    print(
        f"Successful fits: "
        f"{n_success}/{len(results)}"
    )


if __name__ == "__main__":
    main()