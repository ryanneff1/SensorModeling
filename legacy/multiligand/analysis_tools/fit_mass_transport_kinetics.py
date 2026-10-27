#!/usr/bin/env python3
"""
Fit a Sartorius-style mass-transport-limited binding model to saved
association/dissociation biosensor simulations.

Model
-----
For normalized receptor occupancy theta = R / Rmax, Rmax = 1:

    dtheta/dt =
        [kon * C * (1 - theta) - koff * theta]
        --------------------------------------
        [1 + (kon / km) * (1 - theta)]

where:
    C     = bulk analyte concentration (M)
    kon   = effective association rate constant (M^-1 s^-1)
    koff  = effective dissociation rate constant (s^-1)
    km    = effective mass-transport coefficient

Association is fit with C = C_assoc.
Dissociation is fit with C = 0.

The association and dissociation phases are fit jointly to one shared set of
(kon, koff, km) values for each replicate. Each phase is anchored to its own
first observed theta value, while the kinetic parameters are shared.

Because theta is dimensionless and Rmax = 1, km has the same effective units
as kon in this normalized form (M^-1 s^-1).

Outputs
-------
One CSV row per replicate with:
    pore diameter and other geometry metadata
    kon_eff
    koff_eff
    KD_eff = koff_eff / kon_eff
    km_eff
    association, dissociation, and joint R^2
    RMSE and fit diagnostics

Optionally, a long-form CSV containing observed and fitted curves can also be
written.

Example
-------
python fit_mass_transport_kinetics.py \
    --data-root /home/rnt2664/results/260806_nanopore_geometry_sweep \
    --output-csv /home/rnt2664/results/260806_nanopore_geometry_sweep/mass_transport_kinetics.csv \
    --fit-curves-csv /home/rnt2664/results/260806_nanopore_geometry_sweep/mass_transport_fit_curves.csv

Interpretation
--------------
These are effective surface-level parameters. In a spatially resolved,
diffusion-limited simulation they should not automatically be interpreted as
the microscopic input rate constants.

With only one association concentration per replicate, kon and km can be
strongly correlated. Multiple analyte concentrations provide substantially
better identifiability for a mass-transport model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# Broad positive bounds. Optimization is carried out in log10 parameter space.
LOG10_LOWER = np.array([0.0, -8.0, 0.0], dtype=float)
LOG10_UPPER = np.array([12.0, 3.0, 14.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly fit association and dissociation phases to a "
            "mass-transport-limited binding model."
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
            "Optional long-form CSV of observed and fitted theta values for "
            "plotting/diagnostics."
        ),
    )
    parser.add_argument(
        "--association-label",
        default="association",
        help="Value of phase_label identifying association.",
    )
    parser.add_argument(
        "--dissociation-label",
        default="dissociation",
        help="Value of phase_label identifying dissociation.",
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
        default=3000,
        help="Maximum least-squares function evaluations for each start.",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=12,
        help="Number of multistart initial guesses per replicate.",
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


def geometry_metadata(run_directory: Path) -> Dict[str, object]:
    geometry = read_json_if_present(run_directory / "geometry_summary.json")
    manifest = read_json_if_present(run_directory / "manifest.json")

    metadata: Dict[str, object] = {}

    for key in (
        "geometry_name",
        "name",
        "pore_diameter_m",
        "pore_depth_m",
        "pore_pitch_m",
        "pitch_m",
        "rim_z_m",
        "layout",
    ):
        if key in geometry:
            metadata[key] = geometry[key]

    run_metadata = manifest.get("run_metadata", {})
    if isinstance(run_metadata, dict):
        for key in (
            "geometry_name",
            "diameter_m",
            "diameter_nm",
            "pore_diameter_m",
            "pore_depth_m",
            "pitch_m",
            "pore_pitch_m",
            "rim_z_m",
            "layout",
        ):
            if key in run_metadata and key not in metadata:
                metadata[key] = run_metadata[key]

    # Fallback for directories such as diameter_3e-08m.
    if "pore_diameter_m" not in metadata:
        parent_name = run_directory.parent.name
        match = re.search(
            r"diameter[_-]([0-9eE.+-]+)m(?:$|[_-])",
            parent_name,
        )
        if match:
            try:
                metadata["pore_diameter_m"] = float(match.group(1))
            except ValueError:
                pass

    if "pore_diameter_m" in metadata:
        try:
            value = float(metadata["pore_diameter_m"])
            metadata["pore_diameter_m"] = value
            metadata["pore_diameter_nm"] = value * 1e9
        except (TypeError, ValueError):
            pass

    if "pore_depth_m" in metadata:
        try:
            metadata["pore_depth_nm"] = float(metadata["pore_depth_m"]) * 1e9
        except (TypeError, ValueError):
            pass

    return metadata


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


def clean_phase(phase: pd.DataFrame) -> pd.DataFrame:
    data = phase.copy()
    data["t_s"] = pd.to_numeric(data["t_s"], errors="coerce")
    data["theta"] = pd.to_numeric(data["theta"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["t_s", "theta"])
    data = data.sort_values("t_s")

    # Keep one value per time point because solve_ivp requires ordered targets.
    data = data.drop_duplicates(subset="t_s", keep="last").reset_index(drop=True)
    return data


def association_concentration_M(
    association: pd.DataFrame,
    run_directory: Path,
) -> Optional[float]:
    value = first_finite_value(
        association,
        ["phase_ligand_conc_M", "ligand_conc_M"],
    )
    if value is not None:
        return value

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
    Saved microscopic parameters are used ONLY as optimizer starting guesses.
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


def mass_transport_rhs(
    t: float,
    y: np.ndarray,
    *,
    concentration_M: float,
    kon_M_inv_s: float,
    koff_s_inv: float,
    km_M_inv_s: float,
) -> np.ndarray:
    """
    Sartorius-style reduced mass-transport model for Rmax = 1.

    dtheta/dt =
        [kon*C*(1-theta) - koff*theta]
        / [1 + (kon/km)*(1-theta)]
    """
    theta = float(y[0])

    # The optimizer is constrained positive, but guard against numerical issues.
    km_M_inv_s = max(float(km_M_inv_s), np.finfo(float).tiny)

    free_fraction = 1.0 - theta
    denominator = 1.0 + (kon_M_inv_s / km_M_inv_s) * free_fraction

    rate = (
        kon_M_inv_s * concentration_M * free_fraction
        - koff_s_inv * theta
    ) / denominator

    return np.array([rate], dtype=float)


def simulate_phase(
    times_s: np.ndarray,
    theta_initial: float,
    *,
    concentration_M: float,
    kon_M_inv_s: float,
    koff_s_inv: float,
    km_M_inv_s: float,
) -> np.ndarray:
    times_s = np.asarray(times_s, dtype=float)

    if len(times_s) == 0:
        return np.array([], dtype=float)

    tau = times_s - times_s[0]

    if len(times_s) == 1 or tau[-1] == 0:
        return np.full(len(times_s), theta_initial, dtype=float)

    solution = solve_ivp(
        fun=lambda t, y: mass_transport_rhs(
            t,
            y,
            concentration_M=concentration_M,
            kon_M_inv_s=kon_M_inv_s,
            koff_s_inv=koff_s_inv,
            km_M_inv_s=km_M_inv_s,
        ),
        t_span=(0.0, float(tau[-1])),
        y0=np.array([float(theta_initial)], dtype=float),
        t_eval=tau,
        method="LSODA",
        rtol=1e-7,
        atol=1e-10,
    )

    if not solution.success or solution.y.shape[1] != len(times_s):
        raise RuntimeError(
            "ODE integration failed: "
            + (solution.message if solution.message else "unknown error")
        )

    return solution.y[0]


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

    mask = (y > 1e-8) & np.isfinite(y) & np.isfinite(tau)

    if np.count_nonzero(mask) >= 3:
        slope, _ = np.polyfit(tau[mask], np.log(y[mask]), deg=1)
        if np.isfinite(slope) and slope < 0:
            return max(-float(slope), 1e-8)

    duration = max(float(tau[-1]), 1e-6)
    return 1.0 / duration


def simple_assoc_rate_guess(association: pd.DataFrame) -> float:
    t = association["t_s"].to_numpy(dtype=float)
    y = association["theta"].to_numpy(dtype=float)
    tau = t - t[0]

    duration = max(float(tau[-1]), 1e-6)
    y0 = float(y[0])
    yend = float(y[-1])

    # Crude timescale estimate; used only to seed multistart optimization.
    if yend > y0:
        target = y0 + 0.6321205588 * (yend - y0)
        idx = np.flatnonzero(y >= target)
        if len(idx):
            t63 = max(float(tau[idx[0]]), duration / 1000.0)
            return 1.0 / t63

    return 1.0 / duration


def make_initial_guesses(
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

    kobs_assoc_guess = simple_assoc_rate_guess(association)

    if saved_kon is not None and saved_kon > 0:
        kon_guess = float(saved_kon)
    else:
        kinetic_increment = max(
            kobs_assoc_guess - koff_guess,
            0.25 * kobs_assoc_guess,
            1e-6,
        )
        kon_guess = kinetic_increment / concentration_M

    kon_guess = float(
        np.clip(kon_guess, 10 ** LOG10_LOWER[0], 10 ** LOG10_UPPER[0])
    )
    koff_guess = float(
        np.clip(koff_guess, 10 ** LOG10_LOWER[1], 10 ** LOG10_UPPER[1])
    )

    # Sweep transport regimes from strongly transport-limited to nearly
    # reaction-limited. This is important because kon and km can be correlated.
    km_ratios = np.array(
        [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        dtype=float,
    )

    kon_scales = np.array([0.2, 1.0, 5.0], dtype=float)

    candidates: List[np.ndarray] = []

    for kon_scale in kon_scales:
        this_kon = np.clip(
            kon_guess * kon_scale,
            10 ** LOG10_LOWER[0],
            10 ** LOG10_UPPER[0],
        )

        for ratio in km_ratios:
            this_km = np.clip(
                this_kon * ratio,
                10 ** LOG10_LOWER[2],
                10 ** LOG10_UPPER[2],
            )

            candidates.append(
                np.log10([this_kon, koff_guess, this_km])
            )

    # Deduplicate and keep an evenly distributed subset.
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
    Approximate standard errors in log10 parameter space from the Jacobian.
    """
    if fit.jac is None or n_residuals <= n_parameters:
        return np.full(n_parameters, np.nan)

    try:
        jac = np.asarray(fit.jac, dtype=float)
        jtj = jac.T @ jac
        covariance = np.linalg.pinv(jtj)

        residual_variance = (
            2.0 * float(fit.cost) / (n_residuals - n_parameters)
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
            raise KeyError(f"{history_path} is missing required column {column!r}.")

    association = get_phase(history, association_label)
    dissociation = get_phase(history, dissociation_label)

    if len(association) < min_points:
        raise ValueError(
            f"Association has {len(association)} points; need at least {min_points}."
        )

    if len(dissociation) < min_points:
        raise ValueError(
            f"Dissociation has {len(dissociation)} points; need at least {min_points}."
        )

    concentration_M = association_concentration_M(
        association,
        run_directory,
    )

    if concentration_M is None or not np.isfinite(concentration_M):
        raise ValueError("Could not determine association analyte concentration.")

    if concentration_M <= 0:
        raise ValueError(
            f"Association concentration must be > 0; found {concentration_M}."
        )

    t_assoc = association["t_s"].to_numpy(dtype=float)
    y_assoc = association["theta"].to_numpy(dtype=float)

    t_diss = dissociation["t_s"].to_numpy(dtype=float)
    y_diss = dissociation["theta"].to_numpy(dtype=float)

    theta0_assoc = float(y_assoc[0])
    theta0_diss = float(y_diss[0])

    # Keep phase starts as exact anchors rather than residual points.
    # This avoids giving a guaranteed-zero residual extra weight.
    assoc_fit_slice = slice(1, None)
    diss_fit_slice = slice(1, None)

    def predictions(log10_parameters: np.ndarray):
        kon, koff, km = 10.0 ** np.asarray(log10_parameters, dtype=float)

        pred_assoc = simulate_phase(
            t_assoc,
            theta0_assoc,
            concentration_M=concentration_M,
            kon_M_inv_s=kon,
            koff_s_inv=koff,
            km_M_inv_s=km,
        )

        pred_diss = simulate_phase(
            t_diss,
            theta0_diss,
            concentration_M=0.0,
            kon_M_inv_s=kon,
            koff_s_inv=koff,
            km_M_inv_s=km,
        )

        return pred_assoc, pred_diss

    def residuals(log10_parameters: np.ndarray) -> np.ndarray:
        try:
            pred_assoc, pred_diss = predictions(log10_parameters)

            resid = np.concatenate(
                [
                    pred_assoc[assoc_fit_slice] - y_assoc[assoc_fit_slice],
                    pred_diss[diss_fit_slice] - y_diss[diss_fit_slice],
                ]
            )

            if not np.all(np.isfinite(resid)):
                raise FloatingPointError("Non-finite residual.")

            return resid

        except Exception:
            # Large finite penalty keeps least_squares alive for bad trial points.
            n = (len(y_assoc) - 1) + (len(y_diss) - 1)
            return np.full(n, 1e6, dtype=float)

    starts = make_initial_guesses(
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
            x0=np.clip(start, LOG10_LOWER, LOG10_UPPER),
            bounds=(LOG10_LOWER, LOG10_UPPER),
            method="trf",
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        if best_fit is None or fit.cost < best_fit.cost:
            best_fit = fit

    if best_fit is None:
        raise RuntimeError("No optimization attempt was completed.")

    kon, koff, km = 10.0 ** best_fit.x
    pred_assoc, pred_diss = predictions(best_fit.x)

    assoc_r2 = r_squared(y_assoc, pred_assoc)
    diss_r2 = r_squared(y_diss, pred_diss)

    observed_joint = np.concatenate([y_assoc, y_diss])
    predicted_joint = np.concatenate([pred_assoc, pred_diss])

    joint_r2 = r_squared(observed_joint, predicted_joint)
    rmse = float(
        np.sqrt(np.mean((observed_joint - predicted_joint) ** 2))
    )

    log_se = parameter_standard_errors(
        best_fit,
        n_residuals=(len(y_assoc) - 1) + (len(y_diss) - 1),
        n_parameters=3,
    )

    # Delta method: if p = 10^x, sigma_p ~= ln(10) * p * sigma_x.
    param_values = np.array([kon, koff, km], dtype=float)
    param_se = np.log(10.0) * param_values * log_se

    kd_M = koff / kon
    kd_nM = kd_M * 1e9

    result: Dict[str, object] = {
        "replicate": parse_replicate_number(run_directory),
        "run_directory": str(run_directory),
        "history_path": str(history_path),
        "association_concentration_M": concentration_M,
        "association_concentration_nM": concentration_M * 1e9,
        "k_on_eff_M_inv_s": kon,
        "k_on_eff_se_M_inv_s": param_se[0],
        "k_off_eff_s_inv": koff,
        "k_off_eff_se_s_inv": param_se[1],
        "K_D_eff_M": kd_M,
        "K_D_eff_nM": kd_nM,
        "k_m_eff_M_inv_s": km,
        "k_m_eff_se_M_inv_s": param_se[2],
        "k_on_over_k_m": kon / km,
        "association_r_squared": assoc_r2,
        "dissociation_r_squared": diss_r2,
        "joint_r_squared": joint_r2,
        "joint_rmse": rmse,
        "association_n_points": len(association),
        "dissociation_n_points": len(dissociation),
        "association_theta_initial": theta0_assoc,
        "association_theta_final": float(y_assoc[-1]),
        "dissociation_theta_initial": theta0_diss,
        "dissociation_theta_final": float(y_diss[-1]),
        "optimizer_success": bool(best_fit.success),
        "optimizer_status": int(best_fit.status),
        "optimizer_message": str(best_fit.message),
        "optimizer_cost": float(best_fit.cost),
        "optimizer_nfev": int(best_fit.nfev),
        "fit_at_lower_bound": bool(
            np.any(np.isclose(best_fit.x, LOG10_LOWER, atol=1e-4))
        ),
        "fit_at_upper_bound": bool(
            np.any(np.isclose(best_fit.x, LOG10_UPPER, atol=1e-4))
        ),
    }

    result.update(geometry_metadata(run_directory))

    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "replicate": result["replicate"],
                    "phase_label": association_label,
                    "t_s": t_assoc,
                    "theta_observed": y_assoc,
                    "theta_fitted": pred_assoc,
                }
            ),
            pd.DataFrame(
                {
                    "replicate": result["replicate"],
                    "phase_label": dissociation_label,
                    "t_s": t_diss,
                    "theta_observed": y_diss,
                    "theta_fitted": pred_diss,
                }
            ),
        ],
        ignore_index=True,
    )

    for key in (
        "pore_diameter_m",
        "pore_diameter_nm",
        "pore_depth_m",
        "pore_depth_nm",
        "layout",
    ):
        if key in result:
            curves[key] = result[key]

    curves["run_directory"] = str(run_directory)

    return result, curves


def failure_row(history_path: Path, error: Exception) -> Dict[str, object]:
    run_directory = history_path.parent

    row: Dict[str, object] = {
        "replicate": parse_replicate_number(run_directory),
        "run_directory": str(run_directory),
        "history_path": str(history_path),
        "optimizer_success": False,
        "analysis_error": f"{type(error).__name__}: {error}",
    }

    row.update(geometry_metadata(run_directory))
    return row


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    history_paths = sorted(data_root.rglob("history.parquet"))

    if not history_paths:
        raise FileNotFoundError(
            f"No history.parquet files found under {data_root}"
        )

    print(f"Found {len(history_paths)} replicate history files.")

    rows: List[Dict[str, object]] = []
    curve_frames: List[pd.DataFrame] = []

    for index, history_path in enumerate(history_paths, start=1):
        print(f"[{index}/{len(history_paths)}] {history_path.parent}")

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
                f"kon={result['k_on_eff_M_inv_s']:.4g} M^-1 s^-1, "
                f"koff={result['k_off_eff_s_inv']:.4g} s^-1, "
                f"KD={result['K_D_eff_nM']:.4g} nM, "
                f"km={result['k_m_eff_M_inv_s']:.4g}, "
                f"R2_assoc={result['association_r_squared']:.4f}, "
                f"R2_diss={result['dissociation_r_squared']:.4f}"
            )

        except Exception as exc:
            rows.append(failure_row(history_path, exc))
            print(f"    FAILED: {type(exc).__name__}: {exc}")

    results = pd.DataFrame(rows)

    sort_columns = [
        column
        for column in ("pore_diameter_m", "replicate")
        if column in results.columns
    ]

    if sort_columns:
        results = results.sort_values(sort_columns).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    print()
    print(f"Saved parameter table: {output_csv}")

    if args.fit_curves_csv is not None:
        fit_curves_csv = args.fit_curves_csv.expanduser().resolve()
        fit_curves_csv.parent.mkdir(parents=True, exist_ok=True)

        if curve_frames:
            curves = pd.concat(curve_frames, ignore_index=True)

            curve_sort = [
                column
                for column in (
                    "pore_diameter_m",
                    "replicate",
                    "phase_label",
                    "t_s",
                )
                if column in curves.columns
            ]

            if curve_sort:
                curves = curves.sort_values(curve_sort)

            curves.to_csv(fit_curves_csv, index=False)
            print(f"Saved fitted curves:   {fit_curves_csv}")
        else:
            print("No successful fits; fitted-curves CSV was not written.")

    n_success = int(results.get("optimizer_success", pd.Series(dtype=bool)).fillna(False).sum())
    print(f"Successful fits: {n_success}/{len(results)}")


if __name__ == "__main__":
    main()
