#!/usr/bin/env python3
"""
Fit effective association/dissociation kinetics from saved biosensor simulations.

For each replicate, this script:

1. Loads history.parquet.
2. Identifies association and dissociation phases from `phase_label`.
3. Fits each phase to a single-exponential relaxation:

       theta(t) = theta_inf + (theta_0 - theta_inf) * exp(-k_obs * tau)

   where tau is time elapsed since the start of the phase.

4. Uses the standard Langmuir kinetic relationships:

       k_assoc_obs = k_on_eff * C_assoc + k_off_eff
       k_off_eff   = k_diss_obs

   therefore:

       k_on_eff = (k_assoc_obs - k_off_eff) / C_assoc
       K_D_eff  = k_off_eff / k_on_eff

5. Writes one CSV row per replicate, including fit quality and geometry metadata.

Important interpretation
------------------------
These are EFFECTIVE kinetic parameters of the simulated surface. In a
transport-limited, geometry-dependent, or rebinding-dominated system they need
not equal the microscopic input k_on and k_off values.

Example
-------
python fit_effective_kinetics.py \
    --data-root /path/to/results/260806_nanopore_geometry_sweep \
    --output-csv /path/to/results/effective_kinetics.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit association/dissociation exponentials for each saved replicate "
            "and calculate effective k_on, k_off, and K_D."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing replicate folders with history.parquet files.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV file.",
    )

    parser.add_argument(
        "--association-label",
        type=str,
        default="association",
        help="phase_label used for the association phase.",
    )

    parser.add_argument(
        "--dissociation-label",
        type=str,
        default="dissociation",
        help="phase_label used for the dissociation phase.",
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum number of finite history points required for each fit.",
    )

    return parser.parse_args()


def read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_replicate_number(run_directory: Path) -> Optional[int]:
    match = re.search(r"replicate[_-]?(\d+)", run_directory.name, flags=re.I)
    return int(match.group(1)) if match else None


def first_finite_value(
    frame: pd.DataFrame,
    candidate_columns: Iterable[str],
) -> Optional[float]:
    for column in candidate_columns:
        if column not in frame.columns:
            continue

        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values)]

        if not values.empty:
            return float(values.iloc[0])

    return None


def exponential_relaxation(
    tau_s: np.ndarray,
    theta_inf: float,
    k_obs_s_inv: float,
    theta_0: float,
) -> np.ndarray:
    return theta_inf + (theta_0 - theta_inf) * np.exp(
        -k_obs_s_inv * tau_s
    )


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual_ss = float(np.sum((y_true - y_pred) ** 2))
    total_ss = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if total_ss == 0:
        return np.nan

    return 1.0 - residual_ss / total_ss


def fit_phase(
    phase: pd.DataFrame,
    *,
    min_points: int,
) -> Dict[str, float]:
    """
    Fit theta(t) to a single exponential relaxation.

    theta_0 is fixed to the first observed point of the phase. This reduces
    parameter degeneracy and preserves continuity with the simulation history.
    """
    data = phase[["t_s", "theta"]].copy()
    data["t_s"] = pd.to_numeric(data["t_s"], errors="coerce")
    data["theta"] = pd.to_numeric(data["theta"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values("t_s")

    if len(data) < min_points:
        raise ValueError(
            f"Need at least {min_points} finite points; found {len(data)}."
        )

    time_s = data["t_s"].to_numpy(dtype=float)
    theta = data["theta"].to_numpy(dtype=float)

    tau_s = time_s - time_s[0]

    if tau_s[-1] <= 0:
        raise ValueError("Phase duration must be greater than zero.")

    theta_0 = float(theta[0])
    theta_last = float(theta[-1])

    # Characteristic initial rate guess based on the phase duration.
    k_guess = max(1.0 / tau_s[-1], 1e-12)

    # Occupancy is physically constrained to [0, 1]. k_obs must be nonnegative.
    initial_guess = [
        float(np.clip(theta_last, 0.0, 1.0)),
        k_guess,
    ]

    def model(tau, theta_inf, k_obs):
        return exponential_relaxation(
            tau,
            theta_inf=theta_inf,
            k_obs_s_inv=k_obs,
            theta_0=theta_0,
        )

    popt, pcov = curve_fit(
        model,
        tau_s,
        theta,
        p0=initial_guess,
        bounds=([0.0, 0.0], [1.0, np.inf]),
        maxfev=20000,
    )

    theta_inf, k_obs = map(float, popt)
    predicted = model(tau_s, theta_inf, k_obs)

    parameter_se = np.sqrt(np.diag(pcov))
    theta_inf_se = (
        float(parameter_se[0])
        if len(parameter_se) > 0 and np.isfinite(parameter_se[0])
        else np.nan
    )
    k_obs_se = (
        float(parameter_se[1])
        if len(parameter_se) > 1 and np.isfinite(parameter_se[1])
        else np.nan
    )

    residuals = theta - predicted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "fit_success": True,
        "n_points": int(len(data)),
        "phase_start_t_s": float(time_s[0]),
        "phase_end_t_s": float(time_s[-1]),
        "phase_duration_s": float(tau_s[-1]),
        "theta_0": theta_0,
        "theta_inf": theta_inf,
        "theta_inf_se": theta_inf_se,
        "k_obs_s_inv": k_obs,
        "k_obs_se_s_inv": k_obs_se,
        "tau_s": (1.0 / k_obs) if k_obs > 0 else np.inf,
        "r_squared": r_squared(theta, predicted),
        "rmse": rmse,
    }


def failed_fit(reason: str) -> Dict[str, object]:
    return {
        "fit_success": False,
        "fit_error": reason,
        "n_points": np.nan,
        "phase_start_t_s": np.nan,
        "phase_end_t_s": np.nan,
        "phase_duration_s": np.nan,
        "theta_0": np.nan,
        "theta_inf": np.nan,
        "theta_inf_se": np.nan,
        "k_obs_s_inv": np.nan,
        "k_obs_se_s_inv": np.nan,
        "tau_s": np.nan,
        "r_squared": np.nan,
        "rmse": np.nan,
    }


def get_phase(history: pd.DataFrame, label: str) -> pd.DataFrame:
    if "phase_label" not in history.columns:
        raise KeyError(
            "history.parquet does not contain 'phase_label'; "
            "phase-specific fitting cannot be performed."
        )

    phase = history.loc[
        history["phase_label"].astype(str).str.lower() == label.lower()
    ].copy()

    if phase.empty:
        raise ValueError(f"No history rows found for phase_label={label!r}.")

    return phase


def geometry_metadata(run_directory: Path) -> Dict[str, object]:
    """
    Read useful geometry metadata from saved JSON files and directory names.
    """
    geometry = read_json_if_present(run_directory / "geometry_summary.json")
    manifest = read_json_if_present(run_directory / "manifest.json")

    metadata: Dict[str, object] = {}

    # Prefer explicit saved geometry metadata.
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

    # Some save implementations put run metadata in manifest.json.
    run_metadata = manifest.get("run_metadata", {})
    if isinstance(run_metadata, dict):
        for key in (
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

    # Fall back to directory naming, e.g. diameter_3e-08m.
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
            diameter_m = float(metadata["pore_diameter_m"])
            metadata["pore_diameter_m"] = diameter_m
            metadata["pore_diameter_nm"] = diameter_m * 1e9
        except (TypeError, ValueError):
            pass

    if "pore_depth_m" in metadata:
        try:
            metadata["pore_depth_nm"] = float(metadata["pore_depth_m"]) * 1e9
        except (TypeError, ValueError):
            pass

    return metadata


def extract_association_concentration_M(
    association: pd.DataFrame,
    run_directory: Path,
) -> Optional[float]:
    concentration = first_finite_value(
        association,
        [
            "phase_ligand_conc_M",
            "ligand_conc_M",
        ],
    )

    if concentration is not None:
        return concentration

    parameters = read_json_if_present(run_directory / "parameters.json")

    value = parameters.get("ligand_conc_M")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    manifest = read_json_if_present(run_directory / "manifest.json")
    run_metadata = manifest.get("run_metadata", {})

    if isinstance(run_metadata, dict):
        association_params = run_metadata.get("association_params", {})
        if isinstance(association_params, dict):
            value = association_params.get("ligand_conc_M")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    return None


def analyze_replicate(
    history_path: Path,
    *,
    association_label: str,
    dissociation_label: str,
    min_points: int,
) -> Dict[str, object]:
    run_directory = history_path.parent
    history = pd.read_parquet(history_path)

    required = {"t_s", "theta"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(
            f"{history_path} is missing required columns: {sorted(missing)}"
        )

    replicate = parse_replicate_number(run_directory)

    result: Dict[str, object] = {
        "replicate": replicate,
        "run_directory": str(run_directory),
        "history_path": str(history_path),
    }

    result.update(geometry_metadata(run_directory))

    try:
        association = get_phase(history, association_label)
    except Exception as exc:
        association = pd.DataFrame()
        association_fit = failed_fit(str(exc))
    else:
        try:
            association_fit = fit_phase(
                association,
                min_points=min_points,
            )
        except Exception as exc:
            association_fit = failed_fit(str(exc))

    try:
        dissociation = get_phase(history, dissociation_label)
    except Exception as exc:
        dissociation = pd.DataFrame()
        dissociation_fit = failed_fit(str(exc))
    else:
        try:
            dissociation_fit = fit_phase(
                dissociation,
                min_points=min_points,
            )
        except Exception as exc:
            dissociation_fit = failed_fit(str(exc))

    concentration_M = (
        extract_association_concentration_M(association, run_directory)
        if not association.empty
        else None
    )

    k_assoc_obs = association_fit["k_obs_s_inv"]
    k_off_eff = dissociation_fit["k_obs_s_inv"]

    k_on_eff = np.nan
    kd_eff_M = np.nan

    if (
        association_fit["fit_success"]
        and dissociation_fit["fit_success"]
        and concentration_M is not None
        and np.isfinite(concentration_M)
        and concentration_M > 0
        and np.isfinite(k_assoc_obs)
        and np.isfinite(k_off_eff)
    ):
        k_on_eff = (float(k_assoc_obs) - float(k_off_eff)) / float(
            concentration_M
        )

        # A negative value means the simple Langmuir decomposition is not
        # physically compatible with the fitted rates.
        if k_on_eff <= 0:
            k_on_eff = np.nan
        else:
            kd_eff_M = float(k_off_eff) / k_on_eff

    result.update(
        {
            "association_concentration_M": concentration_M,
            "association_concentration_nM": (
                concentration_M * 1e9
                if concentration_M is not None
                else np.nan
            ),
            "k_assoc_obs_s_inv": association_fit["k_obs_s_inv"],
            "k_assoc_obs_se_s_inv": association_fit["k_obs_se_s_inv"],
            "k_off_eff_s_inv": k_off_eff,
            "k_off_eff_se_s_inv": dissociation_fit["k_obs_se_s_inv"],
            "k_on_eff_M_inv_s": k_on_eff,
            "K_D_eff_M": kd_eff_M,
            "K_D_eff_nM": (
                kd_eff_M * 1e9
                if np.isfinite(kd_eff_M)
                else np.nan
            ),
            "association_r_squared": association_fit["r_squared"],
            "dissociation_r_squared": dissociation_fit["r_squared"],
            "association_rmse": association_fit["rmse"],
            "dissociation_rmse": dissociation_fit["rmse"],
            "association_theta_0": association_fit["theta_0"],
            "association_theta_inf": association_fit["theta_inf"],
            "dissociation_theta_0": dissociation_fit["theta_0"],
            "dissociation_theta_inf": dissociation_fit["theta_inf"],
            "association_n_points": association_fit["n_points"],
            "dissociation_n_points": dissociation_fit["n_points"],
            "association_fit_success": association_fit["fit_success"],
            "dissociation_fit_success": dissociation_fit["fit_success"],
            "association_fit_error": association_fit.get("fit_error", ""),
            "dissociation_fit_error": dissociation_fit.get("fit_error", ""),
        }
    )

    return result


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

    rows = []

    for index, history_path in enumerate(history_paths, start=1):
        print(
            f"[{index}/{len(history_paths)}] "
            f"Fitting {history_path.parent}"
        )

        try:
            row = analyze_replicate(
                history_path,
                association_label=args.association_label,
                dissociation_label=args.dissociation_label,
                min_points=args.min_points,
            )
        except Exception as exc:
            row = {
                "replicate": parse_replicate_number(history_path.parent),
                "run_directory": str(history_path.parent),
                "history_path": str(history_path),
                "analysis_error": str(exc),
            }
            row.update(geometry_metadata(history_path.parent))

        rows.append(row)

    results = pd.DataFrame(rows)

    sort_columns = [
        column
        for column in ("pore_diameter_m", "replicate")
        if column in results.columns
    ]

    if sort_columns:
        results = results.sort_values(sort_columns)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    successful = 0
    if {
        "association_fit_success",
        "dissociation_fit_success",
    }.issubset(results.columns):
        successful = int(
            (
                results["association_fit_success"].fillna(False)
                & results["dissociation_fit_success"].fillna(False)
            ).sum()
        )

    print()
    print(f"Saved: {output_csv}")
    print(f"Rows: {len(results)}")
    print(f"Replicates with both fits successful: {successful}")


if __name__ == "__main__":
    main()