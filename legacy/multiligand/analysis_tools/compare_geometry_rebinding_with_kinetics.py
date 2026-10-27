#!/usr/bin/env python3
"""
General rebinding analysis for saved biosensor parameter sweeps.

This script recursively scans one simulation result tree, automatically
identifies the swept parameter, extracts association/dissociation/total
rebinding counts for every replicate, and plots rebinding versus the swept
parameter.

Sweep detection priority
------------------------
1. manifest.json -> run_metadata.sweep_parameter / sweep_value
2. manifest.json -> run_metadata.sweep_variable and matching value
3. known parameter-sweep directory naming conventions
4. legacy nanopore geometry metadata
5. association concentration from history
6. generic parent-directory condition label

Recognized sweep families include:
    pore_diameter_m
    pore_depth_m / pore height
    association_concentration_M
    receptor_density_m2
    spike_density_m2
    branch_levels
    branches_per_segment_mean

The rebinding counter is cumulative across resumed multi-phase simulations.
Phase-specific counts are therefore calculated from cumulative-counter
differences.

Notebook example
----------------
from analyze_rebinding_sweep import (
    load_rebinding_sweep,
    summarize_rebinding,
    plot_rebinding_sweep,
)

data = load_rebinding_sweep(
    "/path/to/sweep/results"
)

fig, axes = plot_rebinding_sweep(data)

Command-line example
--------------------
python analyze_rebinding_sweep.py \
    --data-root /path/to/sweep/results \
    --output rebinding_vs_sweep.png \
    --summary-csv rebinding_summary.csv \
    --replicate-csv rebinding_replicates.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REBINDING_COLUMN_CANDIDATES = (
    "rebinding_events_total",
    "rebindings_total",
)

CONCENTRATION_COLUMN_CANDIDATES = (
    "phase_ligand_conc_M",
    "ligand_conc_M",
    "association_concentration_M",
)

SWEEP_PARAMETER_INFO = {
    "pore_diameter_m": {
        "label": "Pore diameter",
        "unit": "m",
        "display_scale": 1e9,
        "display_unit": "nm",
        "kind": "numeric",
    },
    "pore_depth_m": {
        "label": "Pore height/depth",
        "unit": "m",
        "display_scale": 1e9,
        "display_unit": "nm",
        "kind": "numeric",
    },
    "pore_height_m": {
        "label": "Pore height",
        "unit": "m",
        "display_scale": 1e9,
        "display_unit": "nm",
        "kind": "numeric",
    },
    "association_concentration_M": {
        "label": "Association concentration",
        "unit": "M",
        "display_scale": 1e9,
        "display_unit": "nM",
        "kind": "numeric",
    },
    "ligand_conc_M": {
        "label": "Ligand concentration",
        "unit": "M",
        "display_scale": 1e9,
        "display_unit": "nM",
        "kind": "numeric",
    },
    "receptor_density_m2": {
        "label": "Receptor density",
        "unit": "m^-2",
        "display_scale": 1.0,
        "display_unit": r"m$^{-2}$",
        "kind": "numeric",
    },
    "spike_density_m2": {
        "label": "Primary spike density",
        "unit": "m^-2",
        "display_scale": 1.0,
        "display_unit": r"m$^{-2}$",
        "kind": "numeric",
    },
    "branch_levels": {
        "label": "Branch levels",
        "unit": "",
        "display_scale": 1.0,
        "display_unit": "",
        "kind": "integer",
    },
    "branches_per_segment_mean": {
        "label": "Branches per segment",
        "unit": "",
        "display_scale": 1.0,
        "display_unit": "",
        "kind": "numeric",
    },
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    return value if isinstance(value, dict) else {}


def _find_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            "Could not find any expected column: "
            + ", ".join(candidates)
        )

    return None


def _parse_replicate(run_directory: Path) -> Optional[int]:
    match = re.search(
        r"replicate[_-]?(\d+)",
        run_directory.name,
        flags=re.I,
    )
    return int(match.group(1)) if match else None


def _first_finite_value(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> Optional[float]:
    for column in columns:
        if column not in frame.columns:
            continue

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        values = values[
            np.isfinite(values.to_numpy(dtype=float))
        ]

        if not values.empty:
            return float(values.iloc[0])

    return None


def _association_concentration(
    history: pd.DataFrame,
    run_directory: Path,
) -> Optional[float]:
    if "phase_label" in history.columns:
        association = history.loc[
            history["phase_label"]
            .astype(str)
            .str.lower()
            .eq("association")
        ]
    else:
        association = history

    value = _first_finite_value(
        association,
        CONCENTRATION_COLUMN_CANDIDATES,
    )
    if value is not None:
        return value

    manifest = _read_json(
        run_directory / "manifest.json"
    )
    run_metadata = manifest.get(
        "run_metadata",
        {},
    )

    if isinstance(run_metadata, dict):
        for key in (
            "association_concentration_M",
            "ligand_conc_M",
        ):
            value = run_metadata.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

        association_params = run_metadata.get(
            "association_params",
            {},
        )
        if isinstance(association_params, dict):
            value = association_params.get(
                "ligand_conc_M"
            )
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    parameters = _read_json(
        run_directory / "parameters.json"
    )
    value = parameters.get("ligand_conc_M")

    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    return None


def _clean_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    if "t_s" not in history.columns:
        raise KeyError(
            "History does not contain 't_s'."
        )

    rebinding_column = _find_column(
        history,
        REBINDING_COLUMN_CANDIDATES,
    )

    keep = [
        "t_s",
        rebinding_column,
    ]

    if "phase_label" in history.columns:
        keep.append("phase_label")

    for column in CONCENTRATION_COLUMN_CANDIDATES:
        if (
            column in history.columns
            and column not in keep
        ):
            keep.append(column)

    data = history[keep].copy()

    data["t_s"] = pd.to_numeric(
        data["t_s"],
        errors="coerce",
    )

    data[rebinding_column] = pd.to_numeric(
        data[rebinding_column],
        errors="coerce",
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=["t_s", rebinding_column],
    )

    data = data.sort_values(
        "t_s"
    ).reset_index(drop=True)

    return data


def rebinding_counts_from_history(
    history: pd.DataFrame,
) -> Dict[str, float]:
    """
    Extract association, dissociation, and total rebinding counts.

    ``rebinding_events_total`` is cumulative across the full resumed run.
    """
    history = _clean_history(history)

    column = _find_column(
        history,
        REBINDING_COLUMN_CANDIDATES,
    )

    cumulative = history[column].to_numpy(
        dtype=float
    )

    total = float(
        cumulative[-1] - cumulative[0]
    )

    association = np.nan
    dissociation = np.nan

    if "phase_label" in history.columns:
        labels = (
            history["phase_label"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        association_rows = history.loc[
            labels.eq("association")
        ]
        dissociation_rows = history.loc[
            labels.eq("dissociation")
        ]

        if not association_rows.empty:
            association_values = (
                association_rows[column]
                .to_numpy(dtype=float)
            )
            association = float(
                association_values[-1]
                - association_values[0]
            )

        if not dissociation_rows.empty:
            dissociation_values = (
                dissociation_rows[column]
                .to_numpy(dtype=float)
            )
            dissociation = float(
                dissociation_values[-1]
                - dissociation_values[0]
            )

            if not association_rows.empty:
                association_end = float(
                    association_rows[column].iloc[-1]
                )
                dissociation = float(
                    cumulative[-1]
                    - association_end
                )

    return {
        "association_rebinding_events": association,
        "dissociation_rebinding_events": dissociation,
        "total_rebinding_events": total,
    }


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if np.isfinite(number) else None


def _manifest_run_metadata(
    run_directory: Path,
) -> Dict[str, Any]:
    manifest = _read_json(
        run_directory / "manifest.json"
    )
    metadata = manifest.get(
        "run_metadata",
        {},
    )
    return metadata if isinstance(metadata, dict) else {}


def _metadata_sweep_identity(
    run_directory: Path,
) -> Optional[Tuple[str, Any]]:
    metadata = _manifest_run_metadata(
        run_directory
    )

    parameter = metadata.get(
        "sweep_parameter"
    )

    if (
        parameter is not None
        and "sweep_value" in metadata
    ):
        return str(parameter), metadata["sweep_value"]

    sweep_variable = metadata.get(
        "sweep_variable"
    )

    if sweep_variable is not None:
        parameter = str(sweep_variable)

        if parameter in metadata:
            return parameter, metadata[parameter]

        aliases = {
            "pore_depth_m": (
                "pore_depth_m",
                "pore_height_m",
                "pore_height_nm",
            ),
            "pore_diameter_m": (
                "pore_diameter_m",
                "pore_diameter_nm",
                "diameter_m",
                "diameter_nm",
            ),
            "association_concentration_M": (
                "association_concentration_M",
                "ligand_conc_M",
            ),
            "receptor_density_m2": (
                "receptor_density_m2",
            ),
            "spike_density_m2": (
                "spike_density_m2",
            ),
            "branch_levels": (
                "branch_levels",
            ),
            "branches_per_segment_mean": (
                "branches_per_segment_mean",
            ),
        }

        for candidate in aliases.get(
            parameter,
            (),
        ):
            if candidate not in metadata:
                continue

            value = metadata[candidate]

            if (
                candidate.endswith("_nm")
                and parameter.endswith("_m")
            ):
                number = _finite_float(value)
                if number is not None:
                    value = number * 1e-9

            return parameter, value

    return None


def _directory_sweep_identity(
    run_directory: Path,
) -> Optional[Tuple[str, Any]]:
    candidate_names = [
        run_directory.parent.name,
        *[
            parent.name
            for parent in run_directory.parents
        ][:3],
    ]

    patterns = [
        (
            r"^diameter_([0-9eE.+-]+)m$",
            "pore_diameter_m",
            lambda x: float(x),
        ),
        (
            r"^height_([0-9eE.+-]+)nm$",
            "pore_depth_m",
            lambda x: float(x) * 1e-9,
        ),
        (
            r"^concentration_([0-9eE.+-]+)M$",
            "association_concentration_M",
            lambda x: float(x),
        ),
        (
            r"^density_([0-9eE.+-]+)_per_m2$",
            "receptor_density_m2",
            lambda x: float(x),
        ),
        (
            r"^receptor_density_m2_([0-9eE.+-]+)$",
            "receptor_density_m2",
            lambda x: float(x),
        ),
        (
            r"^spike_density_m2_([0-9eE.+-]+)$",
            "spike_density_m2",
            lambda x: float(x),
        ),
        (
            r"^branch_levels_([0-9eE.+-]+)$",
            "branch_levels",
            lambda x: int(round(float(x))),
        ),
        (
            r"^branches_per_segment_mean_([0-9eE.+-]+)$",
            "branches_per_segment_mean",
            lambda x: float(x),
        ),
    ]

    for name in candidate_names:
        for pattern, parameter, converter in patterns:
            match = re.fullmatch(
                pattern,
                name,
                flags=re.I,
            )

            if match:
                return (
                    parameter,
                    converter(match.group(1)),
                )

    return None


def _legacy_geometry_sweep_identity(
    run_directory: Path,
) -> Optional[Tuple[str, Any]]:
    summary = _read_json(
        run_directory / "geometry_summary.json"
    )

    if "pore_diameter_m" in summary:
        value = _finite_float(
            summary["pore_diameter_m"]
        )
        if value is not None:
            return "pore_diameter_m", value

    return None


def _history_sweep_identity(
    history: pd.DataFrame,
) -> Optional[Tuple[str, Any]]:
    concentration = _first_finite_value(
        history,
        CONCENTRATION_COLUMN_CANDIDATES,
    )

    if concentration is not None:
        return (
            "association_concentration_M",
            concentration,
        )

    return None


def _detect_sweep_identity(
    run_directory: Path,
    history: pd.DataFrame,
) -> Tuple[str, Any]:
    for detector in (
        lambda: _metadata_sweep_identity(
            run_directory
        ),
        lambda: _directory_sweep_identity(
            run_directory
        ),
        lambda: _legacy_geometry_sweep_identity(
            run_directory
        ),
        lambda: _history_sweep_identity(
            history
        ),
    ):
        result = detector()

        if result is not None:
            return result

    return (
        "condition",
        run_directory.parent.name,
    )


def _sweep_info(
    parameter: str,
) -> Dict[str, Any]:
    return SWEEP_PARAMETER_INFO.get(
        parameter,
        {
            "label": (
                parameter
                .replace("_", " ")
                .strip()
                .capitalize()
            ),
            "unit": "",
            "display_scale": 1.0,
            "display_unit": "",
            "kind": "numeric",
        },
    )


def _display_sweep_value(
    parameter: str,
    value: Any,
) -> Any:
    info = _sweep_info(parameter)
    number = _finite_float(value)

    if number is None:
        return value

    if info["kind"] == "integer":
        return int(round(number))

    return number * float(
        info["display_scale"]
    )


def _sweep_label(
    parameter: str,
) -> str:
    info = _sweep_info(parameter)
    unit = str(info["display_unit"])

    if unit:
        return f"{info['label']} ({unit})"

    return str(info["label"])


def _metadata_columns(
    run_directory: Path,
) -> Dict[str, Any]:
    """
    Propagate useful geometry/run metadata into the replicate table.
    """
    metadata = _manifest_run_metadata(
        run_directory
    )

    keys = (
        "geometry_type",
        "geometry_name",
        "pore_diameter_m",
        "pore_depth_m",
        "pore_pitch_m",
        "rim_z_m",
        "layout",
        "receptor_density_m2",
        "actual_receptor_density_m2",
        "spike_density_m2",
        "branch_levels",
        "branches_per_segment_mean",
        "geometry_seed",
        "nanospike_primary_count",
        "nanospike_segment_count",
        "nanospike_canopy_z_m",
        "roughness_factor",
        "true_exposed_area_m2",
        "projected_area_m2",
        "reservoir_interface_z_m",
    )

    result = {
        key: metadata[key]
        for key in keys
        if key in metadata
    }

    return result


def load_rebinding_sweep(
    data_root: str | Path,
    *,
    sweep_parameter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load one saved parameter sweep and extract per-replicate rebinding counts.

    Parameters
    ----------
    data_root
        Root directory containing condition/replicate result folders.

    sweep_parameter
        Optional manual override. Normally this should be omitted because the
        script detects the sweep automatically from saved run metadata.

    Returns
    -------
    pd.DataFrame
        One row per replicate, with canonical columns:
            sweep_parameter
            sweep_parameter_label
            sweep_value
            sweep_value_numeric
            sweep_value_display
            replicate
            association_concentration_M
            association_rebinding_events
            dissociation_rebinding_events
            total_rebinding_events
    """
    data_root = (
        Path(data_root)
        .expanduser()
        .resolve()
    )

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root does not exist: {data_root}"
        )

    history_paths = sorted(
        list(data_root.rglob("history.parquet"))
        + list(data_root.rglob("history.csv"))
    )

    if not history_paths:
        raise FileNotFoundError(
            f"No history.parquet or history.csv files found under {data_root}"
        )

    rows: List[dict] = []

    for history_path in history_paths:
        run_directory = history_path.parent

        if history_path.suffix.lower() == ".parquet":
            history = pd.read_parquet(
                history_path
            )
        else:
            history = pd.read_csv(
                history_path
            )

        detected_parameter, detected_value = (
            _detect_sweep_identity(
                run_directory,
                history,
            )
        )

        if sweep_parameter is not None:
            requested = str(sweep_parameter)

            if detected_parameter != requested:
                metadata = _manifest_run_metadata(
                    run_directory
                )

                if requested in metadata:
                    detected_parameter = requested
                    detected_value = metadata[
                        requested
                    ]
                else:
                    raise ValueError(
                        f"Manual sweep parameter {requested!r} was requested, "
                        f"but no value could be found for {run_directory}."
                    )

        concentration_M = (
            _association_concentration(
                history,
                run_directory,
            )
        )

        counts = rebinding_counts_from_history(
            history
        )

        numeric_value = _finite_float(
            detected_value
        )

        row = {
            "sweep_parameter": (
                detected_parameter
            ),
            "sweep_parameter_label": (
                _sweep_info(
                    detected_parameter
                )["label"]
            ),
            "sweep_parameter_unit": (
                _sweep_info(
                    detected_parameter
                )["unit"]
            ),
            "sweep_value": detected_value,
            "sweep_value_numeric": (
                numeric_value
            ),
            "sweep_value_display": (
                _display_sweep_value(
                    detected_parameter,
                    detected_value,
                )
            ),
            "replicate": _parse_replicate(
                run_directory
            ),
            "association_concentration_M": (
                concentration_M
            ),
            "run_directory": str(
                run_directory
            ),
            "history_path": str(
                history_path
            ),
            **counts,
            **_metadata_columns(
                run_directory
            ),
        }

        rows.append(row)

    data = pd.DataFrame(rows)

    parameters = (
        data["sweep_parameter"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(parameters) != 1:
        raise ValueError(
            "The result tree contains more than one detected sweep parameter: "
            f"{parameters}. Analyze each sweep root separately or pass a more "
            "specific data_root."
        )

    sort_columns = [
        column
        for column in (
            "sweep_value_numeric",
            "sweep_value_display",
            "replicate",
        )
        if column in data.columns
    ]

    return data.sort_values(
        sort_columns,
        na_position="last",
    ).reset_index(drop=True)


def _phase_column(
    phase: str,
) -> str:
    mapping = {
        "association": (
            "association_rebinding_events"
        ),
        "dissociation": (
            "dissociation_rebinding_events"
        ),
        "total": (
            "total_rebinding_events"
        ),
    }

    try:
        return mapping[phase.lower()]
    except KeyError as exc:
        raise ValueError(
            "phase must be 'association', "
            "'dissociation', or 'total'."
        ) from exc


def summarize_rebinding(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean, SD, SEM, and n for each sweep value and phase.
    """
    required = {
        "sweep_parameter",
        "sweep_value",
        "sweep_value_display",
    }

    missing = required.difference(
        data.columns
    )

    if missing:
        raise KeyError(
            "Data are missing required sweep column(s): "
            + ", ".join(sorted(missing))
        )

    rows = []

    grouping = (
        data[
            [
                "sweep_parameter",
                "sweep_value",
                "sweep_value_numeric",
                "sweep_value_display",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "sweep_value_numeric",
                "sweep_value_display",
            ],
            na_position="last",
        )
    )

    for condition in grouping.itertuples(
        index=False
    ):
        subset = data.loc[
            data["sweep_value"].astype(str)
            == str(condition.sweep_value)
        ]

        for phase in (
            "association",
            "dissociation",
            "total",
        ):
            column = _phase_column(phase)

            values = pd.to_numeric(
                subset[column],
                errors="coerce",
            ).to_numpy(dtype=float)

            values = values[
                np.isfinite(values)
            ]

            n = len(values)

            if n == 0:
                mean = sd = sem = np.nan
            elif n == 1:
                mean = float(values[0])
                sd = sem = 0.0
            else:
                mean = float(
                    np.mean(values)
                )
                sd = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                sem = sd / np.sqrt(n)

            rows.append(
                {
                    "sweep_parameter": (
                        condition.sweep_parameter
                    ),
                    "sweep_value": (
                        condition.sweep_value
                    ),
                    "sweep_value_numeric": (
                        condition.sweep_value_numeric
                    ),
                    "sweep_value_display": (
                        condition.sweep_value_display
                    ),
                    "phase": phase,
                    "mean": mean,
                    "sd": sd,
                    "sem": sem,
                    "n": n,
                }
            )

    return pd.DataFrame(rows)


def _ordered_conditions(
    data: pd.DataFrame,
) -> pd.DataFrame:
    return (
        data[
            [
                "sweep_value",
                "sweep_value_numeric",
                "sweep_value_display",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "sweep_value_numeric",
                "sweep_value_display",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def plot_rebinding_parameter(
    data: pd.DataFrame,
    *,
    phase: str = "total",
    ax=None,
    show_replicates: bool = True,
    show_sem: bool = True,
    point_jitter: float = 0.015,
    log_x: bool = False,
    log_y: bool = False,
):
    """
    Plot one rebinding phase against the automatically detected sweep parameter.
    """
    if ax is None:
        _, ax = plt.subplots(
            figsize=(7.5, 5.2)
        )

    parameter = str(
        data["sweep_parameter"].iloc[0]
    )

    column = _phase_column(phase)

    conditions = _ordered_conditions(
        data
    )

    summary = summarize_rebinding(
        data
    )
    summary = summary.loc[
        summary["phase"].eq(
            phase.lower()
        )
    ]

    numeric_axis = (
        conditions["sweep_value_numeric"]
        .notna()
        .all()
    )

    rng = np.random.default_rng(0)

    for index, condition in enumerate(
        conditions.itertuples(index=False)
    ):
        subset = data.loc[
            data["sweep_value"].astype(str)
            == str(condition.sweep_value)
        ]

        values = pd.to_numeric(
            subset[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        values = values[
            np.isfinite(values)
        ]

        if not len(values):
            continue

        match = summary.loc[
            summary["sweep_value"].astype(str)
            == str(condition.sweep_value)
        ]

        mean = float(
            match["mean"].iloc[0]
        )
        sem = (
            float(
                match["sem"].iloc[0]
            )
            if show_sem
            else 0.0
        )

        if numeric_axis:
            x = float(
                condition.sweep_value_display
            )

            if len(conditions) > 1:
                x_values = pd.to_numeric(
                    conditions[
                        "sweep_value_display"
                    ],
                    errors="coerce",
                ).to_numpy(dtype=float)

                finite_x = x_values[
                    np.isfinite(x_values)
                ]

                if len(finite_x) > 1:
                    scale = (
                        np.nanmax(finite_x)
                        - np.nanmin(finite_x)
                    )
                else:
                    scale = max(abs(x), 1.0)
            else:
                scale = max(abs(x), 1.0)

            jitter_scale = (
                point_jitter
                * max(scale, 1e-12)
            )

            if show_replicates:
                jitter = rng.uniform(
                    -jitter_scale,
                    jitter_scale,
                    size=len(values),
                )

                ax.scatter(
                    np.full(len(values), x)
                    + jitter,
                    values,
                    s=36,
                    alpha=0.65,
                    zorder=2,
                )

            ax.errorbar(
                x,
                mean,
                yerr=sem,
                fmt="o",
                markersize=8,
                capsize=4,
                linewidth=1.8,
                zorder=3,
            )

        else:
            x = index

            if show_replicates:
                jitter = rng.uniform(
                    -0.10,
                    0.10,
                    size=len(values),
                )

                ax.scatter(
                    np.full(len(values), x)
                    + jitter,
                    values,
                    s=36,
                    alpha=0.65,
                    zorder=2,
                )

            ax.errorbar(
                x,
                mean,
                yerr=sem,
                fmt="o",
                markersize=8,
                capsize=4,
                linewidth=1.8,
                zorder=3,
            )

    if numeric_axis:
        ax.set_xlabel(
            _sweep_label(parameter)
        )
    else:
        ax.set_xticks(
            range(len(conditions))
        )
        ax.set_xticklabels(
            conditions[
                "sweep_value_display"
            ].astype(str),
            rotation=35,
            ha="right",
        )
        ax.set_xlabel(
            _sweep_label(parameter)
        )

    if log_x and numeric_axis:
        x_values = pd.to_numeric(
            conditions[
                "sweep_value_display"
            ],
            errors="coerce",
        ).to_numpy(dtype=float)

        if np.all(x_values > 0):
            ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    phase_title = {
        "association": (
            "Association rebinding events"
        ),
        "dissociation": (
            "Dissociation rebinding events"
        ),
        "total": (
            "Total rebinding events"
        ),
    }[phase.lower()]

    ax.set_title(
        f"{phase_title} vs "
        f"{_sweep_info(parameter)['label']}"
    )

    ax.set_ylabel(
        "Number of rebinding events"
    )

    ax.grid(
        alpha=0.20,
    )

    return ax


def plot_rebinding_sweep(
    data: pd.DataFrame,
    *,
    show_replicates: bool = True,
    show_sem: bool = True,
    log_x: bool = False,
    log_y: bool = False,
    figsize: Tuple[float, float] = (
        15.0,
        4.8,
    ),
):
    """
    Create association, dissociation, and total rebinding panels.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
    )

    for ax, phase in zip(
        axes,
        (
            "association",
            "dissociation",
            "total",
        ),
    ):
        plot_rebinding_parameter(
            data,
            phase=phase,
            ax=ax,
            show_replicates=show_replicates,
            show_sem=show_sem,
            log_x=log_x,
            log_y=log_y,
        )

    parameter = str(
        data["sweep_parameter"].iloc[0]
    )

    fig.suptitle(
        "Rebinding versus "
        f"{_sweep_info(parameter)['label']}",
        y=1.02,
    )

    fig.tight_layout()

    return fig, axes


# -----------------------------------------------------------------------------
# Rebinding versus fitted kinetic parameters
# -----------------------------------------------------------------------------

KINETIC_PARAMETER_COLUMNS = {
    "k_on": (
        "k_on_eff_M_inv_s",
        "k_on_M_inv_s",
        "k_on",
    ),
    "k_off": (
        "k_off_eff_s_inv",
        "k_off_s_inv",
        "k_off_s",
        "k_off",
    ),
    "K_D": (
        "K_D_eff_M",
        "Kd_M",
        "K_D_M",
        "K_D",
    ),
}

KINETIC_PARAMETER_LABELS = {
    "k_on": r"$k_{\mathrm{on}}$ (M$^{-1}$ s$^{-1}$)",
    "k_off": r"$k_{\mathrm{off}}$ (s$^{-1}$)",
    "K_D": r"$K_D$ (M)",
}


def _kinetic_column(
    dataframe: pd.DataFrame,
    parameter: str,
) -> str:
    if parameter not in KINETIC_PARAMETER_COLUMNS:
        raise ValueError(
            "parameter must be 'k_on', 'k_off', or 'K_D'."
        )

    return _find_column(
        dataframe,
        KINETIC_PARAMETER_COLUMNS[parameter],
    )


def _kinetic_success_mask(
    dataframe: pd.DataFrame,
) -> pd.Series:
    for column in (
        "optimizer_success",
        "fit_success",
        "association_fit_success",
    ):
        if column not in dataframe.columns:
            continue

        values = dataframe[column]

        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False)

        return (
            values.astype(str)
            .str.strip()
            .str.lower()
            .isin(("true", "1", "yes"))
        )

    return pd.Series(
        True,
        index=dataframe.index,
        dtype=bool,
    )


def load_rebinding_with_kinetics(
    rebinding_data: pd.DataFrame,
    kinetics_csv: str | Path,
) -> pd.DataFrame:
    """
    Merge sweep-aware rebinding counts with sweep-aware kinetic-fit output.

    The preferred merge key is:
        sweep_parameter + sweep_value + replicate

    This matches the output of ``fit_1to1_kinetics_sweep_aware.py``.
    """
    kinetics_csv = (
        Path(kinetics_csv)
        .expanduser()
        .resolve()
    )

    kinetics = pd.read_csv(
        kinetics_csv
    )

    kinetics = kinetics.loc[
        _kinetic_success_mask(
            kinetics
        )
    ].copy()

    required = {
        "sweep_parameter",
        "sweep_value",
        "replicate",
    }

    missing = required.difference(
        kinetics.columns
    )

    if missing:
        raise KeyError(
            "The kinetic CSV is not sweep-aware. Missing column(s): "
            + ", ".join(sorted(missing))
            + ". Re-run the sweep-aware kinetic fitter."
        )

    standardized = kinetics[
        [
            "sweep_parameter",
            "sweep_value",
            "replicate",
        ]
    ].copy()

    for parameter in (
        "k_on",
        "k_off",
        "K_D",
    ):
        column = _kinetic_column(
            kinetics,
            parameter,
        )

        standardized[parameter] = (
            pd.to_numeric(
                kinetics[column],
                errors="coerce",
            )
        )

    left = rebinding_data.copy()
    right = standardized.copy()

    left["_sweep_key"] = (
        left["sweep_value"]
        .astype(str)
    )
    right["_sweep_key"] = (
        right["sweep_value"]
        .astype(str)
    )

    left["replicate"] = pd.to_numeric(
        left["replicate"],
        errors="coerce",
    )
    right["replicate"] = pd.to_numeric(
        right["replicate"],
        errors="coerce",
    )

    merged = left.merge(
        right[
            [
                "sweep_parameter",
                "_sweep_key",
                "replicate",
                "k_on",
                "k_off",
                "K_D",
            ]
        ],
        on=[
            "sweep_parameter",
            "_sweep_key",
            "replicate",
        ],
        how="left",
        validate="many_to_one",
    )

    return merged.drop(
        columns=["_sweep_key"]
    )


def _linear_fit_statistics(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    x = np.asarray(
        x,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r_squared": np.nan,
            "n": len(x),
        }

    slope, intercept = np.polyfit(
        x,
        y,
        deg=1,
    )

    predicted = slope * x + intercept

    ss_res = float(
        np.sum(
            (y - predicted) ** 2
        )
    )
    ss_tot = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    r_squared = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "n": len(x),
    }


def plot_rebinding_vs_kinetic_parameter(
    data: pd.DataFrame,
    *,
    parameter: str = "k_off",
    phase: str = "total",
    show_regression: bool = True,
    log_x: bool = True,
    log_y: bool = False,
    figsize: Tuple[float, float] = (
        7.0,
        5.5,
    ),
):
    """
    Scatter plot of per-replicate rebinding count versus a fitted parameter.
    """
    if parameter not in KINETIC_PARAMETER_COLUMNS:
        raise ValueError(
            "parameter must be 'k_on', 'k_off', or 'K_D'."
        )

    rebinding_column = _phase_column(
        phase
    )

    plot_data = data[
        [
            parameter,
            rebinding_column,
            "sweep_value_display",
        ]
    ].copy()

    plot_data[parameter] = pd.to_numeric(
        plot_data[parameter],
        errors="coerce",
    )
    plot_data[rebinding_column] = pd.to_numeric(
        plot_data[rebinding_column],
        errors="coerce",
    )

    finite = (
        np.isfinite(
            plot_data[
                parameter
            ].to_numpy(dtype=float)
        )
        & np.isfinite(
            plot_data[
                rebinding_column
            ].to_numpy(dtype=float)
        )
    )

    if log_x:
        finite &= (
            plot_data[
                parameter
            ].to_numpy(dtype=float)
            > 0
        )

    if log_y:
        finite &= (
            plot_data[
                rebinding_column
            ].to_numpy(dtype=float)
            > 0
        )

    plot_data = plot_data.loc[
        finite
    ].copy()

    if plot_data.empty:
        raise ValueError(
            "No finite matched rebinding/kinetic values remain for plotting."
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.scatter(
        plot_data[parameter],
        plot_data[rebinding_column],
        s=55,
        alpha=0.75,
    )

    if show_regression and len(plot_data) >= 2:
        x_raw = plot_data[
            parameter
        ].to_numpy(dtype=float)

        y_raw = plot_data[
            rebinding_column
        ].to_numpy(dtype=float)

        x_fit = (
            np.log10(x_raw)
            if log_x
            else x_raw
        )

        y_fit = (
            np.log10(y_raw)
            if log_y
            else y_raw
        )

        stats = _linear_fit_statistics(
            x_fit,
            y_fit,
        )

        if np.isfinite(
            stats["slope"]
        ):
            tx = np.linspace(
                np.min(x_fit),
                np.max(x_fit),
                100,
            )

            ty = (
                stats["slope"] * tx
                + stats["intercept"]
            )

            xx = (
                10.0 ** tx
                if log_x
                else tx
            )
            yy = (
                10.0 ** ty
                if log_y
                else ty
            )

            ax.plot(
                xx,
                yy,
                linestyle="--",
                linewidth=1.8,
                label=(
                    "Trend "
                    f"(R²={stats['r_squared']:.2f})"
                ),
            )
            ax.legend()

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(
        KINETIC_PARAMETER_LABELS[
            parameter
        ]
    )

    ax.set_ylabel(
        "Number of rebinding events"
    )

    phase_title = {
        "association": "Association",
        "dissociation": "Dissociation",
        "total": "Total",
    }[phase.lower()]

    ax.set_title(
        f"{phase_title} rebinding vs "
        f"{KINETIC_PARAMETER_LABELS[parameter].split(' (')[0]}"
    )

    ax.grid(
        alpha=0.20,
    )

    fig.tight_layout()

    return fig, ax


def plot_rebinding_vs_all_kinetics(
    data: pd.DataFrame,
    *,
    phase: str = "total",
    show_regression: bool = True,
    log_x: bool = True,
    log_y: bool = False,
    figsize: Tuple[float, float] = (
        15.0,
        4.8,
    ),
):
    """
    Three-panel figure of rebinding versus k_on, k_off, and K_D.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
    )

    for ax, parameter in zip(
        axes,
        (
            "k_on",
            "k_off",
            "K_D",
        ),
    ):
        rebinding_column = _phase_column(
            phase
        )

        frame = data[
            [
                parameter,
                rebinding_column,
            ]
        ].copy()

        frame[parameter] = pd.to_numeric(
            frame[parameter],
            errors="coerce",
        )
        frame[rebinding_column] = pd.to_numeric(
            frame[rebinding_column],
            errors="coerce",
        )

        finite = (
            np.isfinite(
                frame[
                    parameter
                ].to_numpy(dtype=float)
            )
            & np.isfinite(
                frame[
                    rebinding_column
                ].to_numpy(dtype=float)
            )
        )

        if log_x:
            finite &= (
                frame[
                    parameter
                ].to_numpy(dtype=float)
                > 0
            )

        if log_y:
            finite &= (
                frame[
                    rebinding_column
                ].to_numpy(dtype=float)
                > 0
            )

        frame = frame.loc[
            finite
        ].copy()

        if frame.empty:
            continue

        ax.scatter(
            frame[parameter],
            frame[rebinding_column],
            s=48,
            alpha=0.75,
        )

        if (
            show_regression
            and len(frame) >= 2
        ):
            x_raw = frame[
                parameter
            ].to_numpy(dtype=float)

            y_raw = frame[
                rebinding_column
            ].to_numpy(dtype=float)

            x_fit = (
                np.log10(x_raw)
                if log_x
                else x_raw
            )

            y_fit = (
                np.log10(y_raw)
                if log_y
                else y_raw
            )

            stats = _linear_fit_statistics(
                x_fit,
                y_fit,
            )

            if np.isfinite(
                stats["slope"]
            ):
                tx = np.linspace(
                    np.min(x_fit),
                    np.max(x_fit),
                    100,
                )
                ty = (
                    stats["slope"] * tx
                    + stats["intercept"]
                )

                xx = (
                    10.0 ** tx
                    if log_x
                    else tx
                )
                yy = (
                    10.0 ** ty
                    if log_y
                    else ty
                )

                ax.plot(
                    xx,
                    yy,
                    linestyle="--",
                    linewidth=1.7,
                    label=(
                        f"Trend "
                        f"(R²={stats['r_squared']:.2f})"
                    ),
                )

        if log_x:
            ax.set_xscale("log")

        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel(
            KINETIC_PARAMETER_LABELS[
                parameter
            ]
        )
        ax.set_ylabel(
            "Rebinding events"
        )
        ax.set_title(
            KINETIC_PARAMETER_LABELS[
                parameter
            ].split(" (")[0]
        )
        ax.grid(
            alpha=0.20
        )

        if show_regression:
            ax.legend()

    phase_title = {
        "association": "Association",
        "dissociation": "Dissociation",
        "total": "Total",
    }[phase.lower()]

    fig.suptitle(
        f"{phase_title} rebinding versus fitted kinetics",
        y=1.03,
    )

    fig.tight_layout()

    return fig, axes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze rebinding events for an automatically detected "
            "biosensor parameter sweep."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sweep-parameter",
        type=str,
        default=None,
        help=(
            "Optional manual sweep-parameter override. "
            "Normally automatic detection should be used."
        ),
    )

    parser.add_argument(
        "--phase",
        choices=(
            "association",
            "dissociation",
            "total",
        ),
        default="total",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output for the selected-phase figure."
        ),
    )

    parser.add_argument(
        "--all-phases-output",
        type=Path,
        default=None,
        help=(
            "Optional output for the three-panel association/dissociation/"
            "total figure."
        ),
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--replicate-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--kinetics-csv",
        type=Path,
        default=None,
        help=(
            "Optional sweep-aware 1:1 kinetic-fit CSV. If supplied, "
            "rebinding is merged with k_on, k_off, and K_D by sweep value "
            "and replicate."
        ),
    )

    parser.add_argument(
        "--kinetics-output",
        type=Path,
        default=None,
        help=(
            "Optional three-panel output for rebinding vs k_on/k_off/K_D. "
            "Requires --kinetics-csv."
        ),
    )

    parser.add_argument(
        "--no-replicates",
        action="store_true",
    )

    parser.add_argument(
        "--no-sem",
        action="store_true",
    )

    parser.add_argument(
        "--log-x",
        action="store_true",
    )

    parser.add_argument(
        "--log-y",
        action="store_true",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser


def _save_figure(
    fig,
    path: Path,
    *,
    dpi: int,
) -> None:
    path = (
        path
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    print(
        f"Saved figure: {path}"
    )


def main() -> None:
    args = build_parser().parse_args()

    data = load_rebinding_sweep(
        args.data_root,
        sweep_parameter=args.sweep_parameter,
    )

    parameter = str(
        data["sweep_parameter"].iloc[0]
    )

    print(
        "Detected sweep parameter: "
        f"{parameter} "
        f"({_sweep_info(parameter)['label']})"
    )

    print(
        "Sweep values: "
        + ", ".join(
            map(
                str,
                _ordered_conditions(data)[
                    "sweep_value_display"
                ].tolist(),
            )
        )
    )

    selected_fig, selected_ax = plt.subplots(
        figsize=(7.5, 5.2)
    )

    plot_rebinding_parameter(
        data,
        phase=args.phase,
        ax=selected_ax,
        show_replicates=not args.no_replicates,
        show_sem=not args.no_sem,
        log_x=args.log_x,
        log_y=args.log_y,
    )

    selected_fig.tight_layout()

    if args.output is None:
        plt.show()
    else:
        _save_figure(
            selected_fig,
            args.output,
            dpi=args.dpi,
        )

    if args.all_phases_output is not None:
        phase_fig, _ = plot_rebinding_sweep(
            data,
            show_replicates=not args.no_replicates,
            show_sem=not args.no_sem,
            log_x=args.log_x,
            log_y=args.log_y,
        )

        _save_figure(
            phase_fig,
            args.all_phases_output,
            dpi=args.dpi,
        )

    if args.summary_csv is not None:
        summary = summarize_rebinding(
            data
        )

        path = (
            args.summary_csv
            .expanduser()
            .resolve()
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary.to_csv(
            path,
            index=False,
        )

        print(
            f"Saved summary CSV: {path}"
        )

    if args.replicate_csv is not None:
        path = (
            args.replicate_csv
            .expanduser()
            .resolve()
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data.to_csv(
            path,
            index=False,
        )

        print(
            f"Saved replicate CSV: {path}"
        )

    if args.kinetics_csv is not None:
        merged = load_rebinding_with_kinetics(
            data,
            args.kinetics_csv,
        )

        if args.kinetics_output is not None:
            kinetics_fig, _ = (
                plot_rebinding_vs_all_kinetics(
                    merged,
                    phase=args.phase,
                )
            )

            _save_figure(
                kinetics_fig,
                args.kinetics_output,
                dpi=args.dpi,
            )


if __name__ == "__main__":
    main()


__all__ = [
    "rebinding_counts_from_history",
    "load_rebinding_sweep",
    "summarize_rebinding",
    "plot_rebinding_parameter",
    "plot_rebinding_sweep",
    "load_rebinding_with_kinetics",
    "plot_rebinding_vs_kinetic_parameter",
    "plot_rebinding_vs_all_kinetics",
]
