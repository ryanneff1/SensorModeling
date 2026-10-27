#!/usr/bin/env python3
"""
Plot planar concentration-sweep fractional occupancy versus time.

Supports:
- mode="replicates": individual replicate traces
- mode="mean": mean trace with SD, SEM, 95% CI, or no error band

The script recursively scans for history.parquet files and groups runs by
association ligand concentration.

Notebook example
----------------
from plot_planar_concentration_sweep import load_concentration_sweep, plot_concentration_sweep

data = load_concentration_sweep(
    "/home/rnt2664/SensorModeling/results/260812_planar_concentration_sweep"
)

fig, ax = plot_concentration_sweep(
    data,
    mode="mean",
    error="sem",
)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _parse_replicate(run_directory: Path) -> Optional[int]:
    match = re.search(r"replicate[_-]?(\d+)", run_directory.name, flags=re.I)
    return int(match.group(1)) if match else None


def _parse_concentration_from_directory(run_directory: Path) -> Optional[float]:
    for parent in (run_directory.parent, *run_directory.parents):
        match = re.fullmatch(r"concentration_([0-9eE.+-]+)M", parent.name)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def _first_finite_value(frame: pd.DataFrame, columns: Iterable[str]) -> Optional[float]:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values.to_numpy(dtype=float))]
        if not values.empty:
            return float(values.iloc[0])
    return None


def _association_concentration(history: pd.DataFrame, run_directory: Path) -> Optional[float]:
    if "phase_label" in history.columns:
        association = history.loc[
            history["phase_label"].astype(str).str.lower().eq("association")
        ]
    else:
        association = history

    concentration = _first_finite_value(
        association,
        ("phase_ligand_conc_M", "ligand_conc_M", "association_concentration_M"),
    )
    if concentration is not None:
        return concentration

    manifest = _read_json(run_directory / "manifest.json")
    run_metadata = manifest.get("run_metadata", {})
    if isinstance(run_metadata, dict):
        for key in ("association_concentration_M", "ligand_conc_M"):
            value = run_metadata.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

        association_params = run_metadata.get("association_params", {})
        if isinstance(association_params, dict):
            value = association_params.get("ligand_conc_M")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    concentration = _parse_concentration_from_directory(run_directory)
    if concentration is not None:
        return concentration

    parameters = _read_json(run_directory / "parameters.json")
    value = parameters.get("ligand_conc_M")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    return None


def _clean_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"t_s", "theta"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(
            "History is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    keep = [
        c
        for c in (
            "t_s",
            "theta",
            "phase_label",
            "phase_start_t_s",
            "phase_ligand_conc_M",
            "ligand_conc_M",
        )
        if c in history.columns
    ]

    result = history[keep].copy()
    result["t_s"] = pd.to_numeric(result["t_s"], errors="coerce")
    result["theta"] = pd.to_numeric(result["theta"], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["t_s", "theta"]).sort_values("t_s")

    # Resume points may duplicate the association endpoint at the phase switch.
    result = result.drop_duplicates(subset=["t_s"], keep="last")
    return result.reset_index(drop=True)


def load_concentration_sweep(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    history_paths = sorted(data_root.rglob("history.parquet"))
    if not history_paths:
        raise FileNotFoundError(
            f"No history.parquet files found under {data_root}"
        )

    frames: List[pd.DataFrame] = []

    for history_path in history_paths:
        run_directory = history_path.parent
        history = _clean_history(pd.read_parquet(history_path))

        concentration_M = _association_concentration(history, run_directory)
        if concentration_M is None:
            raise ValueError(
                f"Could not determine association concentration for {history_path}"
            )

        history["concentration_M"] = float(concentration_M)
        history["concentration_nM"] = float(concentration_M) * 1e9
        history["replicate"] = _parse_replicate(run_directory)
        history["run_directory"] = str(run_directory)
        history["history_path"] = str(history_path)
        frames.append(history)

    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(
        ["concentration_M", "replicate", "t_s"]
    ).reset_index(drop=True)


def _phase_boundary_time(data: pd.DataFrame) -> Optional[float]:
    if "phase_label" not in data.columns:
        return None

    labels = data["phase_label"].astype(str).str.lower()
    dissociation = data.loc[labels.eq("dissociation"), "t_s"]

    if dissociation.empty:
        return None

    return float(dissociation.min())


def _common_grid(
    replicate_frames: Sequence[pd.DataFrame],
    n_points: Optional[int] = None,
) -> np.ndarray:
    if not replicate_frames:
        raise ValueError("No replicate data supplied.")

    time_vectors = [
        frame["t_s"].to_numpy(dtype=float)
        for frame in replicate_frames
    ]

    first = time_vectors[0]
    if all(
        len(values) == len(first)
        and np.allclose(values, first, rtol=1e-10, atol=1e-12)
        for values in time_vectors[1:]
    ):
        return first.copy()

    start = max(float(values[0]) for values in time_vectors)
    end = min(float(values[-1]) for values in time_vectors)

    if end <= start:
        raise ValueError("Replicates do not share an overlapping time range.")

    if n_points is None:
        counts = [
            np.count_nonzero((values >= start) & (values <= end))
            for values in time_vectors
        ]
        n_points = int(np.median(counts))

    return np.linspace(start, end, max(int(n_points), 2))


def summarize_replicates(
    data: pd.DataFrame,
    concentration_M: float,
    n_grid_points: Optional[int] = None,
) -> pd.DataFrame:
    subset = data.loc[
        np.isclose(
            data["concentration_M"].to_numpy(dtype=float),
            float(concentration_M),
            rtol=1e-10,
            atol=0.0,
        )
    ].copy()

    if subset.empty:
        raise ValueError(f"No data found for {concentration_M:g} M.")

    replicate_frames = [
        frame.sort_values("t_s")
        for _, frame in subset.groupby("replicate", dropna=False)
    ]

    grid = _common_grid(replicate_frames, n_points=n_grid_points)

    matrix = np.vstack(
        [
            np.interp(
                grid,
                frame["t_s"].to_numpy(dtype=float),
                frame["theta"].to_numpy(dtype=float),
            )
            for frame in replicate_frames
        ]
    )

    n = matrix.shape[0]
    mean = np.mean(matrix, axis=0)

    if n > 1:
        sd = np.std(matrix, axis=0, ddof=1)
    else:
        sd = np.zeros_like(mean)

    sem = sd / np.sqrt(max(n, 1))
    ci95 = 1.96 * sem

    return pd.DataFrame(
        {
            "t_s": grid,
            "mean": mean,
            "sd": sd,
            "sem": sem,
            "ci95": ci95,
            "n": n,
        }
    )


def _format_concentration(concentration_M: float) -> str:
    nM = concentration_M * 1e9
    if nM >= 1:
        return f"{nM:g} nM"

    pM = concentration_M * 1e12
    if pM >= 1:
        return f"{pM:g} pM"

    return f"{concentration_M:g} M"


def _selected_concentrations(
    data: pd.DataFrame,
    concentrations_M: Optional[Sequence[float]],
) -> List[float]:
    available = sorted(float(v) for v in pd.unique(data["concentration_M"]))

    if concentrations_M is None:
        return available

    selected: List[float] = []
    for requested in concentrations_M:
        matches = [
            value
            for value in available
            if np.isclose(value, float(requested), rtol=1e-9, atol=0.0)
        ]
        if not matches:
            raise ValueError(
                f"Requested concentration {requested:g} M was not found."
            )
        selected.append(matches[0])

    return selected


def plot_concentration_sweep(
    data: pd.DataFrame,
    *,
    mode: str = "mean",
    error: str = "sem",
    concentrations_M: Optional[Sequence[float]] = None,
    n_grid_points: Optional[int] = None,
    show_phase_boundary: bool = True,
    show_phase_labels: bool = False,
    figsize: Tuple[float, float] = (9.0, 5.5),
    y_lim: Tuple[float, float] = (0.0, 1.05),
    x_lim: Optional[Tuple[float, float]] = None,
    title: str = "Planar sensor concentration sweep",
    show_subtitle: bool = False,
):
    mode = mode.lower()
    error = error.lower()

    if mode not in {"replicates", "mean"}:
        raise ValueError("mode must be 'replicates' or 'mean'.")

    if error not in {"sd", "sem", "ci95", "none"}:
        raise ValueError("error must be 'sd', 'sem', 'ci95', or 'none'.")

    concentrations = _selected_concentrations(data, concentrations_M)

    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for i, concentration_M in enumerate(concentrations):
        color = default_colors[i % len(default_colors)] if default_colors else None

        subset = data.loc[
            np.isclose(
                data["concentration_M"].to_numpy(dtype=float),
                concentration_M,
                rtol=1e-10,
                atol=0.0,
            )
        ]

        concentration_label = _format_concentration(concentration_M)

        if mode == "replicates":
            groups = list(subset.groupby("replicate", dropna=False))

            for replicate_index, (replicate, frame) in enumerate(groups):
                frame = frame.sort_values("t_s")
                ax.plot(
                    frame["t_s"],
                    frame["theta"],
                    color=color,
                    linewidth=1.3,
                    alpha=0.6,
                    label=(
                        concentration_label
                        if replicate_index == 0
                        else "_nolegend_"
                    ),
                )

        else:
            summary = summarize_replicates(
                data,
                concentration_M=concentration_M,
                n_grid_points=n_grid_points,
            )

            ax.plot(
                summary["t_s"],
                summary["mean"],
                color=color,
                linewidth=2.0,
                label=concentration_label,
            )

            if error != "none":
                spread = summary[error].to_numpy(dtype=float)
                mean = summary["mean"].to_numpy(dtype=float)

                ax.fill_between(
                    summary["t_s"],
                    np.clip(mean - spread, 0.0, 1.0),
                    np.clip(mean + spread, 0.0, 1.0),
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )

    phase_boundary = _phase_boundary_time(data)

    if show_phase_boundary and phase_boundary is not None:
        ax.axvline(
            phase_boundary,
            linestyle=":",
            linewidth=1.5,
        )

    if show_phase_labels and phase_boundary is not None:
        x_min = float(data["t_s"].min())
        x_max = float(data["t_s"].max())

        ax.text(
            0.5 * (x_min + phase_boundary),
            1.015,
            "association",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
        )
        ax.text(
            0.5 * (phase_boundary + x_max),
            1.015,
            "dissociation",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
        )
    if show_subtitle:
        if mode == "mean":
            subtitle = {
                "sd": "Mean ± SD",
                "sem": "Mean ± SEM",
                "ci95": "Mean ± 95% CI",
                "none": "Mean",
            }[error]
        else:
            subtitle = "Individual replicates"

        ax.set_title(f"{title}\n{subtitle}")
    else:
        ax.set_title(title)
        
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction of bound receptors")
    ax.set_ylim(*y_lim)
    ax.set_xlim(*(x_lim if x_lim is not None else (None, None)))
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()

    return fig, ax


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fractional occupancy versus time for a planar "
            "ligand-concentration sweep."
        )
    )

    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("replicates", "mean"),
        default="mean",
    )
    parser.add_argument(
        "--error",
        choices=("sd", "sem", "ci95", "none"),
        default="sem",
    )
    parser.add_argument(
        "--concentrations-M",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--n-grid-points",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--title",
        default="Planar sensor concentration sweep",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .png, .pdf, .svg, etc. output path.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-phase-boundary", action="store_true")
    parser.add_argument("--no-phase-labels", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    data = load_concentration_sweep(args.data_root)

    fig, _ = plot_concentration_sweep(
        data,
        mode=args.mode,
        error=args.error,
        concentrations_M=args.concentrations_M,
        n_grid_points=args.n_grid_points,
        show_phase_boundary=not args.no_phase_boundary,
        show_phase_labels=not args.no_phase_labels,
        title=args.title,
    )

    if args.output is None:
        plt.show()
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
