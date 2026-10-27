from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError as exc:
    raise ImportError(
        "biosensor_history_viewer requires ipywidgets. "
        "Install it with `pip install ipywidgets` or "
        "`conda install -c conda-forge ipywidgets`."
    ) from exc

PathLike = Union[str, Path]


# Canonical sweep parameters used by the protocol scripts created for this
# project. The viewer also has a generic fallback for future sweep parameters.
SWEEP_PARAMETER_INFO = {
    "pore_diameter_m": {
        "display": "Pore diameter",
        "kind": "length_m",
    },
    "pore_depth_m": {
        "display": "Pore height/depth",
        "kind": "length_m",
    },
    "pore_height_m": {
        "display": "Pore height",
        "kind": "length_m",
    },
    "association_concentration_M": {
        "display": "Association concentration",
        "kind": "concentration_M",
    },
    "ligand_conc_M": {
        "display": "Ligand concentration",
        "kind": "concentration_M",
    },
    "receptor_density_m2": {
        "display": "Receptor density",
        "kind": "receptor_density",
    },
    "spike_density_m2": {
        "display": "Primary spike density",
        "kind": "spike_density",
    },
    "branch_levels": {
        "display": "Branch levels",
        "kind": "integer",
    },
    "branches_per_segment_mean": {
        "display": "Branches per segment",
        "kind": "number",
    },
}


@dataclass(frozen=True)
class RunRecord:
    sweep_parameter: str
    sweep_value: Any
    sweep_value_sort: float
    condition_key: str
    condition_label: str
    replicate: int
    run_directory: Path
    history_path: Path

    # Backward-compatible aliases for notebooks that used the original viewer.
    @property
    def geometry_key(self) -> str:
        return self.condition_key

    @property
    def geometry_label(self) -> str:
        return self.condition_label


def _read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    return value if isinstance(value, dict) else {}


def _parse_replicate(run_directory: Path) -> int:
    match = re.search(
        r"replicate[_-]?(\d+)",
        run_directory.name,
        flags=re.I,
    )

    if match:
        return int(match.group(1))

    return 1


def _manifest_run_metadata(run_directory: Path) -> dict:
    """Return save_simulation_results() run_metadata when available."""
    manifest = _read_json_if_present(
        run_directory / "manifest.json"
    )

    metadata = manifest.get("run_metadata", {})

    return metadata if isinstance(metadata, dict) else {}


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if np.isfinite(number) else None


def _parameter_display_name(parameter: str) -> str:
    info = SWEEP_PARAMETER_INFO.get(parameter)

    if info is not None:
        return str(info["display"])

    return parameter.replace("_", " ").strip().capitalize()


def _format_scientific(value: float) -> str:
    return f"{float(value):.4g}"


def _format_sweep_value(parameter: str, value: Any) -> str:
    info = SWEEP_PARAMETER_INFO.get(
        parameter,
        {"kind": "number"},
    )
    kind = info["kind"]

    number = _finite_float(value)

    if number is None:
        return str(value)

    if kind == "length_m":
        return f"{number * 1e9:g} nm"

    if kind == "concentration_M":
        magnitude = abs(number)

        if magnitude == 0:
            return "0 M"
        if magnitude < 1e-9:
            return f"{number * 1e12:g} pM"
        if magnitude < 1e-6:
            return f"{number * 1e9:g} nM"
        if magnitude < 1e-3:
            return f"{number * 1e6:g} µM"

        return f"{number:g} M"

    if kind == "receptor_density":
        return f"{_format_scientific(number)} receptors m⁻²"

    if kind == "spike_density":
        return f"{_format_scientific(number)} spikes m⁻²"

    if kind == "integer":
        return str(int(round(number)))

    return f"{number:g}"


def _condition_label(parameter: str, value: Any) -> str:
    return (
        f"{_parameter_display_name(parameter)} = "
        f"{_format_sweep_value(parameter, value)}"
    )


def _condition_key(parameter: str, value: Any) -> str:
    number = _finite_float(value)

    if number is None:
        token = str(value)
    else:
        token = f"{number:.17g}"

    return f"{parameter}={token}"


def _sweep_value_sort(value: Any) -> float:
    number = _finite_float(value)

    if number is not None:
        return number

    # Non-numeric sweep values remain usable, but sort after numeric values.
    return np.inf


def _metadata_sweep_identity(
    run_directory: Path,
) -> Optional[Tuple[str, Any]]:
    """
    Detect a sweep from run_metadata written by the protocol scripts.

    This is the highest-priority source because it records the protocol's
    intended swept parameter rather than merely describing the final geometry.
    """
    metadata = _manifest_run_metadata(run_directory)

    parameter = metadata.get("sweep_parameter")

    if parameter is not None and "sweep_value" in metadata:
        return str(parameter), metadata["sweep_value"]

    sweep_variable = metadata.get("sweep_variable")

    if sweep_variable is not None:
        parameter = str(sweep_variable)

        # Direct value with the same name.
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
            ),
            "association_concentration_M": (
                "association_concentration_M",
                "ligand_conc_M",
            ),
            "receptor_density_m2": (
                "receptor_density_m2",
            ),
        }

        for candidate in aliases.get(parameter, ()):
            if candidate not in metadata:
                continue

            value = metadata[candidate]

            # Convert explicit nm aliases back to canonical meters.
            if candidate.endswith("_nm") and parameter.endswith("_m"):
                number = _finite_float(value)
                if number is not None:
                    value = number * 1e-9

            return parameter, value

    return None


def _directory_sweep_identity(
    run_directory: Path,
) -> Optional[Tuple[str, Any]]:
    """
    Recognize directory conventions used by the project's sweep protocols.

    Examples
    --------
    diameter_3e-08m
    height_50nm
    concentration_1e-08M
    density_1e14_per_m2
    spike_density_m2_5e14
    branch_levels_2
    branches_per_segment_mean_1.5
    """
    candidate_names = [
        run_directory.parent.name,
        *[
            parent.name
            for parent in run_directory.parents
            if parent != run_directory.parent
        ][:2],
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
            r"^pore[_-]?height_([0-9eE.+-]+)nm$",
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
            match = re.fullmatch(pattern, name, flags=re.I)

            if match:
                return parameter, converter(match.group(1))

    return None


def _geometry_summary_sweep_identity(
    run_directory: Path,
) -> Optional[Tuple[str, Any]]:
    """
    Backward-compatible geometry fallback.

    This intentionally runs *after* run metadata and directory detection.
    A nanopore-height sweep, for example, also has a pore diameter in its
    geometry summary; choosing diameter first would misidentify that sweep.
    """
    summary = _read_json_if_present(
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
    history_path: Path,
) -> Optional[Tuple[str, Any]]:
    """
    Last-resort detection from history metadata.

    Association concentration is the safest useful parameter that can be
    recovered directly from a history. Geometry sweeps should normally have
    already been recognized from run_metadata or directory names.
    """
    try:
        if history_path.suffix.lower() == ".parquet":
            history = pd.read_parquet(
                history_path,
                columns=[
                    "phase_label",
                    "phase_ligand_conc_M",
                ],
            )
        else:
            history = pd.read_csv(
                history_path,
                usecols=lambda c: c in {
                    "phase_label",
                    "phase_ligand_conc_M",
                },
            )
    except Exception:
        return None

    if "phase_ligand_conc_M" not in history.columns:
        return None

    if "phase_label" in history.columns:
        association = history.loc[
            history["phase_label"]
            .astype(str)
            .str.lower()
            .eq("association")
        ]

        if association.empty:
            association = history
    else:
        association = history

    values = pd.to_numeric(
        association["phase_ligand_conc_M"],
        errors="coerce",
    )
    values = values[np.isfinite(values)]

    if len(values):
        return (
            "association_concentration_M",
            float(values.iloc[0]),
        )

    return None


def _detect_sweep_identity(
    run_directory: Path,
    history_path: Path,
) -> Tuple[str, Any]:
    """
    Automatically identify the parameter represented by one saved condition.

    Priority:
        1. run_metadata in manifest.json
        2. protocol directory naming convention
        3. geometry_summary.json (legacy nanopore sweep)
        4. phase metadata in history
        5. generic parent-directory condition
    """
    for detector in (
        lambda: _metadata_sweep_identity(run_directory),
        lambda: _directory_sweep_identity(run_directory),
        lambda: _geometry_summary_sweep_identity(run_directory),
        lambda: _history_sweep_identity(history_path),
    ):
        result = detector()

        if result is not None:
            return result

    parent_name = run_directory.parent.name
    return "condition", parent_name


def scan_saved_runs(data_root: PathLike) -> pd.DataFrame:
    """
    Recursively discover saved simulations and infer their sweep parameter.

    Both history.parquet and history.csv are supported.
    """
    data_root = Path(data_root).expanduser().resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_root}"
        )

    history_paths = sorted(
        list(data_root.rglob("history.parquet"))
        + list(data_root.rglob("history.csv"))
    )

    records: List[RunRecord] = []

    for history_path in history_paths:
        run_directory = history_path.parent

        parameter, value = _detect_sweep_identity(
            run_directory,
            history_path,
        )

        records.append(
            RunRecord(
                sweep_parameter=str(parameter),
                sweep_value=value,
                sweep_value_sort=_sweep_value_sort(value),
                condition_key=_condition_key(
                    str(parameter),
                    value,
                ),
                condition_label=_condition_label(
                    str(parameter),
                    value,
                ),
                replicate=_parse_replicate(
                    run_directory
                ),
                run_directory=run_directory,
                history_path=history_path,
            )
        )

    if not records:
        raise FileNotFoundError(
            f"No history.parquet or history.csv files were found under {data_root}"
        )

    catalog = pd.DataFrame(
        [
            {
                "sweep_parameter": record.sweep_parameter,
                "sweep_parameter_label": _parameter_display_name(
                    record.sweep_parameter
                ),
                "sweep_value": record.sweep_value,
                "sweep_value_sort": record.sweep_value_sort,
                "condition_key": record.condition_key,
                "condition_label": record.condition_label,
                # Original-column aliases.
                "geometry_key": record.condition_key,
                "geometry_label": record.condition_label,
                "replicate": record.replicate,
                "run_directory": record.run_directory,
                "history_path": record.history_path,
            }
            for record in records
        ]
    )

    return catalog.sort_values(
        [
            "sweep_parameter_label",
            "sweep_value_sort",
            "condition_label",
            "replicate",
            "history_path",
        ]
    ).reset_index(drop=True)


def _phase_ranges(history: pd.DataFrame) -> List[dict]:
    if history.empty:
        return []

    if "phase_label" not in history.columns:
        return [
            {
                "label": "simulation",
                "start_t_s": float(history["t_s"].iloc[0]),
                "end_t_s": float(history["t_s"].iloc[-1]),
                "rows": history,
            }
        ]

    labels = history["phase_label"].fillna("simulation").astype(str).to_numpy()
    change_indices = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change_indices))
    stops = np.concatenate((change_indices, [len(history)]))

    ranges = []
    for start, stop in zip(starts, stops):
        phase = history.iloc[start:stop]
        if phase.empty:
            continue
        ranges.append(
            {
                "label": str(labels[start]),
                "start_t_s": float(phase["t_s"].iloc[0]),
                "end_t_s": float(phase["t_s"].iloc[-1]),
                "rows": phase,
            }
        )
    return ranges


def _first_finite_value(data: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    for column in candidates:
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values):
            return float(values.iloc[0])
    return None


def langmuir_equilibrium_for_phase(
    phase_history: pd.DataFrame,
    *,
    fallback_parameters: Optional[dict] = None,
) -> Optional[float]:
    """Calculate theta_eq = C*k_on / (C*k_on + k_off)."""
    fallback_parameters = fallback_parameters or {}

    concentration_M = _first_finite_value(
        phase_history,
        ["phase_ligand_conc_M", "ligand_conc_M"],
    )
    k_on = _first_finite_value(
        phase_history,
        ["phase_k_on_M_inv_s", "k_on_M_inv_s"],
    )
    k_off = _first_finite_value(
        phase_history,
        ["phase_k_off_s", "k_off_s"],
    )

    if concentration_M is None:
        value = fallback_parameters.get("ligand_conc_M")
        concentration_M = None if value is None else float(value)
    if k_on is None:
        value = fallback_parameters.get("k_on_M_inv_s")
        k_on = None if value is None else float(value)
    if k_off is None:
        value = fallback_parameters.get("k_off_s")
        k_off = None if value is None else float(value)

    if concentration_M is None or k_on is None or k_off is None:
        return None
    if concentration_M < 0 or k_on < 0 or k_off < 0:
        return None
    if k_on == 0:
        return 0.0

    denominator = concentration_M * k_on + k_off
    if denominator == 0:
        return np.nan

    return float(concentration_M * k_on / denominator)



class HistoryViewer:
    def __init__(
        self,
        data_root: PathLike,
        *,
        figsize: Tuple[float, float] = (9.0, 5.5),
        line_width: float = 1.8,
        ylim: Tuple[float, float] = (0.0, 1.05),
        xlim: Optional[Tuple[float, float]] = None,
    ):
        self.data_root = Path(
            data_root
        ).expanduser().resolve()

        self.catalog = scan_saved_runs(
            self.data_root
        )

        self.figsize = figsize
        self.line_width = line_width
        self.ylim = ylim
        self.xlim = xlim
        self._history_cache: Dict[
            Path,
            pd.DataFrame,
        ] = {}

        parameter_options = (
            self.catalog[
                [
                    "sweep_parameter",
                    "sweep_parameter_label",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "sweep_parameter_label"
            )
        )

        self.sweep_parameter_dropdown = widgets.Dropdown(
            options=[
                (
                    row.sweep_parameter_label,
                    row.sweep_parameter,
                )
                for row in parameter_options.itertuples(
                    index=False
                )
            ],
            description="Sweep:",
            layout=widgets.Layout(
                width="100%"
            ),
        )

        self.condition_dropdown = widgets.Dropdown(
            options=[],
            description="Value:",
            layout=widgets.Layout(
                width="100%"
            ),
        )

        # Backward compatibility with code that accessed geometry_dropdown.
        self.geometry_dropdown = self.condition_dropdown

        self.replicate_select = widgets.SelectMultiple(
            options=[],
            description="Replicates:",
            rows=8,
            layout=widgets.Layout(
                width="100%"
            ),
        )

        self.plot_mode = widgets.ToggleButtons(
            options=[
                ("Replicates", "replicates"),
                ("Mean ± SEM", "mean_sem"),
            ],
            value="replicates",
            description="Plot:",
        )

        self.show_langmuir = widgets.Checkbox(
            value=True,
            description="Show Langmuir equilibrium",
        )

        self.show_phase_labels = widgets.Checkbox(
            value=False,
            description="Show phase labels",
        )

        self.show_phase_lines = widgets.Checkbox(
            value=True,
            description="Show phase boundaries",
        )

        self.show_legend = widgets.Checkbox(
            value=True,
            description="Show legend",
        )

        self.select_all_button = widgets.Button(
            description="Select all"
        )
        self.clear_button = widgets.Button(
            description="Clear"
        )

        self.sweep_info = widgets.HTML()
        self.output = widgets.Output()

        self.sweep_parameter_dropdown.observe(
            self._sweep_parameter_changed,
            names="value",
        )
        self.condition_dropdown.observe(
            self._condition_changed,
            names="value",
        )
        self.replicate_select.observe(
            self._selection_changed,
            names="value",
        )
        self.plot_mode.observe(
            self._selection_changed,
            names="value",
        )
        self.show_langmuir.observe(
            self._selection_changed,
            names="value",
        )
        self.show_phase_labels.observe(
            self._selection_changed,
            names="value",
        )
        self.show_phase_lines.observe(
            self._selection_changed,
            names="value",
        )
        self.show_legend.observe(
            self._selection_changed,
            names="value",
        )

        self.select_all_button.on_click(
            self._select_all
        )
        self.clear_button.on_click(
            self._clear_selection
        )

        controls = widgets.VBox(
            [
                self.sweep_parameter_dropdown,
                self.condition_dropdown,
                self.sweep_info,
                self.replicate_select,
                widgets.HBox(
                    [
                        self.select_all_button,
                        self.clear_button,
                    ]
                ),
                self.plot_mode,
                self.show_langmuir,
                self.show_phase_lines,
                self.show_phase_labels,
                self.show_legend,
            ],
            layout=widgets.Layout(
                width="300px",
                flex="0 0 300px",
            ),
        )

        self.widget = widgets.VBox(
            [
                widgets.HTML(
                    value=(
                        "<b>Saved simulation viewer</b><br>"
                        "<span style='font-size:90%'>"
                        f"{len(self.catalog)} runs found under "
                        f"{self.data_root}"
                        "</span>"
                    )
                ),
                widgets.HBox(
                    [
                        controls,
                        self.output,
                    ],
                    layout=widgets.Layout(
                        align_items="flex-start",
                        width="100%",
                    ),
                ),
            ]
        )

        self._refresh_conditions(
            select_first=True
        )
        self._refresh_replicates(
            select_all=True
        )
        self._refresh_sweep_info()

    def _load_history(
        self,
        path: Path,
    ) -> pd.DataFrame:
        path = Path(path)

        if path not in self._history_cache:
            if path.suffix.lower() == ".parquet":
                history = pd.read_parquet(
                    path
                )
            elif path.suffix.lower() == ".csv":
                history = pd.read_csv(
                    path
                )
            else:
                raise ValueError(
                    f"Unsupported history format: {path}"
                )

            if "t_s" not in history.columns:
                raise KeyError(
                    f"{path} does not contain a 't_s' column."
                )

            if "theta" not in history.columns:
                raise KeyError(
                    f"{path} does not contain a 'theta' column. "
                    "The viewer expects theta to be the fraction "
                    "of bound receptors."
                )

            self._history_cache[path] = (
                history.sort_values(
                    "t_s"
                ).reset_index(drop=True)
            )

        return self._history_cache[path]

    def _catalog_for_parameter(
        self,
    ) -> pd.DataFrame:
        return self.catalog.loc[
            self.catalog[
                "sweep_parameter"
            ]
            == self.sweep_parameter_dropdown.value
        ]

    def _records_for_condition(
        self,
    ) -> pd.DataFrame:
        catalog = self._catalog_for_parameter()

        return catalog.loc[
            catalog[
                "condition_key"
            ]
            == self.condition_dropdown.value
        ].sort_values("replicate")

    # Original method name retained for notebook compatibility.
    def _records_for_geometry(
        self,
    ) -> pd.DataFrame:
        return self._records_for_condition()

    def _refresh_conditions(
        self,
        *,
        select_first: bool,
    ) -> None:
        records = self._catalog_for_parameter()

        options_frame = (
            records[
                [
                    "condition_key",
                    "condition_label",
                    "sweep_value_sort",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "sweep_value_sort",
                    "condition_label",
                ]
            )
        )

        options = [
            (
                row.condition_label,
                row.condition_key,
            )
            for row in options_frame.itertuples(
                index=False
            )
        ]

        old_value = self.condition_dropdown.value

        self.condition_dropdown.options = options

        valid_values = {
            value
            for _, value in options
        }

        if (
            old_value in valid_values
            and not select_first
        ):
            self.condition_dropdown.value = old_value
        elif options:
            self.condition_dropdown.value = options[0][1]

    def _refresh_replicates(
        self,
        *,
        select_all: bool,
    ) -> None:
        records = self._records_for_condition()

        options = [
            (
                f"Replicate {int(rep)}",
                int(rep),
            )
            for rep in records[
                "replicate"
            ]
            .drop_duplicates()
            .tolist()
        ]

        self.replicate_select.options = options

        if select_all:
            self.replicate_select.value = tuple(
                value
                for _, value in options
            )

    def _refresh_sweep_info(
        self,
    ) -> None:
        parameter = (
            self.sweep_parameter_dropdown.value
        )

        if parameter is None:
            self.sweep_info.value = ""
            return

        records = self._catalog_for_parameter()

        n_conditions = records[
            "condition_key"
        ].nunique()

        parameter_label = (
            _parameter_display_name(parameter)
        )

        self.sweep_info.value = (
            "<span style='font-size:88%'>"
            f"Detected <b>{parameter_label}</b> sweep "
            f"with {n_conditions} condition"
            f"{'' if n_conditions == 1 else 's'}."
            "</span>"
        )

    def _sweep_parameter_changed(
        self,
        change,
    ) -> None:
        self._refresh_conditions(
            select_first=True
        )
        self._refresh_replicates(
            select_all=True
        )
        self._refresh_sweep_info()
        self.redraw()

    def _condition_changed(
        self,
        change,
    ) -> None:
        self._refresh_replicates(
            select_all=True
        )
        self.redraw()

    # Original callback name retained for compatibility.
    def _geometry_changed(
        self,
        change,
    ) -> None:
        self._condition_changed(change)

    def _selection_changed(
        self,
        change,
    ) -> None:
        self.redraw()

    def _select_all(
        self,
        button,
    ) -> None:
        self.replicate_select.value = tuple(
            value
            for _, value
            in self.replicate_select.options
        )

    def _clear_selection(
        self,
        button,
    ) -> None:
        self.replicate_select.value = ()

    def _plot_replicates(
        self,
        ax,
        records: pd.DataFrame,
    ) -> None:
        for row in records.itertuples(
            index=False
        ):
            history = self._load_history(
                row.history_path
            )

            ax.plot(
                history["t_s"],
                history["theta"],
                linewidth=self.line_width,
                label=(
                    f"Replicate "
                    f"{int(row.replicate)}"
                ),
            )

    def _plot_mean_sem(
        self,
        ax,
        records: pd.DataFrame,
    ) -> None:
        """
        Plot mean occupancy with SEM.

        Histories are interpolated onto the union of recorded time points so
        this remains robust when replicate record grids differ slightly.
        """
        histories = [
            self._load_history(
                row.history_path
            )
            for row in records.itertuples(
                index=False
            )
        ]

        if not histories:
            return

        common_time = np.unique(
            np.concatenate(
                [
                    h["t_s"].to_numpy(
                        dtype=float
                    )
                    for h in histories
                ]
            )
        )

        interpolated = []

        for history in histories:
            time = history[
                "t_s"
            ].to_numpy(dtype=float)

            theta = history[
                "theta"
            ].to_numpy(dtype=float)

            values = np.interp(
                common_time,
                time,
                theta,
                left=np.nan,
                right=np.nan,
            )

            # np.interp extrapolates edge values by default; explicitly mark
            # points outside the observed range as NaN.
            values[
                common_time < np.min(time)
            ] = np.nan
            values[
                common_time > np.max(time)
            ] = np.nan

            interpolated.append(
                values
            )

        matrix = np.vstack(
            interpolated
        )

        mean = np.nanmean(
            matrix,
            axis=0,
        )

        n = np.sum(
            np.isfinite(matrix),
            axis=0,
        )

        sd = np.nanstd(
            matrix,
            axis=0,
            ddof=1,
        )

        sem = np.divide(
            sd,
            np.sqrt(n),
            out=np.zeros_like(
                sd,
                dtype=float,
            ),
            where=n > 1,
        )

        ax.plot(
            common_time,
            mean,
            linewidth=max(
                2.0,
                self.line_width,
            ),
            label=(
                f"Mean (n={len(histories)})"
            ),
        )

        ax.fill_between(
            common_time,
            mean - sem,
            mean + sem,
            alpha=0.25,
            label="SEM",
        )

    def redraw(self) -> None:
        selected_replicates = set(
            self.replicate_select.value
        )

        records = self._records_for_condition()

        records = records.loc[
            records[
                "replicate"
            ].isin(
                selected_replicates
            )
        ]

        with self.output:
            self.output.clear_output(
                wait=True
            )

            if records.empty:
                print(
                    "Select one or more replicates."
                )
                return

            fig, ax = plt.subplots(
                figsize=self.figsize
            )

            reference_row = next(
                records.itertuples(
                    index=False
                )
            )

            reference_history = (
                self._load_history(
                    reference_row.history_path
                )
            )

            reference_run_directory = Path(
                reference_row.run_directory
            )

            if (
                self.plot_mode.value
                == "mean_sem"
            ):
                self._plot_mean_sem(
                    ax,
                    records,
                )
            else:
                self._plot_replicates(
                    ax,
                    records,
                )

            phases = _phase_ranges(
                reference_history
            )

            parameters = _read_json_if_present(
                reference_run_directory
                / "parameters.json"
            )

            if self.show_langmuir.value:
                used_labels = set()

                for phase in phases:
                    theta_eq = (
                        langmuir_equilibrium_for_phase(
                            phase["rows"],
                            fallback_parameters=parameters,
                        )
                    )

                    if (
                        theta_eq is None
                        or not np.isfinite(
                            theta_eq
                        )
                    ):
                        continue

                    label = (
                        f"Langmuir eq., "
                        f"{phase['label']}: "
                        f"{theta_eq:.3f}"
                    )

                    plot_label = (
                        label
                        if label
                        not in used_labels
                        else "_nolegend_"
                    )

                    used_labels.add(
                        label
                    )

                    ax.hlines(
                        theta_eq,
                        phase["start_t_s"],
                        phase["end_t_s"],
                        linestyles="--",
                        linewidth=2.0,
                        label=plot_label,
                    )

            if (
                self.show_phase_lines.value
                and len(phases) > 1
            ):
                for phase in phases[1:]:
                    ax.axvline(
                        phase["start_t_s"],
                        linestyle=":",
                        linewidth=1.4,
                    )

            if self.show_phase_labels.value:
                for phase in phases:
                    midpoint = 0.5 * (
                        phase["start_t_s"]
                        + phase["end_t_s"]
                    )

                    ax.text(
                        midpoint,
                        0.9,
                        phase["label"],
                        transform=(
                            ax.get_xaxis_transform()
                        ),
                        ha="center",
                        va="bottom",
                    )

            condition_label = records[
                "condition_label"
            ].iloc[0]

            ax.set_title(
                "Bound receptor fraction — "
                f"{condition_label}"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(
                "Fraction of bound receptors"
            )
            ax.set_ylim(
                *self.ylim
            )

            if self.xlim is not None:
                ax.set_xlim(
                    *self.xlim
                )

            ax.grid(
                alpha=0.25
            )

            if self.show_legend.value:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(
                        1.05,
                        1.0,
                    ),
                    borderaxespad=0.0,
                )

            fig.tight_layout()
            plt.show()

    def display(self):
        display(
            self.widget
        )
        self.redraw()
        return self


def launch_history_viewer(
    data_root: PathLike,
    *,
    figsize: Tuple[float, float] = (
        9.0,
        5.5,
    ),
    line_width: float = 1.8,
    ylim: Tuple[float, float] = (
        0.0,
        1.05,
    ),
    xlim: Optional[
        Tuple[float, float]
    ] = None,
) -> HistoryViewer:
    viewer = HistoryViewer(
        data_root,
        figsize=figsize,
        line_width=line_width,
        ylim=ylim,
        xlim=xlim,
    )
    viewer.display()
    return viewer


__all__ = [
    "RunRecord",
    "SWEEP_PARAMETER_INFO",
    "scan_saved_runs",
    "langmuir_equilibrium_for_phase",
    "HistoryViewer",
    "launch_history_viewer",
]
