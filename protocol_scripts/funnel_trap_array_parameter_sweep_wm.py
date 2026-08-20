#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ipyparallel as ipp
import numpy as np

from utils.biosensor_mc import Params, run_simulation
from utils.generate_geometries import make_funnel_trap_array_geometry
from utils.save_simulation import save_simulation_results


SUPPORTED_SWEEP_PARAMETERS = {
    "surface_z_m",
    "cavity_radius_m",
    "funnel_depth_m",
    "funnel_mouth_radius_m",
    "funnel_throat_radius_m",
    "pitch_m",
    "throat_length_m",
    "cavity_overlap_m",
    "edge_margin_m",
}

GEOMETRY_ARGUMENTS = {
    "surface_z_m",
    "cavity_radius_m",
    "funnel_depth_m",
    "funnel_mouth_radius_m",
    "funnel_throat_radius_m",
    "pitch_m",
    "throat_length_m",
    "cavity_overlap_m",
    "layout",
    "edge_margin_m",
    "reactive_region",
    "name",
}


def load_params_json(path: str | Path) -> Params:
    path = Path(path).expanduser().resolve()

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if "parameters" in raw and isinstance(raw["parameters"], dict):
        raw = raw["parameters"]

    valid = {field.name for field in fields(Params)}
    kwargs = {k: v for k, v in raw.items() if k in valid}

    if "open_boundaries" in kwargs:
        kwargs["open_boundaries"] = tuple(kwargs["open_boundaries"])

    return Params(**kwargs)


def load_geometry_json(
    path: str | Path,
) -> Tuple[Dict[str, Any], str, List[float], Dict[str, Any]]:
    path = Path(path).expanduser().resolve()

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if raw.get("geometry_type", "funnel_trap_array") != "funnel_trap_array":
        raise ValueError("geometry_type must be 'funnel_trap_array'.")

    geometry = raw.get("geometry")
    sweep = raw.get("sweep")

    if not isinstance(geometry, dict):
        raise ValueError("geometry JSON must contain a 'geometry' object.")

    if not isinstance(sweep, dict):
        raise ValueError("geometry JSON must contain a 'sweep' object.")

    unknown = set(geometry).difference(GEOMETRY_ARGUMENTS)
    if unknown:
        raise ValueError(
            "Unknown geometry argument(s): "
            + ", ".join(sorted(unknown))
        )

    sweep_parameter = sweep.get("parameter")
    sweep_values = sweep.get("values")

    if sweep_parameter not in SUPPORTED_SWEEP_PARAMETERS:
        raise ValueError(
            "Unsupported sweep parameter. Supported parameters: "
            + ", ".join(sorted(SUPPORTED_SWEEP_PARAMETERS))
        )

    if not isinstance(sweep_values, list) or not sweep_values:
        raise ValueError("sweep.values must be a non-empty array.")

    # JSON null means "use the geometry generator's default".
    geometry = {
        key: value
        for key, value in geometry.items()
        if value is not None
    }

    normalized = []

    for value in sweep_values:
        number = float(value)

        if not math.isfinite(number):
            raise ValueError("All sweep values must be finite.")

        if sweep_parameter in {
            "surface_z_m",
            "throat_length_m",
            "cavity_overlap_m",
            "edge_margin_m",
        }:
            if number < 0:
                raise ValueError(
                    f"{sweep_parameter} cannot be negative."
                )
        elif number <= 0:
            raise ValueError(
                f"{sweep_parameter} sweep values must be positive."
            )

        normalized.append(number)

    if len(set(normalized)) != len(normalized):
        raise ValueError("Sweep values must not contain duplicates.")

    return geometry, str(sweep_parameter), normalized, raw


def replicate_seeds(base_seed: int, n_replicates: int) -> List[int]:
    children = np.random.SeedSequence(
        int(base_seed)
    ).spawn(int(n_replicates))

    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    ]


def condition_directory_name(
    parameter: str,
    value: float,
) -> str:
    token = f"{float(value):.8g}".replace("+", "")
    return f"{parameter}_{token}"


def build_tasks(
    args,
    base_params: Params,
    geometry_base: Dict[str, Any],
    sweep_parameter: str,
    sweep_values: List[float],
) -> List[Dict[str, Any]]:
    seeds = replicate_seeds(
        base_params.seed,
        args.n_replicates,
    )

    association_concentration_M = (
        float(base_params.ligand_conc_M)
        if args.association_concentration_M is None
        else float(args.association_concentration_M)
    )

    tasks = []

    for sweep_value in sweep_values:
        geometry_kwargs = dict(geometry_base)
        geometry_kwargs[sweep_parameter] = float(sweep_value)

        for replicate, seed in enumerate(seeds, start=1):
            run_directory = (
                args.output_root
                / condition_directory_name(
                    sweep_parameter,
                    sweep_value,
                )
                / f"replicate_{replicate:03d}"
            )

            tasks.append(
                {
                    "base_params": asdict(base_params),
                    "geometry_kwargs": geometry_kwargs,
                    "sweep_parameter": sweep_parameter,
                    "sweep_value": float(sweep_value),
                    "replicate": replicate,
                    "seed": seed,
                    "association_s": float(args.association_s),
                    "dissociation_s": float(args.dissociation_s),
                    "association_concentration_M": association_concentration_M,
                    "dissociation_concentration_M": float(
                        args.dissociation_concentration_M
                    ),
                    "reservoir_offset_layers": int(
                        args.reservoir_offset_layers
                    ),
                    "record_every_s": args.record_every_s,
                    "association_frames": int(args.association_frames),
                    "dissociation_frames": int(args.dissociation_frames),
                    "table_format": args.table_format,
                    "run_directory": str(run_directory),
                    "overwrite": bool(args.overwrite),
                }
            )

    return tasks


def run_protocol_task(task: Dict[str, Any]) -> Dict[str, Any]:
    from dataclasses import replace
    from pathlib import Path
    import traceback

    from utils.biosensor_mc import Params, run_simulation
    from utils.generate_geometries import make_funnel_trap_array_geometry
    from utils.save_simulation import save_simulation_results

    run_directory = Path(task["run_directory"])

    try:
        history_path = (
            run_directory
            / f"history.{task['table_format']}"
        )

        if history_path.exists() and not task["overwrite"]:
            return {
                "status": "skipped",
                "sweep_parameter": task["sweep_parameter"],
                "sweep_value": task["sweep_value"],
                "replicate": task["replicate"],
                "run_directory": str(run_directory),
            }

        params_dict = dict(task["base_params"])

        if "open_boundaries" in params_dict:
            params_dict["open_boundaries"] = tuple(
                params_dict["open_boundaries"]
            )

        base_params = Params(**params_dict)

        association_params = replace(
            base_params,
            seed=int(task["seed"]),
            ligand_conc_M=float(
                task["association_concentration_M"]
            ),
            use_well_mixed_reservoir=True,
            reservoir_offset_layers=int(
                task["reservoir_offset_layers"]
            ),
        )

        dissociation_params = replace(
            association_params,
            ligand_conc_M=float(
                task["dissociation_concentration_M"]
            ),
        )

        geometry = make_funnel_trap_array_geometry(
            association_params,
            **dict(task["geometry_kwargs"]),
        )

        history, state, G, state_frames = run_simulation(
            association_params,
            seconds=float(task["association_s"]),
            record_every_s=task["record_every_s"],
            return_state=True,
            show_progress=False,
            verbose=False,
            save_state_frames=True,
            n_state_frames=max(
                1,
                int(task["association_frames"]),
            ),
            geometry=geometry,
            phase_label="association",
        )

        (
            history,
            state,
            G,
            state_frames,
            rebinding_events,
        ) = run_simulation(
            dissociation_params,
            seconds=float(task["dissociation_s"]),
            record_every_s=task["record_every_s"],
            return_state=True,
            show_progress=False,
            verbose=False,
            save_state_frames=True,
            n_state_frames=max(
                1,
                int(task["dissociation_frames"]),
            ),
            geometry=geometry,
            return_rebinding_events=True,
            initial_state=state,
            history=history,
            state_frames=state_frames,
            copy_initial_state=False,
            reseed_on_resume=False,
            phase_label="dissociation",
        )

        run_metadata = {
            "protocol": (
                "funnel_trap_array_parameter_sweep_well_mixed"
            ),
            "geometry_type": "funnel_trap_array",
            "sweep_parameter": task["sweep_parameter"],
            "sweep_value": task["sweep_value"],
            "geometry_parameters": task["geometry_kwargs"],
            "replicate": int(task["replicate"]),
            "seed": int(task["seed"]),
            "association_s": float(task["association_s"]),
            "dissociation_s": float(task["dissociation_s"]),
            "association_concentration_M": float(
                task["association_concentration_M"]
            ),
            "dissociation_concentration_M": float(
                task["dissociation_concentration_M"]
            ),
            "use_well_mixed_reservoir": True,
            "reservoir_offset_layers": int(
                task["reservoir_offset_layers"]
            ),
            "reservoir_interface_z_m": (
                None
                if G.reservoir_interface_z_m is None
                else float(G.reservoir_interface_z_m)
            ),
            "funnel_trap_array_n_traps": int(
                getattr(
                    geometry,
                    "funnel_trap_array_n_traps",
                    0,
                )
            ),
            "funnel_trap_array_projected_area_fraction": float(
                getattr(
                    geometry,
                    "funnel_trap_array_projected_area_fraction",
                    np.nan,
                )
            ),
            "funnel_trap_throat_to_cavity_ratio": float(
                getattr(
                    geometry,
                    "funnel_trap_throat_to_cavity_ratio",
                    np.nan,
                )
            ),
            "funnel_trap_mouth_to_throat_ratio": float(
                getattr(
                    geometry,
                    "funnel_trap_mouth_to_throat_ratio",
                    np.nan,
                )
            ),
            "funnel_trap_throat_aspect_ratio": float(
                getattr(
                    geometry,
                    "funnel_trap_throat_aspect_ratio",
                    np.nan,
                )
            ),
        }

        save_simulation_results(
            run_directory,
            P=dissociation_params,
            history=history,
            state=state,
            G=G,
            state_frames=state_frames,
            rebinding_events=rebinding_events,
            run_metadata=run_metadata,
            table_format=task["table_format"],
            overwrite=bool(task["overwrite"]),
        )

        final_row = history.iloc[-1]

        return {
            "status": "completed",
            "sweep_parameter": task["sweep_parameter"],
            "sweep_value": task["sweep_value"],
            "replicate": int(task["replicate"]),
            "run_directory": str(run_directory),
            "final_theta": float(final_row["theta"]),
            "rebinding_events_total": int(
                final_row.get(
                    "rebinding_events_total",
                    0,
                )
            ),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "sweep_parameter": task.get("sweep_parameter"),
            "sweep_value": task.get("sweep_value"),
            "replicate": task.get("replicate"),
            "run_directory": str(run_directory),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument("--geometry-json", type=Path, required=True)
    parser.add_argument("--n-replicates", type=int, required=True)
    parser.add_argument("--n-workers", type=int, required=True)

    parser.add_argument("--association-s", type=float, required=True)
    parser.add_argument("--dissociation-s", type=float, required=True)

    parser.add_argument(
        "--association-concentration-M",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--dissociation-concentration-M",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--reservoir-offset-layers",
        type=int,
        default=1,
    )

    parser.add_argument("--record-every-s", type=float, default=None)
    parser.add_argument("--association-frames", type=int, default=20)
    parser.add_argument("--dissociation-frames", type=int, default=20)

    parser.add_argument(
        "--table-format",
        choices=("parquet", "csv"),
        default="parquet",
    )

    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")

    if args.n_workers < 1:
        raise ValueError("n_workers must be at least 1.")

    if args.reservoir_offset_layers < 1:
        raise ValueError(
            "reservoir_offset_layers must be at least 1."
        )

    base_params = load_params_json(
        args.params_json
    )

    (
        geometry_base,
        sweep_parameter,
        sweep_values,
        geometry_json_raw,
    ) = load_geometry_json(
        args.geometry_json
    )

    args.output_root = (
        args.output_root.expanduser().resolve()
    )
    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    tasks = build_tasks(
        args,
        base_params,
        geometry_base,
        sweep_parameter,
        sweep_values,
    )

    print("=" * 72)
    print(
        "Funnel-trap-array parameter sweep "
        "with well-mixed reservoir"
    )
    print("=" * 72)
    print(f"Sweep parameter : {sweep_parameter}")
    print(f"Sweep values    : {sweep_values}")
    print(f"Replicates      : {args.n_replicates}")
    print(f"Tasks           : {len(tasks)}")
    print(f"Workers         : {args.n_workers}")
    print(f"Geometry JSON   : {args.geometry_json}")
    print(f"Output root     : {args.output_root}")
    print("=" * 72)

    with ipp.Cluster(
        n=int(args.n_workers)
    ) as client:
        client.wait_for_engines(
            int(args.n_workers)
        )
        client[:].use_cloudpickle()

        view = client.load_balanced_view()

        results = view.map_async(
            run_protocol_task,
            tasks,
        ).get()

    summary = {
        "protocol": (
            "funnel_trap_array_parameter_sweep_well_mixed"
        ),
        "sweep_parameter": sweep_parameter,
        "sweep_values": sweep_values,
        "geometry_config": geometry_json_raw,
        "n_replicates": int(args.n_replicates),
        "n_workers": int(args.n_workers),
        "association_s": float(args.association_s),
        "dissociation_s": float(args.dissociation_s),
        "use_well_mixed_reservoir": True,
        "reservoir_offset_layers": int(
            args.reservoir_offset_layers
        ),
        "results": results,
    }

    summary_path = (
        args.output_root / "task_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    failed = [
        result
        for result in results
        if result.get("status") == "failed"
    ]

    print(
        f"Completed: "
        f"{sum(r.get('status') == 'completed' for r in results)}"
    )
    print(
        f"Skipped:   "
        f"{sum(r.get('status') == 'skipped' for r in results)}"
    )
    print(f"Failed:    {len(failed)}")
    print(f"Summary:   {summary_path}")

    if failed:
        for result in failed:
            print(
                f"FAILED "
                f"{result.get('sweep_parameter')}="
                f"{result.get('sweep_value')}, "
                f"replicate={result.get('replicate')}: "
                f"{result.get('error')}"
            )

        raise RuntimeError(
            f"{len(failed)} task(s) failed. "
            "See task_summary.json for tracebacks."
        )


if __name__ == "__main__":
    main()
