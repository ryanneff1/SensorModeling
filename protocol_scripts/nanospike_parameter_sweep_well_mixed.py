#!/usr/bin/env python3
"""Parallel dendritic-nanospike morphology sweep with a well-mixed reservoir."""
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

from utils.biosensor_mc import Params

SUPPORTED_SWEEP_PARAMETERS = {
    "spike_density_m2",
    "branch_levels",
    "branches_per_segment_mean",
}

GEOMETRY_ARGUMENTS = {
    "base_z_m", "spike_density_m2", "n_primary_spikes",
    "use_poisson_spike_count", "primary_height_mean_m",
    "primary_height_sd_m", "primary_base_radius_mean_m",
    "primary_base_radius_sd_m", "primary_tip_radius_mean_m",
    "primary_tip_radius_sd_m", "primary_tilt_mean_deg",
    "primary_tilt_sd_deg", "branch_levels", "branches_per_segment_mean",
    "branch_origin_fraction_range", "branch_length_fraction_mean",
    "branch_length_fraction_sd", "branch_angle_mean_deg",
    "branch_angle_sd_deg", "branch_radius_fraction",
    "branch_tip_radius_fraction", "min_primary_spacing_m",
    "edge_margin_m", "geometry_seed", "max_total_segments", "name",
}


def load_params_json(path: str | Path) -> Params:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The parameter JSON must contain a JSON object.")
    if "parameters" in raw and isinstance(raw["parameters"], dict):
        raw = raw["parameters"]
    valid = {field.name for field in fields(Params)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    if "open_boundaries" in kwargs:
        kwargs["open_boundaries"] = tuple(kwargs["open_boundaries"])
    return Params(**kwargs)


def load_geometry_json(path: str | Path) -> Tuple[Dict[str, Any], str, List[Any], Dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The geometry JSON must contain a JSON object.")
    if raw.get("geometry_type", "dendritic_nanospike_planar") != "dendritic_nanospike_planar":
        raise ValueError("geometry_type must be 'dendritic_nanospike_planar'.")
    geometry = raw.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("The geometry JSON must contain a 'geometry' object.")
    unknown = set(geometry).difference(GEOMETRY_ARGUMENTS)
    if unknown:
        raise ValueError("Unknown geometry argument(s): " + ", ".join(sorted(unknown)))
    sweep = raw.get("sweep")
    if not isinstance(sweep, dict):
        raise ValueError("The geometry JSON must contain a 'sweep' object.")
    parameter = sweep.get("parameter")
    values = sweep.get("values")
    if parameter not in SUPPORTED_SWEEP_PARAMETERS:
        raise ValueError("Unsupported sweep parameter: " + str(parameter))
    if not isinstance(values, list) or not values:
        raise ValueError("sweep.values must be a non-empty JSON array.")
    geometry = dict(geometry)

    # Treat JSON null as "use the geometry generator's default value".
    geometry = {
        key: value
        for key, value in geometry.items()
        if value is not None
    }

    # JSON stores tuples as lists.
    if "branch_origin_fraction_range" in geometry:
        geometry["branch_origin_fraction_range"] = tuple(
            geometry["branch_origin_fraction_range"]
        )

    return geometry, sweep_parameter, sweep_values, raw


def validate_sweep(geometry: Dict[str, Any], parameter: str, values: List[Any]) -> List[Any]:
    normalized: List[Any] = []
    if parameter == "branch_levels":
        for value in values:
            number = float(value)
            integer = int(round(number))
            if not math.isfinite(number) or not np.isclose(number, integer) or integer < 0:
                raise ValueError("branch_levels values must be nonnegative integers.")
            normalized.append(integer)
    elif parameter == "spike_density_m2":
        if geometry.get("n_primary_spikes") is not None:
            raise ValueError("Set n_primary_spikes to null for a spike_density_m2 sweep.")
        for value in values:
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("spike_density_m2 values must be finite and nonnegative.")
            normalized.append(number)
    else:
        for value in values:
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("branches_per_segment_mean values must be finite and nonnegative.")
            normalized.append(number)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Sweep values must not contain duplicates.")
    return normalized


def replicate_seeds(base_seed: int, n_replicates: int) -> List[int]:
    children = np.random.SeedSequence(int(base_seed)).spawn(int(n_replicates))
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def condition_directory_name(parameter: str, value: Any) -> str:
    token = str(int(value)) if parameter == "branch_levels" else f"{float(value):.8g}".replace("+", "")
    return f"{parameter}_{token}"


def build_tasks(args, base_params, geometry_base, parameter, values):
    seeds = replicate_seeds(base_params.seed, args.n_replicates)
    association_concentration_M = (
        float(base_params.ligand_conc_M)
        if args.association_concentration_M is None
        else float(args.association_concentration_M)
    )
    tasks = []
    for value in values:
        geometry_kwargs = dict(geometry_base)
        geometry_kwargs[parameter] = value
        for replicate, seed in enumerate(seeds, start=1):
            run_directory = args.output_root / condition_directory_name(parameter, value) / f"replicate_{replicate:03d}"
            tasks.append({
                "base_params": asdict(base_params),
                "geometry_kwargs": geometry_kwargs,
                "sweep_parameter": parameter,
                "sweep_value": value,
                "replicate": replicate,
                "seed": seed,
                "association_s": float(args.association_s),
                "dissociation_s": float(args.dissociation_s),
                "association_concentration_M": association_concentration_M,
                "dissociation_concentration_M": float(args.dissociation_concentration_M),
                "reservoir_offset_layers": int(args.reservoir_offset_layers),
                "record_every_s": None if args.record_every_s is None else float(args.record_every_s),
                "association_frames": int(args.association_frames),
                "dissociation_frames": int(args.dissociation_frames),
                "table_format": str(args.table_format),
                "run_directory": str(run_directory),
                "overwrite": bool(args.overwrite),
            })
    return tasks


def run_protocol_task(task: Dict[str, Any]) -> Dict[str, Any]:
    from dataclasses import replace
    from pathlib import Path
    import numpy as np
    import traceback
    from utils.biosensor_mc import Params, run_simulation
    from utils.generate_geometries import make_dendritic_nanospike_geometry
    from utils.save_simulation import save_simulation_results

    run_directory = Path(task["run_directory"])
    try:
        history_path = run_directory / f"history.{task['table_format']}"
        if history_path.exists() and not task["overwrite"]:
            return {"status": "skipped", "sweep_parameter": task["sweep_parameter"], "sweep_value": task["sweep_value"], "replicate": task["replicate"], "run_directory": str(run_directory)}

        params_dict = dict(task["base_params"])
        if "open_boundaries" in params_dict:
            params_dict["open_boundaries"] = tuple(params_dict["open_boundaries"])
        base_params = Params(**params_dict)
        association_params = replace(
            base_params,
            seed=int(task["seed"]),
            ligand_conc_M=float(task["association_concentration_M"]),
            use_well_mixed_reservoir=True,
            reservoir_offset_layers=int(task["reservoir_offset_layers"]),
        )
        dissociation_params = replace(
            association_params,
            ligand_conc_M=float(task["dissociation_concentration_M"]),
        )

        geometry_kwargs = dict(task["geometry_kwargs"])
        if "branch_origin_fraction_range" in geometry_kwargs:
            geometry_kwargs["branch_origin_fraction_range"] = tuple(geometry_kwargs["branch_origin_fraction_range"])
        geometry = make_dendritic_nanospike_geometry(association_params, **geometry_kwargs)

        history, state, G, state_frames = run_simulation(
            association_params,
            seconds=float(task["association_s"]),
            record_every_s=task["record_every_s"],
            return_state=True,
            show_progress=False,
            verbose=False,
            save_state_frames=True,
            n_state_frames=max(1, int(task["association_frames"])),
            geometry=geometry,
            phase_label="association",
        )

        history, state, G, state_frames, rebinding_events = run_simulation(
            dissociation_params,
            seconds=float(task["dissociation_s"]),
            record_every_s=task["record_every_s"],
            return_state=True,
            show_progress=False,
            verbose=False,
            save_state_frames=True,
            n_state_frames=max(1, int(task["dissociation_frames"])),
            geometry=geometry,
            return_rebinding_events=True,
            initial_state=state,
            history=history,
            state_frames=state_frames,
            copy_initial_state=False,
            reseed_on_resume=False,
            phase_label="dissociation",
        )

        true_area_m2 = float(getattr(geometry, "nanospike_total_exposed_area_m2", np.sum(geometry.surface_area_m2)))
        projected_area_m2 = float(getattr(geometry, "nanospike_projected_area_m2", association_params.Lx_m * association_params.Ly_m))
        roughness_factor = float(getattr(geometry, "roughness_factor", true_area_m2 / projected_area_m2))

        run_metadata = {
            "protocol": "nanospike_parameter_sweep_well_mixed",
            "geometry_type": "dendritic_nanospike_planar",
            "sweep_parameter": task["sweep_parameter"],
            "sweep_value": task["sweep_value"],
            "geometry_parameters": geometry_kwargs,
            "geometry_seed": geometry_kwargs.get("geometry_seed", association_params.seed),
            "replicate": int(task["replicate"]),
            "seed": int(task["seed"]),
            "association_s": float(task["association_s"]),
            "dissociation_s": float(task["dissociation_s"]),
            "association_concentration_M": float(task["association_concentration_M"]),
            "dissociation_concentration_M": float(task["dissociation_concentration_M"]),
            "use_well_mixed_reservoir": True,
            "reservoir_offset_layers": int(task["reservoir_offset_layers"]),
            "reservoir_interface_z_m": None if G.reservoir_interface_z_m is None else float(G.reservoir_interface_z_m),
            "nanospike_canopy_z_m": float(getattr(geometry, "nanospike_canopy_z_m", np.nan)),
            "nanospike_primary_count": int(getattr(geometry, "nanospike_n_primary_generated", 0)),
            "nanospike_segment_count": int(getattr(geometry, "nanospike_n_segments", 0)),
            "true_exposed_area_m2": true_area_m2,
            "projected_area_m2": projected_area_m2,
            "roughness_factor": roughness_factor,
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
            "seed": int(task["seed"]),
            "run_directory": str(run_directory),
            "roughness_factor": roughness_factor,
            "nanospike_primary_count": int(getattr(geometry, "nanospike_n_primary_generated", 0)),
            "nanospike_segment_count": int(getattr(geometry, "nanospike_n_segments", 0)),
            "nanospike_canopy_z_m": float(getattr(geometry, "nanospike_canopy_z_m", np.nan)),
            "reservoir_interface_z_m": None if G.reservoir_interface_z_m is None else float(G.reservoir_interface_z_m),
            "final_theta": float(final_row["theta"]),
            "rebinding_events_total": int(final_row.get("rebinding_events_total", 0)),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "sweep_parameter": task.get("sweep_parameter"),
            "sweep_value": task.get("sweep_value"),
            "replicate": task.get("replicate"),
            "seed": task.get("seed"),
            "run_directory": str(run_directory),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel dendritic-nanospike geometry sweep using a geometry JSON.")
    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument("--geometry-json", type=Path, required=True)
    parser.add_argument("--n-replicates", type=int, required=True)
    parser.add_argument("--n-workers", type=int, required=True)
    parser.add_argument("--association-s", type=float, required=True)
    parser.add_argument("--dissociation-s", type=float, required=True)
    parser.add_argument("--association-concentration-M", type=float, default=None)
    parser.add_argument("--dissociation-concentration-M", type=float, default=0.0)
    parser.add_argument("--reservoir-offset-layers", type=int, default=1)
    parser.add_argument("--record-every-s", type=float, default=None)
    parser.add_argument("--association-frames", type=int, default=20)
    parser.add_argument("--dissociation-frames", type=int, default=20)
    parser.add_argument("--table-format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_replicates < 1 or args.n_workers < 1:
        raise ValueError("n_replicates and n_workers must both be at least 1.")
    if args.association_s < 0 or args.dissociation_s < 0:
        raise ValueError("Phase durations cannot be negative.")
    if args.reservoir_offset_layers < 1:
        raise ValueError("reservoir_offset_layers must be at least 1.")

    base_params = load_params_json(args.params_json)
    geometry_base, parameter, values, geometry_raw = load_geometry_json(args.geometry_json)
    values = validate_sweep(geometry_base, parameter, values)
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args, base_params, geometry_base, parameter, values)

    print("=" * 72)
    print("Dendritic nanospike parameter sweep with well-mixed reservoir")
    print("=" * 72)
    print(f"Sweep parameter       : {parameter}")
    print(f"Sweep values          : {values}")
    print(f"Geometry seed         : {geometry_base.get('geometry_seed', base_params.seed)}")
    print(f"Replicates            : {args.n_replicates}")
    print(f"Tasks                 : {len(tasks)}")
    print(f"Workers               : {args.n_workers}")
    print(f"Reservoir offset      : {args.reservoir_offset_layers} layer(s)")
    print(f"Geometry JSON         : {args.geometry_json}")
    print(f"Output root           : {args.output_root}")
    print("=" * 72)

    with ipp.Cluster(n=int(args.n_workers)) as client:
        client.wait_for_engines(int(args.n_workers))
        client[:].use_cloudpickle()
        view = client.load_balanced_view()
        results = view.map_async(run_protocol_task, tasks).get()

    summary = {
        "protocol": "nanospike_parameter_sweep_well_mixed",
        "sweep_parameter": parameter,
        "sweep_values": values,
        "geometry_config": geometry_raw,
        "association_concentration_M": float(base_params.ligand_conc_M) if args.association_concentration_M is None else float(args.association_concentration_M),
        "dissociation_concentration_M": float(args.dissociation_concentration_M),
        "association_s": float(args.association_s),
        "dissociation_s": float(args.dissociation_s),
        "n_replicates": int(args.n_replicates),
        "n_workers": int(args.n_workers),
        "use_well_mixed_reservoir": True,
        "reservoir_offset_layers": int(args.reservoir_offset_layers),
        "results": results,
    }
    summary_path = args.output_root / "task_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    completed = sum(r.get("status") == "completed" for r in results)
    skipped = sum(r.get("status") == "skipped" for r in results)
    failed = [r for r in results if r.get("status") == "failed"]
    print(f"Completed : {completed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {len(failed)}")
    print(f"Summary   : {summary_path}")
    if failed:
        for r in failed:
            print(f"FAILED {r.get('sweep_parameter')}={r.get('sweep_value')}, replicate={r.get('replicate')}: {r.get('error')}")
        raise RuntimeError(f"{len(failed)} task(s) failed. See task_summary.json.")


if __name__ == "__main__":
    main()
