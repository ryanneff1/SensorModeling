#!/usr/bin/env python3
"""
Parallel nanopore-height sweep using the well-mixed-reservoir transport model.

Each task corresponds to:
    nanopore height (implemented as pore_depth_m) x replicate

Replicate seeds are generated once and reused across all pore heights so the
sweep is paired across geometry conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

import ipyparallel as ipp
import numpy as np

from utils.biosensor_mc import Params, run_simulation
from utils.generate_geometries import make_nanopore_array_geometry
from utils.save_simulation import save_simulation_results


def _load_params_json(path: str | Path) -> Params:
    path = Path(path).expanduser().resolve()

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("The parameter JSON must contain a JSON object.")

    if "parameters" in raw and isinstance(raw["parameters"], dict):
        raw = raw["parameters"]

    valid_names = {field.name for field in fields(Params)}
    kwargs = {
        key: value
        for key, value in raw.items()
        if key in valid_names
    }

    if "open_boundaries" in kwargs:
        kwargs["open_boundaries"] = tuple(kwargs["open_boundaries"])

    return Params(**kwargs)


def _replicate_seeds(base_seed: int, n_replicates: int) -> List[int]:
    seed_sequence = np.random.SeedSequence(int(base_seed))
    children = seed_sequence.spawn(int(n_replicates))

    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    ]


def _height_directory_name(height_nm: float) -> str:
    return f"height_{height_nm:g}nm"


def _replicate_directory_name(replicate: int) -> str:
    return f"replicate_{int(replicate):03d}"


def _validate_geometry_settings(
    heights_nm: Iterable[float],
    diameter_nm: float,
    pitch_nm: float,
    rim_z_nm: float,
) -> None:
    heights_nm = np.asarray(list(heights_nm), dtype=float)

    if heights_nm.size == 0:
        raise ValueError("At least one nanopore height is required.")

    if not np.all(np.isfinite(heights_nm)):
        raise ValueError("All nanopore heights must be finite.")

    if np.any(heights_nm <= 0):
        raise ValueError("All nanopore heights must be positive.")

    if diameter_nm <= 0:
        raise ValueError("diameter_nm must be positive.")

    if pitch_nm <= 0:
        raise ValueError("pitch_nm must be positive.")

    if rim_z_nm < 0:
        raise ValueError("rim_z_nm must be nonnegative.")

    if np.max(heights_nm) > rim_z_nm:
        raise ValueError(
            "Every nanopore height/depth must be <= rim_z_nm because the "
            "pore bottom cannot extend below z=0. "
            f"Maximum requested height={np.max(heights_nm):g} nm, "
            f"rim_z_nm={rim_z_nm:g} nm."
        )

    if pitch_nm < diameter_nm:
        raise ValueError(
            "pitch_nm must be at least as large as diameter_nm."
        )


def _build_tasks(args, base_params: Params) -> List[Dict[str, Any]]:
    seeds = _replicate_seeds(
        base_seed=base_params.seed,
        n_replicates=args.n_replicates,
    )

    association_concentration_M = (
        float(base_params.ligand_conc_M)
        if args.association_concentration_M is None
        else float(args.association_concentration_M)
    )

    tasks: List[Dict[str, Any]] = []

    for height_nm in args.heights_nm:
        for replicate, seed in enumerate(seeds, start=1):
            run_directory = (
                args.output_root
                / _height_directory_name(float(height_nm))
                / _replicate_directory_name(replicate)
            )

            tasks.append(
                {
                    "base_params": asdict(base_params),
                    "height_nm": float(height_nm),
                    "diameter_nm": float(args.diameter_nm),
                    "pitch_nm": float(args.pitch_nm),
                    "rim_z_nm": float(args.rim_z_nm),
                    "edge_margin_nm": (
                        None
                        if args.edge_margin_nm is None
                        else float(args.edge_margin_nm)
                    ),
                    "layout": str(args.layout),
                    "replicate": int(replicate),
                    "seed": int(seed),
                    "association_s": float(args.association_s),
                    "dissociation_s": float(args.dissociation_s),
                    "association_concentration_M": association_concentration_M,
                    "dissociation_concentration_M": float(
                        args.dissociation_concentration_M
                    ),
                    "reservoir_offset_layers": int(
                        args.reservoir_offset_layers
                    ),
                    "record_every_s": (
                        None
                        if args.record_every_s is None
                        else float(args.record_every_s)
                    ),
                    "association_frames": int(args.association_frames),
                    "dissociation_frames": int(args.dissociation_frames),
                    "table_format": str(args.table_format),
                    "run_directory": str(run_directory),
                    "overwrite": bool(args.overwrite),
                }
            )

    return tasks


def run_protocol_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one pore-height x replicate simulation task."""
    from dataclasses import replace
    from pathlib import Path

    from utils.biosensor_mc import Params, run_simulation
    from utils.generate_geometries import make_nanopore_array_geometry
    from utils.save_simulation import save_simulation_results

    run_directory = Path(task["run_directory"])

    try:
        history_path = run_directory / f"history.{task['table_format']}"

        if history_path.exists() and not task["overwrite"]:
            return {
                "status": "skipped",
                "height_nm": task["height_nm"],
                "replicate": task["replicate"],
                "seed": task["seed"],
                "run_directory": str(run_directory),
                "message": "Output already exists.",
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
            ligand_conc_M=float(task["association_concentration_M"]),
            use_well_mixed_reservoir=True,
            reservoir_offset_layers=int(task["reservoir_offset_layers"]),
        )

        dissociation_params = replace(
            association_params,
            ligand_conc_M=float(task["dissociation_concentration_M"]),
        )

        height_m = float(task["height_nm"]) * 1e-9
        diameter_m = float(task["diameter_nm"]) * 1e-9
        pitch_m = float(task["pitch_nm"]) * 1e-9
        rim_z_m = float(task["rim_z_nm"]) * 1e-9

        edge_margin_m = (
            None
            if task["edge_margin_nm"] is None
            else float(task["edge_margin_nm"]) * 1e-9
        )

        geometry = make_nanopore_array_geometry(
            association_params,
            pore_diameter_m=diameter_m,
            pore_depth_m=height_m,
            pitch_m=pitch_m,
            rim_z_m=rim_z_m,
            layout=task["layout"],
            edge_margin_m=edge_margin_m,
            name=f"nanopore_height_{task['height_nm']:g}nm",
        )

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
            return_rebinding_events=False,
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

        run_metadata = {
            "protocol": "nanopore_height_sweep_well_mixed",
            "geometry_type": "nanopore",
            "sweep_variable": "pore_depth_m",
            "pore_height_nm": float(task["height_nm"]),
            "pore_depth_m": height_m,
            "pore_diameter_nm": float(task["diameter_nm"]),
            "pore_diameter_m": diameter_m,
            "pore_pitch_nm": float(task["pitch_nm"]),
            "pore_pitch_m": pitch_m,
            "rim_z_nm": float(task["rim_z_nm"]),
            "rim_z_m": rim_z_m,
            "edge_margin_nm": task["edge_margin_nm"],
            "layout": task["layout"],
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
            "sensor_envelope_z_index": int(
                G.sensor_envelope_z_index
            ),
            "reservoir_explicit_max_z_index": (
                None
                if G.reservoir_explicit_max_z_index is None
                else int(G.reservoir_explicit_max_z_index)
            ),
            "record_every_s": task["record_every_s"],
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
            "height_nm": float(task["height_nm"]),
            "replicate": int(task["replicate"]),
            "seed": int(task["seed"]),
            "run_directory": str(run_directory),
            "final_time_s": float(final_row["t_s"]),
            "final_theta": float(final_row["theta"]),
            "binding_events_total": int(
                final_row.get("binding_events_total", 0)
            ),
            "rebinding_events_total": int(
                final_row.get("rebinding_events_total", 0)
            ),
            "entered_from_well_mixed_bulk_total": int(
                final_row.get(
                    "entered_from_well_mixed_bulk_total",
                    0,
                )
            ),
            "lost_to_well_mixed_bulk_total": int(
                final_row.get(
                    "lost_to_well_mixed_bulk_total",
                    0,
                )
            ),
            "reservoir_interface_z_m": (
                None
                if G.reservoir_interface_z_m is None
                else float(G.reservoir_interface_z_m)
            ),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "height_nm": task.get("height_nm"),
            "replicate": task.get("replicate"),
            "seed": task.get("seed"),
            "run_directory": str(run_directory),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _write_task_summary(
    output_root: Path,
    results: List[Dict[str, Any]],
    args,
    base_params: Params,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "protocol": "nanopore_height_sweep_well_mixed",
        "base_seed": int(base_params.seed),
        "n_replicates": int(args.n_replicates),
        "n_workers": int(args.n_workers),
        "heights_nm": [float(value) for value in args.heights_nm],
        "diameter_nm": float(args.diameter_nm),
        "pitch_nm": float(args.pitch_nm),
        "rim_z_nm": float(args.rim_z_nm),
        "layout": str(args.layout),
        "edge_margin_nm": (
            None
            if args.edge_margin_nm is None
            else float(args.edge_margin_nm)
        ),
        "association_s": float(args.association_s),
        "dissociation_s": float(args.dissociation_s),
        "association_concentration_M": (
            float(base_params.ligand_conc_M)
            if args.association_concentration_M is None
            else float(args.association_concentration_M)
        ),
        "dissociation_concentration_M": float(
            args.dissociation_concentration_M
        ),
        "use_well_mixed_reservoir": True,
        "reservoir_offset_layers": int(args.reservoir_offset_layers),
        "results": results,
    }

    path = output_root / "task_summary.json"

    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return path


def build_parser() -> argparse.ArgumentParser:
    default_workers = max(
        1,
        int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run a parallel nanopore-height sweep using the internal "
            "well-mixed reservoir transport model."
        )
    )

    parser.add_argument("--params-json", type=Path, required=True)

    parser.add_argument(
        "--heights-nm",
        type=float,
        nargs="+",
        required=True,
        help=(
            "Nanopore heights/depths in nm. Passed as pore_depth_m to "
            "make_nanopore_array_geometry."
        ),
    )

    parser.add_argument("--diameter-nm", type=float, required=True)
    parser.add_argument("--pitch-nm", type=float, required=True)

    parser.add_argument(
        "--rim-z-nm",
        type=float,
        required=True,
        help=(
            "Upper slab/rim height above z=0 in nm. Must be >= the maximum "
            "swept pore height."
        ),
    )

    parser.add_argument(
        "--layout",
        choices=("square", "hexagonal", "hex"),
        default="square",
    )

    parser.add_argument("--edge-margin-nm", type=float, default=None)
    parser.add_argument("--n-replicates", type=int, default=5)

    parser.add_argument(
        "--n-workers",
        type=int,
        default=default_workers,
        help="Defaults to SLURM_CPUS_PER_TASK when available.",
    )

    parser.add_argument("--association-s", type=float, required=True)
    parser.add_argument("--dissociation-s", type=float, required=True)

    parser.add_argument(
        "--association-concentration-M",
        type=float,
        default=None,
        help="Defaults to ligand_conc_M in the parameter JSON.",
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
    parser.add_argument("--association-frames", type=int, default=30)
    parser.add_argument("--dissociation-frames", type=int, default=40)

    parser.add_argument(
        "--table-format",
        choices=("parquet", "csv"),
        default="parquet",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("nanopore_height_sweep_well_mixed"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")

    if args.n_workers < 1:
        raise ValueError("n_workers must be at least 1.")

    if args.association_s < 0 or args.dissociation_s < 0:
        raise ValueError("Phase durations cannot be negative.")

    if args.reservoir_offset_layers < 1:
        raise ValueError("reservoir_offset_layers must be at least 1.")

    _validate_geometry_settings(
        heights_nm=args.heights_nm,
        diameter_nm=args.diameter_nm,
        pitch_nm=args.pitch_nm,
        rim_z_nm=args.rim_z_nm,
    )

    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    base_params = _load_params_json(args.params_json)

    # The highest solid voxel is at approximately rim_z. offset=1 leaves one
    # fluid layer above that envelope and requires another lattice node above
    # it so that the internal reservoir interface remains inside the box.
    required_top_m = (
        float(args.rim_z_nm) * 1e-9
        + (int(args.reservoir_offset_layers) + 1) * base_params.a_m
    )

    if required_top_m > base_params.H_m + 1e-15:
        raise ValueError(
            "The simulation box is too short for the requested pore rim plus "
            "the internal reservoir interface. Increase H_m in the base "
            "parameters JSON. "
            f"Current H_m={base_params.H_m:.3e} m; approximately "
            f"{required_top_m:.3e} m is required."
        )

    tasks = _build_tasks(args, base_params)

    print("=" * 72)
    print("Nanopore height sweep with well-mixed reservoir")
    print("=" * 72)
    print(f"Tasks                 : {len(tasks)}")
    print(f"Heights (nm)          : {args.heights_nm}")
    print(f"Replicates per height : {args.n_replicates}")
    print(f"Workers               : {args.n_workers}")
    print(f"Pore diameter (nm)    : {args.diameter_nm:g}")
    print(f"Pore pitch (nm)       : {args.pitch_nm:g}")
    print(f"Rim z (nm)            : {args.rim_z_nm:g}")
    print(f"Reservoir offset      : {args.reservoir_offset_layers} layer(s)")
    print(f"Output root           : {args.output_root}")
    print("=" * 72)

    with ipp.Cluster(n=int(args.n_workers)) as client:
        client.wait_for_engines(int(args.n_workers))
        client[:].use_cloudpickle()

        view = client.load_balanced_view()
        async_result = view.map_async(run_protocol_task, tasks)
        results = async_result.get()

    summary_path = _write_task_summary(
        output_root=args.output_root,
        results=results,
        args=args,
        base_params=base_params,
    )

    completed = sum(
        result.get("status") == "completed"
        for result in results
    )
    skipped = sum(
        result.get("status") == "skipped"
        for result in results
    )
    failed = [
        result
        for result in results
        if result.get("status") == "failed"
    ]

    print()
    print("=" * 72)
    print("Sweep complete")
    print("=" * 72)
    print(f"Completed : {completed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {len(failed)}")
    print(f"Summary   : {summary_path}")

    if failed:
        print()
        print("Failed tasks:")

        for result in failed:
            print(
                f"  height={result.get('height_nm')} nm, "
                f"replicate={result.get('replicate')}: "
                f"{result.get('error')}"
            )

        raise RuntimeError(
            f"{len(failed)} task(s) failed. "
            "See task_summary.json for full tracebacks."
        )


if __name__ == "__main__":
    main()
