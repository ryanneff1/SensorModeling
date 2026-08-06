#!/usr/bin/env python3
"""Run nanopore association/dissociation simulations with IPyParallel.

One independent task is created for every nanopore-diameter/replicate pair.
Within each task, association and dissociation are run sequentially so the
second phase begins from the exact microscopic endpoint of the first phase.
Each task saves its own history, final state checkpoint, geometry, state
frames, and rebinding-event table.

The script is intended to run inside a single Slurm allocation with multiple
CPU cores. The Slurm submission script should pass all protocol values and set
``--n-workers`` to the number of IPyParallel engines to launch.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Prevent one engine from starting additional BLAS/OpenMP worker threads.
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import ipyparallel as ipp
import numpy as np

from utils.biosensor_mc import Params


PROTOCOL_NAME = "nanopore_geometry_sweep"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run association and dissociation simulations for multiple "
            "nanopore diameters and replicates using IPyParallel."
        )
    )

    parser.add_argument(
        "--params-json",
        type=Path,
        required=True,
        help=(
            "JSON file containing keyword arguments for utils.biosensor_mc.Params. "
            "The ligand concentration and seed are overridden for each task."
        ),
    )
    parser.add_argument(
        "--diameters-m",
        type=float,
        nargs="+",
        required=True,
        help="Nanopore diameters in meters.",
    )
    parser.add_argument("--n-replicates", type=int, required=True)
    parser.add_argument("--n-workers", type=int, required=True)

    parser.add_argument("--association-s", type=float, required=True)
    parser.add_argument("--dissociation-s", type=float, required=True)
    parser.add_argument(
        "--association-concentration-M",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--dissociation-concentration-M",
        type=float,
        required=True,
    )

    parser.add_argument("--pore-depth-m", type=float, required=True)
    parser.add_argument("--pitch-m", type=float, required=True)
    parser.add_argument("--rim-z-m", type=float, required=True)
    parser.add_argument(
        "--layout",
        choices=("square", "hexagonal"),
        default="square",
    )
    parser.add_argument("--edge-margin-m", type=float, default=None)

    parser.add_argument("--record-every-s", type=float, required=True)
    parser.add_argument("--association-frames", type=int, required=True)
    parser.add_argument("--dissociation-frames", type=int, required=True)
    parser.add_argument(
        "--table-format",
        choices=("parquet", "csv"),
        default="parquet",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-task output directories.",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.n_replicates < 1:
        raise ValueError("--n-replicates must be at least 1.")

    if args.n_workers < 1:
        raise ValueError("--n-workers must be at least 1.")

    if len(set(args.diameters_m)) != len(args.diameters_m):
        raise ValueError("--diameters-m must not contain duplicates.")

    positive_values = {
        "diameter": min(args.diameters_m),
        "association duration": args.association_s,
        "dissociation duration": args.dissociation_s,
        "pore depth": args.pore_depth_m,
        "pitch": args.pitch_m,
        "rim height": args.rim_z_m,
        "record interval": args.record_every_s,
    }

    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if args.association_concentration_M < 0:
        raise ValueError("Association concentration cannot be negative.")

    if args.dissociation_concentration_M < 0:
        raise ValueError("Dissociation concentration cannot be negative.")

    if args.association_frames < 1 or args.dissociation_frames < 1:
        raise ValueError("Frame counts must be at least 1.")

    if args.edge_margin_m is not None and args.edge_margin_m < 0:
        raise ValueError("Edge margin cannot be negative.")


def load_base_params(path: Path) -> Params:
    with path.expanduser().open("r", encoding="utf-8") as file:
        values = json.load(file)

    if not isinstance(values, dict):
        raise TypeError("The Params JSON file must contain one JSON object.")

    return Params(**values)


def replicate_seeds(base_seed: int, n_replicates: int) -> List[int]:
    """Generate one reproducible seed per replicate, paired across diameters."""
    children = np.random.SeedSequence(int(base_seed)).spawn(n_replicates)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    ]


def build_tasks(
    args: argparse.Namespace,
    base_params: Params,
) -> List[Dict[str, Any]]:
    seeds = replicate_seeds(base_params.seed, args.n_replicates)
    tasks: List[Dict[str, Any]] = []

    for diameter_m in args.diameters_m:
        for replicate_index, seed in enumerate(seeds, start=1):
            tasks.append(
                {
                    "base_params": asdict(base_params),
                    "diameter_m": float(diameter_m),
                    "replicate": int(replicate_index),
                    "seed": int(seed),
                    "association_seconds": float(args.association_s),
                    "dissociation_seconds": float(args.dissociation_s),
                    "association_concentration_M": (
                        float(args.association_concentration_M)
                    ),
                    "dissociation_concentration_M": (
                        float(args.dissociation_concentration_M)
                    ),
                    "pore_depth_m": float(args.pore_depth_m),
                    "pitch_m": float(args.pitch_m),
                    "rim_z_m": float(args.rim_z_m),
                    "layout": args.layout,
                    "edge_margin_m": (
                        None
                        if args.edge_margin_m is None
                        else float(args.edge_margin_m)
                    ),
                    "record_every_s": float(args.record_every_s),
                    "association_frames": int(args.association_frames),
                    "dissociation_frames": int(args.dissociation_frames),
                    "table_format": args.table_format,
                    "output_root": str(args.output_root),
                    "overwrite": bool(args.overwrite),
                }
            )

    return tasks


def run_protocol_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run and save one diameter/replicate association-dissociation task."""
    # Imports are local so every IPyParallel engine initializes its own model
    # modules after receiving the task.
    from dataclasses import asdict, replace
    from pathlib import Path

    from utils.biosensor_mc import Params, run_simulation
    from utils.generate_geometries import make_nanopore_array_geometry
    from utils.save_simulation import save_simulation_results

    diameter_m = float(task["diameter_m"])
    replicate = int(task["replicate"])

    diameter_text = f"{diameter_m}".rstrip("0").rstrip(".")
    diameter_text = diameter_text.replace(".", "p")

    output_directory = (
        Path(task["output_root"])
        / f"diameter_{diameter_text}m"
        / f"replicate_{replicate:03d}"
    )

    association_params = Params(**task["base_params"])
    association_params = replace(
        association_params,
        seed=int(task["seed"]),
        ligand_conc_M=float(task["association_concentration_M"]),
    )

    dissociation_params = replace(
        association_params,
        ligand_conc_M=float(task["dissociation_concentration_M"]),
    )

    geometry = make_nanopore_array_geometry(
        P=association_params,
        pore_diameter_m=diameter_m,
        pore_depth_m=float(task["pore_depth_m"]),
        pitch_m=float(task["pitch_m"]),
        rim_z_m=float(task["rim_z_m"]),
        layout=str(task["layout"]),
        edge_margin_m=task["edge_margin_m"],
        name=f"nanopore_array_{diameter_text}m",
    )

    history, state, derived, state_frames = run_simulation(
        P=association_params,
        seconds=float(task["association_seconds"]),
        geometry=geometry,
        record_every_s=float(task["record_every_s"]),
        return_state=True,
        save_state_frames=True,
        n_state_frames=int(task["association_frames"]),
        show_progress=False,
        verbose=False,
        phase_label="association",
    )

    history, state, derived, state_frames, events = run_simulation(
        P=dissociation_params,
        seconds=float(task["dissociation_seconds"]),
        geometry=geometry,
        record_every_s=float(task["record_every_s"]),
        initial_state=state,
        history=history,
        state_frames=state_frames,
        copy_initial_state=False,
        reseed_on_resume=False,
        return_state=True,
        save_state_frames=True,
        n_state_frames=int(task["dissociation_frames"]),
        return_rebinding_events=True,
        show_progress=False,
        verbose=False,
        phase_label="dissociation",
    )

    save_simulation_results(
        output_directory=output_directory,
        P=association_params,
        history=history,
        state=state,
        G=derived,
        state_frames=state_frames,
        rebinding_events=events,
        run_metadata={
            "protocol": PROTOCOL_NAME,
            "diameter_m": diameter_m,
            "replicate": replicate,
            "seed": int(task["seed"]),
            "association_params": asdict(association_params),
            "dissociation_params": asdict(dissociation_params),
            "pore_depth_m": float(task["pore_depth_m"]),
            "pitch_m": float(task["pitch_m"]),
            "rim_z_m": float(task["rim_z_m"]),
            "layout": str(task["layout"]),
            "edge_margin_m": task["edge_margin_m"],
        },
        table_format=str(task["table_format"]),
        overwrite=bool(task["overwrite"]),
    )

    final = history.iloc[-1]

    return {
        "diameter_m": diameter_m,
        "replicate": replicate,
        "seed": int(task["seed"]),
        "output_directory": str(output_directory),
        "final_time_s": float(final["t_s"]),
        "final_occupancy": float(final["theta"]),
        "binding_events_total": int(final["binding_events_total"]),
        "unbinding_events_total": int(final["unbinding_events_total"]),
        "rebinding_events_total": int(final["rebinding_events_total"]),
    }


def save_task_summary(results: Sequence[Dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "task_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(list(results), file, indent=2, sort_keys=True)


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    args.params_json = args.params_json.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    base_params = load_base_params(args.params_json)
    tasks = build_tasks(args, base_params)
    n_workers = min(args.n_workers, len(tasks))

    print(
        f"Running {len(tasks)} diameter/replicate tasks "
        f"with {n_workers} IPyParallel engines."
    )

    # This starts local engines inside the current Slurm allocation. A
    # LoadBalancedView assigns each independent simulation chain to the next
    # available engine.
    with ipp.Cluster(n=n_workers) as client:
        client.wait_for_engines(n_workers)

        # The worker is defined in this script. Cloudpickle allows the engines
        # to receive it without requiring this file to be installed as a Python
        # package, while utils.* must still be importable on every engine.
        client[:].use_cloudpickle()

        view = client.load_balanced_view()
        async_result = view.map_async(run_protocol_task, tasks)
        results = async_result.get()

    save_task_summary(results, args.output_root)

    print(f"Completed {len(results)} tasks.")
    print(f"Summary: {args.output_root / 'task_summary.json'}")


if __name__ == "__main__":
    main()
