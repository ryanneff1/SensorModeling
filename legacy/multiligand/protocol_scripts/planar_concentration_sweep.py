#!/usr/bin/env python3
"""Run planar-surface concentration-sweep simulations with IPyParallel.

One independent task is created for every association-concentration/replicate
pair. Within each task, association and dissociation are run sequentially so
the dissociation phase begins from the exact microscopic endpoint of the
association phase.

Each task saves its own combined history, final microscopic state checkpoint,
planar geometry, state frames, rebinding-event table, and run metadata.

Example
-------
python planar_concentration_sweep.py \
    --params-json configs/planar_base_params.json \
    --concentrations-M 1e-9 3e-9 1e-8 3e-8 1e-7 \
    --n-replicates 5 \
    --n-workers 16 \
    --association-s 30 \
    --dissociation-s 30 \
    --dissociation-concentration-M 0 \
    --surface-z-m 0 \
    --record-every-s 0.05 \
    --association-frames 40 \
    --dissociation-frames 40 \
    --table-format parquet \
    --output-root results/planar_concentration_sweep
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

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


PROTOCOL_NAME = "planar_concentration_sweep"
PROTOCOL_VERSION = "2026-08-12-v1"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run association/dissociation simulations across ligand "
            "concentrations on a planar sensor surface using IPyParallel."
        )
    )

    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument(
        "--concentrations-M",
        type=float,
        nargs="+",
        required=True,
        help="Association-phase ligand concentrations in molar units.",
    )
    parser.add_argument("--n-replicates", type=int, required=True)
    parser.add_argument("--n-workers", type=int, required=True)

    parser.add_argument("--association-s", type=float, required=True)
    parser.add_argument("--dissociation-s", type=float, required=True)
    parser.add_argument(
        "--dissociation-concentration-M",
        type=float,
        default=0.0,
        help="Ligand concentration during wash/dissociation in M.",
    )

    parser.add_argument(
        "--surface-z-m",
        type=float,
        default=0.0,
        help="Planar sensor surface height above z=0, in meters.",
    )

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


def load_base_params(path: Path) -> Params:
    with path.expanduser().open("r", encoding="utf-8") as file:
        values = json.load(file)

    if not isinstance(values, dict):
        raise TypeError("The Params JSON file must contain one JSON object.")

    return Params(**values)


def validate_arguments(args: argparse.Namespace, base_params: Params) -> None:
    if args.n_replicates < 1:
        raise ValueError("--n-replicates must be at least 1.")

    if args.n_workers < 1:
        raise ValueError("--n-workers must be at least 1.")

    concentrations = np.asarray(args.concentrations_M, dtype=float)

    if concentrations.size == 0:
        raise ValueError("--concentrations-M must contain at least one value.")

    if not np.all(np.isfinite(concentrations)):
        raise ValueError("--concentrations-M must contain finite values.")

    if np.any(concentrations <= 0):
        raise ValueError("All association concentrations must be positive.")

    if len(set(args.concentrations_M)) != len(args.concentrations_M):
        raise ValueError("--concentrations-M must not contain duplicates.")

    if args.association_s <= 0:
        raise ValueError("--association-s must be positive.")

    if args.dissociation_s <= 0:
        raise ValueError("--dissociation-s must be positive.")

    if args.dissociation_concentration_M < 0:
        raise ValueError("--dissociation-concentration-M cannot be negative.")

    if args.record_every_s <= 0:
        raise ValueError("--record-every-s must be positive.")

    if args.association_frames < 1 or args.dissociation_frames < 1:
        raise ValueError("Frame counts must be at least 1.")

    if args.surface_z_m < 0:
        raise ValueError("--surface-z-m cannot be negative.")

    if args.surface_z_m >= base_params.H_m:
        raise ValueError(
            "--surface-z-m must be less than Params.H_m so fluid remains "
            "above the planar sensor."
        )


def replicate_seeds(base_seed: int, n_replicates: int) -> List[int]:
    """Generate one reproducible seed per replicate, paired across concentrations."""
    children = np.random.SeedSequence(int(base_seed)).spawn(n_replicates)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    ]


def concentration_directory_name(concentration_M: float) -> str:
    concentration_text = f"{float(concentration_M):.8g}"
    return f"concentration_{concentration_text}M"


def build_tasks(
    args: argparse.Namespace,
    base_params: Params,
) -> List[Dict[str, Any]]:
    seeds = replicate_seeds(base_params.seed, args.n_replicates)
    tasks: List[Dict[str, Any]] = []

    for concentration_M in args.concentrations_M:
        for replicate_index, seed in enumerate(seeds, start=1):
            tasks.append(
                {
                    "base_params": asdict(base_params),
                    "association_concentration_M": float(concentration_M),
                    "dissociation_concentration_M": float(
                        args.dissociation_concentration_M
                    ),
                    "replicate": int(replicate_index),
                    "seed": int(seed),
                    "association_s": float(args.association_s),
                    "dissociation_s": float(args.dissociation_s),
                    "surface_z_m": float(args.surface_z_m),
                    "record_every_s": float(args.record_every_s),
                    "association_frames": int(args.association_frames),
                    "dissociation_frames": int(args.dissociation_frames),
                    "table_format": str(args.table_format),
                    "output_root": str(args.output_root),
                    "overwrite": bool(args.overwrite),
                }
            )

    return tasks


def run_protocol_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run and save one concentration/replicate task."""
    from dataclasses import asdict, replace
    from pathlib import Path

    from utils.biosensor_mc import Params, run_simulation
    from utils.generate_geometries import make_flat_geometry
    from utils.save_simulation import save_simulation_results

    concentration_M = float(task["association_concentration_M"])
    concentration_nM = concentration_M * 1e9
    replicate = int(task["replicate"])

    output_directory = (
        Path(task["output_root"])
        / concentration_directory_name(concentration_M)
        / f"replicate_{replicate:03d}"
    )

    association_params = Params(**task["base_params"])
    association_params = replace(
        association_params,
        seed=int(task["seed"]),
        ligand_conc_M=concentration_M,
    )

    dissociation_params = replace(
        association_params,
        ligand_conc_M=float(task["dissociation_concentration_M"]),
    )

    geometry = make_flat_geometry(
        P=association_params,
        surface_z_m=float(task["surface_z_m"]),
        name="planar_surface",
    )

    history, state, derived, state_frames = run_simulation(
        P=association_params,
        seconds=float(task["association_s"]),
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
        seconds=float(task["dissociation_s"]),
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
            "protocol_version": PROTOCOL_VERSION,
            "geometry_type": "planar",
            "geometry_name": "planar_surface",
            "surface_z_m": float(task["surface_z_m"]),
            "association_concentration_M": concentration_M,
            "association_concentration_nM": concentration_nM,
            "dissociation_concentration_M": float(
                task["dissociation_concentration_M"]
            ),
            "dissociation_concentration_nM": float(
                task["dissociation_concentration_M"]
            )
            * 1e9,
            "replicate": replicate,
            "seed": int(task["seed"]),
            "association_s": float(task["association_s"]),
            "dissociation_s": float(task["dissociation_s"]),
            "association_params": asdict(association_params),
            "dissociation_params": asdict(dissociation_params),
        },
        table_format=str(task["table_format"]),
        overwrite=bool(task["overwrite"]),
    )

    final = history.iloc[-1]

    return {
        "association_concentration_M": concentration_M,
        "association_concentration_nM": concentration_nM,
        "replicate": replicate,
        "seed": int(task["seed"]),
        "output_directory": str(output_directory),
        "final_time_s": float(final["t_s"]),
        "final_occupancy": float(final["theta"]),
        "binding_events_total": int(final["binding_events_total"]),
        "unbinding_events_total": int(final["unbinding_events_total"]),
        "rebinding_events_total": int(final["rebinding_events_total"]),
    }


def save_task_summary(
    results: Sequence[Dict[str, Any]],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "task_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(list(results), file, indent=2, sort_keys=True)


def main() -> None:
    args = parse_arguments()

    args.params_json = args.params_json.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    base_params = load_base_params(args.params_json)
    validate_arguments(args, base_params)

    args.output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args, base_params)
    n_workers = min(args.n_workers, len(tasks))

    print(
        f"Running {len(tasks)} concentration/replicate tasks "
        f"with {n_workers} IPyParallel engines."
    )
    print(
        "Association concentrations (M): "
        + ", ".join(f"{value:.6g}" for value in args.concentrations_M)
    )
    print(
        "Association concentrations (nM): "
        + ", ".join(f"{value * 1e9:.6g}" for value in args.concentrations_M)
    )

    with ipp.Cluster(n=n_workers) as client:
        client.wait_for_engines(n_workers)
        client[:].use_cloudpickle()

        view = client.load_balanced_view()
        async_result = view.map_async(run_protocol_task, tasks)
        results = async_result.get()

    save_task_summary(results, args.output_root)

    print(f"Completed {len(results)} tasks.")
    print(f"Summary: {args.output_root / 'task_summary.json'}")


if __name__ == "__main__":
    main()
