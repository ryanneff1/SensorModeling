#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Dict, List

import ipyparallel as ipp
import numpy as np

from utils.biosensor_mc import Params, run_simulation
from utils.generate_geometries import make_flat_geometry
from utils.save_simulation import save_simulation_results


def load_params_json(path: str | Path) -> Params:
    path = Path(path).expanduser().resolve()
    with path.open('r', encoding='utf-8') as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError('The parameter JSON must contain a JSON object.')
    if 'parameters' in raw and isinstance(raw['parameters'], dict):
        raw = raw['parameters']
    valid_names = {field.name for field in fields(Params)}
    kwargs = {k: v for k, v in raw.items() if k in valid_names}
    if 'open_boundaries' in kwargs:
        kwargs['open_boundaries'] = tuple(kwargs['open_boundaries'])
    return Params(**kwargs)


def replicate_seeds(base_seed: int, n_replicates: int) -> List[int]:
    children = np.random.SeedSequence(int(base_seed)).spawn(int(n_replicates))
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def density_directory_name(density_m2: float) -> str:
    token = f'{float(density_m2):.8g}'.replace('+', '')
    return f'density_{token}_per_m2'


def build_tasks(args, base_params: Params) -> List[Dict[str, Any]]:
    seeds = replicate_seeds(base_params.seed, args.n_replicates)
    association_concentration_M = (
        float(base_params.ligand_conc_M)
        if args.association_concentration_M is None
        else float(args.association_concentration_M)
    )
    tasks: List[Dict[str, Any]] = []
    for density_m2 in args.receptor_densities_m2:
        for replicate, seed in enumerate(seeds, start=1):
            run_directory = (
                args.output_root
                / density_directory_name(density_m2)
                / f'replicate_{replicate:03d}'
            )
            tasks.append({
                'base_params': asdict(base_params),
                'receptor_density_m2': float(density_m2),
                'replicate': int(replicate),
                'seed': int(seed),
                'association_concentration_M': association_concentration_M,
                'dissociation_concentration_M': float(args.dissociation_concentration_M),
                'association_s': float(args.association_s),
                'dissociation_s': float(args.dissociation_s),
                'surface_z_m': float(args.surface_z_m),
                'reservoir_offset_layers': int(args.reservoir_offset_layers),
                'record_every_s': None if args.record_every_s is None else float(args.record_every_s),
                'association_frames': int(args.association_frames),
                'dissociation_frames': int(args.dissociation_frames),
                'table_format': str(args.table_format),
                'run_directory': str(run_directory),
                'overwrite': bool(args.overwrite),
            })
    return tasks


def run_protocol_task(task: Dict[str, Any]) -> Dict[str, Any]:
    from dataclasses import replace
    from pathlib import Path
    import traceback
    from utils.biosensor_mc import Params, run_simulation
    from utils.generate_geometries import make_flat_geometry
    from utils.save_simulation import save_simulation_results

    run_directory = Path(task['run_directory'])
    try:
        history_path = run_directory / f"history.{task['table_format']}"
        if history_path.exists() and not task['overwrite']:
            return {
                'status': 'skipped',
                'receptor_density_m2': task['receptor_density_m2'],
                'replicate': task['replicate'],
                'seed': task['seed'],
                'run_directory': str(run_directory),
            }

        params_dict = dict(task['base_params'])
        if 'open_boundaries' in params_dict:
            params_dict['open_boundaries'] = tuple(params_dict['open_boundaries'])
        base_params = Params(**params_dict)

        association_params = replace(
            base_params,
            seed=int(task['seed']),
            receptor_density_m2=float(task['receptor_density_m2']),
            receptor_count_override=None,
            ligand_conc_M=float(task['association_concentration_M']),
            use_well_mixed_reservoir=True,
            reservoir_offset_layers=int(task['reservoir_offset_layers']),
        )
        dissociation_params = replace(
            association_params,
            ligand_conc_M=float(task['dissociation_concentration_M']),
        )

        geometry = make_flat_geometry(
            association_params,
            surface_z_m=float(task['surface_z_m']),
            name='planar_receptor_density_sweep',
        )

        history, state, G, state_frames = run_simulation(
            association_params,
            seconds=float(task['association_s']),
            record_every_s=task['record_every_s'],
            return_state=True,
            show_progress=False,
            verbose=False,
            save_state_frames=True,
            n_state_frames=max(1, int(task['association_frames'])),
            geometry=geometry,
            phase_label='association',
        )

        history, state, G, state_frames, rebinding_events = run_simulation(
            dissociation_params,
            seconds=float(task['dissociation_s']),
            record_every_s=task['record_every_s'],
            return_state=True,
            show_progress=False,
            verbose=False,
            save_state_frames=True,
            n_state_frames=max(1, int(task['dissociation_frames'])),
            geometry=geometry,
            return_rebinding_events=True,
            initial_state=state,
            history=history,
            state_frames=state_frames,
            copy_initial_state=False,
            reseed_on_resume=False,
            phase_label='dissociation',
        )

        actual_density = float(G.NR / G.sensing_area_m2) if G.sensing_area_m2 > 0 else None
        run_metadata = {
            'protocol': 'planar_receptor_density_sweep_well_mixed',
            'geometry_type': 'planar',
            'sweep_variable': 'receptor_density_m2',
            'receptor_density_m2': float(task['receptor_density_m2']),
            'actual_receptor_density_m2': actual_density,
            'receptor_count': int(G.NR),
            'surface_z_m': float(task['surface_z_m']),
            'association_concentration_M': float(task['association_concentration_M']),
            'dissociation_concentration_M': float(task['dissociation_concentration_M']),
            'replicate': int(task['replicate']),
            'seed': int(task['seed']),
            'association_s': float(task['association_s']),
            'dissociation_s': float(task['dissociation_s']),
            'use_well_mixed_reservoir': True,
            'reservoir_offset_layers': int(task['reservoir_offset_layers']),
            'reservoir_interface_z_m': None if G.reservoir_interface_z_m is None else float(G.reservoir_interface_z_m),
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
            table_format=task['table_format'],
            overwrite=bool(task['overwrite']),
        )

        final_row = history.iloc[-1]
        return {
            'status': 'completed',
            'receptor_density_m2': float(task['receptor_density_m2']),
            'actual_receptor_density_m2': actual_density,
            'receptor_count': int(G.NR),
            'replicate': int(task['replicate']),
            'seed': int(task['seed']),
            'run_directory': str(run_directory),
            'final_theta': float(final_row['theta']),
            'rebinding_events_total': int(final_row.get('rebinding_events_total', 0)),
        }
    except Exception as exc:
        return {
            'status': 'failed',
            'receptor_density_m2': task.get('receptor_density_m2'),
            'replicate': task.get('replicate'),
            'seed': task.get('seed'),
            'run_directory': str(run_directory),
            'error': repr(exc),
            'traceback': traceback.format_exc(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Parallel planar receptor-density sweep using the well-mixed reservoir.')
    parser.add_argument('--params-json', type=Path, required=True)
    parser.add_argument('--receptor-densities-m2', type=float, nargs='+', required=True)
    parser.add_argument('--n-replicates', type=int, default=5)
    parser.add_argument('--n-workers', type=int, required=True)
    parser.add_argument('--association-s', type=float, required=True)
    parser.add_argument('--dissociation-s', type=float, required=True)
    parser.add_argument('--association-concentration-M', type=float, default=None)
    parser.add_argument('--dissociation-concentration-M', type=float, default=0.0)
    parser.add_argument('--surface-z-m', type=float, default=0.0)
    parser.add_argument('--reservoir-offset-layers', type=int, default=1)
    parser.add_argument('--record-every-s', type=float, default=None)
    parser.add_argument('--association-frames', type=int, default=20)
    parser.add_argument('--dissociation-frames', type=int, default=20)
    parser.add_argument('--table-format', choices=('parquet', 'csv'), default='parquet')
    parser.add_argument('--output-root', type=Path, default=Path('planar_receptor_density_sweep_well_mixed'))
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_replicates < 1 or args.n_workers < 1:
        raise ValueError('n_replicates and n_workers must be at least 1.')
    if args.association_s < 0 or args.dissociation_s < 0:
        raise ValueError('Phase durations cannot be negative.')
    if args.reservoir_offset_layers < 1:
        raise ValueError('reservoir_offset_layers must be at least 1.')

    densities = np.asarray(args.receptor_densities_m2, dtype=float)
    if densities.size == 0 or np.any(~np.isfinite(densities)) or np.any(densities < 0):
        raise ValueError('Receptor densities must be finite and nonnegative.')

    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base_params = load_params_json(args.params_json)

    envelope_index = int(round(float(args.surface_z_m) / base_params.a_m))
    explicit_max_index = envelope_index + int(args.reservoir_offset_layers)
    Nz = int(round(base_params.H_m / base_params.a_m))
    if explicit_max_index >= Nz:
        raise ValueError('H_m is too small for the planar surface plus reservoir offset.')

    tasks = build_tasks(args, base_params)
    association_concentration_M = (
        base_params.ligand_conc_M
        if args.association_concentration_M is None
        else args.association_concentration_M
    )

    print('=' * 72)
    print('Planar receptor-density sweep with well-mixed reservoir')
    print('=' * 72)
    print(f'Receptor densities    : {args.receptor_densities_m2}')
    print(f'Association conc. (M) : {association_concentration_M}')
    print(f'Replicates            : {args.n_replicates}')
    print(f'Tasks                 : {len(tasks)}')
    print(f'Workers               : {args.n_workers}')
    print(f'Surface z (m)         : {args.surface_z_m:.3e}')
    print(f'Reservoir offset      : {args.reservoir_offset_layers}')
    print(f'Output root           : {args.output_root}')
    print('=' * 72)

    with ipp.Cluster(n=int(args.n_workers)) as client:
        client.wait_for_engines(int(args.n_workers))
        client[:].use_cloudpickle()
        results = client.load_balanced_view().map_async(run_protocol_task, tasks).get()

    summary = {
        'protocol': 'planar_receptor_density_sweep_well_mixed',
        'receptor_densities_m2': [float(v) for v in args.receptor_densities_m2],
        'association_concentration_M': float(association_concentration_M),
        'dissociation_concentration_M': float(args.dissociation_concentration_M),
        'surface_z_m': float(args.surface_z_m),
        'n_replicates': int(args.n_replicates),
        'n_workers': int(args.n_workers),
        'use_well_mixed_reservoir': True,
        'reservoir_offset_layers': int(args.reservoir_offset_layers),
        'results': results,
    }
    summary_path = args.output_root / 'task_summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    failed = [r for r in results if r.get('status') == 'failed']
    print(f"Completed: {sum(r.get('status') == 'completed' for r in results)}")
    print(f"Skipped:   {sum(r.get('status') == 'skipped' for r in results)}")
    print(f'Failed:    {len(failed)}')
    print(f'Summary:   {summary_path}')
    if failed:
        for result in failed:
            print(f"FAILED density={result.get('receptor_density_m2')}, replicate={result.get('replicate')}: {result.get('error')}")
        raise RuntimeError(f'{len(failed)} task(s) failed. See task_summary.json.')


if __name__ == '__main__':
    main()