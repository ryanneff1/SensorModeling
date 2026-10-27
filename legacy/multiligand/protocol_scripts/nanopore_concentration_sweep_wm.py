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
from utils.generate_geometries import make_nanopore_array_geometry
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


def concentration_directory_name(concentration_M: float) -> str:
    token = f'{float(concentration_M):.8g}'.replace('+', '')
    return f'concentration_{token}M'


def build_tasks(args, base_params: Params) -> List[Dict[str, Any]]:
    seeds = replicate_seeds(base_params.seed, args.n_replicates)
    tasks: List[Dict[str, Any]] = []
    for concentration_M in args.concentrations_M:
        for replicate, seed in enumerate(seeds, start=1):
            run_directory = (
                args.output_root
                / concentration_directory_name(concentration_M)
                / f'replicate_{replicate:03d}'
            )
            tasks.append({
                'base_params': asdict(base_params),
                'association_concentration_M': float(concentration_M),
                'dissociation_concentration_M': float(args.dissociation_concentration_M),
                'diameter_nm': float(args.diameter_nm),
                'height_nm': float(args.height_nm),
                'pitch_nm': float(args.pitch_nm),
                'rim_z_nm': float(args.rim_z_nm),
                'layout': str(args.layout),
                'edge_margin_nm': None if args.edge_margin_nm is None else float(args.edge_margin_nm),
                'replicate': int(replicate),
                'seed': int(seed),
                'association_s': float(args.association_s),
                'dissociation_s': float(args.dissociation_s),
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
    from utils.generate_geometries import make_nanopore_array_geometry
    from utils.save_simulation import save_simulation_results

    run_directory = Path(task['run_directory'])
    try:
        history_path = run_directory / f"history.{task['table_format']}"
        if history_path.exists() and not task['overwrite']:
            return {
                'status': 'skipped',
                'association_concentration_M': task['association_concentration_M'],
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
            ligand_conc_M=float(task['association_concentration_M']),
            use_well_mixed_reservoir=True,
            reservoir_offset_layers=int(task['reservoir_offset_layers']),
        )
        dissociation_params = replace(
            association_params,
            ligand_conc_M=float(task['dissociation_concentration_M']),
        )

        geometry = make_nanopore_array_geometry(
            association_params,
            pore_diameter_m=float(task['diameter_nm']) * 1e-9,
            pore_depth_m=float(task['height_nm']) * 1e-9,
            pitch_m=float(task['pitch_nm']) * 1e-9,
            rim_z_m=float(task['rim_z_nm']) * 1e-9,
            layout=task['layout'],
            edge_margin_m=None if task['edge_margin_nm'] is None else float(task['edge_margin_nm']) * 1e-9,
            name='nanopore_concentration_sweep',
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

        run_metadata = {
            'protocol': 'nanopore_concentration_sweep_well_mixed',
            'geometry_type': 'nanopore',
            'sweep_variable': 'association_concentration_M',
            'association_concentration_M': float(task['association_concentration_M']),
            'dissociation_concentration_M': float(task['dissociation_concentration_M']),
            'pore_diameter_m': float(task['diameter_nm']) * 1e-9,
            'pore_diameter_nm': float(task['diameter_nm']),
            'pore_depth_m': float(task['height_nm']) * 1e-9,
            'pore_height_nm': float(task['height_nm']),
            'pore_pitch_m': float(task['pitch_nm']) * 1e-9,
            'pore_pitch_nm': float(task['pitch_nm']),
            'rim_z_m': float(task['rim_z_nm']) * 1e-9,
            'rim_z_nm': float(task['rim_z_nm']),
            'layout': task['layout'],
            'edge_margin_nm': task['edge_margin_nm'],
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
            'association_concentration_M': float(task['association_concentration_M']),
            'replicate': int(task['replicate']),
            'seed': int(task['seed']),
            'run_directory': str(run_directory),
            'final_theta': float(final_row['theta']),
            'rebinding_events_total': int(final_row.get('rebinding_events_total', 0)),
        }
    except Exception as exc:
        return {
            'status': 'failed',
            'association_concentration_M': task.get('association_concentration_M'),
            'replicate': task.get('replicate'),
            'seed': task.get('seed'),
            'run_directory': str(run_directory),
            'error': repr(exc),
            'traceback': traceback.format_exc(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Parallel concentration sweep for one nanoporous geometry using the well-mixed reservoir.')
    parser.add_argument('--params-json', type=Path, required=True)
    parser.add_argument('--concentrations-M', type=float, nargs='+', required=True)
    parser.add_argument('--diameter-nm', type=float, required=True)
    parser.add_argument('--height-nm', type=float, required=True, help='Pore height/depth in nm.')
    parser.add_argument('--pitch-nm', type=float, required=True)
    parser.add_argument('--rim-z-nm', type=float, required=True)
    parser.add_argument('--layout', choices=('square', 'hexagonal', 'hex'), default='square')
    parser.add_argument('--edge-margin-nm', type=float, default=None)
    parser.add_argument('--n-replicates', type=int, default=5)
    parser.add_argument('--n-workers', type=int, required=True)
    parser.add_argument('--association-s', type=float, required=True)
    parser.add_argument('--dissociation-s', type=float, required=True)
    parser.add_argument('--dissociation-concentration-M', type=float, default=0.0)
    parser.add_argument('--reservoir-offset-layers', type=int, default=1)
    parser.add_argument('--record-every-s', type=float, default=None)
    parser.add_argument('--association-frames', type=int, default=20)
    parser.add_argument('--dissociation-frames', type=int, default=20)
    parser.add_argument('--table-format', choices=('parquet', 'csv'), default='parquet')
    parser.add_argument('--output-root', type=Path, default=Path('nanopore_concentration_sweep_well_mixed'))
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

    concentrations = np.asarray(args.concentrations_M, dtype=float)
    if concentrations.size == 0 or np.any(~np.isfinite(concentrations)) or np.any(concentrations < 0):
        raise ValueError('Association concentrations must be finite and nonnegative.')
    if args.diameter_nm <= 0 or args.height_nm <= 0 or args.pitch_nm <= 0:
        raise ValueError('diameter, height, and pitch must be positive.')
    if args.pitch_nm < args.diameter_nm:
        raise ValueError('pitch_nm must be >= diameter_nm.')
    if args.height_nm > args.rim_z_nm:
        raise ValueError('height_nm must be <= rim_z_nm.')

    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base_params = load_params_json(args.params_json)

    required_top_m = float(args.rim_z_nm) * 1e-9 + (int(args.reservoir_offset_layers) + 1) * base_params.a_m
    if required_top_m > base_params.H_m + 1e-15:
        raise ValueError(
            'H_m is too small for the pore rim plus the internal reservoir interface. '
            f'Current H_m={base_params.H_m:.3e} m; approximately {required_top_m:.3e} m is required.'
        )

    tasks = build_tasks(args, base_params)
    print('=' * 72)
    print('Nanopore concentration sweep with well-mixed reservoir')
    print('=' * 72)
    print(f'Concentrations (M) : {args.concentrations_M}')
    print(f'Replicates         : {args.n_replicates}')
    print(f'Tasks              : {len(tasks)}')
    print(f'Workers            : {args.n_workers}')
    print(f'Diameter (nm)      : {args.diameter_nm:g}')
    print(f'Height/depth (nm)  : {args.height_nm:g}')
    print(f'Pitch (nm)         : {args.pitch_nm:g}')
    print(f'Rim z (nm)         : {args.rim_z_nm:g}')
    print(f'Reservoir offset   : {args.reservoir_offset_layers}')
    print(f'Output root        : {args.output_root}')
    print('=' * 72)

    with ipp.Cluster(n=int(args.n_workers)) as client:
        client.wait_for_engines(int(args.n_workers))
        client[:].use_cloudpickle()
        results = client.load_balanced_view().map_async(run_protocol_task, tasks).get()

    summary = {
        'protocol': 'nanopore_concentration_sweep_well_mixed',
        'concentrations_M': [float(v) for v in args.concentrations_M],
        'n_replicates': int(args.n_replicates),
        'n_workers': int(args.n_workers),
        'diameter_nm': float(args.diameter_nm),
        'height_nm': float(args.height_nm),
        'pitch_nm': float(args.pitch_nm),
        'rim_z_nm': float(args.rim_z_nm),
        'layout': args.layout,
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
            print(f"FAILED concentration={result.get('association_concentration_M')} M, replicate={result.get('replicate')}: {result.get('error')}")
        raise RuntimeError(f'{len(failed)} task(s) failed. See task_summary.json.')


if __name__ == '__main__':
    main()
