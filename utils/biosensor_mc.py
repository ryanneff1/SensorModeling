# biosensor_mc.py
#
# Lattice Monte Carlo biosensor model with generalized voxelized sensor geometry.
# Spatial quantities use SI units: m, m^2, m^3, and m^2/s.
# Concentration and kinetic constants remain in M, M^-1 s^-1, and s^-1.
#
# Geometry convention
# -------------------
# The x and y lattice indices run from 0 to Nx-1 and 0 to Ny-1.
# The z lattice index runs from 0 to Nz. This retains the original planar
# convention in which the flat sensor occupies z=0 and fluid occupies z=1..Nz.
# A SensorGeometry supplies a Boolean solid_mask with shape (Nx, Ny, Nz + 1).
# Ligands may occupy only bulk-accessible fluid sites.

from __future__ import annotations

import copy
import warnings
from collections import deque
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover - used only when scipy is unavailable
    distance_transform_edt = None


MODEL_VERSION = "2026-08-17-well-mixed-reservoir-v1"

NA = 6.02214076e23
L_PER_M3 = 1e3

BOUNDARY_FACES = (
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
)

MOVE_VECTORS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int32,
)

FACE_DIRECTIONS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


# -----------------------------------------------------------------------------
# Parameters and geometry data structures
# -----------------------------------------------------------------------------


@dataclass
class Params:
    Lx_m: float = 2e-6
    Ly_m: float = 2e-6
    H_m: float = 0.2e-6
    a_m: float = 0.02e-6

    D_m2_s: float = 1e-11

    receptor_density_m2: float = 1e14
    receptor_count_override: Optional[int] = None
    ligand_conc_M: float = 10e-9

    k_on_M_inv_s: float = 1e7
    k_off_s: float = 0.1

    dt_s: Optional[float] = None
    reaction_volume_voxels: int = 1

    # escape_distance_m is the geometry-independent parameter. The older
    # escape_height_m name is retained for backward compatibility.
    escape_distance_m: Optional[float] = None
    escape_height_m: float = 0.1e-6

    use_poisson_ligand_count: bool = True
    allow_multiple_receptors_per_site: bool = False

    # These faces exchange ligands with a bulk reservoir. Other outer faces
    # are reflecting. The default reproduces the original flat-sensor model.
    open_boundaries: Tuple[str, ...] = (
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_max",
    )

    # Optional reduced-order mixing model. When enabled, the explicitly
    # simulated diffusion domain ends reservoir_offset_layers lattice spacings
    # above the highest solid sensor voxel. A constant-concentration,
    # well-mixed reservoir occupies the region above that internal interface.
    #
    # reservoir_offset_layers=1 leaves one complete explicit fluid lattice
    # layer between the upper sensor envelope and the reservoir. In this mode,
    # the outer box boundaries are reflecting; all bulk exchange occurs through
    # the internal reservoir interface.
    use_well_mixed_reservoir: bool = False
    reservoir_offset_layers: int = 1

    seed: int = 1


@dataclass
class SensorGeometry:
    """Voxelized sensor geometry and its exposed solid-fluid interfaces."""

    name: str
    solid_mask: np.ndarray

    surface_solid_xyz: np.ndarray
    surface_fluid_xyz: np.ndarray
    surface_normals: np.ndarray
    surface_centers_m: np.ndarray
    surface_area_m2: np.ndarray

    # True for faces that are chemically reactive. Faces adjacent to fluid
    # that is not connected to the bulk are filtered later during derive().
    reactive_face_mask: np.ndarray

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.solid_mask.shape)

    @property
    def n_surface_faces(self) -> int:
        return int(self.surface_solid_xyz.shape[0])

    @property
    def n_reactive_faces(self) -> int:
        return int(np.count_nonzero(self.reactive_face_mask))


@dataclass
class Derived:
    Nx: int
    Ny: int
    Nz: int
    grid_shape: Tuple[int, int, int]

    geometry: SensorGeometry
    accessible_fluid_mask: np.ndarray
    accessible_fluid_xyz: np.ndarray
    boundary_fluid_sites: Dict[str, np.ndarray]

    reactive_face_ids: np.ndarray
    reaction_site_mask: np.ndarray
    distance_to_reactive_surface_m: np.ndarray

    volume_m3: float
    sensing_area_m2: float
    NR: int
    mean_ligands: float
    bulk_conc_m3: float
    D_m2_s: float
    a_m: float
    open_boundaries: Tuple[str, ...]

    use_well_mixed_reservoir: bool
    reservoir_offset_layers: int
    sensor_envelope_z_index: int
    reservoir_explicit_max_z_index: Optional[int]
    reservoir_interface_z_m: Optional[float]
    reservoir_boundary_sites: np.ndarray
    reservoir_injection_mean: float

    face_injection_means: Dict[str, float]
    total_injection_mean: float

    dt_s: float
    move_probs: np.ndarray
    kon_exp_per_receptor: float
    p_off: float
    Kd_M: float
    escape_distance_m: float


@dataclass
class State:
    rng: np.random.Generator

    receptor_face_id: np.ndarray
    receptor_xyz: np.ndarray
    receptor_xy: np.ndarray
    receptor_release_xyz: np.ndarray
    receptor_normal: np.ndarray
    receptor_ligand: np.ndarray

    site_to_receptors: List[List[int]]
    receptor_grid: np.ndarray
    use_receptor_grid: bool

    ligand_uid: np.ndarray
    ligand_xyz: np.ndarray
    ligand_receptor: np.ndarray
    ligand_active: np.ndarray

    last_unbound_receptor: np.ndarray
    rebind_watch_active: np.ndarray
    rebind_event_id: np.ndarray
    rebind_unbind_time_s: np.ndarray
    rebind_unbind_xyz: np.ndarray
    rebind_max_surface_distance_m: np.ndarray
    rebind_free_steps: np.ndarray

    ligand_n_bindings: np.ndarray
    ligand_n_rebindings: np.ndarray
    ligand_n_self_rebindings: np.ndarray
    ligand_n_cross_rebindings: np.ndarray

    n_ligands_created: int
    next_unbind_event_id: int
    rebinding_records: List[Dict]

    step_count: int
    t_s: float
    event_counts: Dict[str, int]



# Geometry functions are imported only after Params and SensorGeometry exist.
# This avoids a circular-import failure because generate_geometries.py imports
# those data structures from this module. Private helper functions are imported
# explicitly because wildcard imports do not include underscore-prefixed names.
from utils.generate_geometries import (
    _boundary_sites,
    _bulk_accessible_fluid_mask,
    _distance_to_reactive_surface,
    _grid_counts,
    _validate_open_boundaries,
    geometry_from_solid_mask,
    make_cylindrical_post_geometry,
    make_cylindrical_well_geometry,
    make_flat_geometry,
    make_height_field_geometry,
    make_implicit_geometry,
    make_nanopore_array_geometry,
    make_spherical_bowl_geometry,
    make_spherical_cap_geometry,
)

# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def derive(
    P: Params,
    geometry: Optional[SensorGeometry] = None,
) -> Derived:
    Nx, Ny, Nz = _grid_counts(P)
    grid_shape = (Nx, Ny, Nz + 1)

    if P.D_m2_s <= 0:
        raise ValueError("D_m2_s must be positive.")

    if P.reaction_volume_voxels < 1:
        raise ValueError("reaction_volume_voxels must be at least 1.")

    requested_open_boundaries = _validate_open_boundaries(P.open_boundaries)

    if geometry is None:
        geometry = make_flat_geometry(P)

    if geometry.shape != grid_shape:
        raise ValueError(
            f"Geometry shape {geometry.shape} does not match expected "
            f"lattice shape {grid_shape}."
        )

    if int(P.reservoir_offset_layers) < 1:
        raise ValueError("reservoir_offset_layers must be at least 1.")

    fluid_mask = ~geometry.solid_mask

    # Determine bulk connectivity before truncating the explicit diffusion
    # domain. This preserves the existing exclusion of sealed fluid cavities.
    bulk_accessible_fluid_mask = _bulk_accessible_fluid_mask(
        fluid_mask,
        requested_open_boundaries,
    )

    solid_xyz = np.argwhere(geometry.solid_mask)
    if solid_xyz.size == 0:
        raise ValueError("The geometry contains no solid sensor sites.")

    sensor_envelope_z_index = int(np.max(solid_xyz[:, 2]))
    use_well_mixed_reservoir = bool(P.use_well_mixed_reservoir)
    reservoir_offset_layers = int(P.reservoir_offset_layers)

    reservoir_explicit_max_z_index: Optional[int]
    reservoir_interface_z_m: Optional[float]
    reservoir_boundary_sites: np.ndarray
    reservoir_injection_mean = 0.0

    if use_well_mixed_reservoir:
        # The upper surface face of the highest solid voxel is located at
        # (sensor_envelope_z_index + 0.5) * a. offset=1 keeps the fluid node at
        # envelope+1 explicit and places the reservoir interface at
        # (envelope+1.5) * a, exactly one lattice spacing above the surface.
        reservoir_explicit_max_z_index = (
            sensor_envelope_z_index + reservoir_offset_layers
        )

        if reservoir_explicit_max_z_index >= Nz:
            raise ValueError(
                "The well-mixed reservoir interface must lie inside the "
                "simulation box. Increase H_m or decrease "
                "reservoir_offset_layers. "
                f"Sensor envelope z index={sensor_envelope_z_index}, "
                f"explicit max z index={reservoir_explicit_max_z_index}, "
                f"Nz={Nz}."
            )

        accessible_fluid_mask = bulk_accessible_fluid_mask.copy()
        accessible_fluid_mask[
            :,
            :,
            reservoir_explicit_max_z_index + 1 :,
        ] = False

        plane_mask = accessible_fluid_mask[
            :,
            :,
            reservoir_explicit_max_z_index,
        ]
        local_xy = np.argwhere(plane_mask).astype(np.int32)

        if local_xy.size == 0:
            raise ValueError(
                "No bulk-accessible fluid sites lie on the requested "
                "well-mixed reservoir interface."
            )

        reservoir_boundary_sites = np.column_stack(
            [
                local_xy[:, 0],
                local_xy[:, 1],
                np.full(
                    local_xy.shape[0],
                    reservoir_explicit_max_z_index,
                    dtype=np.int32,
                ),
            ]
        ).astype(np.int32)

        reservoir_interface_z_m = (
            reservoir_explicit_max_z_index + 0.5
        ) * P.a_m

        # The internal reservoir replaces exchange at the outer box faces.
        open_boundaries: Tuple[str, ...] = ()
    else:
        accessible_fluid_mask = bulk_accessible_fluid_mask
        reservoir_explicit_max_z_index = None
        reservoir_interface_z_m = None
        reservoir_boundary_sites = np.empty((0, 3), dtype=np.int32)
        open_boundaries = requested_open_boundaries

    accessible_fluid_xyz = np.argwhere(accessible_fluid_mask).astype(np.int32)

    if accessible_fluid_xyz.size == 0:
        raise ValueError("The geometry contains no bulk-accessible fluid sites.")

    face_fluid_xyz = geometry.surface_fluid_xyz
    face_accessible = accessible_fluid_mask[
        face_fluid_xyz[:, 0],
        face_fluid_xyz[:, 1],
        face_fluid_xyz[:, 2],
    ]
    active_reactive_face_mask = geometry.reactive_face_mask & face_accessible
    reactive_face_ids = np.flatnonzero(active_reactive_face_mask).astype(np.int64)

    if reactive_face_ids.size == 0 and (
        P.receptor_count_override not in (None, 0) or P.receptor_density_m2 > 0
    ):
        raise ValueError(
            "No reactive sensor faces are adjacent to bulk-accessible fluid."
        )

    sensing_area_m2 = float(
        np.sum(geometry.surface_area_m2[reactive_face_ids])
    )
    volume_m3 = float(accessible_fluid_xyz.shape[0] * P.a_m**3)

    if P.receptor_count_override is None:
        NR = int(round(P.receptor_density_m2 * sensing_area_m2))
    else:
        NR = int(P.receptor_count_override)

    if NR < 0:
        raise ValueError("The receptor count cannot be negative.")

    if (
        not P.allow_multiple_receptors_per_site
        and NR > reactive_face_ids.size
    ):
        raise ValueError(
            "The requested receptor count exceeds one receptor per reactive "
            "surface face. Decrease receptor density, refine the lattice, or "
            "set allow_multiple_receptors_per_site=True."
        )

    bulk_conc_m3 = P.ligand_conc_M * NA * L_PER_M3
    mean_ligands = bulk_conc_m3 * volume_m3

    if P.dt_s is None:
        dt_s = 0.95 * P.a_m**2 / (6 * P.D_m2_s)
    else:
        dt_s = float(P.dt_s)

    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    p_axis = P.D_m2_s * dt_s / P.a_m**2
    p_stay = 1.0 - 6.0 * p_axis

    if p_stay < -1e-12:
        raise ValueError(
            "dt_s is too large for the six-neighbor diffusion kernel. "
            "Require dt_s <= a_m^2 / (6 D_m2_s)."
        )

    move_probs = np.array(
        [max(0.0, p_stay), p_axis, p_axis, p_axis, p_axis, p_axis, p_axis],
        dtype=float,
    )
    move_probs /= move_probs.sum()

    V_rxn_L = P.reaction_volume_voxels * P.a_m**3 * L_PER_M3

    if P.k_on_M_inv_s > 0:
        kon_exp_per_receptor = P.k_on_M_inv_s * dt_s / (NA * V_rxn_L)
        Kd_M = P.k_off_s / P.k_on_M_inv_s
    else:
        kon_exp_per_receptor = 0.0
        Kd_M = np.inf

    p_off = 1.0 - np.exp(-P.k_off_s * dt_s)

    escape_distance_m = (
        float(P.escape_distance_m)
        if P.escape_distance_m is not None
        else float(P.escape_height_m)
    )

    if escape_distance_m <= 0:
        raise ValueError("escape_distance_m must be positive.")

    reaction_site_mask = np.zeros(grid_shape, dtype=bool)

    if reactive_face_ids.size:
        reaction_xyz = geometry.surface_fluid_xyz[reactive_face_ids]
        reaction_site_mask[
            reaction_xyz[:, 0],
            reaction_xyz[:, 1],
            reaction_xyz[:, 2],
        ] = True

        distance_to_reactive_surface_m = _distance_to_reactive_surface(
            grid_shape,
            geometry.surface_solid_xyz[reactive_face_ids],
            P.a_m,
        )
    else:
        distance_to_reactive_surface_m = np.full(grid_shape, np.inf)

    boundary_fluid_sites: Dict[str, np.ndarray] = {}
    face_injection_means: Dict[str, float] = {}
    exchange_prefactor = bulk_conc_m3 * P.D_m2_s * dt_s / P.a_m

    if use_well_mixed_reservoir:
        reservoir_area_m2 = reservoir_boundary_sites.shape[0] * P.a_m**2
        reservoir_injection_mean = exchange_prefactor * reservoir_area_m2

    for face in BOUNDARY_FACES:
        if face in open_boundaries:
            sites = _boundary_sites(accessible_fluid_mask, face)
        else:
            sites = np.empty((0, 3), dtype=np.int32)

        boundary_fluid_sites[face] = sites
        boundary_area_m2 = sites.shape[0] * P.a_m**2
        face_injection_means[face] = exchange_prefactor * boundary_area_m2

    total_injection_mean = (
        float(reservoir_injection_mean)
        if use_well_mixed_reservoir
        else float(sum(face_injection_means.values()))
    )

    return Derived(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        grid_shape=grid_shape,
        geometry=geometry,
        accessible_fluid_mask=accessible_fluid_mask,
        accessible_fluid_xyz=accessible_fluid_xyz,
        boundary_fluid_sites=boundary_fluid_sites,
        reactive_face_ids=reactive_face_ids,
        reaction_site_mask=reaction_site_mask,
        distance_to_reactive_surface_m=distance_to_reactive_surface_m,
        volume_m3=volume_m3,
        sensing_area_m2=sensing_area_m2,
        NR=NR,
        mean_ligands=mean_ligands,
        bulk_conc_m3=bulk_conc_m3,
        D_m2_s=P.D_m2_s,
        a_m=P.a_m,
        open_boundaries=open_boundaries,
        use_well_mixed_reservoir=use_well_mixed_reservoir,
        reservoir_offset_layers=reservoir_offset_layers,
        sensor_envelope_z_index=sensor_envelope_z_index,
        reservoir_explicit_max_z_index=reservoir_explicit_max_z_index,
        reservoir_interface_z_m=reservoir_interface_z_m,
        reservoir_boundary_sites=reservoir_boundary_sites,
        reservoir_injection_mean=float(reservoir_injection_mean),
        face_injection_means=face_injection_means,
        total_injection_mean=total_injection_mean,
        dt_s=dt_s,
        move_probs=move_probs,
        kon_exp_per_receptor=kon_exp_per_receptor,
        p_off=p_off,
        Kd_M=Kd_M,
        escape_distance_m=escape_distance_m,
    )


def _flat_site_index(xyz: np.ndarray, G: Derived) -> np.ndarray:
    xyz = np.asarray(xyz)
    return (
        (xyz[..., 0] * G.Ny + xyz[..., 1]) * (G.Nz + 1)
        + xyz[..., 2]
    ).astype(np.int64)


def _initial_event_counts() -> Dict[str, int]:
    counts = {
        "bindings": 0,
        "unbindings": 0,
        "rebindings": 0,
        "self_rebindings": 0,
        "cross_rebindings": 0,
        "local_rebinding_escapes": 0,
        "rebind_watch_bulk_losses": 0,
        "rebind_watch_well_mixed_bulk_losses": 0,
        "lost_to_bulk": 0,
        "entered_from_bulk": 0,
        "lost_to_well_mixed_bulk": 0,
        "entered_from_well_mixed_bulk": 0,
    }

    for face in BOUNDARY_FACES:
        counts[f"lost_{face}"] = 0
        counts[f"entered_{face}"] = 0

    return counts


def initialize(P: Params, G: Derived) -> State:
    rng = np.random.default_rng(P.seed)
    n_grid_sites = int(np.prod(G.grid_shape))

    if G.NR > 0:
        if P.allow_multiple_receptors_per_site:
            chosen_local = rng.integers(
                0,
                G.reactive_face_ids.size,
                size=G.NR,
            )
        else:
            chosen_local = rng.choice(
                G.reactive_face_ids.size,
                size=G.NR,
                replace=False,
            )

        receptor_face_id = G.reactive_face_ids[chosen_local].astype(np.int64)
        receptor_xyz = G.geometry.surface_solid_xyz[receptor_face_id].copy()
        receptor_release_xyz = G.geometry.surface_fluid_xyz[receptor_face_id].copy()
        receptor_normal = G.geometry.surface_normals[receptor_face_id].copy()
    else:
        receptor_face_id = np.empty(0, dtype=np.int64)
        receptor_xyz = np.empty((0, 3), dtype=np.int32)
        receptor_release_xyz = np.empty((0, 3), dtype=np.int32)
        receptor_normal = np.empty((0, 3), dtype=np.int8)

    site_to_receptors: List[List[int]] = [[] for _ in range(n_grid_sites)]

    if G.NR:
        release_sites = _flat_site_index(receptor_release_xyz, G)

        for receptor_id, site_id in enumerate(release_sites):
            site_to_receptors[int(site_id)].append(int(receptor_id))
    else:
        release_sites = np.empty(0, dtype=np.int64)

    receptor_grid = np.full(n_grid_sites, -1, dtype=np.int64)
    use_receptor_grid = True

    for site_id, receptors in enumerate(site_to_receptors):
        if len(receptors) == 1:
            receptor_grid[site_id] = receptors[0]
        elif len(receptors) > 1:
            receptor_grid[site_id] = -2
            use_receptor_grid = False

    receptor_ligand = np.full(G.NR, -1, dtype=np.int64)

    if P.use_poisson_ligand_count:
        N_L = int(rng.poisson(G.mean_ligands))
    else:
        N_L = int(round(G.mean_ligands))

    if N_L > 0:
        chosen_fluid = rng.integers(
            0,
            G.accessible_fluid_xyz.shape[0],
            size=N_L,
        )
        ligand_xyz = G.accessible_fluid_xyz[chosen_fluid].copy()
    else:
        ligand_xyz = np.empty((0, 3), dtype=np.int32)

    return State(
        rng=rng,
        receptor_face_id=receptor_face_id,
        receptor_xyz=receptor_xyz,
        receptor_xy=receptor_xyz[:, :2].copy(),
        receptor_release_xyz=receptor_release_xyz,
        receptor_normal=receptor_normal,
        receptor_ligand=receptor_ligand,
        site_to_receptors=site_to_receptors,
        receptor_grid=receptor_grid,
        use_receptor_grid=use_receptor_grid,
        ligand_uid=np.arange(N_L, dtype=np.int64),
        ligand_xyz=ligand_xyz,
        ligand_receptor=np.full(N_L, -1, dtype=np.int64),
        ligand_active=np.ones(N_L, dtype=bool),
        last_unbound_receptor=np.full(N_L, -1, dtype=np.int64),
        rebind_watch_active=np.zeros(N_L, dtype=bool),
        rebind_event_id=np.full(N_L, -1, dtype=np.int64),
        rebind_unbind_time_s=np.full(N_L, np.nan),
        rebind_unbind_xyz=np.full((N_L, 3), -1, dtype=np.int32),
        rebind_max_surface_distance_m=np.zeros(N_L, dtype=float),
        rebind_free_steps=np.zeros(N_L, dtype=np.int64),
        ligand_n_bindings=np.zeros(N_L, dtype=np.int64),
        ligand_n_rebindings=np.zeros(N_L, dtype=np.int64),
        ligand_n_self_rebindings=np.zeros(N_L, dtype=np.int64),
        ligand_n_cross_rebindings=np.zeros(N_L, dtype=np.int64),
        n_ligands_created=N_L,
        next_unbind_event_id=0,
        rebinding_records=[],
        step_count=0,
        t_s=0.0,
        event_counts=_initial_event_counts(),
    )




# -----------------------------------------------------------------------------
# Restart and continuation helpers
# -----------------------------------------------------------------------------


STATE_CHECKPOINT_VERSION = 1


def state_to_checkpoint(S: State) -> Dict[str, Any]:
    """
    Convert a State into a restart checkpoint.

    The checkpoint contains the complete microscopic state, cumulative event
    counters, open rebinding watches, and the NumPy random-number-generator
    state. It can be serialized with pickle and later supplied to
    ``run_simulation(initial_state=checkpoint)``.
    """
    state_fields: Dict[str, Any] = {}

    for field_info in fields(State):
        name = field_info.name

        if name == "rng":
            continue

        value = getattr(S, name)
        state_fields[name] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)

    return {
        "checkpoint_version": STATE_CHECKPOINT_VERSION,
        "bit_generator": type(S.rng.bit_generator).__name__,
        "rng_state": copy.deepcopy(S.rng.bit_generator.state),
        "state_fields": state_fields,
    }


def state_from_checkpoint(checkpoint: Mapping[str, Any]) -> State:
    """Reconstruct a State from ``state_to_checkpoint`` output."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping.")

    version = int(checkpoint.get("checkpoint_version", -1))

    if version != STATE_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version}; expected "
            f"{STATE_CHECKPOINT_VERSION}."
        )

    bit_generator_name = str(checkpoint.get("bit_generator", "PCG64"))
    bit_generator_class = getattr(np.random, bit_generator_name, None)

    if bit_generator_class is None:
        raise ValueError(
            f"NumPy does not provide the saved bit generator "
            f"{bit_generator_name!r}."
        )

    rng = np.random.Generator(bit_generator_class())
    rng.bit_generator.state = copy.deepcopy(checkpoint["rng_state"])

    saved_fields = checkpoint.get("state_fields")

    if not isinstance(saved_fields, Mapping):
        raise ValueError("checkpoint is missing the 'state_fields' mapping.")

    kwargs: Dict[str, Any] = {"rng": rng}
    missing = []

    for field_info in fields(State):
        name = field_info.name

        if name == "rng":
            continue

        if name not in saved_fields:
            missing.append(name)
            continue

        value = saved_fields[name]
        kwargs[name] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)

    if missing:
        raise ValueError(
            "checkpoint is missing State fields: " + ", ".join(missing)
        )

    return State(**kwargs)


def clone_state(S: State) -> State:
    """Return a deep, trajectory-identical copy of a simulation State."""
    if not isinstance(S, State):
        raise TypeError("S must be a State instance.")

    return state_from_checkpoint(state_to_checkpoint(S))


def _coerce_initial_state(
    initial_state: Union[State, Mapping[str, Any]],
    copy_initial_state: bool,
) -> State:
    if isinstance(initial_state, State):
        return clone_state(initial_state) if copy_initial_state else initial_state

    if isinstance(initial_state, Mapping):
        return state_from_checkpoint(initial_state)

    raise TypeError(
        "initial_state must be a State or a checkpoint mapping returned by "
        "state_to_checkpoint()."
    )


def _rebuild_receptor_site_lookup(S: State, G: Derived) -> None:
    """Rebuild geometry-dependent receptor lookup tables for a resumed state."""
    n_grid_sites = int(np.prod(G.grid_shape))
    site_to_receptors: List[List[int]] = [[] for _ in range(n_grid_sites)]

    if S.receptor_release_xyz.shape[0] > 0:
        release_sites = _flat_site_index(S.receptor_release_xyz, G)

        for receptor_id, site_id in enumerate(release_sites):
            site_to_receptors[int(site_id)].append(int(receptor_id))

    receptor_grid = np.full(n_grid_sites, -1, dtype=np.int64)
    use_receptor_grid = True

    for site_id, receptors in enumerate(site_to_receptors):
        if len(receptors) == 1:
            receptor_grid[site_id] = receptors[0]
        elif len(receptors) > 1:
            receptor_grid[site_id] = -2
            use_receptor_grid = False

    S.site_to_receptors = site_to_receptors
    S.receptor_grid = receptor_grid
    S.use_receptor_grid = use_receptor_grid


def _validate_resume_state(S: State, G: Derived) -> None:
    """
    Validate that a saved microscopic state is compatible with a new phase.

    Dynamic parameters such as concentration, D, k_on, k_off, dt, and open
    boundary exchange rates may change. The lattice, sensor geometry, receptor
    count, and receptor placement must remain unchanged.
    """
    n_receptors = int(S.receptor_face_id.shape[0])

    if n_receptors != G.NR:
        raise ValueError(
            "The resumed State contains "
            f"{n_receptors} receptors, but the new parameters/geometry derive "
            f"{G.NR}. Keep receptor density/count and geometry unchanged when "
            "continuing a microscopic state."
        )

    receptor_fields = (
        "receptor_xyz",
        "receptor_xy",
        "receptor_release_xyz",
        "receptor_normal",
        "receptor_ligand",
    )

    for name in receptor_fields:
        value = getattr(S, name)

        if value.shape[0] != n_receptors:
            raise ValueError(
                f"State field {name!r} has {value.shape[0]} receptor rows; "
                f"expected {n_receptors}."
            )

    if n_receptors:
        if np.any(S.receptor_face_id < 0) or np.any(
            S.receptor_face_id >= G.geometry.n_surface_faces
        ):
            raise ValueError("The resumed receptor_face_id array is invalid.")

        active_face_set = set(map(int, G.reactive_face_ids.tolist()))
        invalid_faces = [
            int(face_id)
            for face_id in S.receptor_face_id
            if int(face_id) not in active_face_set
        ]

        if invalid_faces:
            raise ValueError(
                "The new geometry does not contain all receptor-bearing "
                "reactive faces from the resumed State."
            )

        expected_xyz = G.geometry.surface_solid_xyz[S.receptor_face_id]
        expected_release = G.geometry.surface_fluid_xyz[S.receptor_face_id]
        expected_normals = G.geometry.surface_normals[S.receptor_face_id]

        if not np.array_equal(S.receptor_xyz, expected_xyz):
            raise ValueError(
                "The sensor geometry changed: resumed receptor solid sites no "
                "longer match the new geometry."
            )

        if not np.array_equal(S.receptor_release_xyz, expected_release):
            raise ValueError(
                "The sensor geometry changed: resumed receptor release sites "
                "no longer match the new geometry."
            )

        if not np.array_equal(S.receptor_normal, expected_normals):
            raise ValueError(
                "The sensor geometry changed: resumed receptor normals no "
                "longer match the new geometry."
            )

    ligand_fields = (
        "ligand_uid",
        "ligand_xyz",
        "ligand_receptor",
        "ligand_active",
        "last_unbound_receptor",
        "rebind_watch_active",
        "rebind_event_id",
        "rebind_unbind_time_s",
        "rebind_unbind_xyz",
        "rebind_max_surface_distance_m",
        "rebind_free_steps",
        "ligand_n_bindings",
        "ligand_n_rebindings",
        "ligand_n_self_rebindings",
        "ligand_n_cross_rebindings",
    )
    n_ligands = int(S.ligand_uid.shape[0])

    for name in ligand_fields:
        value = getattr(S, name)

        if value.shape[0] != n_ligands:
            raise ValueError(
                f"State field {name!r} has {value.shape[0]} ligand rows; "
                f"expected {n_ligands}."
            )

    if S.ligand_xyz.shape != (n_ligands, 3):
        raise ValueError("ligand_xyz must have shape (N_ligands, 3).")

    if S.rebind_unbind_xyz.shape != (n_ligands, 3):
        raise ValueError("rebind_unbind_xyz must have shape (N_ligands, 3).")

    if n_ligands and np.unique(S.ligand_uid).size != n_ligands:
        raise ValueError("ligand_uid values must be unique.")

    if n_ligands and S.n_ligands_created <= int(np.max(S.ligand_uid)):
        raise ValueError(
            "n_ligands_created must exceed every existing ligand UID."
        )

    receptor_ligand = S.receptor_ligand
    invalid_bound_receptors = (
        (receptor_ligand < -1) | (receptor_ligand >= n_ligands)
    )

    if np.any(invalid_bound_receptors):
        raise ValueError("receptor_ligand contains invalid ligand indices.")

    invalid_ligand_receptors = (
        (S.ligand_receptor < -1) | (S.ligand_receptor >= n_receptors)
    )

    if np.any(invalid_ligand_receptors):
        raise ValueError("ligand_receptor contains invalid receptor indices.")

    for receptor_id in np.flatnonzero(receptor_ligand >= 0):
        ligand_id = int(receptor_ligand[receptor_id])

        if not S.ligand_active[ligand_id]:
            raise ValueError("An occupied receptor references an inactive ligand.")

        if int(S.ligand_receptor[ligand_id]) != int(receptor_id):
            raise ValueError(
                "receptor_ligand and ligand_receptor are not mutually "
                "consistent."
            )

        if not np.array_equal(
            S.ligand_xyz[ligand_id],
            S.receptor_xyz[receptor_id],
        ):
            raise ValueError(
                "A bound ligand is not located at its receptor solid site."
            )

    for ligand_id in np.flatnonzero(
        S.ligand_active & (S.ligand_receptor >= 0)
    ):
        receptor_id = int(S.ligand_receptor[ligand_id])

        if int(S.receptor_ligand[receptor_id]) != int(ligand_id):
            raise ValueError(
                "ligand_receptor and receptor_ligand are not mutually "
                "consistent."
            )

    free_ids = np.flatnonzero(S.ligand_active & (S.ligand_receptor < 0))

    if free_ids.size:
        free_xyz = S.ligand_xyz[free_ids]
        inside = (
            (free_xyz[:, 0] >= 0)
            & (free_xyz[:, 0] < G.Nx)
            & (free_xyz[:, 1] >= 0)
            & (free_xyz[:, 1] < G.Ny)
            & (free_xyz[:, 2] >= 0)
            & (free_xyz[:, 2] <= G.Nz)
        )

        if not np.all(inside):
            raise ValueError(
                "An active free ligand lies outside the new lattice."
            )

        valid_xyz = free_xyz[inside]

        if valid_xyz.size and not np.all(
            G.accessible_fluid_mask[
                valid_xyz[:, 0],
                valid_xyz[:, 1],
                valid_xyz[:, 2],
            ]
        ):
            raise ValueError(
                "An active free ligand is not in bulk-accessible fluid under "
                "the new geometry/boundary configuration."
            )

    invalid_watches = S.rebind_watch_active & (
        ~S.ligand_active | (S.ligand_receptor >= 0)
    )

    if np.any(invalid_watches):
        raise ValueError(
            "Open rebinding watches must belong to active free ligands."
        )

    defaults = _initial_event_counts()

    for key, value in defaults.items():
        S.event_counts.setdefault(key, value)

    _rebuild_receptor_site_lookup(S, G)


def _prepare_prior_history(
    history: Optional[pd.DataFrame],
    S: State,
) -> pd.DataFrame:
    if history is None:
        return pd.DataFrame()

    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame or None.")

    prior = history.copy(deep=True)

    if prior.empty:
        return prior

    if "t_s" not in prior.columns:
        raise ValueError("history must contain a 't_s' column.")

    last_time = float(prior.iloc[-1]["t_s"])
    tolerance = max(1e-15, 1e-10 * max(1.0, abs(S.t_s)))

    if not np.isclose(last_time, S.t_s, rtol=1e-10, atol=tolerance):
        raise ValueError(
            "The final history time does not match initial_state.t_s. "
            f"history ends at {last_time:.16g} s, while the State is at "
            f"{S.t_s:.16g} s."
        )

    if "step" in prior.columns:
        last_step = int(prior.iloc[-1]["step"])

        if last_step != int(S.step_count):
            raise ValueError(
                "The final history step does not match "
                f"initial_state.step_count ({last_step} != {S.step_count})."
            )

    return prior


def _merge_history(
    prior_history: pd.DataFrame,
    segment_history: pd.DataFrame,
) -> pd.DataFrame:
    if prior_history.empty:
        return segment_history.reset_index(drop=True)

    if segment_history.empty:
        return prior_history.reset_index(drop=True)

    first_new = segment_history.iloc[0]
    last_old = prior_history.iloc[-1]
    same_time = np.isclose(
        float(first_new["t_s"]),
        float(last_old["t_s"]),
        rtol=1e-10,
        atol=1e-15,
    )
    same_step = (
        "step" not in prior_history.columns
        or "step" not in segment_history.columns
        or int(first_new["step"]) == int(last_old["step"])
    )

    if same_time and same_step:
        segment_history = segment_history.iloc[1:]

    return pd.concat(
        [prior_history, segment_history],
        ignore_index=True,
        sort=False,
    )


def _snapshot_with_phase(
    S: State,
    G: Derived,
    P: Params,
    phase_label: str,
    phase_start_t_s: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = snapshot(S, G)
    row.update(
        {
            "phase_label": phase_label,
            "phase_start_t_s": float(phase_start_t_s),
            "phase_elapsed_s": float(S.t_s - phase_start_t_s),
            "phase_geometry_name": G.geometry.name,
            "phase_D_m2_s": float(P.D_m2_s),
            "phase_ligand_conc_M": float(P.ligand_conc_M),
            "phase_k_on_M_inv_s": float(P.k_on_M_inv_s),
            "phase_k_off_s": float(P.k_off_s),
            "phase_dt_s": float(G.dt_s),
            "phase_use_well_mixed_reservoir": bool(
                G.use_well_mixed_reservoir
            ),
            "phase_reservoir_offset_layers": int(
                G.reservoir_offset_layers
            ),
            "phase_reservoir_interface_z_m": (
                float(G.reservoir_interface_z_m)
                if G.reservoir_interface_z_m is not None
                else np.nan
            ),
        }
    )
    return row


# -----------------------------------------------------------------------------
# Bulk exchange
# -----------------------------------------------------------------------------


def _zero_counts() -> Dict[str, int]:
    return {face: 0 for face in BOUNDARY_FACES}


def _sample_zero_truncated_poisson(
    rng: np.random.Generator,
    lam: float,
) -> int:
    if lam <= 0:
        return 0

    p0 = np.exp(-lam)
    target = p0 + rng.random() * (1.0 - p0)
    probability = p0
    cdf = p0
    k = 0

    while cdf < target:
        k += 1
        probability *= lam / k
        cdf += probability

        if k > max(1000, int(lam + 20 * np.sqrt(lam + 1))):
            return int(rng.poisson(lam)) or 1

    return max(1, k)


def _sample_entry_counts(
    rng: np.random.Generator,
    G: Derived,
    force_at_least_one: bool = False,
) -> Dict[str, int]:
    if G.total_injection_mean <= 0:
        return _zero_counts()

    means = np.array(
        [G.face_injection_means[face] for face in BOUNDARY_FACES],
        dtype=float,
    )

    if not force_at_least_one:
        sampled = rng.poisson(means)
        return {
            face: int(n)
            for face, n in zip(BOUNDARY_FACES, sampled)
        }

    n_total = _sample_zero_truncated_poisson(
        rng,
        G.total_injection_mean,
    )

    if n_total <= 0:
        return _zero_counts()

    probabilities = means / means.sum()
    sampled = rng.multinomial(n_total, probabilities)

    return {
        face: int(n)
        for face, n in zip(BOUNDARY_FACES, sampled)
    }


def _append_new_ligand_arrays(
    S: State,
    xyz_new: np.ndarray,
) -> None:
    n_new = int(xyz_new.shape[0])

    if n_new <= 0:
        return

    uid_start = S.n_ligands_created
    uid_new = np.arange(uid_start, uid_start + n_new, dtype=np.int64)

    S.ligand_uid = np.concatenate([S.ligand_uid, uid_new])
    S.ligand_xyz = np.vstack([S.ligand_xyz, xyz_new.astype(np.int32)])
    S.ligand_receptor = np.concatenate(
        [S.ligand_receptor, np.full(n_new, -1, dtype=np.int64)]
    )
    S.ligand_active = np.concatenate(
        [S.ligand_active, np.ones(n_new, dtype=bool)]
    )
    S.last_unbound_receptor = np.concatenate(
        [S.last_unbound_receptor, np.full(n_new, -1, dtype=np.int64)]
    )
    S.rebind_watch_active = np.concatenate(
        [S.rebind_watch_active, np.zeros(n_new, dtype=bool)]
    )
    S.rebind_event_id = np.concatenate(
        [S.rebind_event_id, np.full(n_new, -1, dtype=np.int64)]
    )
    S.rebind_unbind_time_s = np.concatenate(
        [S.rebind_unbind_time_s, np.full(n_new, np.nan)]
    )
    S.rebind_unbind_xyz = np.vstack(
        [S.rebind_unbind_xyz, np.full((n_new, 3), -1, dtype=np.int32)]
    )
    S.rebind_max_surface_distance_m = np.concatenate(
        [S.rebind_max_surface_distance_m, np.zeros(n_new, dtype=float)]
    )
    S.rebind_free_steps = np.concatenate(
        [S.rebind_free_steps, np.zeros(n_new, dtype=np.int64)]
    )
    S.ligand_n_bindings = np.concatenate(
        [S.ligand_n_bindings, np.zeros(n_new, dtype=np.int64)]
    )
    S.ligand_n_rebindings = np.concatenate(
        [S.ligand_n_rebindings, np.zeros(n_new, dtype=np.int64)]
    )
    S.ligand_n_self_rebindings = np.concatenate(
        [S.ligand_n_self_rebindings, np.zeros(n_new, dtype=np.int64)]
    )
    S.ligand_n_cross_rebindings = np.concatenate(
        [S.ligand_n_cross_rebindings, np.zeros(n_new, dtype=np.int64)]
    )

    S.n_ligands_created += n_new


def add_ligands_from_bulk(
    S: State,
    G: Derived,
    force_at_least_one: bool = False,
) -> Dict[str, int]:
    counts = _sample_entry_counts(
        S.rng,
        G,
        force_at_least_one=force_at_least_one,
    )
    n_new = int(sum(counts.values()))

    if n_new <= 0:
        return counts

    xyz_new = np.empty((n_new, 3), dtype=np.int32)
    start = 0

    for face in BOUNDARY_FACES:
        n = counts[face]

        if n <= 0:
            continue

        sites = G.boundary_fluid_sites[face]

        if sites.shape[0] == 0:
            raise RuntimeError(
                f"Sampled entry from {face}, but no accessible sites exist."
            )

        selected = S.rng.integers(0, sites.shape[0], size=n)
        stop = start + n
        xyz_new[start:stop] = sites[selected]
        start = stop

    _append_new_ligand_arrays(S, xyz_new)
    return counts


def register_entry_counts(S: State, counts: Dict[str, int]) -> None:
    n_enter = int(sum(counts.values()))
    S.event_counts["entered_from_bulk"] += n_enter

    for face in BOUNDARY_FACES:
        S.event_counts[f"entered_{face}"] += int(counts[face])


def add_ligands_from_well_mixed_bulk(
    S: State,
    G: Derived,
    force_at_least_one: bool = False,
) -> int:
    """Inject ligands through the internal constant-concentration reservoir."""
    if not G.use_well_mixed_reservoir:
        return 0

    mean = float(G.reservoir_injection_mean)
    if mean <= 0 or G.reservoir_boundary_sites.shape[0] == 0:
        return 0

    if force_at_least_one:
        n_new = _sample_zero_truncated_poisson(S.rng, mean)
    else:
        n_new = int(S.rng.poisson(mean))

    if n_new <= 0:
        return 0

    selected = S.rng.integers(
        0,
        G.reservoir_boundary_sites.shape[0],
        size=n_new,
    )
    xyz_new = G.reservoir_boundary_sites[selected].copy()
    _append_new_ligand_arrays(S, xyz_new)
    return n_new


def register_well_mixed_entry_count(S: State, n_enter: int) -> None:
    """Register entries through the internal well-mixed interface."""
    n_enter = int(n_enter)
    if n_enter <= 0:
        return

    S.event_counts["entered_from_bulk"] += n_enter
    S.event_counts["entered_from_well_mixed_bulk"] += n_enter


def _add_ligands_from_active_reservoir(S: State, G: Derived) -> None:
    """Inject from the selected bulk-exchange model."""
    if G.use_well_mixed_reservoir:
        n_enter = add_ligands_from_well_mixed_bulk(S, G)
        register_well_mixed_entry_count(S, n_enter)
    else:
        entered_counts = add_ligands_from_bulk(S, G)
        register_entry_counts(S, entered_counts)


# -----------------------------------------------------------------------------
# Rebinding event records
# -----------------------------------------------------------------------------


def _surface_distance_at_ligand(
    S: State,
    G: Derived,
    ligand_id: int,
) -> float:
    xyz = S.ligand_xyz[ligand_id]

    if (
        0 <= xyz[0] < G.Nx
        and 0 <= xyz[1] < G.Ny
        and 0 <= xyz[2] <= G.Nz
    ):
        return float(
            G.distance_to_reactive_surface_m[xyz[0], xyz[1], xyz[2]]
        )

    return float("nan")


def _start_rebinding_watch(
    S: State,
    G: Derived,
    ligand_id: int,
    receptor_id: int,
    event_time_s: float,
) -> None:
    S.last_unbound_receptor[ligand_id] = receptor_id
    S.rebind_watch_active[ligand_id] = True
    S.rebind_event_id[ligand_id] = S.next_unbind_event_id
    S.next_unbind_event_id += 1

    S.rebind_unbind_time_s[ligand_id] = event_time_s
    S.rebind_unbind_xyz[ligand_id] = S.ligand_xyz[ligand_id]
    S.rebind_max_surface_distance_m[ligand_id] = _surface_distance_at_ligand(
        S,
        G,
        ligand_id,
    )
    S.rebind_free_steps[ligand_id] = 0


def _finish_rebinding_watch(
    S: State,
    G: Derived,
    ligand_id: int,
    outcome: str,
    event_time_s: float,
    new_receptor_id: int = -1,
) -> None:
    if not S.rebind_watch_active[ligand_id]:
        return

    previous_receptor_id = int(S.last_unbound_receptor[ligand_id])
    event_id = int(S.rebind_event_id[ligand_id])
    ligand_uid = int(S.ligand_uid[ligand_id])
    unbind_xyz = S.rebind_unbind_xyz[ligand_id].astype(float)
    end_xyz = S.ligand_xyz[ligand_id].astype(float)

    is_self = outcome == "self_rebinding"
    is_cross = outcome == "cross_rebinding"
    is_rebinding = is_self or is_cross

    if is_rebinding and new_receptor_id >= 0 and previous_receptor_id >= 0:
        receptor_distance_m = float(
            np.linalg.norm(
                (
                    S.receptor_xyz[new_receptor_id]
                    - S.receptor_xyz[previous_receptor_id]
                )
                * G.a_m
            )
        )
        new_face_id = int(S.receptor_face_id[new_receptor_id])
    else:
        receptor_distance_m = np.nan
        new_face_id = -1

    previous_face_id = (
        int(S.receptor_face_id[previous_receptor_id])
        if previous_receptor_id >= 0
        else -1
    )

    record = {
        "unbind_event_id": event_id,
        "ligand_uid": ligand_uid,
        "outcome": outcome,
        "is_rebinding": bool(is_rebinding),
        "is_self_rebinding": bool(is_self),
        "is_cross_rebinding": bool(is_cross),
        "previous_receptor_id": previous_receptor_id,
        "new_receptor_id": int(new_receptor_id),
        "previous_face_id": previous_face_id,
        "new_face_id": new_face_id,
        "t_unbind_s": float(S.rebind_unbind_time_s[ligand_id]),
        "t_end_s": float(event_time_s),
        "free_excursion_time_s": float(
            event_time_s - S.rebind_unbind_time_s[ligand_id]
        ),
        "free_steps": int(S.rebind_free_steps[ligand_id]),
        "max_surface_distance_m": float(
            S.rebind_max_surface_distance_m[ligand_id]
        ),
        "receptor_distance_m": receptor_distance_m,
        "unbind_x_m": float(unbind_xyz[0] * G.a_m),
        "unbind_y_m": float(unbind_xyz[1] * G.a_m),
        "unbind_z_m": float(unbind_xyz[2] * G.a_m),
        "end_x_m": float(end_xyz[0] * G.a_m),
        "end_y_m": float(end_xyz[1] * G.a_m),
        "end_z_m": float(end_xyz[2] * G.a_m),
    }
    S.rebinding_records.append(record)

    S.rebind_watch_active[ligand_id] = False
    S.rebind_event_id[ligand_id] = -1


def rebinding_events_dataframe(
    S: State,
    G: Derived,
    include_open: bool = True,
) -> pd.DataFrame:
    """Return one row per completed or currently open dissociation excursion."""

    records = list(S.rebinding_records)

    if include_open:
        open_ligands = np.flatnonzero(S.rebind_watch_active)

        for ligand_id in open_ligands:
            previous_receptor_id = int(S.last_unbound_receptor[ligand_id])
            previous_face_id = (
                int(S.receptor_face_id[previous_receptor_id])
                if previous_receptor_id >= 0
                else -1
            )
            unbind_xyz = S.rebind_unbind_xyz[ligand_id].astype(float)
            end_xyz = S.ligand_xyz[ligand_id].astype(float)
            a_m = G.a_m

            records.append(
                {
                    "unbind_event_id": int(S.rebind_event_id[ligand_id]),
                    "ligand_uid": int(S.ligand_uid[ligand_id]),
                    "outcome": "open",
                    "is_rebinding": False,
                    "is_self_rebinding": False,
                    "is_cross_rebinding": False,
                    "previous_receptor_id": previous_receptor_id,
                    "new_receptor_id": -1,
                    "previous_face_id": previous_face_id,
                    "new_face_id": -1,
                    "t_unbind_s": float(S.rebind_unbind_time_s[ligand_id]),
                    "t_end_s": float(S.t_s),
                    "free_excursion_time_s": float(
                        S.t_s - S.rebind_unbind_time_s[ligand_id]
                    ),
                    "free_steps": int(S.rebind_free_steps[ligand_id]),
                    "max_surface_distance_m": float(
                        S.rebind_max_surface_distance_m[ligand_id]
                    ),
                    "receptor_distance_m": np.nan,
                    "unbind_x_m": float(unbind_xyz[0] * a_m),
                    "unbind_y_m": float(unbind_xyz[1] * a_m),
                    "unbind_z_m": float(unbind_xyz[2] * a_m),
                    "end_x_m": float(end_xyz[0] * a_m),
                    "end_y_m": float(end_xyz[1] * a_m),
                    "end_z_m": float(end_xyz[2] * a_m),
                }
            )

    columns = [
        "unbind_event_id",
        "ligand_uid",
        "outcome",
        "is_rebinding",
        "is_self_rebinding",
        "is_cross_rebinding",
        "previous_receptor_id",
        "new_receptor_id",
        "previous_face_id",
        "new_face_id",
        "t_unbind_s",
        "t_end_s",
        "free_excursion_time_s",
        "free_steps",
        "max_surface_distance_m",
        "receptor_distance_m",
        "unbind_x_m",
        "unbind_y_m",
        "unbind_z_m",
        "end_x_m",
        "end_y_m",
        "end_z_m",
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame.from_records(records)[columns].sort_values(
        "unbind_event_id"
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Memory compaction and snapshots
# -----------------------------------------------------------------------------


def state_is_empty(S: State) -> bool:
    return (
        np.count_nonzero(S.ligand_active) == 0
        and np.count_nonzero(S.receptor_ligand >= 0) == 0
    )


def compact_ligands(S: State) -> int:
    """Remove inactive ligand slots and remap bound-ligand indices."""

    keep = S.ligand_active
    n_old = keep.size
    n_keep = int(np.count_nonzero(keep))

    if n_keep == n_old:
        return 0

    old_to_new = np.full(n_old, -1, dtype=np.int64)
    old_to_new[keep] = np.arange(n_keep, dtype=np.int64)

    bound_receptors = np.flatnonzero(S.receptor_ligand >= 0)

    if bound_receptors.size:
        old_ligand_ids = S.receptor_ligand[bound_receptors]
        S.receptor_ligand[bound_receptors] = old_to_new[old_ligand_ids]

    ligand_array_names = (
        "ligand_uid",
        "ligand_xyz",
        "ligand_receptor",
        "ligand_active",
        "last_unbound_receptor",
        "rebind_watch_active",
        "rebind_event_id",
        "rebind_unbind_time_s",
        "rebind_unbind_xyz",
        "rebind_max_surface_distance_m",
        "rebind_free_steps",
        "ligand_n_bindings",
        "ligand_n_rebindings",
        "ligand_n_self_rebindings",
        "ligand_n_cross_rebindings",
    )

    for name in ligand_array_names:
        setattr(S, name, getattr(S, name)[keep].copy())

    return n_old - n_keep


def snapshot(S: State, G: Derived) -> Dict[str, float]:
    bound = int(np.count_nonzero(S.receptor_ligand >= 0))
    active_free = S.ligand_active & (S.ligand_receptor < 0)

    return {
        "step": S.step_count,
        "t_s": S.t_s,
        "B": bound,
        "theta": bound / G.NR if G.NR > 0 else np.nan,
        "N_free": int(np.count_nonzero(active_free)),
        "N_bound": bound,
        "N_active_ligands": int(np.count_nonzero(S.ligand_active)),
        "N_ligand_slots": int(S.ligand_xyz.shape[0]),
        "N_total_ligands_ever": int(S.n_ligands_created),
        "N_open_rebinding_watches": int(
            np.count_nonzero(S.rebind_watch_active)
        ),
        "binding_events_total": int(S.event_counts["bindings"]),
        "unbinding_events_total": int(S.event_counts["unbindings"]),
        "rebinding_events_total": int(S.event_counts["rebindings"]),
        "self_rebindings_total": int(S.event_counts["self_rebindings"]),
        "cross_rebindings_total": int(S.event_counts["cross_rebindings"]),
        "local_rebinding_escapes_total": int(
            S.event_counts["local_rebinding_escapes"]
        ),
        "rebind_watch_bulk_losses_total": int(
            S.event_counts["rebind_watch_bulk_losses"]
        ),
        "rebind_watch_well_mixed_bulk_losses_total": int(
            S.event_counts["rebind_watch_well_mixed_bulk_losses"]
        ),
        "lost_to_bulk_total": int(S.event_counts["lost_to_bulk"]),
        "entered_from_bulk_total": int(S.event_counts["entered_from_bulk"]),
        "lost_to_well_mixed_bulk_total": int(
            S.event_counts["lost_to_well_mixed_bulk"]
        ),
        "entered_from_well_mixed_bulk_total": int(
            S.event_counts["entered_from_well_mixed_bulk"]
        ),
        **{
            f"lost_{face}_total": int(S.event_counts[f"lost_{face}"])
            for face in BOUNDARY_FACES
        },
        **{
            f"entered_{face}_total": int(
                S.event_counts[f"entered_{face}"]
            )
            for face in BOUNDARY_FACES
        },
    }


# -----------------------------------------------------------------------------
# Core Monte Carlo step
# -----------------------------------------------------------------------------


def _outside_face(xyz: np.ndarray, G: Derived) -> Optional[str]:
    if xyz[0] < 0:
        return "x_min"
    if xyz[0] >= G.Nx:
        return "x_max"
    if xyz[1] < 0:
        return "y_min"
    if xyz[1] >= G.Ny:
        return "y_max"
    if xyz[2] < 0:
        return "z_min"
    if xyz[2] > G.Nz:
        return "z_max"
    return None


def _diffuse_free_ligands(S: State, G: Derived, event_time_s: float) -> None:
    free_ids = np.flatnonzero(S.ligand_active & (S.ligand_receptor < 0))

    if free_ids.size == 0:
        return

    moves = S.rng.choice(7, size=free_ids.size, p=G.move_probs)
    old_positions = S.ligand_xyz[free_ids].copy()
    proposed = old_positions + MOVE_VECTORS[moves]

    lost_by_face: Dict[str, List[int]] = {face: [] for face in BOUNDARY_FACES}
    lost_to_well_mixed_bulk: List[int] = []

    for local_index, ligand_id in enumerate(free_ids):
        proposed_xyz = proposed[local_index]

        # Once a free ligand crosses upward into the well-mixed bulk it leaves
        # the explicit microscopic domain and cannot return.
        if (
            G.use_well_mixed_reservoir
            and G.reservoir_explicit_max_z_index is not None
            and proposed_xyz[2] > G.reservoir_explicit_max_z_index
        ):
            S.ligand_xyz[ligand_id] = proposed_xyz
            lost_to_well_mixed_bulk.append(int(ligand_id))
            continue

        face = _outside_face(proposed_xyz, G)

        if face is not None:
            if face in G.open_boundaries:
                S.ligand_xyz[ligand_id] = proposed_xyz
                lost_by_face[face].append(int(ligand_id))
            continue

        x, y, z = map(int, proposed_xyz)

        if not G.accessible_fluid_mask[x, y, z]:
            continue

        if G.geometry.solid_mask[x, y, z]:
            continue

        S.ligand_xyz[ligand_id] = proposed_xyz

    if lost_to_well_mixed_bulk:
        ids_array = np.asarray(lost_to_well_mixed_bulk, dtype=np.int64)
        watched_ids = ids_array[S.rebind_watch_active[ids_array]]

        for ligand_id in watched_ids:
            _finish_rebinding_watch(
                S,
                G,
                int(ligand_id),
                outcome="well_mixed_bulk_loss",
                event_time_s=event_time_s,
            )

        n_watched = int(watched_ids.size)
        n_lost = int(ids_array.size)

        S.event_counts["rebind_watch_well_mixed_bulk_losses"] += n_watched
        S.event_counts["rebind_watch_bulk_losses"] += n_watched
        S.event_counts["lost_to_well_mixed_bulk"] += n_lost
        S.event_counts["lost_to_bulk"] += n_lost
        S.ligand_active[ids_array] = False

    all_lost: List[int] = []

    for face, ids in lost_by_face.items():
        if not ids:
            continue

        ids_array = np.asarray(ids, dtype=np.int64)

        for ligand_id in ids_array[S.rebind_watch_active[ids_array]]:
            _finish_rebinding_watch(
                S,
                G,
                int(ligand_id),
                outcome="bulk_loss",
                event_time_s=event_time_s,
            )
            S.event_counts["rebind_watch_bulk_losses"] += 1

        S.event_counts[f"lost_{face}"] += len(ids)
        all_lost.extend(ids)

    if all_lost:
        lost_ids = np.asarray(all_lost, dtype=np.int64)
        S.ligand_active[lost_ids] = False
        S.event_counts["lost_to_bulk"] += int(lost_ids.size)

    watched = np.flatnonzero(
        S.ligand_active
        & (S.ligand_receptor < 0)
        & S.rebind_watch_active
    )

    if watched.size == 0:
        return

    watched_xyz = S.ligand_xyz[watched]
    distances = G.distance_to_reactive_surface_m[
        watched_xyz[:, 0],
        watched_xyz[:, 1],
        watched_xyz[:, 2],
    ]
    S.rebind_max_surface_distance_m[watched] = np.maximum(
        S.rebind_max_surface_distance_m[watched],
        distances,
    )
    S.rebind_free_steps[watched] += 1

    escaped = watched[distances >= G.escape_distance_m]

    for ligand_id in escaped:
        _finish_rebinding_watch(
            S,
            G,
            int(ligand_id),
            outcome="local_escape",
            event_time_s=event_time_s,
        )
        S.event_counts["local_rebinding_escapes"] += 1


def _attempt_binding(S: State, G: Derived, event_time_s: float) -> None:
    if G.NR == 0 or G.kon_exp_per_receptor <= 0:
        return

    free_ids = np.flatnonzero(S.ligand_active & (S.ligand_receptor < 0))

    if free_ids.size == 0:
        return

    free_xyz = S.ligand_xyz[free_ids]
    near_mask = G.reaction_site_mask[
        free_xyz[:, 0],
        free_xyz[:, 1],
        free_xyz[:, 2],
    ]
    near_ids = free_ids[near_mask]

    if near_ids.size == 0:
        return

    S.rng.shuffle(near_ids)

    bind_events = 0
    rebind_events = 0
    self_rebind_events = 0
    cross_rebind_events = 0

    for ligand_id in near_ids:
        site_id = int(_flat_site_index(S.ligand_xyz[ligand_id], G))
        grid_receptor = int(S.receptor_grid[site_id])

        if grid_receptor >= 0:
            if S.receptor_ligand[grid_receptor] >= 0:
                continue
            unbound_receptors = [grid_receptor]
        else:
            receptors = S.site_to_receptors[site_id]

            if not receptors:
                continue

            unbound_receptors = [
                receptor_id
                for receptor_id in receptors
                if S.receptor_ligand[receptor_id] < 0
            ]

            if not unbound_receptors:
                continue

        p_bind = 1.0 - np.exp(
            -G.kon_exp_per_receptor * len(unbound_receptors)
        )

        if S.rng.random() >= p_bind:
            continue

        if len(unbound_receptors) == 1:
            receptor_id = unbound_receptors[0]
        else:
            receptor_id = unbound_receptors[
                int(S.rng.integers(0, len(unbound_receptors)))
            ]

        S.ligand_receptor[ligand_id] = receptor_id
        S.receptor_ligand[receptor_id] = ligand_id
        S.ligand_xyz[ligand_id] = S.receptor_xyz[receptor_id]
        S.ligand_n_bindings[ligand_id] += 1
        bind_events += 1

        if S.rebind_watch_active[ligand_id]:
            previous_receptor = int(S.last_unbound_receptor[ligand_id])

            if previous_receptor == receptor_id:
                outcome = "self_rebinding"
                S.ligand_n_self_rebindings[ligand_id] += 1
                self_rebind_events += 1
            else:
                outcome = "cross_rebinding"
                S.ligand_n_cross_rebindings[ligand_id] += 1
                cross_rebind_events += 1

            S.ligand_n_rebindings[ligand_id] += 1
            rebind_events += 1
            _finish_rebinding_watch(
                S,
                G,
                int(ligand_id),
                outcome=outcome,
                event_time_s=event_time_s,
                new_receptor_id=receptor_id,
            )

    S.event_counts["bindings"] += bind_events
    S.event_counts["rebindings"] += rebind_events
    S.event_counts["self_rebindings"] += self_rebind_events
    S.event_counts["cross_rebindings"] += cross_rebind_events


def _attempt_dissociation(
    S: State,
    G: Derived,
    bound_receptors_start: np.ndarray,
    event_time_s: float,
) -> None:
    if bound_receptors_start.size == 0 or G.p_off <= 0:
        return

    S.rng.shuffle(bound_receptors_start)
    unbind_events = 0

    for receptor_id in bound_receptors_start:
        ligand_id = int(S.receptor_ligand[receptor_id])

        if ligand_id < 0:
            continue

        if S.rng.random() >= G.p_off:
            continue

        S.receptor_ligand[receptor_id] = -1
        S.ligand_receptor[ligand_id] = -1
        S.ligand_xyz[ligand_id] = S.receptor_release_xyz[receptor_id]

        _start_rebinding_watch(
            S,
            G,
            ligand_id,
            int(receptor_id),
            event_time_s,
        )
        unbind_events += 1

    S.event_counts["unbindings"] += unbind_events


def step(S: State, G: Derived) -> None:
    event_time_s = S.t_s + G.dt_s
    bound_receptors_start = np.flatnonzero(S.receptor_ligand >= 0)

    if state_is_empty(S):
        _add_ligands_from_active_reservoir(S, G)
        S.step_count += 1
        S.t_s = event_time_s
        return

    _diffuse_free_ligands(S, G, event_time_s)
    _attempt_binding(S, G, event_time_s)
    _attempt_dissociation(
        S,
        G,
        bound_receptors_start,
        event_time_s,
    )

    _add_ligands_from_active_reservoir(S, G)

    S.step_count += 1
    S.t_s = event_time_s


# -----------------------------------------------------------------------------
# State capture for visualization
# -----------------------------------------------------------------------------


def capture_state_frame(
    S: State,
    G: Derived,
    P: Params,
    phase_label: Optional[str] = None,
    phase_start_t_s: Optional[float] = None,
) -> Dict:
    bound_mask = S.receptor_ligand >= 0
    active_free_mask = S.ligand_active & (S.ligand_receptor < 0)

    receptor_xyz_m = S.receptor_xyz.copy() * P.a_m
    ligand_xyz_m = S.ligand_xyz[active_free_mask].copy() * P.a_m

    frame = {
        "t_s": S.t_s,
        "step": S.step_count,
        "geometry_name": G.geometry.name,
        "grid_shape": G.grid_shape,
        "solid_mask": G.geometry.solid_mask,
        "a_m": P.a_m,
        "Lx_m": G.Nx * P.a_m,
        "Ly_m": G.Ny * P.a_m,
        "H_m": G.Nz * P.a_m,
        "receptor_xyz_m": receptor_xyz_m,
        "receptor_xy_m": receptor_xyz_m[:, :2],
        "receptor_surface_center_m": (
            G.geometry.surface_centers_m[S.receptor_face_id].copy()
        ),
        "receptor_release_xyz_m": S.receptor_release_xyz.copy() * P.a_m,
        "receptor_normal": S.receptor_normal.copy(),
        "receptor_bound": bound_mask.copy(),
        "ligand_xyz_m": ligand_xyz_m,
        "B": int(np.count_nonzero(bound_mask)),
        "theta": (
            float(np.count_nonzero(bound_mask) / G.NR)
            if G.NR > 0
            else np.nan
        ),
        "N_free": int(np.count_nonzero(active_free_mask)),
        "N_active_ligands": int(np.count_nonzero(S.ligand_active)),
        "N_ligand_slots": int(S.ligand_xyz.shape[0]),
        "N_total_ligands_ever": int(S.n_ligands_created),
        "N_open_rebinding_watches": int(
            np.count_nonzero(S.rebind_watch_active)
        ),
        "use_well_mixed_reservoir": bool(G.use_well_mixed_reservoir),
        "reservoir_offset_layers": int(G.reservoir_offset_layers),
        "reservoir_interface_z_m": (
            float(G.reservoir_interface_z_m)
            if G.reservoir_interface_z_m is not None
            else np.nan
        ),
    }

    if phase_label is not None:
        frame["phase_label"] = str(phase_label)

    if phase_start_t_s is not None:
        frame["phase_start_t_s"] = float(phase_start_t_s)
        frame["phase_elapsed_s"] = float(S.t_s - phase_start_t_s)

    return frame


# -----------------------------------------------------------------------------
# Simulation runner
# -----------------------------------------------------------------------------


def run_simulation(
    P: Params,
    seconds: float,
    record_every_s: Optional[float] = None,
    return_state: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    save_state_frames: bool = False,
    n_state_frames: int = 20,
    compact_every: Optional[int] = 10_000,
    compact_inactive_fraction: float = 0.5,
    geometry: Optional[SensorGeometry] = None,
    return_rebinding_events: bool = False,
    initial_state: Optional[Union[State, Mapping[str, Any]]] = None,
    history: Optional[pd.DataFrame] = None,
    state_frames: Optional[Sequence[Dict]] = None,
    copy_initial_state: bool = True,
    reseed_on_resume: bool = False,
    phase_label: Optional[str] = None,
):
    """
    Run a new simulation phase or continue an existing microscopic state.

    Parameters added for continuation
    ---------------------------------
    initial_state
        A ``State`` returned by a previous call with ``return_state=True``, or
        a checkpoint mapping returned by ``state_to_checkpoint``. This is the
        information that permits exact microscopic continuation.
    history
        Optional previous history DataFrame. New history rows are appended and
        the duplicate phase-boundary row is removed. History alone cannot
        reconstruct ligand positions, receptor assignments, rebinding watches,
        or RNG state, so a nonempty history requires ``initial_state``.
    state_frames
        Optional previously captured visualization frames. New frames are
        appended to a new list. Frames alone are intentionally not used as a
        restart state because visualization frames omit full ligand identities,
        receptor-ligand mappings, event records, and RNG state.
    copy_initial_state
        If True, continue from a deep copy so the supplied State is not mutated.
        Set False to continue in place and reduce memory use.
    reseed_on_resume
        If False, preserve the saved RNG stream for exact continuation. If
        True, replace it with ``np.random.default_rng(P.seed)``.
    phase_label
        Optional label such as ``"wash"`` or ``"high_concentration"`` added
        to new history rows and state frames.

    Notes
    -----
    Dynamic parameters may change between phases, including ligand
    concentration, D, k_on, k_off, dt, and bulk boundary exchange. Lattice
    dimensions, spacing, sensor geometry, receptor count, and receptor
    placement must remain compatible with the saved State.
    """
    from tqdm import tqdm

    if seconds < 0:
        raise ValueError("seconds cannot be negative.")

    has_prior_history = history is not None and not history.empty
    has_prior_frames = state_frames is not None and len(state_frames) > 0

    if initial_state is None and (has_prior_history or has_prior_frames):
        raise ValueError(
            "A history DataFrame or visualization frames cannot reconstruct "
            "the exact microscopic simulation state. Pass the State returned "
            "by the prior run using initial_state=previous_state (or pass a "
            "state_to_checkpoint() mapping)."
        )

    G = derive(P, geometry=geometry)
    resumed = initial_state is not None

    if resumed:
        S = _coerce_initial_state(
            initial_state,
            copy_initial_state=copy_initial_state,
        )
        _validate_resume_state(S, G)

        if reseed_on_resume:
            S.rng = np.random.default_rng(P.seed)
    else:
        S = initialize(P, G)

    prior_history = _prepare_prior_history(history, S)

    # Supplying an existing frame list implies that the caller wants new frames
    # appended, even if save_state_frames was accidentally left False.
    if state_frames is not None:
        save_state_frames = True

    if phase_label is None:
        phase_label = "continued" if resumed else "initial"
    else:
        phase_label = str(phase_label)

    phase_start_t_s = float(S.t_s)
    phase_start_step = int(S.step_count)

    if verbose:
        print("=" * 68)
        print("Simulation continuation" if resumed else "Simulation initialization")
        print("=" * 68)
        print(f"Phase label          : {phase_label}")
        print(f"Start time           : {S.t_s:.6e} s")
        print(f"Start step           : {S.step_count:,}")
        print(f"Geometry             : {G.geometry.name}")
        print(f"Grid nodes           : {G.Nx} x {G.Ny} x {G.Nz + 1}")
        print(f"Accessible fluid     : {G.accessible_fluid_xyz.shape[0]:,} sites")
        print(f"Reactive faces       : {G.reactive_face_ids.size:,}")
        print(f"Reactive area        : {G.sensing_area_m2:.3e} m^2")
        print(f"Accessible volume    : {G.volume_m3:.3e} m^3")
        print(f"Receptors            : {G.NR:,}")
        print(f"Active ligands       : {np.count_nonzero(S.ligand_active):,}")
        print(f"Mean reservoir count : {G.mean_ligands:.3e}")
        print(f"Bulk concentration   : {G.bulk_conc_m3:.3e} molecules/m^3")
        print(f"D                    : {G.D_m2_s:.3e} m^2/s")
        print(f"Mean bulk entries    : {G.total_injection_mean:.3e} ligands/step")
        print(f"Well-mixed reservoir : {G.use_well_mixed_reservoir}")
        if G.use_well_mixed_reservoir:
            print(
                f"Reservoir interface  : {G.reservoir_interface_z_m:.3e} m "
                f"(offset={G.reservoir_offset_layers} layer(s))"
            )
            print(
                f"Reservoir sites      : "
                f"{G.reservoir_boundary_sites.shape[0]:,}"
            )
        print(f"Receptor density     : {P.receptor_density_m2:.3e} receptors/m^2")
        print(f"Ligand concentration : {P.ligand_conc_M:.3e} M")
        print(f"KD                   : {G.Kd_M:.3e} M")
        print(f"dt                   : {G.dt_s:.3e} s")
        print(f"Escape distance      : {G.escape_distance_m:.3e} m")
        print(f"Fast receptor grid   : {S.use_receptor_grid}")
        print(f"RNG preserved        : {resumed and not reseed_on_resume}")
        print("=" * 68)

    step_ratio = seconds / G.dt_s
    nearest_step_count = int(round(step_ratio))

    if np.isclose(step_ratio, nearest_step_count, rtol=1e-12, atol=1e-12):
        n_steps = nearest_step_count
    else:
        n_steps = int(np.ceil(step_ratio))

    if record_every_s is None:
        record_every = max(1, n_steps // 200) if n_steps > 0 else 1
    else:
        record_every = max(1, int(round(record_every_s / G.dt_s)))

    output_state_frames: Optional[List[Dict]]

    if save_state_frames:
        n_state_frames = max(1, int(n_state_frames))
        frame_steps = np.unique(
            np.linspace(0, n_steps, n_state_frames, dtype=int)
        )
        frame_steps_set = set(map(int, frame_steps.tolist()))
        output_state_frames = list(state_frames) if state_frames is not None else []
        initial_frame = capture_state_frame(
            S,
            G,
            P,
            phase_label=phase_label,
            phase_start_t_s=phase_start_t_s,
        )

        append_initial_frame = True

        if output_state_frames:
            last_frame = output_state_frames[-1]
            last_time = float(last_frame.get("t_s", np.nan))
            last_step = int(last_frame.get("step", -1))
            append_initial_frame = not (
                np.isclose(last_time, S.t_s, rtol=1e-10, atol=1e-15)
                and last_step == S.step_count
            )

            if np.isfinite(last_time) and last_time > S.t_s + 1e-15:
                raise ValueError(
                    "The final supplied state frame occurs after the "
                    "initial_state time."
                )

        if append_initial_frame:
            output_state_frames.append(initial_frame)
    else:
        frame_steps_set = set()
        output_state_frames = None

    rows = [
        _snapshot_with_phase(
            S,
            G,
            P,
            phase_label=phase_label,
            phase_start_t_s=phase_start_t_s,
        )
    ]
    iterator = range(n_steps)

    if show_progress:
        iterator = tqdm(
            iterator,
            total=n_steps,
            desc=f"Running {phase_label}",
            unit="step",
            mininterval=0.5,
            miniters=1000,
            leave=True,
            dynamic_ncols=True,
        )

    for local_step_index in iterator:
        step(S, G)
        segment_step = local_step_index + 1

        if compact_every is not None and compact_every > 0:
            if S.step_count % compact_every == 0:
                total_slots = S.ligand_active.size

                if total_slots > 0:
                    inactive = int(np.count_nonzero(~S.ligand_active))

                    if inactive / total_slots >= compact_inactive_fraction:
                        compact_ligands(S)

        if segment_step % record_every == 0:
            rows.append(
                _snapshot_with_phase(
                    S,
                    G,
                    P,
                    phase_label=phase_label,
                    phase_start_t_s=phase_start_t_s,
                )
            )

        if save_state_frames and segment_step in frame_steps_set:
            output_state_frames.append(
                capture_state_frame(
                    S,
                    G,
                    P,
                    phase_label=phase_label,
                    phase_start_t_s=phase_start_t_s,
                )
            )

    if rows[-1]["step"] != S.step_count:
        rows.append(
            _snapshot_with_phase(
                S,
                G,
                P,
                phase_label=phase_label,
                phase_start_t_s=phase_start_t_s,
            )
        )

    # Guard against n_state_frames=1, where linspace selects only the phase
    # boundary. A positive-duration phase should still end with a final frame.
    if save_state_frames and n_steps > 0:
        last_frame_time = (
            float(output_state_frames[-1]["t_s"])
            if output_state_frames
            else -np.inf
        )

        if not np.isclose(last_frame_time, S.t_s, rtol=1e-10, atol=1e-15):
            output_state_frames.append(
                capture_state_frame(
                    S,
                    G,
                    P,
                    phase_label=phase_label,
                    phase_start_t_s=phase_start_t_s,
                )
            )

    segment_history = pd.DataFrame(rows)
    output_history = _merge_history(prior_history, segment_history)
    events = (
        rebinding_events_dataframe(S, G, include_open=True)
        if return_rebinding_events
        else None
    )

    if verbose:
        print(
            f"Completed phase {phase_label!r}: "
            f"{S.step_count - phase_start_step:,} steps, "
            f"t = {S.t_s:.6e} s."
        )

    if return_state and save_state_frames and return_rebinding_events:
        return output_history, S, G, output_state_frames, events

    if return_state and save_state_frames:
        return output_history, S, G, output_state_frames

    if return_state and return_rebinding_events:
        return output_history, S, G, events

    if save_state_frames and return_rebinding_events:
        return output_history, output_state_frames, events

    if return_state:
        return output_history, S, G

    if save_state_frames:
        return output_history, output_state_frames

    if return_rebinding_events:
        return output_history, events

    return output_history


__all__ = [
    "MODEL_VERSION",
    "Params",
    "SensorGeometry",
    "Derived",
    "State",
    "STATE_CHECKPOINT_VERSION",
    "geometry_from_solid_mask",
    "make_implicit_geometry",
    "make_height_field_geometry",
    "make_flat_geometry",
    "make_spherical_cap_geometry",
    "make_spherical_bowl_geometry",
    "make_cylindrical_post_geometry",
    "make_cylindrical_well_geometry",
    "make_nanopore_array_geometry",
    "derive",
    "initialize",
    "step",
    "snapshot",
    "capture_state_frame",
    "compact_ligands",
    "state_is_empty",
    "state_to_checkpoint",
    "state_from_checkpoint",
    "clone_state",
    "rebinding_events_dataframe",
    "add_ligands_from_well_mixed_bulk",
    "run_simulation",
]