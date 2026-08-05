"""
biosensor_mc_2d.py

Two-dimensional lattice Monte Carlo model of ligand binding to a
functionalized biosensor surface.

Geometry
--------
The simulated lattice is an x-z cross-section:

    z = H      bulk boundary
      ^
      |
      |        free ligands diffuse in x and z
      |
    z = 1      reaction-adjacent lattice layer
    z = 0      sensing surface containing receptors
      +------------------------------> x

Because molar concentration and the bimolecular association constant require
physical volume, the 2D lattice is interpreted as a slice with an out-of-plane
thickness ``depth_nm``. Ligands do not diffuse in that omitted dimension, but
``depth_nm`` is used to calculate:

* the physical simulation volume,
* the physical sensing area,
* receptor number from receptors/nm^2,
* ligand number from molar concentration,
* reaction volume, and
* bulk-exchange fluxes.

The public API intentionally follows the original 3D script closely:

    P = Params(...)
    hist = run_simulation(P, seconds=1.0)

or

    hist, state, derived, frames = run_simulation(
        P,
        seconds=1.0,
        return_state=True,
        save_state_frames=True,
    )
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


NA = 6.02214076e23
L_PER_NM3 = 1e-24
NM2_PER_CM2 = 1e14


@dataclass
class Params:
    # Explicit 2D dimensions.
    Lx_nm: float = 10.0
    H_nm: float = 10.0
    a_nm: float = 1.0

    # Effective thickness of the omitted y dimension. This preserves physical
    # concentration, receptor-density, and kinetic units in the 2D reduction.
    depth_nm: float = 10.0

    D_cm2_s: float = 1e-6

    receptor_density_nm2: float = 0.1
    ligand_conc_M: float = 10e-9

    k_on_M_inv_s: float = 1e7
    k_off_s: float = 0.1

    dt_s: Optional[float] = None
    reaction_volume_voxels: int = 1
    escape_height_nm: float = 5.0

    use_poisson_ligand_count: bool = True
    allow_multiple_receptors_per_site: bool = False
    seed: int = 1


@dataclass
class Derived:
    Nx: int
    Nz: int
    Lx_eff_nm: float
    H_eff_nm: float
    depth_nm: float
    volume_nm3: float
    sensing_area_nm2: float
    NR: int
    mean_ligands: float
    bulk_conc_nm3: float
    D_nm2_s: float
    face_injection_means: Dict[str, float]
    total_injection_mean: float
    dt_s: float
    move_probs: np.ndarray
    kon_exp_per_receptor: float
    p_off: float
    Kd_M: float
    escape_z_layer: int
    use_receptor_grid: bool


@dataclass
class State:
    rng: np.random.Generator

    # Receptors lie on z = 0 and therefore require only an x coordinate.
    receptor_x: np.ndarray
    site_to_receptors: List[List[int]]
    receptor_grid: np.ndarray
    receptor_ligand: np.ndarray

    # Ligand coordinates are stored as columns [x, z].
    ligand_xz: np.ndarray
    ligand_receptor: np.ndarray
    ligand_active: np.ndarray

    last_unbound_receptor: np.ndarray
    rebind_watch_active: np.ndarray

    ligand_n_bindings: np.ndarray
    ligand_n_rebindings: np.ndarray
    ligand_n_self_rebindings: np.ndarray
    ligand_n_cross_rebindings: np.ndarray

    n_ligands_created: int

    step_count: int
    t_s: float
    event_counts: Dict[str, int]


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------


def derive(P: Params) -> Derived:
    """Validate parameters and calculate lattice-derived quantities."""

    if P.a_nm <= 0:
        raise ValueError("a_nm must be positive.")
    if P.depth_nm <= 0:
        raise ValueError("depth_nm must be positive.")
    if P.D_cm2_s <= 0:
        raise ValueError("D_cm2_s must be positive.")
    if P.receptor_density_nm2 < 0:
        raise ValueError("receptor_density_nm2 cannot be negative.")
    if P.ligand_conc_M < 0:
        raise ValueError("ligand_conc_M cannot be negative.")
    if P.k_on_M_inv_s < 0 or P.k_off_s < 0:
        raise ValueError("k_on_M_inv_s and k_off_s cannot be negative.")
    if P.reaction_volume_voxels < 1:
        raise ValueError("reaction_volume_voxels must be at least 1.")

    Nx = int(round(P.Lx_nm / P.a_nm))
    Nz = int(round(P.H_nm / P.a_nm))

    if min(Nx, Nz) < 1:
        raise ValueError("Lx_nm and H_nm must each be at least one lattice spacing.")

    Lx_eff_nm = Nx * P.a_nm
    H_eff_nm = Nz * P.a_nm

    # The 2D lattice represents a slab with thickness depth_nm.
    sensing_area_nm2 = Lx_eff_nm * P.depth_nm
    volume_nm3 = sensing_area_nm2 * H_eff_nm

    D_nm2_s = P.D_cm2_s * NM2_PER_CM2

    surface_sites = Nx
    NR = int(round(P.receptor_density_nm2 * sensing_area_nm2))

    if NR < 1:
        raise ValueError(
            "Fewer than one receptor. Increase Lx_nm, depth_nm, or "
            "receptor_density_nm2."
        )

    if not P.allow_multiple_receptors_per_site and NR > surface_sites:
        raise ValueError(
            "The collapsed 2D surface contains more receptors than x lattice "
            "sites. Decrease receptor density or depth_nm, decrease a_nm, or "
            "set allow_multiple_receptors_per_site=True."
        )

    bulk_conc_nm3 = P.ligand_conc_M * NA * L_PER_NM3
    mean_ligands = bulk_conc_nm3 * volume_nm3

    # For a 2D nearest-neighbor random walk, four directional moves each have
    # probability D*dt/a^2, so stability requires 4*D*dt/a^2 <= 1.
    if P.dt_s is None:
        dt_s = 0.95 * P.a_nm**2 / (4 * D_nm2_s)
    else:
        dt_s = float(P.dt_s)

    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    p_axis = D_nm2_s * dt_s / P.a_nm**2
    p_stay = 1.0 - 4.0 * p_axis

    if p_stay < -1e-12:
        max_dt = P.a_nm**2 / (4 * D_nm2_s)
        raise ValueError(
            f"dt_s is too large for 2D diffusion stability. Use dt_s <= {max_dt:.3e} s."
        )

    # move index: 0 stay, 1 +x, 2 -x, 3 +z, 4 -z
    move_probs = np.array(
        [max(0.0, p_stay), p_axis, p_axis, p_axis, p_axis],
        dtype=float,
    )
    move_probs /= move_probs.sum()

    # One 2D reaction lattice cell corresponds to a physical slab volume
    # a_nm * a_nm * depth_nm.
    V_rxn_L = (
        P.reaction_volume_voxels
        * P.a_nm**2
        * P.depth_nm
        * L_PER_NM3
    )

    if P.k_on_M_inv_s > 0:
        kon_exp_per_receptor = P.k_on_M_inv_s * dt_s / (NA * V_rxn_L)
        Kd_M = P.k_off_s / P.k_on_M_inv_s
    else:
        kon_exp_per_receptor = 0.0
        Kd_M = np.inf

    p_off = 1.0 - np.exp(-P.k_off_s * dt_s)

    escape_z_layer = int(np.ceil(P.escape_height_nm / P.a_nm))
    escape_z_layer = max(2, min(escape_z_layer, Nz))

    # Diffusive exchange with a fixed-concentration reservoir. The boundary
    # measures are physical face areas of the represented slab.
    exchange_prefactor = bulk_conc_nm3 * D_nm2_s * dt_s / P.a_nm

    area_x_faces_nm2 = H_eff_nm * P.depth_nm
    area_z_top_nm2 = Lx_eff_nm * P.depth_nm

    face_injection_means = {
        "x_min": exchange_prefactor * area_x_faces_nm2,
        "x_max": exchange_prefactor * area_x_faces_nm2,
        "z_max": exchange_prefactor * area_z_top_nm2,
    }

    total_injection_mean = float(sum(face_injection_means.values()))

    return Derived(
        Nx=Nx,
        Nz=Nz,
        Lx_eff_nm=Lx_eff_nm,
        H_eff_nm=H_eff_nm,
        depth_nm=P.depth_nm,
        volume_nm3=volume_nm3,
        sensing_area_nm2=sensing_area_nm2,
        NR=NR,
        mean_ligands=mean_ligands,
        bulk_conc_nm3=bulk_conc_nm3,
        D_nm2_s=D_nm2_s,
        face_injection_means=face_injection_means,
        total_injection_mean=total_injection_mean,
        dt_s=dt_s,
        move_probs=move_probs,
        kon_exp_per_receptor=kon_exp_per_receptor,
        p_off=p_off,
        Kd_M=Kd_M,
        escape_z_layer=escape_z_layer,
        use_receptor_grid=not P.allow_multiple_receptors_per_site,
    )


def initialize(P: Params, G: Derived) -> State:
    """Create receptors and the initial equilibrium bulk ligand population."""

    rng = np.random.default_rng(P.seed)
    n_sites = G.Nx

    if P.allow_multiple_receptors_per_site:
        receptor_sites = rng.integers(0, n_sites, size=G.NR)
    else:
        receptor_sites = rng.choice(n_sites, size=G.NR, replace=False)

    receptor_x = np.asarray(receptor_sites, dtype=np.int32)

    site_to_receptors: List[List[int]] = [[] for _ in range(n_sites)]
    for receptor_id, site in enumerate(receptor_sites):
        site_to_receptors[int(site)].append(int(receptor_id))

    receptor_grid = np.full(n_sites, -1, dtype=np.int64)
    if not P.allow_multiple_receptors_per_site:
        receptor_grid[receptor_sites] = np.arange(G.NR, dtype=np.int64)

    receptor_ligand = np.full(G.NR, -1, dtype=np.int64)

    if P.use_poisson_ligand_count:
        N_L = int(rng.poisson(G.mean_ligands))
    else:
        N_L = int(round(G.mean_ligands))

    ligand_xz = np.empty((N_L, 2), dtype=np.int32)
    if N_L > 0:
        ligand_xz[:, 0] = rng.integers(0, G.Nx, size=N_L)
        ligand_xz[:, 1] = rng.integers(1, G.Nz + 1, size=N_L)

    return State(
        rng=rng,
        receptor_x=receptor_x,
        site_to_receptors=site_to_receptors,
        receptor_grid=receptor_grid,
        receptor_ligand=receptor_ligand,
        ligand_xz=ligand_xz,
        ligand_receptor=np.full(N_L, -1, dtype=np.int64),
        ligand_active=np.ones(N_L, dtype=bool),
        last_unbound_receptor=np.full(N_L, -1, dtype=np.int64),
        rebind_watch_active=np.zeros(N_L, dtype=bool),
        ligand_n_bindings=np.zeros(N_L, dtype=np.int64),
        ligand_n_rebindings=np.zeros(N_L, dtype=np.int64),
        ligand_n_self_rebindings=np.zeros(N_L, dtype=np.int64),
        ligand_n_cross_rebindings=np.zeros(N_L, dtype=np.int64),
        n_ligands_created=N_L,
        step_count=0,
        t_s=0.0,
        event_counts={
            "bindings": 0,
            "unbindings": 0,
            "rebindings": 0,
            "self_rebindings": 0,
            "cross_rebindings": 0,
            "lost_to_bulk": 0,
            "lost_x_min": 0,
            "lost_x_max": 0,
            "lost_z_max": 0,
            "entered_from_bulk": 0,
            "entered_x_min": 0,
            "entered_x_max": 0,
            "entered_z_max": 0,
        },
    )


# ---------------------------------------------------------------------
# Bulk exchange
# ---------------------------------------------------------------------


def _zero_counts() -> Dict[str, int]:
    return {"x_min": 0, "x_max": 0, "z_max": 0}


def _sample_zero_truncated_poisson(
    rng: np.random.Generator,
    lam: float,
) -> int:
    """Sample Poisson(lam) conditioned on a value greater than zero."""

    if lam <= 0:
        return 0

    p0 = np.exp(-lam)
    target = p0 + rng.random() * (1.0 - p0)

    p = p0
    cdf = p0
    k = 0

    while cdf < target:
        k += 1
        p *= lam / k
        cdf += p

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

    faces = list(G.face_injection_means)
    means = np.array([G.face_injection_means[face] for face in faces], dtype=float)

    if not force_at_least_one:
        sampled = rng.poisson(means)
        return {face: int(n) for face, n in zip(faces, sampled)}

    n_total = _sample_zero_truncated_poisson(rng, G.total_injection_mean)
    if n_total <= 0:
        return _zero_counts()

    sampled = rng.multinomial(n_total, means / means.sum())
    return {face: int(n) for face, n in zip(faces, sampled)}


def add_ligands_from_bulk(
    S: State,
    G: Derived,
    force_at_least_one: bool = False,
) -> Dict[str, int]:
    """Inject reservoir ligands through x-min, x-max, and z-max boundaries."""

    counts = _sample_entry_counts(
        S.rng,
        G,
        force_at_least_one=force_at_least_one,
    )
    n_new = sum(counts.values())

    if n_new <= 0:
        return counts

    xz_new = np.empty((n_new, 2), dtype=np.int32)
    start = 0

    for face, n in counts.items():
        if n == 0:
            continue

        stop = start + n

        if face == "x_min":
            xz_new[start:stop, 0] = 0
            xz_new[start:stop, 1] = S.rng.integers(1, G.Nz + 1, size=n)

        elif face == "x_max":
            xz_new[start:stop, 0] = G.Nx - 1
            xz_new[start:stop, 1] = S.rng.integers(1, G.Nz + 1, size=n)

        elif face == "z_max":
            xz_new[start:stop, 0] = S.rng.integers(0, G.Nx, size=n)
            xz_new[start:stop, 1] = G.Nz

        start = stop

    S.ligand_xz = np.vstack([S.ligand_xz, xz_new])
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
    return counts


def register_entry_counts(S: State, counts: Dict[str, int]) -> None:
    S.event_counts["entered_from_bulk"] += sum(counts.values())
    S.event_counts["entered_x_min"] += counts["x_min"]
    S.event_counts["entered_x_max"] += counts["x_max"]
    S.event_counts["entered_z_max"] += counts["z_max"]


# ---------------------------------------------------------------------
# Memory compaction and snapshots
# ---------------------------------------------------------------------


def state_is_empty(S: State) -> bool:
    return (
        np.count_nonzero(S.ligand_active) == 0
        and np.count_nonzero(S.receptor_ligand >= 0) == 0
    )


def compact_ligands(S: State) -> int:
    """Delete inactive ligand slots and remap bound-ligand indices."""

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

    S.ligand_xz = S.ligand_xz[keep].copy()
    S.ligand_receptor = S.ligand_receptor[keep].copy()
    S.ligand_active = S.ligand_active[keep].copy()
    S.last_unbound_receptor = S.last_unbound_receptor[keep].copy()
    S.rebind_watch_active = S.rebind_watch_active[keep].copy()
    S.ligand_n_bindings = S.ligand_n_bindings[keep].copy()
    S.ligand_n_rebindings = S.ligand_n_rebindings[keep].copy()
    S.ligand_n_self_rebindings = S.ligand_n_self_rebindings[keep].copy()
    S.ligand_n_cross_rebindings = S.ligand_n_cross_rebindings[keep].copy()

    return n_old - n_keep


def snapshot(S: State, G: Derived) -> Dict[str, float]:
    bound = int(np.count_nonzero(S.receptor_ligand >= 0))
    active_free = S.ligand_active & (S.ligand_receptor < 0)

    return {
        "step": S.step_count,
        "t_s": S.t_s,
        "B": bound,
        "theta": bound / G.NR,
        "N_free": int(np.count_nonzero(active_free)),
        "N_bound": bound,
        "N_active_ligands": int(np.count_nonzero(S.ligand_active)),
        "N_ligand_slots": int(S.ligand_xz.shape[0]),
        "N_total_ligands_ever": int(S.n_ligands_created),
        "binding_events_total": int(S.event_counts["bindings"]),
        "unbinding_events_total": int(S.event_counts["unbindings"]),
        "rebinding_events_total": int(S.event_counts["rebindings"]),
        "self_rebindings_total": int(S.event_counts["self_rebindings"]),
        "cross_rebindings_total": int(S.event_counts["cross_rebindings"]),
        "lost_to_bulk_total": int(S.event_counts["lost_to_bulk"]),
        "entered_from_bulk_total": int(S.event_counts["entered_from_bulk"]),
        "lost_x_min_total": int(S.event_counts["lost_x_min"]),
        "lost_x_max_total": int(S.event_counts["lost_x_max"]),
        "lost_z_max_total": int(S.event_counts["lost_z_max"]),
        "entered_x_min_total": int(S.event_counts["entered_x_min"]),
        "entered_x_max_total": int(S.event_counts["entered_x_max"]),
        "entered_z_max_total": int(S.event_counts["entered_z_max"]),
    }


# ---------------------------------------------------------------------
# Core Monte Carlo step
# ---------------------------------------------------------------------


def step(S: State, G: Derived) -> None:
    """Advance the 2D model by one diffusion-reaction timestep."""

    rng = S.rng
    bound_receptors_start = np.flatnonzero(S.receptor_ligand >= 0)

    # If the domain is empty, only reservoir entry needs to be sampled.
    if np.count_nonzero(S.ligand_active) == 0 and bound_receptors_start.size == 0:
        entered_counts = add_ligands_from_bulk(S, G)
        register_entry_counts(S, entered_counts)
        S.step_count += 1
        S.t_s += G.dt_s
        return

    # ------------------------------------------------------------------
    # 1. Diffuse active, unbound ligands in x and z.
    # ------------------------------------------------------------------
    free_ids = np.flatnonzero(S.ligand_active & (S.ligand_receptor < 0))

    if free_ids.size:
        moves = rng.choice(5, size=free_ids.size, p=G.move_probs)
        pos = S.ligand_xz[free_ids].copy()

        pos[moves == 1, 0] += 1
        pos[moves == 2, 0] -= 1
        pos[moves == 3, 1] += 1

        # z = 0 is the sensing face and remains reflecting for free ligands.
        pos[moves == 4, 1] = np.maximum(pos[moves == 4, 1] - 1, 1)

        S.ligand_xz[free_ids] = pos

        lost_x_min = free_ids[S.ligand_xz[free_ids, 0] < 0]
        lost_x_max = free_ids[S.ligand_xz[free_ids, 0] >= G.Nx]
        lost_z_max = free_ids[S.ligand_xz[free_ids, 1] > G.Nz]

        lost_ids = np.unique(
            np.concatenate([lost_x_min, lost_x_max, lost_z_max])
        )

        if lost_ids.size:
            S.ligand_active[lost_ids] = False
            S.rebind_watch_active[lost_ids] = False

            S.event_counts["lost_x_min"] += int(lost_x_min.size)
            S.event_counts["lost_x_max"] += int(lost_x_max.size)
            S.event_counts["lost_z_max"] += int(lost_z_max.size)
            S.event_counts["lost_to_bulk"] += int(lost_ids.size)

        still_free = np.flatnonzero(S.ligand_active & (S.ligand_receptor < 0))
        watched = still_free[S.rebind_watch_active[still_free]]

        if watched.size:
            escaped = watched[S.ligand_xz[watched, 1] >= G.escape_z_layer]
            S.rebind_watch_active[escaped] = False

    # ------------------------------------------------------------------
    # 2. Bind ligands occupying the z = 1 layer above a receptor site.
    # ------------------------------------------------------------------
    bind_events = 0
    rebind_events = 0
    self_rebind_events = 0
    cross_rebind_events = 0

    free_ids = np.flatnonzero(S.ligand_active & (S.ligand_receptor < 0))
    near_ids = free_ids[S.ligand_xz[free_ids, 1] == 1]

    if near_ids.size:
        rng.shuffle(near_ids)

        if G.use_receptor_grid:
            p_bind_single = 1.0 - np.exp(-G.kon_exp_per_receptor)

            for ligand_id in near_ids:
                x = int(S.ligand_xz[ligand_id, 0])
                receptor_id = int(S.receptor_grid[x])

                if receptor_id < 0:
                    continue
                if S.receptor_ligand[receptor_id] >= 0:
                    continue
                if rng.random() >= p_bind_single:
                    continue

                S.ligand_receptor[ligand_id] = receptor_id
                S.receptor_ligand[receptor_id] = ligand_id
                S.ligand_xz[ligand_id] = np.array([x, 0], dtype=np.int32)

                S.ligand_n_bindings[ligand_id] += 1
                bind_events += 1

                if S.rebind_watch_active[ligand_id]:
                    S.ligand_n_rebindings[ligand_id] += 1
                    rebind_events += 1

                    if S.last_unbound_receptor[ligand_id] == receptor_id:
                        S.ligand_n_self_rebindings[ligand_id] += 1
                        self_rebind_events += 1
                    else:
                        S.ligand_n_cross_rebindings[ligand_id] += 1
                        cross_rebind_events += 1

                    S.rebind_watch_active[ligand_id] = False

        else:
            # Multiple collapsed receptors may share the same x lattice site.
            for ligand_id in near_ids:
                x = int(S.ligand_xz[ligand_id, 0])
                receptors = S.site_to_receptors[x]

                if not receptors:
                    continue

                unbound = [
                    receptor_id
                    for receptor_id in receptors
                    if S.receptor_ligand[receptor_id] < 0
                ]

                if not unbound:
                    continue

                p_bind = 1.0 - np.exp(
                    -G.kon_exp_per_receptor * len(unbound)
                )
                if rng.random() >= p_bind:
                    continue

                receptor_id = unbound[int(rng.integers(0, len(unbound)))]

                S.ligand_receptor[ligand_id] = receptor_id
                S.receptor_ligand[receptor_id] = ligand_id
                S.ligand_xz[ligand_id] = np.array([x, 0], dtype=np.int32)

                S.ligand_n_bindings[ligand_id] += 1
                bind_events += 1

                if S.rebind_watch_active[ligand_id]:
                    S.ligand_n_rebindings[ligand_id] += 1
                    rebind_events += 1

                    if S.last_unbound_receptor[ligand_id] == receptor_id:
                        S.ligand_n_self_rebindings[ligand_id] += 1
                        self_rebind_events += 1
                    else:
                        S.ligand_n_cross_rebindings[ligand_id] += 1
                        cross_rebind_events += 1

                    S.rebind_watch_active[ligand_id] = False

    # ------------------------------------------------------------------
    # 3. Dissociate receptors that were bound at the start of the step.
    # Newly bound complexes cannot dissociate until the next step.
    # ------------------------------------------------------------------
    unbind_events = 0

    if bound_receptors_start.size:
        rng.shuffle(bound_receptors_start)

        for receptor_id in bound_receptors_start:
            ligand_id = int(S.receptor_ligand[receptor_id])

            if ligand_id < 0:
                continue
            if rng.random() >= G.p_off:
                continue

            x = int(S.receptor_x[receptor_id])

            S.receptor_ligand[receptor_id] = -1
            S.ligand_receptor[ligand_id] = -1
            S.ligand_xz[ligand_id] = np.array([x, 1], dtype=np.int32)

            S.last_unbound_receptor[ligand_id] = receptor_id
            S.rebind_watch_active[ligand_id] = True
            unbind_events += 1

    # ------------------------------------------------------------------
    # 4. Inject reservoir ligands at the end of the timestep.
    # ------------------------------------------------------------------
    entered_counts = add_ligands_from_bulk(S, G)
    register_entry_counts(S, entered_counts)

    S.step_count += 1
    S.t_s += G.dt_s

    S.event_counts["bindings"] += bind_events
    S.event_counts["unbindings"] += unbind_events
    S.event_counts["rebindings"] += rebind_events
    S.event_counts["self_rebindings"] += self_rebind_events
    S.event_counts["cross_rebindings"] += cross_rebind_events


# ---------------------------------------------------------------------
# State capture for visualization
# ---------------------------------------------------------------------


def capture_state_frame(S: State, G: Derived, P: Params) -> Dict:
    """Return a plotting-friendly copy of the current 2D state."""

    bound_mask = S.receptor_ligand >= 0
    active_free_mask = S.ligand_active & (S.ligand_receptor < 0)

    return {
        "t_s": S.t_s,
        "step": S.step_count,
        "Lx_nm": G.Lx_eff_nm,
        "H_nm": G.H_eff_nm,
        "depth_nm": P.depth_nm,
        "a_nm": P.a_nm,
        "receptor_x_nm": S.receptor_x.copy() * P.a_nm,
        "receptor_bound": bound_mask.copy(),
        "ligand_xz_nm": S.ligand_xz[active_free_mask].copy() * P.a_nm,
        "B": int(np.count_nonzero(bound_mask)),
        "theta": float(np.count_nonzero(bound_mask) / G.NR),
        "N_free": int(np.count_nonzero(active_free_mask)),
        "N_active_ligands": int(np.count_nonzero(S.ligand_active)),
        "N_ligand_slots": int(S.ligand_xz.shape[0]),
        "N_total_ligands_ever": int(S.n_ligands_created),
    }


# ---------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------


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
):
    """Run the 2D lattice Monte Carlo simulation.

    Returns
    -------
    pandas.DataFrame
        Time history when both optional return flags are False.
    (hist, state, derived)
        When ``return_state=True``.
    (hist, state_frames)
        When ``save_state_frames=True``.
    (hist, state, derived, state_frames)
        When both optional return flags are True.
    """

    if seconds < 0:
        raise ValueError("seconds cannot be negative.")
    if not 0 <= compact_inactive_fraction <= 1:
        raise ValueError("compact_inactive_fraction must be between 0 and 1.")

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - only used when tqdm is unavailable
        tqdm = None

    G = derive(P)
    S = initialize(P, G)

    if verbose:
        print("=" * 64)
        print("2D simulation initialization")
        print("=" * 64)
        print(f"Grid                         : {G.Nx} x {G.Nz} (x x z)")
        print(f"Out-of-plane depth          : {G.depth_nm:.3f} nm")
        print(f"Sensing area represented    : {G.sensing_area_nm2:.3f} nm^2")
        print(f"Simulation volume represented: {G.volume_nm3:.3f} nm^3")
        print(f"Receptors                    : {G.NR:,}")
        print(f"Initial ligands              : {S.ligand_xz.shape[0]:,}")
        print(f"Mean ligands                 : {G.mean_ligands:.3e}")
        print(f"Bulk concentration           : {G.bulk_conc_nm3:.3e} molecules/nm^3")
        print(f"D                            : {P.D_cm2_s:.3e} cm^2/s")
        print(f"D                            : {G.D_nm2_s:.3e} nm^2/s")
        print(f"Mean bulk entries            : {G.total_injection_mean:.3e} ligands/step")
        print(f"Receptor density             : {P.receptor_density_nm2:.3f} receptors/nm^2")
        print(f"Ligand concentration         : {P.ligand_conc_M * 1e9:.3f} nM")
        print(f"KD                           : {G.Kd_M * 1e9:.3f} nM")
        print(f"dt                           : {G.dt_s:.3e} s")
        print(f"Fast receptor grid           : {G.use_receptor_grid}")
        print("=" * 64)

    n_steps = int(np.ceil(seconds / G.dt_s))

    if record_every_s is None:
        record_every = max(1, n_steps // 200) if n_steps else 1
    else:
        if record_every_s <= 0:
            raise ValueError("record_every_s must be positive when provided.")
        record_every = max(1, int(round(record_every_s / G.dt_s)))

    if save_state_frames:
        n_state_frames = max(1, int(n_state_frames))
        frame_steps = np.unique(np.linspace(0, n_steps, n_state_frames, dtype=int))
        frame_steps_set = set(frame_steps.tolist())
        state_frames: Optional[List[Dict]] = [capture_state_frame(S, G, P)]
    else:
        frame_steps_set = set()
        state_frames = None

    rows = [snapshot(S, G)]

    iterator = range(n_steps)
    if show_progress and tqdm is not None:
        iterator = tqdm(
            iterator,
            total=n_steps,
            desc="Running 2D simulation",
            unit="step",
            mininterval=0.5,
            miniters=1000,
            leave=True,
            dynamic_ncols=True,
        )

    for _ in iterator:
        step(S, G)

        if compact_every is not None and compact_every > 0:
            if S.step_count % compact_every == 0:
                total_slots = S.ligand_active.size
                if total_slots > 0:
                    inactive = int(np.count_nonzero(~S.ligand_active))
                    if inactive / total_slots >= compact_inactive_fraction:
                        compact_ligands(S)

        if S.step_count % record_every == 0:
            rows.append(snapshot(S, G))

        if save_state_frames and S.step_count in frame_steps_set:
            # Avoid duplicating the explicitly saved initial frame.
            if S.step_count != 0:
                state_frames.append(capture_state_frame(S, G, P))

    # Ensure the final state is represented when the recording interval did
    # not happen to land exactly on the final step.
    if rows[-1]["step"] != S.step_count:
        rows.append(snapshot(S, G))

    hist = pd.DataFrame(rows)

    if return_state and save_state_frames:
        return hist, S, G, state_frames
    if return_state:
        return hist, S, G
    if save_state_frames:
        return hist, state_frames
    return hist


# ---------------------------------------------------------------------
# Optional convenience plotting
# ---------------------------------------------------------------------


def plot_history(hist: pd.DataFrame, ax=None):
    """Plot receptor occupancy versus simulation time."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(hist["t_s"], hist["theta"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction of receptors bound")
    ax.set_ylim(0, 1)
    return ax


def plot_state_frame(frame: Dict, ax=None, ligand_size: float = 20, receptor_size: float = 50):
    """Plot one state frame in the x-z simulation plane."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    receptor_x = np.asarray(frame["receptor_x_nm"])
    receptor_bound = np.asarray(frame["receptor_bound"], dtype=bool)
    ligands = np.asarray(frame["ligand_xz_nm"])

    if np.any(~receptor_bound):
        ax.scatter(
            receptor_x[~receptor_bound],
            np.zeros(np.count_nonzero(~receptor_bound)),
            s=receptor_size,
            alpha=0.4,
            label="unbound receptors",
        )

    if np.any(receptor_bound):
        ax.scatter(
            receptor_x[receptor_bound],
            np.zeros(np.count_nonzero(receptor_bound)),
            s=receptor_size,
            marker="^",
            alpha=0.9,
            label="bound receptors",
        )

    if ligands.size:
        ax.scatter(
            ligands[:, 0],
            ligands[:, 1],
            s=ligand_size,
            alpha=0.6,
            label="free ligands",
        )

    ax.set_xlim(-0.5 * frame["a_nm"], frame["Lx_nm"] - 0.5 * frame["a_nm"])
    ax.set_ylim(-0.5 * frame["a_nm"], frame["H_nm"] + 0.5 * frame["a_nm"])
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title(
        f"t = {frame['t_s']:.3g} s, bound fraction = {frame['theta']:.3f}"
    )
    ax.legend(loc="best")
    return ax


if __name__ == "__main__":
    # A small example that can be run directly. The elevated concentration is
    # used only so that a short demonstration contains several ligand events.
    example_params = Params(
        Lx_nm=50.0,
        H_nm=25.0,
        a_nm=1.0,
        depth_nm=10.0,
        receptor_density_nm2=0.05,
        ligand_conc_M=1e-4,
        k_on_M_inv_s=1e7,
        k_off_s=0.1,
        allow_multiple_receptors_per_site=False,
        seed=1,
    )

    example_history = run_simulation(
        example_params,
        seconds=1e-5,
        show_progress=False,
        verbose=True,
    )
    print(example_history.tail())