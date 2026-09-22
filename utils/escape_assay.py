"""Single-molecule escape and rebinding assays on voxelized sensor geometries.

The assay follows independent copies of one released molecule.  Unlike the
concentration simulations in :mod:`utils.biosensor_mc`, trajectories in an
escape assay never interact with one another and no molecules enter from the
bulk.  The diffusion and reaction probabilities are nevertheless taken from
the same :class:`~utils.biosensor_mc.Derived` object, so the two simulations
use the same lattice physics.

Typical use
-----------
::

    from utils.biosensor_mc import Params
    from utils.escape_assay import ReleaseLocation, run_escape_assay
    from utils.generate_geometries import make_spherical_bowl_geometry

    P = Params(Lx_m=400e-9, Ly_m=400e-9, H_m=300e-9, a_m=5e-9)
    geometry = make_spherical_bowl_geometry(P, radius_m=150e-9,
                                             depth_m=100e-9)
    result = run_escape_assay(
        P,
        geometry,
        release_locations=[ReleaseLocation.from_xyz_m("bowl_bottom",
                                                       (200e-9, 200e-9, 5e-9))],
        n_trials_per_location=10_000,
        max_time_s=1.0,
        escape_mode="z_plane",
        escape_z_m=150e-9,
    )

``result.trajectories`` contains one row per molecule and
``result.survival`` contains a location-stratified Kaplan--Meier estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.biosensor_mc import (
    MOVE_VECTORS,
    Derived,
    Params,
    SensorGeometry,
    derive,
)


@dataclass(frozen=True)
class ReleaseLocation:
    """A named single-molecule starting location.

    Exactly one of ``face_id`` and ``xyz_m`` must be supplied.  A face release
    starts in the accessible fluid voxel adjacent to that surface face and
    treats that face as the molecule's source receptor.  A coordinate release
    is snapped to the nearest accessible fluid voxel and has no source
    receptor unless ``source_face_id`` is explicitly supplied.
    """

    label: str
    face_id: Optional[int] = None
    xyz_m: Optional[Tuple[float, float, float]] = None
    source_face_id: Optional[int] = None

    def __post_init__(self) -> None:
        if (self.face_id is None) == (self.xyz_m is None):
            raise ValueError("Specify exactly one of face_id and xyz_m.")
        if not str(self.label):
            raise ValueError("ReleaseLocation.label must not be empty.")
        if self.xyz_m is not None and len(self.xyz_m) != 3:
            raise ValueError("xyz_m must contain exactly three coordinates.")

    @classmethod
    def from_face(cls, label: str, face_id: int) -> "ReleaseLocation":
        return cls(label=label, face_id=int(face_id))

    @classmethod
    def from_xyz_m(
        cls,
        label: str,
        xyz_m: Sequence[float],
        source_face_id: Optional[int] = None,
    ) -> "ReleaseLocation":
        return cls(
            label=label,
            xyz_m=tuple(float(value) for value in xyz_m),
            source_face_id=source_face_id,
        )


@dataclass
class EscapeAssayResult:
    """Tables and model metadata returned by :func:`run_escape_assay`."""

    params: Params
    derived: Derived
    releases: pd.DataFrame
    receptors: pd.DataFrame
    trajectories: pd.DataFrame
    survival: pd.DataFrame


def surface_release_locations(
    geometry: SensorGeometry,
    face_ids: Optional[Iterable[int]] = None,
    prefix: str = "face",
) -> list[ReleaseLocation]:
    """Create release locations for selected (or all reactive) surface faces."""

    if face_ids is None:
        ids = np.flatnonzero(geometry.reactive_face_mask)
    else:
        ids = np.asarray(list(face_ids), dtype=np.int64)
    return [ReleaseLocation.from_face(f"{prefix}_{int(i)}", int(i)) for i in ids]


def sample_receptor_faces(
    P: Params,
    geometry: SensorGeometry,
    *,
    eligible_face_ids: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample one receptor layout from the requested surface density.

    Receptors are sampled uniformly by surface face. All faces currently have
    area ``a_m**2``, but the count is calculated from the actual summed face
    area so this remains correct for future geometry implementations.

    ``eligible_face_ids`` can restrict receptors to a selected part of a
    structure. When it is omitted, all bulk-accessible reactive faces are
    eligible. ``receptor_count_override`` takes precedence over density, as in
    :func:`utils.biosensor_mc.derive`.
    """

    G = derive(P, geometry)
    active_faces = G.reactive_face_ids
    active_set = set(map(int, active_faces))

    if eligible_face_ids is None:
        eligible = active_faces.copy()
    else:
        eligible = np.asarray(eligible_face_ids, dtype=np.int64)
        if eligible.ndim != 1 or np.unique(eligible).size != eligible.size:
            raise ValueError(
                "eligible_face_ids must be a one-dimensional sequence of unique IDs."
            )
        invalid = [int(face) for face in eligible if int(face) not in active_set]
        if invalid:
            raise ValueError(
                f"eligible_face_ids contains non-reactive or inaccessible faces: {invalid}"
            )

    if P.receptor_count_override is None:
        eligible_area_m2 = float(np.sum(geometry.surface_area_m2[eligible]))
        receptor_count = int(round(P.receptor_density_m2 * eligible_area_m2))
    else:
        receptor_count = int(P.receptor_count_override)

    if receptor_count < 0:
        raise ValueError("The receptor count cannot be negative.")
    if receptor_count == 0:
        return np.empty(0, dtype=np.int64)
    if eligible.size == 0:
        raise ValueError("No eligible reactive faces are available for receptors.")
    if not P.allow_multiple_receptors_per_site and receptor_count > eligible.size:
        raise ValueError(
            "The requested receptor count exceeds one receptor per eligible face."
        )

    rng = np.random.default_rng(P.seed if seed is None else seed)
    return np.asarray(
        rng.choice(
            eligible,
            size=receptor_count,
            replace=bool(P.allow_multiple_receptors_per_site),
        ),
        dtype=np.int64,
    )


def _flat_site(xyz: np.ndarray, G: Derived) -> np.ndarray:
    xyz = np.asarray(xyz)
    return ((xyz[..., 0] * G.Ny + xyz[..., 1]) * (G.Nz + 1) + xyz[..., 2]).astype(np.int64)


def _resolve_releases(
    locations: Sequence[ReleaseLocation],
    G: Derived,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    xyz_out = []
    source_faces = []
    accessible = G.accessible_fluid_xyz

    for location_id, location in enumerate(locations):
        if location.face_id is not None:
            face_id = int(location.face_id)
            if face_id < 0 or face_id >= G.geometry.n_surface_faces:
                raise ValueError(f"Invalid face_id {face_id} for {location.label!r}.")
            xyz = G.geometry.surface_fluid_xyz[face_id].astype(np.int32, copy=True)
            source_face = face_id
        else:
            requested_m = np.asarray(location.xyz_m, dtype=float)
            if not np.all(np.isfinite(requested_m)):
                raise ValueError(f"Non-finite xyz_m for {location.label!r}.")
            # Nearest accessible voxel in Euclidean lattice distance.  This
            # also handles coordinates initially placed inside voxelized solid.
            requested_lattice = requested_m / G.a_m
            nearest = int(np.argmin(np.sum((accessible - requested_lattice) ** 2, axis=1)))
            xyz = accessible[nearest].copy()
            source_face = -1 if location.source_face_id is None else int(location.source_face_id)

        if not G.accessible_fluid_mask[tuple(xyz)]:
            raise ValueError(f"Release {location.label!r} is not in accessible fluid.")
        if source_face < -1 or source_face >= G.geometry.n_surface_faces:
            raise ValueError(f"Invalid source_face_id {source_face} for {location.label!r}.")

        xyz_out.append(xyz)
        source_faces.append(source_face)
        rows.append({
            "location_id": location_id,
            "label": location.label,
            "source_face_id": source_face,
            "release_x_index": int(xyz[0]),
            "release_y_index": int(xyz[1]),
            "release_z_index": int(xyz[2]),
            "release_x_m": float(xyz[0] * G.a_m),
            "release_y_m": float(xyz[1] * G.a_m),
            "release_z_m": float(xyz[2] * G.a_m),
        })

    return pd.DataFrame(rows), np.asarray(xyz_out, dtype=np.int32), np.asarray(source_faces, dtype=np.int64)


def _choose_receptor_faces(
    P: Params,
    G: Derived,
    source_faces: np.ndarray,
    receptor_face_ids: Optional[Sequence[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    reactive = G.reactive_face_ids
    reactive_set = set(map(int, reactive))
    forced = np.unique(source_faces[source_faces >= 0]).astype(np.int64)

    bad_forced = [int(face) for face in forced if int(face) not in reactive_set]
    if bad_forced:
        raise ValueError(f"Source receptor faces are not active reactive faces: {bad_forced}")

    if receptor_face_ids is not None:
        selected = np.asarray(receptor_face_ids, dtype=np.int64)
        if selected.ndim != 1 or np.unique(selected).size != selected.size:
            raise ValueError("receptor_face_ids must be a one-dimensional sequence of unique IDs.")
        bad = [int(face) for face in selected if int(face) not in reactive_set]
        if bad:
            raise ValueError(f"receptor_face_ids contains non-reactive faces: {bad}")
        return np.unique(np.concatenate([selected, forced])).astype(np.int64)

    target = max(int(G.NR), int(forced.size))
    if target == 0:
        return forced
    if not P.allow_multiple_receptors_per_site and target > reactive.size:
        raise ValueError("Requested receptor count exceeds the number of reactive faces.")

    remaining = np.asarray([face for face in reactive if int(face) not in set(map(int, forced))], dtype=np.int64)
    n_needed = target - forced.size
    if n_needed <= 0:
        return forced
    replace = bool(P.allow_multiple_receptors_per_site and n_needed > remaining.size)
    sampled = rng.choice(remaining if remaining.size else reactive, size=n_needed, replace=replace)
    return np.concatenate([forced, np.asarray(sampled, dtype=np.int64)])


def _kaplan_meier(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Compute a right-continuous Kaplan--Meier curve for every location."""

    frames = []
    for (location_id, label), group in trajectories.groupby(["location_id", "label"], sort=False):
        times = group["observation_time_s"].to_numpy(float)
        events = group["escaped"].to_numpy(bool)
        event_times = np.unique(times[events])
        survival = 1.0
        rows = [{
            "location_id": int(location_id), "label": label, "t_s": 0.0,
            "n_at_risk": int(len(group)), "n_escaped": 0,
            "survival_probability": 1.0,
        }]
        for t_s in event_times:
            at_risk = int(np.count_nonzero(times >= t_s))
            escaped = int(np.count_nonzero(events & (times == t_s)))
            survival *= 1.0 - escaped / at_risk
            rows.append({
                "location_id": int(location_id), "label": label, "t_s": float(t_s),
                "n_at_risk": at_risk, "n_escaped": escaped,
                "survival_probability": float(survival),
            })
        final_time = float(np.max(times))
        if rows[-1]["t_s"] < final_time:
            rows.append({
                "location_id": int(location_id), "label": label, "t_s": final_time,
                "n_at_risk": int(np.count_nonzero(times >= final_time)),
                "n_escaped": 0, "survival_probability": float(survival),
            })
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_escape_assay(
    P: Params,
    geometry: SensorGeometry,
    release_locations: Sequence[ReleaseLocation],
    n_trials_per_location: int = 1_000,
    max_time_s: float = 1.0,
    *,
    escape_mode: str = "surface_distance",
    escape_distance_m: Optional[float] = None,
    escape_z_m: Optional[float] = None,
    receptor_face_ids: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> EscapeAssayResult:
    """Run independent single-molecule release trajectories.

    Parameters
    ----------
    P, geometry
        The same model parameters and voxel geometry used by the main Monte
        Carlo simulation.  Bulk concentration and bulk injection are ignored.
    release_locations
        Surface-face or physical-coordinate starting locations.
    n_trials_per_location
        Number of statistically independent trajectories at each location.
    max_time_s
        Trajectories still present at this time are right-censored.
    escape_mode
        ``"surface_distance"``: escape on reaching ``escape_distance_m`` from
        the nearest reactive surface; ``"z_plane"``: escape on reaching or
        crossing ``escape_z_m``; ``"domain_exit"``: escape only on crossing an
        open boundary or the well-mixed reservoir interface.  Domain loss is
        always counted as escape in every mode.
    escape_distance_m
        Threshold for ``surface_distance`` mode. Defaults to the value derived
        from ``P.escape_distance_m``/``P.escape_height_m``.
    escape_z_m
        Plane height for ``z_plane`` mode, in meters.
    receptor_face_ids
        Optional exact receptor layout. Otherwise one layout is sampled from
        ``P`` and reused for every trajectory; source receptor faces are always
        included so receptor-release assays genuinely begin at a receptor.
    seed
        Assay RNG seed. Defaults to ``P.seed``.
    """

    if n_trials_per_location < 1:
        raise ValueError("n_trials_per_location must be at least 1.")
    if max_time_s <= 0:
        raise ValueError("max_time_s must be positive.")
    if not release_locations:
        raise ValueError("release_locations must contain at least one location.")
    if len({location.label for location in release_locations}) != len(release_locations):
        raise ValueError("Release-location labels must be unique.")

    mode = str(escape_mode).lower()
    if mode not in {"surface_distance", "z_plane", "domain_exit"}:
        raise ValueError("escape_mode must be 'surface_distance', 'z_plane', or 'domain_exit'.")

    G = derive(P, geometry)
    threshold_distance = G.escape_distance_m if escape_distance_m is None else float(escape_distance_m)
    if mode == "surface_distance" and threshold_distance <= 0:
        raise ValueError("escape_distance_m must be positive.")
    if mode == "z_plane" and (escape_z_m is None or not np.isfinite(escape_z_m)):
        raise ValueError("A finite escape_z_m is required for z_plane mode.")

    rng = np.random.default_rng(P.seed if seed is None else seed)
    releases, release_xyz, source_faces = _resolve_releases(release_locations, G)
    receptor_faces = _choose_receptor_faces(P, G, source_faces, receptor_face_ids, rng)
    receptor_release_xyz = G.geometry.surface_fluid_xyz[receptor_faces]

    # Map a reaction fluid voxel to the independent receptors available there.
    site_to_receptors: dict[int, np.ndarray] = {}
    for receptor_id, site in enumerate(_flat_site(receptor_release_xyz, G)):
        site_to_receptors.setdefault(int(site), []).append(receptor_id)
    site_to_receptors = {site: np.asarray(ids, dtype=np.int64) for site, ids in site_to_receptors.items()}
    receptor_table = pd.DataFrame({
        "receptor_id": np.arange(receptor_faces.size, dtype=np.int64),
        "face_id": receptor_faces,
        "x_m": G.geometry.surface_centers_m[receptor_faces, 0],
        "y_m": G.geometry.surface_centers_m[receptor_faces, 1],
        "z_m": G.geometry.surface_centers_m[receptor_faces, 2],
    })

    n_locations = len(release_locations)
    n_total = n_locations * n_trials_per_location
    location_ids = np.repeat(np.arange(n_locations), n_trials_per_location)
    trial_ids = np.tile(np.arange(n_trials_per_location), n_locations)
    xyz = release_xyz[location_ids].copy()
    source_face_per_molecule = source_faces[location_ids]
    bound_receptor = np.full(n_total, -1, dtype=np.int64)
    active = np.ones(n_total, dtype=bool)
    escaped = np.zeros(n_total, dtype=bool)
    escape_time = np.full(n_total, np.nan)
    escape_reason = np.full(n_total, "censored", dtype=object)
    n_bindings = np.zeros(n_total, dtype=np.int64)
    n_unbindings = np.zeros(n_total, dtype=np.int64)
    n_self = np.zeros(n_total, dtype=np.int64)
    n_cross = np.zeros(n_total, dtype=np.int64)
    total_bound_time = np.zeros(n_total, dtype=float)
    max_surface_distance = G.distance_to_reactive_surface_m[xyz[:, 0], xyz[:, 1], xyz[:, 2]].copy()

    if mode == "surface_distance":
        initially_escaped = max_surface_distance >= threshold_distance
    elif mode == "z_plane":
        initially_escaped = xyz[:, 2] * G.a_m >= float(escape_z_m)
    else:
        initially_escaped = np.zeros(n_total, dtype=bool)
    active[initially_escaped] = False
    escaped[initially_escaped] = True
    escape_time[initially_escaped] = 0.0
    escape_reason[initially_escaped] = mode

    # A partial lattice timestep does not have the same transition kernel.
    # Simulate complete timesteps only and right-censor at the requested time.
    n_steps = int(np.floor(max_time_s / G.dt_s))
    for step_index in range(1, n_steps + 1):
        event_time = step_index * G.dt_s
        bound_start = active & (bound_receptor >= 0)
        total_bound_time[bound_start] += G.dt_s

        free_ids = np.flatnonzero(active & (bound_receptor < 0))
        if free_ids.size:
            moves = rng.choice(7, size=free_ids.size, p=G.move_probs)
            proposed = xyz[free_ids] + MOVE_VECTORS[moves]

            domain_escape = np.zeros(free_ids.size, dtype=bool)
            domain_reason = np.full(free_ids.size, "", dtype=object)
            if G.use_well_mixed_reservoir and G.reservoir_explicit_max_z_index is not None:
                reservoir_loss = proposed[:, 2] > G.reservoir_explicit_max_z_index
                domain_escape[reservoir_loss] = True
                domain_reason[reservoir_loss] = "well_mixed_reservoir"

            outside_masks = {
                "x_min": proposed[:, 0] < 0,
                "x_max": proposed[:, 0] >= G.Nx,
                "y_min": proposed[:, 1] < 0,
                "y_max": proposed[:, 1] >= G.Ny,
                "z_min": proposed[:, 2] < 0,
                "z_max": proposed[:, 2] > G.Nz,
            }
            outside = np.logical_or.reduce(list(outside_masks.values()))
            for face, face_mask in outside_masks.items():
                loss = face_mask & (face in G.open_boundaries)
                domain_escape[loss] = True
                domain_reason[loss] = face

            in_bounds = ~outside
            valid_move = np.zeros(free_ids.size, dtype=bool)
            if np.any(in_bounds):
                bounded = proposed[in_bounds]
                valid_move[in_bounds] = G.accessible_fluid_mask[
                    bounded[:, 0], bounded[:, 1], bounded[:, 2]
                ]
            valid_move &= ~domain_escape
            xyz[free_ids[valid_move]] = proposed[valid_move]

            lost_ids = free_ids[domain_escape]
            if lost_ids.size:
                active[lost_ids] = False
                escaped[lost_ids] = True
                escape_time[lost_ids] = event_time
                escape_reason[lost_ids] = domain_reason[domain_escape]

            remaining = free_ids[~domain_escape]
            if remaining.size:
                remaining_xyz = xyz[remaining]
                distances = G.distance_to_reactive_surface_m[
                    remaining_xyz[:, 0], remaining_xyz[:, 1], remaining_xyz[:, 2]
                ]
                max_surface_distance[remaining] = np.maximum(
                    max_surface_distance[remaining], distances
                )
                if mode == "surface_distance":
                    local_escape = distances >= threshold_distance
                elif mode == "z_plane":
                    local_escape = remaining_xyz[:, 2] * G.a_m >= float(escape_z_m)
                else:
                    local_escape = np.zeros(remaining.size, dtype=bool)
                local_ids = remaining[local_escape]
                active[local_ids] = False
                escaped[local_ids] = True
                escape_time[local_ids] = event_time
                escape_reason[local_ids] = mode

        # Binding follows diffusion, matching biosensor_mc.step(). Each assay
        # trajectory has its own independent copy of every receptor.
        free_ids = np.flatnonzero(active & (bound_receptor < 0))
        if free_ids.size and G.kon_exp_per_receptor > 0 and receptor_faces.size:
            free_sites = _flat_site(xyz[free_ids], G)
            for site in np.unique(free_sites):
                receptors_here = site_to_receptors.get(int(site))
                if receptors_here is None:
                    continue
                candidates = free_ids[free_sites == site]
                p_bind = 1.0 - np.exp(-G.kon_exp_per_receptor * receptors_here.size)
                binding_ids = candidates[rng.random(candidates.size) < p_bind]
                if binding_ids.size == 0:
                    continue
                selected = receptors_here[rng.integers(receptors_here.size, size=binding_ids.size)]
                bound_receptor[binding_ids] = selected
                n_bindings[binding_ids] += 1
                has_source = source_face_per_molecule[binding_ids] >= 0
                is_self = has_source & (
                    receptor_faces[selected] == source_face_per_molecule[binding_ids]
                )
                is_cross = has_source & ~is_self
                n_self[binding_ids[is_self]] += 1
                n_cross[binding_ids[is_cross]] += 1
                selected_faces = receptor_faces[selected]
                xyz[binding_ids] = G.geometry.surface_solid_xyz[selected_faces]

        # Only molecules that were bound at the start of this timestep may
        # dissociate; newly bound molecules remain bound for at least one step.
        dissociation_ids = np.flatnonzero(bound_start & active)
        if dissociation_ids.size and G.p_off > 0:
            unbind = dissociation_ids[rng.random(dissociation_ids.size) < G.p_off]
            receptor_ids = bound_receptor[unbind]
            xyz[unbind] = receptor_release_xyz[receptor_ids]
            bound_receptor[unbind] = -1
            n_unbindings[unbind] += 1

        if not np.any(active):
            break

    observation_time = np.where(escaped, escape_time, max_time_s)
    trajectories = pd.DataFrame({
        "location_id": location_ids,
        "label": releases.loc[location_ids, "label"].to_numpy(),
        "trial_id": trial_ids,
        "escaped": escaped,
        "censored": ~escaped,
        "escape_time_s": escape_time,
        "observation_time_s": observation_time,
        "escape_reason": escape_reason,
        "n_bindings": n_bindings,
        "n_rebindings": n_bindings,
        "n_self_rebindings": n_self,
        "n_cross_rebindings": n_cross,
        "n_unbindings": n_unbindings,
        "total_bound_time_s": total_bound_time,
        "max_surface_distance_m": max_surface_distance,
        "final_x_m": xyz[:, 0] * G.a_m,
        "final_y_m": xyz[:, 1] * G.a_m,
        "final_z_m": xyz[:, 2] * G.a_m,
        "final_bound": active & (bound_receptor >= 0),
    })

    return EscapeAssayResult(
        params=P,
        derived=G,
        releases=releases,
        receptors=receptor_table,
        trajectories=trajectories,
        survival=_kaplan_meier(trajectories),
    )


__all__ = [
    "EscapeAssayResult",
    "ReleaseLocation",
    "run_escape_assay",
    "sample_receptor_faces",
    "surface_release_locations",
]
