"""Single-ligand first-capture assays on voxelized sensor geometries.

Each independent trajectory starts on a plane immediately below a bulk escape
plane. It ends at the first successful receptor association, on return to the
bulk, or at a safety censoring horizon. Dissociation and rebinding are never
simulated, keeping initial capture separate from post-capture retention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.biosensor_mc import (
    L_PER_M3,
    MOVE_VECTORS,
    NA,
    Derived,
    Params,
    SensorGeometry,
    _wrap_periodic_coordinates,
    derive,
)
from utils.escape_assay import sample_receptor_faces


@dataclass
class CaptureAssayResult:
    """Trajectory tables and model metadata from a first-capture assay."""

    params: Params
    derived: Derived
    receptors: pd.DataFrame
    trajectories: pd.DataFrame
    condition_summary: pd.DataFrame
    receptor_summary: pd.DataFrame


def _flat_site(xyz: np.ndarray, G: Derived) -> np.ndarray:
    xyz = np.asarray(xyz)
    return (
        (xyz[..., 0] * G.Ny + xyz[..., 1]) * (G.Nz + 1) + xyz[..., 2]
    ).astype(np.int64)


def _resolve_escape_plane(G: Derived, escape_z_m: Optional[float]) -> float:
    if escape_z_m is None:
        if G.use_well_mixed_reservoir and G.reservoir_interface_z_m is not None:
            value = float(G.reservoir_interface_z_m)
        else:
            raise ValueError(
                "escape_z_m is required unless a well-mixed reservoir interface "
                "is enabled."
            )
    else:
        value = float(escape_z_m)
    if not np.isfinite(value) or value <= 0 or value > G.Nz * G.a_m:
        raise ValueError("escape_z_m must lie inside the positive z domain.")
    return value


def _resolve_injection_layer(
    G: Derived,
    escape_z_m: float,
    injection_z_m: Optional[float],
) -> int:
    if injection_z_m is None:
        # Highest lattice plane strictly below the escape plane.
        layer = int(np.ceil(escape_z_m / G.a_m) - 1)
    else:
        value = float(injection_z_m)
        if not np.isfinite(value):
            raise ValueError("injection_z_m must be finite.")
        layer = int(round(value / G.a_m))
    if not (0 <= layer <= G.Nz):
        raise ValueError("The injection layer lies outside the lattice.")
    if layer * G.a_m >= escape_z_m:
        raise ValueError("The injection layer must be strictly below escape_z_m.")
    return layer


def _initial_positions(
    G: Derived,
    n_trajectories: int,
    injection_z_index: int,
    rng: np.random.Generator,
    initial_xy_m: Optional[Tuple[float, float]],
) -> np.ndarray:
    plane_accessible = G.accessible_fluid_mask[:, :, injection_z_index]
    if initial_xy_m is None:
        # Uniform x,y over the domain. Requiring a complete fluid plane avoids
        # silently biasing starts toward only the accessible portion.
        if not np.all(plane_accessible):
            blocked = int(plane_accessible.size - np.count_nonzero(plane_accessible))
            raise ValueError(
                f"The requested injection plane contains {blocked} non-fluid or "
                "inaccessible sites. Move the injection plane above the sensor "
                "or specify initial_xy_m explicitly."
            )
        xyz = np.empty((n_trajectories, 3), dtype=np.int32)
        xyz[:, 0] = rng.integers(0, G.Nx, size=n_trajectories)
        xyz[:, 1] = rng.integers(0, G.Ny, size=n_trajectories)
        xyz[:, 2] = injection_z_index
        return xyz

    if len(initial_xy_m) != 2 or not np.all(np.isfinite(initial_xy_m)):
        raise ValueError("initial_xy_m must contain two finite coordinates.")
    x_index = int(round(float(initial_xy_m[0]) / G.a_m))
    y_index = int(round(float(initial_xy_m[1]) / G.a_m))
    if "x" in G.periodic_axes:
        x_index %= G.Nx
    if "y" in G.periodic_axes:
        y_index %= G.Ny
    if not (0 <= x_index < G.Nx and 0 <= y_index < G.Ny):
        raise ValueError("initial_xy_m lies outside a nonperiodic lateral boundary.")
    if not plane_accessible[x_index, y_index]:
        raise ValueError("The requested initial coordinate is not accessible fluid.")
    return np.tile(
        np.array([x_index, y_index, injection_z_index], dtype=np.int32),
        (n_trajectories, 1),
    )


def summarize_capture_assay(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row condition summary."""

    captured = trajectories["captured"].to_numpy(bool)
    escaped = trajectories["escaped"].to_numpy(bool)
    capture_times = trajectories.loc[captured, "capture_time_s"]
    escape_times = trajectories.loc[escaped, "escape_time_s"]
    n_total = len(trajectories)
    capture_probability = float(captured.mean())
    capture_probability_se = float(
        np.sqrt(capture_probability * (1.0 - capture_probability) / n_total)
    )
    # Wilson interval remains well behaved when no or every trajectory captures.
    z = 1.959963984540054
    denominator = 1.0 + z**2 / n_total
    wilson_center = (
        capture_probability + z**2 / (2.0 * n_total)
    ) / denominator
    wilson_half_width = z * np.sqrt(
        capture_probability * (1.0 - capture_probability) / n_total
        + z**2 / (4.0 * n_total**2)
    ) / denominator
    return pd.DataFrame(
        [
            {
                "n_trajectories": int(n_total),
                "n_captured": int(captured.sum()),
                "n_escaped": int(escaped.sum()),
                "n_censored": int(trajectories["censored"].sum()),
                "capture_probability": capture_probability,
                "capture_probability_se": capture_probability_se,
                "capture_probability_wilson_95_low": wilson_center - wilson_half_width,
                "capture_probability_wilson_95_high": wilson_center + wilson_half_width,
                "escape_probability": float(escaped.mean()),
                "censoring_fraction": float(trajectories["censored"].mean()),
                "mean_capture_time_given_capture_s": (
                    float(capture_times.mean()) if len(capture_times) else np.nan
                ),
                "median_capture_time_given_capture_s": (
                    float(capture_times.median()) if len(capture_times) else np.nan
                ),
                "mean_escape_time_given_escape_s": (
                    float(escape_times.mean()) if len(escape_times) else np.nan
                ),
                "mean_termination_time_s": float(
                    trajectories["termination_time_s"].mean()
                ),
                "mean_receptor_contact_steps": float(
                    trajectories["n_receptor_contact_steps"].mean()
                ),
            }
        ]
    )


def summarize_receptor_captures(
    trajectories: pd.DataFrame,
    receptors: pd.DataFrame,
) -> pd.DataFrame:
    """Return capture counts and probabilities for every receptor, including zeroes."""

    captured = trajectories[trajectories["captured"]]
    grouped = (
        captured.groupby("capturing_receptor_id", sort=False)
        .agg(
            n_captures=("trajectory_id", "size"),
            mean_capture_time_s=("capture_time_s", "mean"),
            median_capture_time_s=("capture_time_s", "median"),
            mean_contact_steps_before_capture=("n_receptor_contact_steps", "mean"),
        )
        .reset_index()
        .rename(columns={"capturing_receptor_id": "receptor_id"})
    )
    summary = receptors.rename(
        columns={
            "face_id": "source_face_id",
            "x_m": "surface_x_m",
            "y_m": "surface_y_m",
            "z_m": "surface_z_m",
        }
    ).merge(grouped, on="receptor_id", how="left", validate="one_to_one")
    summary["n_captures"] = summary["n_captures"].fillna(0).astype(np.int64)
    # Compatibility with the generic receptor-to-surface interpolation used
    # by the escape-assay visualization helpers.
    summary["location_id"] = summary["receptor_id"]
    summary["label"] = summary["receptor_id"].map(lambda value: f"receptor_{value}")
    n_total = len(trajectories)
    n_captured = int(captured.shape[0])
    summary["capture_probability_all_trajectories"] = (
        summary["n_captures"] / n_total if n_total else np.nan
    )
    summary["fraction_of_captures"] = (
        summary["n_captures"] / n_captured if n_captured else 0.0
    )
    return summary


def run_capture_assay(
    P: Params,
    geometry: SensorGeometry,
    n_trajectories: int = 10_000,
    max_time_s: float = 0.1,
    *,
    escape_z_m: Optional[float] = None,
    injection_z_m: Optional[float] = None,
    initial_xy_m: Optional[Tuple[float, float]] = None,
    receptor_face_ids: Optional[Sequence[int]] = None,
    receptor_seed: Optional[int] = None,
    trajectory_seed: Optional[int] = None,
) -> CaptureAssayResult:
    """Run independent trajectories until first binding or return to bulk.

    Random starts are uniform over all x-y lattice sites on the highest plane
    strictly below ``escape_z_m``. A fixed ``initial_xy_m`` can instead be used
    for every trajectory. Binding is attempted after diffusion exactly as in
    :mod:`utils.biosensor_mc`; a successful attempt terminates the trajectory
    immediately, so dissociation and rebinding cannot affect the result.
    """

    n_trajectories = int(n_trajectories)
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be at least 1.")
    max_time_s = float(max_time_s)
    if not np.isfinite(max_time_s) or max_time_s <= 0:
        raise ValueError("max_time_s must be finite and positive.")

    G = derive(P, geometry)
    escape_z_m = _resolve_escape_plane(G, escape_z_m)
    injection_z_index = _resolve_injection_layer(G, escape_z_m, injection_z_m)
    receptor_seed = P.seed if receptor_seed is None else int(receptor_seed)
    trajectory_seed = P.seed if trajectory_seed is None else int(trajectory_seed)
    receptor_faces = (
        sample_receptor_faces(P, geometry, seed=receptor_seed)
        if receptor_face_ids is None
        else np.asarray(receptor_face_ids, dtype=np.int64)
    )
    if receptor_faces.ndim != 1:
        raise ValueError("receptor_face_ids must be one-dimensional.")
    if receptor_faces.size and (
        np.min(receptor_faces) < 0 or np.max(receptor_faces) >= geometry.n_surface_faces
    ):
        raise ValueError("receptor_face_ids contains an invalid surface face.")
    active_faces = set(map(int, G.reactive_face_ids))
    invalid_faces = [int(face) for face in receptor_faces if int(face) not in active_faces]
    if invalid_faces:
        raise ValueError(f"Receptors occupy inactive faces: {invalid_faces[:10]}")

    receptor_release_xyz = geometry.surface_fluid_xyz[receptor_faces]
    site_to_receptors: dict[int, np.ndarray] = {}
    for receptor_id, site in enumerate(_flat_site(receptor_release_xyz, G)):
        site_to_receptors.setdefault(int(site), []).append(receptor_id)
    site_to_receptors = {
        site: np.asarray(ids, dtype=np.int64) for site, ids in site_to_receptors.items()
    }
    receptors = pd.DataFrame(
        {
            "receptor_id": np.arange(receptor_faces.size, dtype=np.int64),
            "face_id": receptor_faces,
            "x_m": geometry.surface_centers_m[receptor_faces, 0],
            "y_m": geometry.surface_centers_m[receptor_faces, 1],
            "z_m": geometry.surface_centers_m[receptor_faces, 2],
            "release_x_m": receptor_release_xyz[:, 0] * G.a_m,
            "release_y_m": receptor_release_xyz[:, 1] * G.a_m,
            "release_z_m": receptor_release_xyz[:, 2] * G.a_m,
        }
    )

    rng = np.random.default_rng(trajectory_seed)
    xyz = _initial_positions(
        G, n_trajectories, injection_z_index, rng, initial_xy_m
    )
    initial_xyz = xyz.copy()
    active = np.ones(n_trajectories, dtype=bool)
    captured = np.zeros(n_trajectories, dtype=bool)
    escaped = np.zeros(n_trajectories, dtype=bool)
    capture_time = np.full(n_trajectories, np.nan)
    escape_time = np.full(n_trajectories, np.nan)
    capturing_receptor = np.full(n_trajectories, -1, dtype=np.int64)
    contact_steps = np.zeros(n_trajectories, dtype=np.int64)
    binding_attempts = np.zeros(n_trajectories, dtype=np.int64)

    n_steps = int(np.floor(max_time_s / G.dt_s))
    for step_index in range(1, n_steps + 1):
        if not np.any(active):
            break
        event_time = step_index * G.dt_s
        active_ids = np.flatnonzero(active)
        moves = rng.choice(7, size=active_ids.size, p=G.move_probs)
        proposed = _wrap_periodic_coordinates(
            xyz[active_ids] + MOVE_VECTORS[moves], G
        )

        domain_escape = proposed[:, 2] * G.a_m >= escape_z_m
        if G.use_well_mixed_reservoir and G.reservoir_explicit_max_z_index is not None:
            domain_escape |= proposed[:, 2] > G.reservoir_explicit_max_z_index
        for face, face_mask in {
            "x_min": proposed[:, 0] < 0,
            "x_max": proposed[:, 0] >= G.Nx,
            "y_min": proposed[:, 1] < 0,
            "y_max": proposed[:, 1] >= G.Ny,
            "z_min": proposed[:, 2] < 0,
            "z_max": proposed[:, 2] > G.Nz,
        }.items():
            if face in G.open_boundaries:
                domain_escape |= face_mask

        escaped_ids = active_ids[domain_escape]
        if escaped_ids.size:
            active[escaped_ids] = False
            escaped[escaped_ids] = True
            escape_time[escaped_ids] = event_time

        remaining_mask = ~domain_escape
        remaining_ids = active_ids[remaining_mask]
        remaining_proposed = proposed[remaining_mask]
        if remaining_ids.size:
            in_bounds = (
                (remaining_proposed[:, 0] >= 0)
                & (remaining_proposed[:, 0] < G.Nx)
                & (remaining_proposed[:, 1] >= 0)
                & (remaining_proposed[:, 1] < G.Ny)
                & (remaining_proposed[:, 2] >= 0)
                & (remaining_proposed[:, 2] <= G.Nz)
            )
            valid = np.zeros(remaining_ids.size, dtype=bool)
            if np.any(in_bounds):
                bounded = remaining_proposed[in_bounds]
                valid[in_bounds] = G.accessible_fluid_mask[
                    bounded[:, 0], bounded[:, 1], bounded[:, 2]
                ]
            xyz[remaining_ids[valid]] = remaining_proposed[valid]

        # First successful association is an absorbing capture event.
        candidates = np.flatnonzero(active)
        if candidates.size and receptor_faces.size and G.kon_exp_per_receptor > 0:
            candidate_sites = _flat_site(xyz[candidates], G)
            for site in np.unique(candidate_sites):
                receptors_here = site_to_receptors.get(int(site))
                if receptors_here is None:
                    continue
                at_site = candidates[candidate_sites == site]
                contact_steps[at_site] += 1
                binding_attempts[at_site] += 1
                p_bind = 1.0 - np.exp(
                    -G.kon_exp_per_receptor * receptors_here.size
                )
                captured_here = at_site[rng.random(at_site.size) < p_bind]
                if captured_here.size == 0:
                    continue
                selected = receptors_here[
                    rng.integers(receptors_here.size, size=captured_here.size)
                ]
                captured[captured_here] = True
                active[captured_here] = False
                capture_time[captured_here] = event_time
                capturing_receptor[captured_here] = selected
                xyz[captured_here] = geometry.surface_solid_xyz[
                    receptor_faces[selected]
                ]

    censored = active
    termination_time = np.where(
        captured, capture_time, np.where(escaped, escape_time, max_time_s)
    )
    capture_face = np.full(n_trajectories, -1, dtype=np.int64)
    capture_xyz_m = np.full((n_trajectories, 3), np.nan)
    captured_ids = np.flatnonzero(captured)
    if captured_ids.size:
        capture_face[captured_ids] = receptor_faces[capturing_receptor[captured_ids]]
        capture_xyz_m[captured_ids] = geometry.surface_centers_m[
            capture_face[captured_ids]
        ]

    trajectories = pd.DataFrame(
        {
            "trajectory_id": np.arange(n_trajectories, dtype=np.int64),
            "captured": captured,
            "escaped": escaped,
            "censored": censored,
            "termination_reason": np.where(
                captured, "captured", np.where(escaped, "bulk_escape", "censored")
            ),
            "termination_time_s": termination_time,
            "capture_time_s": capture_time,
            "escape_time_s": escape_time,
            "initial_x_index": initial_xyz[:, 0],
            "initial_y_index": initial_xyz[:, 1],
            "initial_z_index": initial_xyz[:, 2],
            "initial_x_m": initial_xyz[:, 0] * G.a_m,
            "initial_y_m": initial_xyz[:, 1] * G.a_m,
            "initial_z_m": initial_xyz[:, 2] * G.a_m,
            "final_x_m": xyz[:, 0] * G.a_m,
            "final_y_m": xyz[:, 1] * G.a_m,
            "final_z_m": xyz[:, 2] * G.a_m,
            "capturing_receptor_id": capturing_receptor,
            "capture_face_id": capture_face,
            "capture_x_m": capture_xyz_m[:, 0],
            "capture_y_m": capture_xyz_m[:, 1],
            "capture_z_m": capture_xyz_m[:, 2],
            "n_receptor_contact_steps": contact_steps,
            "n_binding_attempts": binding_attempts,
        }
    )
    condition_summary = summarize_capture_assay(trajectories)
    condition_summary["escape_z_m"] = escape_z_m
    condition_summary["injection_z_m"] = injection_z_index * G.a_m
    condition_summary["n_receptors"] = receptor_faces.size
    injection_plane_area_m2 = G.Nx * G.Ny * G.a_m**2
    delivery_rate_coefficient = (
        NA * L_PER_M3 * G.D_m2_s * injection_plane_area_m2 / G.a_m
    )
    condition_summary["injection_plane_area_m2"] = injection_plane_area_m2
    condition_summary["delivery_rate_coefficient_M_inv_s"] = delivery_rate_coefficient
    condition_summary["capture_rate_coefficient_M_inv_s"] = (
        delivery_rate_coefficient * condition_summary["capture_probability"]
    )
    receptor_summary = summarize_receptor_captures(trajectories, receptors)
    return CaptureAssayResult(
        params=P,
        derived=G,
        receptors=receptors,
        trajectories=trajectories,
        condition_summary=condition_summary,
        receptor_summary=receptor_summary,
    )


__all__ = [
    "CaptureAssayResult",
    "run_capture_assay",
    "summarize_capture_assay",
    "summarize_receptor_captures",
]
