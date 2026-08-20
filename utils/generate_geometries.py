# generate_geometries.py

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover - used only when scipy is unavailable
    distance_transform_edt = None

from utils.biosensor_mc import *

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
# Grid and geometry construction
# -----------------------------------------------------------------------------


def _grid_counts(P: Params) -> Tuple[int, int, int]:
    if P.a_m <= 0:
        raise ValueError("a_m must be positive.")

    Nx = int(round(P.Lx_m / P.a_m))
    Ny = int(round(P.Ly_m / P.a_m))
    Nz = int(round(P.H_m / P.a_m))

    if min(Nx, Ny, Nz) < 1:
        raise ValueError(
            "Lx_m, Ly_m, and H_m must each be at least one lattice spacing."
        )

    return Nx, Ny, Nz


def _axis_pair_slices(
    shape: Tuple[int, int, int],
    direction: Tuple[int, int, int],
) -> Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice], np.ndarray]:
    src: List[slice] = []
    dst: List[slice] = []
    src_offset = np.zeros(3, dtype=np.int32)

    for axis, delta in enumerate(direction):
        n = shape[axis]

        if delta == 1:
            src.append(slice(0, n - 1))
            dst.append(slice(1, n))
        elif delta == -1:
            src.append(slice(1, n))
            dst.append(slice(0, n - 1))
            src_offset[axis] = 1
        else:
            src.append(slice(0, n))
            dst.append(slice(0, n))

    return tuple(src), tuple(dst), src_offset


# -----------------------------------------------------------------------------
# Geometry-derived masks and boundary sites
# -----------------------------------------------------------------------------


def _validate_open_boundaries(open_boundaries: Sequence[str]) -> Tuple[str, ...]:
    result = tuple(dict.fromkeys(open_boundaries))
    invalid = sorted(set(result) - set(BOUNDARY_FACES))

    if invalid:
        raise ValueError(
            f"Unknown open boundary name(s): {invalid}. "
            f"Allowed names are {BOUNDARY_FACES}."
        )

    return result


def _boundary_sites(mask: np.ndarray, face: str) -> np.ndarray:
    if face == "x_min":
        local = np.argwhere(mask[0, :, :])
        return np.column_stack(
            [np.zeros(local.shape[0], dtype=np.int32), local]
        ).astype(np.int32)

    if face == "x_max":
        local = np.argwhere(mask[-1, :, :])
        return np.column_stack(
            [np.full(local.shape[0], mask.shape[0] - 1, dtype=np.int32), local]
        ).astype(np.int32)

    if face == "y_min":
        local = np.argwhere(mask[:, 0, :])
        return np.column_stack(
            [local[:, 0], np.zeros(local.shape[0], dtype=np.int32), local[:, 1]]
        ).astype(np.int32)

    if face == "y_max":
        local = np.argwhere(mask[:, -1, :])
        return np.column_stack(
            [
                local[:, 0],
                np.full(local.shape[0], mask.shape[1] - 1, dtype=np.int32),
                local[:, 1],
            ]
        ).astype(np.int32)

    if face == "z_min":
        local = np.argwhere(mask[:, :, 0])
        return np.column_stack(
            [local, np.zeros(local.shape[0], dtype=np.int32)]
        ).astype(np.int32)

    if face == "z_max":
        local = np.argwhere(mask[:, :, -1])
        return np.column_stack(
            [local, np.full(local.shape[0], mask.shape[2] - 1, dtype=np.int32)]
        ).astype(np.int32)

    raise ValueError(f"Unknown boundary face: {face}")


def _bulk_accessible_fluid_mask(
    fluid_mask: np.ndarray,
    open_boundaries: Sequence[str],
) -> np.ndarray:
    """Return fluid sites connected to at least one open outer boundary."""

    if len(open_boundaries) == 0:
        return fluid_mask.copy()

    seed_mask = np.zeros_like(fluid_mask, dtype=bool)

    for face in open_boundaries:
        sites = _boundary_sites(fluid_mask, face)

        if sites.size:
            seed_mask[sites[:, 0], sites[:, 1], sites[:, 2]] = True

    if not np.any(seed_mask):
        raise ValueError(
            "No fluid sites touch an open boundary. Either change the geometry "
            "or choose different open_boundaries."
        )

    accessible = np.zeros_like(fluid_mask, dtype=bool)
    queue: deque[Tuple[int, int, int]] = deque(
        map(tuple, np.argwhere(seed_mask).tolist())
    )
    accessible[seed_mask] = True
    nx, ny, nz = fluid_mask.shape

    while queue:
        x, y, z = queue.popleft()

        for dx, dy, dz in FACE_DIRECTIONS:
            xn = x + dx
            yn = y + dy
            zn = z + dz

            if not (0 <= xn < nx and 0 <= yn < ny and 0 <= zn < nz):
                continue

            if accessible[xn, yn, zn] or not fluid_mask[xn, yn, zn]:
                continue

            accessible[xn, yn, zn] = True
            queue.append((xn, yn, zn))

    return accessible


def _distance_to_surface_fallback(
    seed_mask: np.ndarray,
    a_m: float,
) -> np.ndarray:
    """Six-neighbor graph-distance fallback used when SciPy is unavailable."""

    warnings.warn(
        "SciPy is unavailable; using a Manhattan-distance approximation for "
        "rebinding escape distance.",
        RuntimeWarning,
    )

    distance_steps = np.full(seed_mask.shape, np.inf)
    queue: deque[Tuple[int, int, int]] = deque()

    for xyz in np.argwhere(seed_mask):
        xyz_tuple = tuple(map(int, xyz))
        distance_steps[xyz_tuple] = 0.0
        queue.append(xyz_tuple)

    nx, ny, nz = seed_mask.shape

    while queue:
        x, y, z = queue.popleft()
        next_distance = distance_steps[x, y, z] + 1.0

        for dx, dy, dz in FACE_DIRECTIONS:
            xn = x + dx
            yn = y + dy
            zn = z + dz

            if not (0 <= xn < nx and 0 <= yn < ny and 0 <= zn < nz):
                continue

            if next_distance >= distance_steps[xn, yn, zn]:
                continue

            distance_steps[xn, yn, zn] = next_distance
            queue.append((xn, yn, zn))

    return distance_steps * a_m


def _distance_to_reactive_surface(
    shape: Tuple[int, int, int],
    reactive_surface_solid_xyz: np.ndarray,
    a_m: float,
) -> np.ndarray:
    seed_mask = np.zeros(shape, dtype=bool)
    seed_mask[
        reactive_surface_solid_xyz[:, 0],
        reactive_surface_solid_xyz[:, 1],
        reactive_surface_solid_xyz[:, 2],
    ] = True

    if distance_transform_edt is not None:
        return distance_transform_edt(~seed_mask, sampling=a_m)

    return _distance_to_surface_fallback(seed_mask, a_m)


def geometry_from_solid_mask(
    P: Params,
    solid_mask: np.ndarray,
    name: str = "custom",
    reactive_solid_mask: Optional[np.ndarray] = None,
) -> SensorGeometry:
    """
    Construct a SensorGeometry from a Boolean solid-site mask.

    Parameters
    ----------
    P
        Simulation parameters defining the lattice dimensions and spacing.
    solid_mask
        Boolean array with shape (Nx, Ny, Nz + 1). True sites are sensor
        material and cannot be occupied by free ligands.
    name
        Descriptive geometry name.
    reactive_solid_mask
        Optional Boolean array with the same shape. Only exposed faces whose
        solid lattice site is True in this mask are chemically reactive.
        If omitted, every exposed sensor face is reactive.
    """

    Nx, Ny, Nz = _grid_counts(P)
    expected_shape = (Nx, Ny, Nz + 1)

    solid_mask = np.asarray(solid_mask, dtype=bool)

    if solid_mask.shape != expected_shape:
        raise ValueError(
            f"solid_mask must have shape {expected_shape}; got {solid_mask.shape}."
        )

    if not np.any(solid_mask):
        raise ValueError("solid_mask contains no sensor material.")

    if np.all(solid_mask):
        raise ValueError("solid_mask contains no fluid sites.")

    if reactive_solid_mask is None:
        reactive_solid_mask = solid_mask
    else:
        reactive_solid_mask = np.asarray(reactive_solid_mask, dtype=bool)

        if reactive_solid_mask.shape != expected_shape:
            raise ValueError(
                "reactive_solid_mask must have the same shape as solid_mask."
            )

        if np.any(reactive_solid_mask & ~solid_mask):
            raise ValueError(
                "reactive_solid_mask may only mark sites that are also solid."
            )

    solid_chunks: List[np.ndarray] = []
    fluid_chunks: List[np.ndarray] = []
    normal_chunks: List[np.ndarray] = []
    reactive_chunks: List[np.ndarray] = []

    for direction in FACE_DIRECTIONS:
        src_slice, dst_slice, src_offset = _axis_pair_slices(
            expected_shape,
            direction,
        )

        interface = solid_mask[src_slice] & ~solid_mask[dst_slice]
        local_solid_xyz = np.argwhere(interface).astype(np.int32)

        if local_solid_xyz.size == 0:
            continue

        solid_xyz = local_solid_xyz + src_offset
        direction_arr = np.asarray(direction, dtype=np.int32)
        fluid_xyz = solid_xyz + direction_arr

        reactive = reactive_solid_mask[
            solid_xyz[:, 0],
            solid_xyz[:, 1],
            solid_xyz[:, 2],
        ]

        solid_chunks.append(solid_xyz)
        fluid_chunks.append(fluid_xyz)
        normal_chunks.append(
            np.repeat(direction_arr[None, :], solid_xyz.shape[0], axis=0)
        )
        reactive_chunks.append(reactive.astype(bool))

    if not solid_chunks:
        raise ValueError("No internal solid-fluid interfaces were found.")

    surface_solid_xyz = np.vstack(solid_chunks).astype(np.int32)
    surface_fluid_xyz = np.vstack(fluid_chunks).astype(np.int32)
    surface_normals = np.vstack(normal_chunks).astype(np.int8)
    reactive_face_mask = np.concatenate(reactive_chunks).astype(bool)

    surface_centers_m = (
        0.5 * (surface_solid_xyz + surface_fluid_xyz) * P.a_m
    )
    surface_area_m2 = np.full(surface_solid_xyz.shape[0], P.a_m**2)

    return SensorGeometry(
        name=str(name),
        solid_mask=solid_mask.copy(),
        surface_solid_xyz=surface_solid_xyz,
        surface_fluid_xyz=surface_fluid_xyz,
        surface_normals=surface_normals,
        surface_centers_m=surface_centers_m,
        surface_area_m2=surface_area_m2,
        reactive_face_mask=reactive_face_mask,
    )


def make_height_field_geometry(
    P: Params,
    height_m: Union[np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    name: str = "height_field",
    reactive_solid_mask: Optional[np.ndarray] = None,
) -> SensorGeometry:
    """
    Construct a sensor whose upper surface is z = height_m(x, y).

    Every lattice site satisfying z <= height_m(x, y) is solid. The x, y,
    and z coordinates supplied to a callable are in meters.
    """

    Nx, Ny, Nz = _grid_counts(P)
    x_m = np.arange(Nx, dtype=float) * P.a_m
    y_m = np.arange(Ny, dtype=float) * P.a_m
    X_m, Y_m = np.meshgrid(x_m, y_m, indexing="ij")

    if callable(height_m):
        height_array_m = np.asarray(height_m(X_m, Y_m), dtype=float)
    else:
        height_array_m = np.asarray(height_m, dtype=float)

    if height_array_m.shape == ():
        height_array_m = np.full((Nx, Ny), float(height_array_m))

    if height_array_m.shape != (Nx, Ny):
        raise ValueError(
            f"height_m must evaluate to shape {(Nx, Ny)}; "
            f"got {height_array_m.shape}."
        )

    if not np.all(np.isfinite(height_array_m)):
        raise ValueError("height_m contains non-finite values.")

    z_m = np.arange(Nz + 1, dtype=float) * P.a_m
    tolerance = 1e-12 * P.a_m
    solid_mask = z_m[None, None, :] <= height_array_m[:, :, None] + tolerance

    return geometry_from_solid_mask(
        P,
        solid_mask,
        name=name,
        reactive_solid_mask=reactive_solid_mask,
    )


def make_implicit_geometry(
    P: Params,
    phi: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    name: str = "implicit",
    solid_when_leq_zero: bool = True,
    reactive_solid_mask: Optional[np.ndarray] = None,
) -> SensorGeometry:
    """
    Construct a geometry from an implicit function phi(x, y, z).

    By default, phi <= 0 is interpreted as solid sensor material.
    Coordinates supplied to phi are in meters.
    """

    Nx, Ny, Nz = _grid_counts(P)
    X_m = np.arange(Nx, dtype=float)[:, None, None] * P.a_m
    Y_m = np.arange(Ny, dtype=float)[None, :, None] * P.a_m
    Z_m = np.arange(Nz + 1, dtype=float)[None, None, :] * P.a_m

    values = np.asarray(phi(X_m, Y_m, Z_m))
    expected_shape = (Nx, Ny, Nz + 1)

    try:
        values = np.broadcast_to(values, expected_shape)
    except ValueError as exc:
        raise ValueError(
            f"phi must broadcast to shape {expected_shape}; got {values.shape}."
        ) from exc

    solid_mask = values <= 0 if solid_when_leq_zero else values >= 0

    return geometry_from_solid_mask(
        P,
        solid_mask,
        name=name,
        reactive_solid_mask=reactive_solid_mask,
    )

def make_flat_geometry(
    P: Params,
    surface_z_m: float = 0.0,
    name: str = "flat",
) -> SensorGeometry:
    """Construct the original planar sensor geometry."""

    return make_height_field_geometry(P, float(surface_z_m), name=name)


def _default_center_xy(P: Params) -> Tuple[float, float]:
    Nx, Ny, _ = _grid_counts(P)
    return 0.5 * (Nx - 1) * P.a_m, 0.5 * (Ny - 1) * P.a_m


def make_spherical_cap_geometry(
    P: Params,
    radius_m: float,
    cap_height_m: float,
    center_xy_m: Optional[Tuple[float, float]] = None,
    base_z_m: float = 0.0,
    name: str = "spherical_cap",
) -> SensorGeometry:
    """Construct a convex spherical cap rising above a planar base."""

    if radius_m <= 0:
        raise ValueError("radius_m must be positive.")

    if cap_height_m <= 0 or cap_height_m > radius_m:
        raise ValueError("cap_height_m must satisfy 0 < cap_height_m <= radius_m.")

    if center_xy_m is None:
        center_xy_m = _default_center_xy(P)

    xc, yc = map(float, center_xy_m)
    footprint_r2 = 2 * radius_m * cap_height_m - cap_height_m**2
    zc = base_z_m + cap_height_m - radius_m

    def height(X_m: np.ndarray, Y_m: np.ndarray) -> np.ndarray:
        r2 = (X_m - xc) ** 2 + (Y_m - yc) ** 2
        h = np.full_like(r2, base_z_m, dtype=float)
        inside = r2 <= footprint_r2
        h[inside] = zc + np.sqrt(np.maximum(radius_m**2 - r2[inside], 0.0))
        return h

    return make_height_field_geometry(P, height, name=name)


def make_spherical_bowl_geometry(
    P: Params,
    radius_m: float,
    depth_m: float,
    center_xy_m: Optional[Tuple[float, float]] = None,
    rim_z_m: Optional[float] = None,
    name: str = "spherical_bowl",
) -> SensorGeometry:
    """Construct a concave spherical bowl recessed below a planar rim."""

    if radius_m <= 0:
        raise ValueError("radius_m must be positive.")

    if depth_m <= 0 or depth_m > radius_m:
        raise ValueError("depth_m must satisfy 0 < depth_m <= radius_m.")

    if center_xy_m is None:
        center_xy_m = _default_center_xy(P)

    if rim_z_m is None:
        rim_z_m = depth_m

    xc, yc = map(float, center_xy_m)
    footprint_r2 = 2 * radius_m * depth_m - depth_m**2
    zc = rim_z_m + radius_m - depth_m

    def height(X_m: np.ndarray, Y_m: np.ndarray) -> np.ndarray:
        r2 = (X_m - xc) ** 2 + (Y_m - yc) ** 2
        h = np.full_like(r2, rim_z_m, dtype=float)
        inside = r2 <= footprint_r2
        h[inside] = zc - np.sqrt(np.maximum(radius_m**2 - r2[inside], 0.0))
        return h

    return make_height_field_geometry(P, height, name=name)


def make_cylindrical_post_geometry(
    P: Params,
    radius_m: float,
    height_m: float,
    center_xy_m: Optional[Tuple[float, float]] = None,
    base_z_m: float = 0.0,
    name: str = "cylindrical_post",
) -> SensorGeometry:
    """Construct a cylindrical post on a planar base."""

    if radius_m <= 0 or height_m <= 0:
        raise ValueError("radius_m and height_m must be positive.")

    if center_xy_m is None:
        center_xy_m = _default_center_xy(P)

    xc, yc = map(float, center_xy_m)

    def height(X_m: np.ndarray, Y_m: np.ndarray) -> np.ndarray:
        r2 = (X_m - xc) ** 2 + (Y_m - yc) ** 2
        return np.where(r2 <= radius_m**2, base_z_m + height_m, base_z_m)

    return make_height_field_geometry(P, height, name=name)


def make_cylindrical_well_geometry(
    P: Params,
    radius_m: float,
    depth_m: float,
    center_xy_m: Optional[Tuple[float, float]] = None,
    rim_z_m: Optional[float] = None,
    name: str = "cylindrical_well",
) -> SensorGeometry:
    """Construct a cylindrical well recessed below a planar rim."""

    if radius_m <= 0 or depth_m <= 0:
        raise ValueError("radius_m and depth_m must be positive.")

    if center_xy_m is None:
        center_xy_m = _default_center_xy(P)

    if rim_z_m is None:
        rim_z_m = depth_m

    xc, yc = map(float, center_xy_m)
    bottom_z_m = rim_z_m - depth_m

    def height(X_m: np.ndarray, Y_m: np.ndarray) -> np.ndarray:
        r2 = (X_m - xc) ** 2 + (Y_m - yc) ** 2
        return np.where(r2 <= radius_m**2, bottom_z_m, rim_z_m)

    return make_height_field_geometry(P, height, name=name)


def _sample_positive_normal(rng, mean, sd, minimum, maximum=None, max_attempts=100):
    """Sample a bounded approximately normal scalar."""
    mean = float(mean)
    sd = float(sd)
    minimum = float(minimum)
    maximum = None if maximum is None else float(maximum)
    if maximum is not None and maximum < minimum:
        raise ValueError("maximum must be >= minimum")
    if sd <= 0:
        value = mean
    else:
        for _ in range(int(max_attempts)):
            value = float(rng.normal(mean, sd))
            if value >= minimum and (maximum is None or value <= maximum):
                return value
        value = mean
    value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return float(value)


def _nanospike_primary_direction(rng, tilt_mean_deg, tilt_sd_deg):
    tilt_deg = _sample_positive_normal(
        rng, tilt_mean_deg, tilt_sd_deg, minimum=0.0, maximum=85.0
    )
    tilt = np.deg2rad(tilt_deg)
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([
        np.sin(tilt) * np.cos(azimuth),
        np.sin(tilt) * np.sin(azimuth),
        np.cos(tilt),
    ], dtype=float)


def _nanospike_perpendicular_basis(direction):
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(direction[2]) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    e1 = np.cross(direction, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _nanospike_branch_direction(
    rng,
    parent_direction,
    angle_mean_deg,
    angle_sd_deg,
    minimum_upward_z=0.05,
    max_attempts=50,
):
    parent_direction = np.asarray(parent_direction, dtype=float)
    parent_direction /= np.linalg.norm(parent_direction)
    e1, e2 = _nanospike_perpendicular_basis(parent_direction)
    direction = parent_direction.copy()
    for _ in range(int(max_attempts)):
        angle_deg = _sample_positive_normal(
            rng, angle_mean_deg, angle_sd_deg, minimum=5.0, maximum=85.0
        )
        angle = np.deg2rad(angle_deg)
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        transverse = np.cos(azimuth) * e1 + np.sin(azimuth) * e2
        direction = (
            np.cos(angle) * parent_direction
            + np.sin(angle) * transverse
        )
        direction /= np.linalg.norm(direction)
        if direction[2] >= minimum_upward_z:
            return direction
    direction[2] = max(direction[2], minimum_upward_z)
    direction /= np.linalg.norm(direction)
    return direction


def _nanospike_segment_fits_domain(start_m, end_m, radius_m, P):
    start_m = np.asarray(start_m, dtype=float)
    end_m = np.asarray(end_m, dtype=float)
    radius_m = float(radius_m)
    Nx, Ny, _ = _grid_counts(P)
    x_max_m = (Nx - 1) * P.a_m
    y_max_m = (Ny - 1) * P.a_m
    return (
        min(start_m[0], end_m[0]) - radius_m >= 0.0
        and max(start_m[0], end_m[0]) + radius_m <= x_max_m
        and min(start_m[1], end_m[1]) - radius_m >= 0.0
        and max(start_m[1], end_m[1]) + radius_m <= y_max_m
        and max(start_m[2], end_m[2]) + radius_m <= P.H_m
    )


def _voxelize_tapered_nanospike_segment(
    solid_mask,
    P,
    start_m,
    end_m,
    radius_start_m,
    radius_end_m,
):
    """Union a linearly tapered cylindrical segment into a Boolean solid mask."""
    start_m = np.asarray(start_m, dtype=float)
    end_m = np.asarray(end_m, dtype=float)
    axis = end_m - start_m
    axis2 = float(np.dot(axis, axis))
    if axis2 <= 0:
        return

    padding = max(float(radius_start_m), float(radius_end_m)) + P.a_m
    mins = np.floor((np.minimum(start_m, end_m) - padding) / P.a_m).astype(int)
    maxs = np.ceil((np.maximum(start_m, end_m) + padding) / P.a_m).astype(int)
    mins = np.maximum(mins, 0)
    maxs = np.minimum(maxs, np.asarray(solid_mask.shape) - 1)
    if np.any(maxs < mins):
        return

    ix = np.arange(mins[0], maxs[0] + 1)
    iy = np.arange(mins[1], maxs[1] + 1)
    iz = np.arange(mins[2], maxs[2] + 1)
    X, Y, Z = np.meshgrid(ix, iy, iz, indexing='ij')
    points = np.stack([X, Y, Z], axis=-1).astype(float) * P.a_m

    relative = points - start_m
    fraction = np.sum(relative * axis, axis=-1) / axis2
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start_m + fraction[..., None] * axis
    distance2 = np.sum((points - closest) ** 2, axis=-1)
    local_radius = (
        float(radius_start_m)
        + fraction * (float(radius_end_m) - float(radius_start_m))
    )
    inside = distance2 <= local_radius**2

    view = solid_mask[
        mins[0]:maxs[0] + 1,
        mins[1]:maxs[1] + 1,
        mins[2]:maxs[2] + 1,
    ]
    view |= inside


def _sample_nanospike_primary_centers(
    rng,
    n_primary,
    P,
    edge_margin_m,
    min_spacing_m,
    max_attempts_per_spike=1000,
):
    Nx, Ny, _ = _grid_counts(P)
    x_max_m = (Nx - 1) * P.a_m
    y_max_m = (Ny - 1) * P.a_m
    x0, x1 = float(edge_margin_m), x_max_m - float(edge_margin_m)
    y0, y1 = float(edge_margin_m), y_max_m - float(edge_margin_m)
    if x0 > x1 or y0 > y1:
        raise ValueError('edge_margin_m leaves no room for nanospike roots.')

    centers = []
    min_d2 = float(min_spacing_m) ** 2
    for _ in range(int(n_primary)):
        accepted = False
        for _attempt in range(int(max_attempts_per_spike)):
            candidate = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            if centers and min_spacing_m > 0:
                existing = np.asarray(centers)
                if np.any(np.sum((existing - candidate) ** 2, axis=1) < min_d2):
                    continue
            centers.append(candidate)
            accepted = True
            break
        if not accepted:
            warnings.warn(
                'Could not place every requested primary nanospike while '
                'respecting min_primary_spacing_m.',
                RuntimeWarning,
            )
            break
    return np.asarray(centers, dtype=float).reshape(-1, 2)


def make_dendritic_nanospike_geometry(
    P: Params,
    base_z_m: float = 0.0,
    spike_density_m2: float = 1e14,
    n_primary_spikes: Optional[int] = None,
    use_poisson_spike_count: bool = False,
    primary_height_mean_m: float = 120e-9,
    primary_height_sd_m: float = 25e-9,
    primary_base_radius_mean_m: float = 18e-9,
    primary_base_radius_sd_m: float = 4e-9,
    primary_tip_radius_mean_m: float = 4e-9,
    primary_tip_radius_sd_m: float = 1e-9,
    primary_tilt_mean_deg: float = 10.0,
    primary_tilt_sd_deg: float = 6.0,
    branch_levels: int = 2,
    branches_per_segment_mean: float = 1.8,
    branch_origin_fraction_range: Tuple[float, float] = (0.30, 0.85),
    branch_length_fraction_mean: float = 0.45,
    branch_length_fraction_sd: float = 0.10,
    branch_angle_mean_deg: float = 48.0,
    branch_angle_sd_deg: float = 12.0,
    branch_radius_fraction: float = 0.55,
    branch_tip_radius_fraction: float = 0.35,
    min_primary_spacing_m: Optional[float] = None,
    edge_margin_m: Optional[float] = None,
    geometry_seed: Optional[int] = None,
    max_total_segments: int = 5000,
    name: str = 'dendritic_nanospike_planar',
) -> SensorGeometry:
    """Decorate a planar gold surface with stochastic dendritic nanospikes.

    The surface is built as a true 3-D voxel mask, so tilted trunks, side
    branches, overhangs, and inter-spike voids are retained. Primary trunks are
    tapered segments rooted in the planar base. Each eligible segment emits a
    Poisson-distributed number of smaller child branches until ``branch_levels``
    is reached.

    This geometry is directly compatible with the well-mixed-reservoir model:
    ``derive()`` finds the highest solid voxel and places the mixed reservoir
    above that global canopy, while all fluid among the spikes remains part of
    the explicit diffusion domain.

    Set ``geometry_seed`` explicitly when the same physical nanospike morphology
    should be reused across Monte Carlo replicates with different ``P.seed``.
    """
    if base_z_m < 0 or base_z_m >= P.H_m:
        raise ValueError('base_z_m must satisfy 0 <= base_z_m < P.H_m.')
    if spike_density_m2 < 0:
        raise ValueError('spike_density_m2 cannot be negative.')
    if n_primary_spikes is not None and int(n_primary_spikes) < 0:
        raise ValueError('n_primary_spikes cannot be negative.')
    if int(branch_levels) < 0:
        raise ValueError('branch_levels cannot be negative.')
    if float(branches_per_segment_mean) < 0:
        raise ValueError('branches_per_segment_mean cannot be negative.')
    if int(max_total_segments) < 1:
        raise ValueError('max_total_segments must be at least 1.')

    origin_lo, origin_hi = map(float, branch_origin_fraction_range)
    if not (0.0 <= origin_lo < origin_hi <= 1.0):
        raise ValueError(
            'branch_origin_fraction_range must satisfy 0 <= low < high <= 1.'
        )

    Nx, Ny, Nz = _grid_counts(P)
    shape = (Nx, Ny, Nz + 1)
    z_m = np.arange(Nz + 1, dtype=float) * P.a_m
    solid_mask = np.broadcast_to(
        z_m[None, None, :] <= float(base_z_m) + 1e-12 * P.a_m,
        shape,
    ).copy()

    seed = int(P.seed if geometry_seed is None else geometry_seed)
    rng = np.random.default_rng(seed)
    projected_area_m2 = P.Lx_m * P.Ly_m
    expected_primary_count = float(spike_density_m2) * projected_area_m2

    if n_primary_spikes is None:
        n_primary = (
            int(rng.poisson(expected_primary_count))
            if use_poisson_spike_count
            else int(round(expected_primary_count))
        )
    else:
        n_primary = int(n_primary_spikes)

    if min_primary_spacing_m is None:
        min_primary_spacing_m = 2.0 * float(primary_base_radius_mean_m)
    if edge_margin_m is None:
        edge_margin_m = max(
            2.0 * float(primary_base_radius_mean_m),
            0.20 * float(primary_height_mean_m),
        )

    if primary_tip_radius_mean_m < 0.5 * P.a_m:
        warnings.warn(
            'The mean primary tip radius is below half a lattice spacing; '
            'tip dimensions will be resolution-limited.',
            RuntimeWarning,
        )

    primary_centers = _sample_nanospike_primary_centers(
        rng,
        n_primary,
        P,
        edge_margin_m=float(edge_margin_m),
        min_spacing_m=float(min_primary_spacing_m),
    )

    segments = []
    queue = deque()

    def append_segment(start, end, r0, r1, level, parent_id, primary_id):
        sid = len(segments)
        vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(vector))
        direction = vector / length if length > 0 else np.array([0.0, 0.0, 1.0])
        segments.append({
            'segment_id': sid,
            'parent_id': int(parent_id),
            'primary_id': int(primary_id),
            'level': int(level),
            'start_m': np.asarray(start, dtype=float).copy(),
            'end_m': np.asarray(end, dtype=float).copy(),
            'direction': direction.copy(),
            'length_m': length,
            'radius_start_m': float(r0),
            'radius_end_m': float(r1),
        })
        return sid

    # Primary trunks.
    for primary_id, xy in enumerate(primary_centers):
        length = _sample_positive_normal(
            rng,
            primary_height_mean_m,
            primary_height_sd_m,
            minimum=P.a_m,
            maximum=max(P.a_m, P.H_m - base_z_m - P.a_m),
        )
        r0 = _sample_positive_normal(
            rng,
            primary_base_radius_mean_m,
            primary_base_radius_sd_m,
            minimum=0.25 * P.a_m,
        )
        r1 = _sample_positive_normal(
            rng,
            primary_tip_radius_mean_m,
            primary_tip_radius_sd_m,
            minimum=0.10 * P.a_m,
            maximum=r0,
        )
        start = np.array([xy[0], xy[1], float(base_z_m)])
        accepted = False
        for _ in range(100):
            direction = _nanospike_primary_direction(
                rng, primary_tilt_mean_deg, primary_tilt_sd_deg
            )
            end = start + length * direction
            if _nanospike_segment_fits_domain(start, end, r0, P):
                accepted = True
                break
        if not accepted:
            warnings.warn(
                f'Skipping primary nanospike {primary_id}: no sampled '
                'orientation fit inside the domain.',
                RuntimeWarning,
            )
            continue
        sid = append_segment(start, end, r0, r1, 0, -1, primary_id)
        queue.append(sid)

    # Recursive branches.
    while queue and len(segments) < int(max_total_segments):
        parent_id = queue.popleft()
        parent = segments[parent_id]
        level = int(parent['level'])
        if level >= int(branch_levels):
            continue

        n_children = int(rng.poisson(float(branches_per_segment_mean)))
        for _ in range(n_children):
            if len(segments) >= int(max_total_segments):
                break
            fraction = rng.uniform(origin_lo, origin_hi)
            p0 = np.asarray(parent['start_m'])
            p1 = np.asarray(parent['end_m'])
            start = p0 + fraction * (p1 - p0)
            parent_radius = (
                float(parent['radius_start_m'])
                + fraction * (
                    float(parent['radius_end_m'])
                    - float(parent['radius_start_m'])
                )
            )
            length_fraction = _sample_positive_normal(
                rng,
                branch_length_fraction_mean,
                branch_length_fraction_sd,
                minimum=0.15,
                maximum=0.80,
            )
            length = max(P.a_m, float(parent['length_m']) * length_fraction)
            r0 = max(0.10 * P.a_m, parent_radius * float(branch_radius_fraction))
            r1 = max(0.05 * P.a_m, r0 * float(branch_tip_radius_fraction))

            accepted = False
            for _attempt in range(50):
                direction = _nanospike_branch_direction(
                    rng,
                    np.asarray(parent['direction']),
                    branch_angle_mean_deg,
                    branch_angle_sd_deg,
                )
                end = start + length * direction
                if end[2] <= base_z_m + 0.5 * P.a_m:
                    continue
                if _nanospike_segment_fits_domain(start, end, r0, P):
                    accepted = True
                    break
            if not accepted:
                continue

            sid = append_segment(
                start,
                end,
                r0,
                r1,
                level + 1,
                parent_id,
                int(parent['primary_id']),
            )
            queue.append(sid)

    if len(segments) >= int(max_total_segments) and queue:
        warnings.warn(
            'Reached max_total_segments before all eligible branches were expanded.',
            RuntimeWarning,
        )

    for segment in segments:
        _voxelize_tapered_nanospike_segment(
            solid_mask,
            P,
            segment['start_m'],
            segment['end_m'],
            segment['radius_start_m'],
            segment['radius_end_m'],
        )

    geometry = geometry_from_solid_mask(P, solid_mask, name=name)
    total_exposed_area_m2 = float(np.sum(geometry.surface_area_m2))
    roughness_factor = (
        total_exposed_area_m2 / projected_area_m2
        if projected_area_m2 > 0
        else np.nan
    )
    highest_solid_z_index = int(np.max(np.argwhere(solid_mask)[:, 2]))

    geometry.nanospike_geometry_seed = seed
    geometry.nanospike_primary_centers_xy_m = primary_centers
    geometry.nanospike_expected_primary_count = expected_primary_count
    geometry.nanospike_n_primary_requested = int(n_primary)
    geometry.nanospike_n_primary_generated = int(
        sum(int(s['level']) == 0 for s in segments)
    )
    geometry.nanospike_n_segments = int(len(segments))
    geometry.nanospike_branch_levels = int(branch_levels)
    geometry.nanospike_segments = segments
    geometry.nanospike_base_z_m = float(base_z_m)
    geometry.nanospike_canopy_z_m = float(highest_solid_z_index * P.a_m)
    geometry.nanospike_projected_area_m2 = float(projected_area_m2)
    geometry.nanospike_total_exposed_area_m2 = total_exposed_area_m2
    geometry.roughness_factor = float(roughness_factor)

    return geometry



def make_funnel_trap_geometry(
    P: Params,
    surface_z_m: float,
    cavity_radius_m: float,
    funnel_depth_m: float,
    funnel_mouth_radius_m: float,
    funnel_throat_radius_m: float,
    center_xy_m: Optional[Tuple[float, float]] = None,
    cavity_overlap_m: Optional[float] = None,
    throat_length_m: float = 0.0,
    reactive_region: str = "all",
    name: str = "funnel_trap",
) -> SensorGeometry:
    """
    Construct a planar sensor containing a funnel connected to a spherical cavity.

    The sensor is a solid slab occupying z <= ``surface_z_m``. A fluid-accessible
    trap is carved into the slab:

        external bulk
              |
        wide funnel mouth
            \\   /
             \\ /
              |       narrow throat
           .-----.
         .'       '.
        / spherical \
       |   cavity    |
        \\           /
         '---------'

    The funnel is a conical frustum whose radius decreases from
    ``funnel_mouth_radius_m`` at the planar surface to
    ``funnel_throat_radius_m`` at depth ``funnel_depth_m``. The spherical cavity
    is placed directly below the funnel and overlaps it slightly so the voxelized
    fluid space remains connected.

    This geometry is particularly useful with ``use_well_mixed_reservoir=True``:
    the global solid envelope remains the planar surface at ``surface_z_m``, so
    the well-mixed reservoir is placed above the plane while diffusion through
    the funnel, throat, and cavity remains explicit.

    Parameters
    ----------
    P : Params
        Simulation parameters.

    surface_z_m : float
        Height of the planar sensor surface. The gold slab occupies lattice sites
        at or below this height.

    cavity_radius_m : float
        Radius of the spherical fluid cavity.

    funnel_depth_m : float
        Vertical distance from the planar surface to the bottom of the conical
        funnel / top of the optional cylindrical throat.

    funnel_mouth_radius_m : float
        Funnel radius at the planar surface.

    funnel_throat_radius_m : float
        Funnel radius at its narrow end.

    center_xy_m : (float, float) or None
        Lateral center of the trap. If None, use the center of the simulation
        domain.

    cavity_overlap_m : float or None
        Overlap between the spherical cavity and the lower end of the throat.
        This prevents loss of connectivity due to voxelization. If None, use
        one lattice spacing.

    throat_length_m : float
        Optional cylindrical narrow-neck length between the funnel and sphere.
        Increasing this strongly suppresses escape from the cavity.

    reactive_region : {"all", "trap_only"}
        ``"all"``:
            Every exposed gold face is reactive, including the surrounding
            planar surface.

        ``"trap_only"``:
            Receptors are restricted approximately to gold voxels bordering the
            funnel/throat/cavity region. This is useful for isolating confinement
            and rebinding inside the trap.

    name : str
        Geometry name.

    Returns
    -------
    SensorGeometry
        Geometry compatible with derive(), run_simulation(), receptor placement,
        rebinding analysis, and the well-mixed-reservoir model.

    Notes
    -----
    The physical sphere and funnel are rasterized onto the lattice. Feature
    dimensions should generally span several lattice spacings.

    For a strong entropic/first-passage trap, the most influential ratios are

        funnel_throat_radius_m / cavity_radius_m

    and

        throat_length_m / funnel_throat_radius_m.

    A small throat connected to a much larger cavity gives a ligand many more
    possible diffusive states inside the cavity than states that lead directly
    back through the exit.
    """
    Nx, Ny, Nz = _grid_counts(P)
    shape = (Nx, Ny, Nz + 1)

    surface_z_m = float(surface_z_m)
    cavity_radius_m = float(cavity_radius_m)
    funnel_depth_m = float(funnel_depth_m)
    funnel_mouth_radius_m = float(funnel_mouth_radius_m)
    funnel_throat_radius_m = float(funnel_throat_radius_m)
    throat_length_m = float(throat_length_m)

    if not (0.0 <= surface_z_m < P.H_m):
        raise ValueError(
            "surface_z_m must satisfy 0 <= surface_z_m < P.H_m."
        )

    for parameter_name, value in {
        "cavity_radius_m": cavity_radius_m,
        "funnel_depth_m": funnel_depth_m,
        "funnel_mouth_radius_m": funnel_mouth_radius_m,
        "funnel_throat_radius_m": funnel_throat_radius_m,
    }.items():
        if value <= 0:
            raise ValueError(f"{parameter_name} must be positive.")

    if throat_length_m < 0:
        raise ValueError("throat_length_m cannot be negative.")

    if funnel_throat_radius_m > funnel_mouth_radius_m:
        raise ValueError(
            "funnel_throat_radius_m must be <= funnel_mouth_radius_m."
        )

    if funnel_depth_m >= surface_z_m:
        raise ValueError(
            "funnel_depth_m must be smaller than surface_z_m so the funnel "
            "remains inside the solid slab above z=0."
        )

    if reactive_region not in {"all", "trap_only"}:
        raise ValueError(
            "reactive_region must be either 'all' or 'trap_only'."
        )

    if center_xy_m is None:
        center_x_m = 0.5 * (Nx - 1) * P.a_m
        center_y_m = 0.5 * (Ny - 1) * P.a_m
    else:
        center_x_m = float(center_xy_m[0])
        center_y_m = float(center_xy_m[1])

    if cavity_overlap_m is None:
        cavity_overlap_m = P.a_m
    cavity_overlap_m = float(cavity_overlap_m)

    if cavity_overlap_m < 0:
        raise ValueError("cavity_overlap_m cannot be negative.")

    # Funnel runs from the planar surface down to funnel_bottom_z.
    funnel_bottom_z_m = surface_z_m - funnel_depth_m

    # Optional narrow cylindrical throat continues below the funnel.
    throat_bottom_z_m = funnel_bottom_z_m - throat_length_m

    # Place the sphere so its top overlaps the bottom of the throat.
    cavity_center_z_m = (
        throat_bottom_z_m
        - cavity_radius_m
        + cavity_overlap_m
    )

    cavity_bottom_z_m = (
        cavity_center_z_m
        - cavity_radius_m
    )

    if cavity_bottom_z_m < 0:
        raise ValueError(
            "The spherical cavity would extend below z=0. Increase "
            "surface_z_m, reduce funnel_depth_m/throat_length_m/cavity radius, "
            "or otherwise increase the slab thickness."
        )

    # Keep the widest part of the trap away from the lateral domain edges.
    lateral_extent = max(
        funnel_mouth_radius_m,
        cavity_radius_m,
    )

    x_max_m = (Nx - 1) * P.a_m
    y_max_m = (Ny - 1) * P.a_m

    if not (
        lateral_extent <= center_x_m <= x_max_m - lateral_extent
        and lateral_extent <= center_y_m <= y_max_m - lateral_extent
    ):
        raise ValueError(
            "The funnel/cavity does not fit laterally inside the simulation "
            "domain. Move center_xy_m inward or reduce the trap dimensions."
        )

    # ------------------------------------------------------------------
    # Build the initial planar solid slab.
    # ------------------------------------------------------------------
    x_m = np.arange(Nx, dtype=float) * P.a_m
    y_m = np.arange(Ny, dtype=float) * P.a_m
    z_m = np.arange(Nz + 1, dtype=float) * P.a_m

    X, Y, Z = np.meshgrid(
        x_m,
        y_m,
        z_m,
        indexing="ij",
    )

    tolerance = 1e-12 * P.a_m

    solid_mask = (
        Z <= surface_z_m + tolerance
    )

    radial_distance_m = np.sqrt(
        (X - center_x_m) ** 2
        + (Y - center_y_m) ** 2
    )

    # ------------------------------------------------------------------
    # Carve the conical funnel.
    #
    # u = 0 at the wide planar mouth
    # u = 1 at the narrow lower end
    # ------------------------------------------------------------------
    in_funnel_z = (
        (Z <= surface_z_m + tolerance)
        & (Z >= funnel_bottom_z_m - tolerance)
    )

    u = np.clip(
        (surface_z_m - Z)
        / funnel_depth_m,
        0.0,
        1.0,
    )

    funnel_radius_at_z_m = (
        funnel_mouth_radius_m
        + u * (
            funnel_throat_radius_m
            - funnel_mouth_radius_m
        )
    )

    funnel_void = (
        in_funnel_z
        & (
            radial_distance_m
            <= funnel_radius_at_z_m + tolerance
        )
    )

    # ------------------------------------------------------------------
    # Carve the optional cylindrical throat.
    # ------------------------------------------------------------------
    if throat_length_m > 0:
        throat_void = (
            (Z <= funnel_bottom_z_m + tolerance)
            & (Z >= throat_bottom_z_m - tolerance)
            & (
                radial_distance_m
                <= funnel_throat_radius_m + tolerance
            )
        )
    else:
        throat_void = np.zeros(
            shape,
            dtype=bool,
        )

    # ------------------------------------------------------------------
    # Carve the spherical cavity.
    # ------------------------------------------------------------------
    cavity_distance_squared_m2 = (
        (X - center_x_m) ** 2
        + (Y - center_y_m) ** 2
        + (Z - cavity_center_z_m) ** 2
    )

    cavity_void = (
        cavity_distance_squared_m2
        <= cavity_radius_m**2 + tolerance
    )

    trap_void = (
        funnel_void
        | throat_void
        | cavity_void
    )

    solid_mask = (
        solid_mask
        & ~trap_void
    )

    # ------------------------------------------------------------------
    # Optionally restrict receptor placement to the trap neighborhood.
    #
    # geometry_from_solid_mask() accepts a solid-site-level reactive mask,
    # so we mark solid voxels close to the carved trap void. This selects
    # funnel, throat, and cavity walls while suppressing most of the external
    # planar surface.
    # ------------------------------------------------------------------
    if reactive_region == "all":
        reactive_solid_mask = None

    else:
        # Mark solid voxels that are face-adjacent to any trap-void voxel.
        trap_neighbor = np.zeros(
            shape,
            dtype=bool,
        )

        for direction in FACE_DIRECTIONS:
            dx, dy, dz = direction

            src_x = slice(
                max(0, -dx),
                min(Nx, Nx - dx),
            )
            src_y = slice(
                max(0, -dy),
                min(Ny, Ny - dy),
            )
            src_z = slice(
                max(0, -dz),
                min(Nz + 1, Nz + 1 - dz),
            )

            dst_x = slice(
                max(0, dx),
                min(Nx, Nx + dx),
            )
            dst_y = slice(
                max(0, dy),
                min(Ny, Ny + dy),
            )
            dst_z = slice(
                max(0, dz),
                min(Nz + 1, Nz + 1 + dz),
            )

            trap_neighbor[dst_x, dst_y, dst_z] |= (
                trap_void[src_x, src_y, src_z]
            )

        reactive_solid_mask = (
            solid_mask
            & trap_neighbor
        )

    geometry = geometry_from_solid_mask(
        P,
        solid_mask,
        name=name,
        reactive_solid_mask=reactive_solid_mask,
    )

    # ------------------------------------------------------------------
    # Metadata for visualization and parameter sweeps.
    # ------------------------------------------------------------------
    geometry.funnel_trap_center_xy_m = (
        float(center_x_m),
        float(center_y_m),
    )
    geometry.funnel_trap_surface_z_m = (
        surface_z_m
    )
    geometry.funnel_trap_cavity_radius_m = (
        cavity_radius_m
    )
    geometry.funnel_trap_cavity_center_z_m = (
        float(cavity_center_z_m)
    )
    geometry.funnel_trap_cavity_bottom_z_m = (
        float(cavity_bottom_z_m)
    )
    geometry.funnel_trap_funnel_depth_m = (
        funnel_depth_m
    )
    geometry.funnel_trap_funnel_mouth_radius_m = (
        funnel_mouth_radius_m
    )
    geometry.funnel_trap_funnel_throat_radius_m = (
        funnel_throat_radius_m
    )
    geometry.funnel_trap_funnel_bottom_z_m = (
        float(funnel_bottom_z_m)
    )
    geometry.funnel_trap_throat_length_m = (
        throat_length_m
    )
    geometry.funnel_trap_throat_bottom_z_m = (
        float(throat_bottom_z_m)
    )
    geometry.funnel_trap_cavity_overlap_m = (
        cavity_overlap_m
    )
    geometry.funnel_trap_reactive_region = (
        reactive_region
    )

    geometry.funnel_trap_void_voxel_count = int(
        np.count_nonzero(trap_void)
    )
    geometry.funnel_trap_cavity_voxel_count = int(
        np.count_nonzero(cavity_void)
    )
    geometry.funnel_trap_funnel_voxel_count = int(
        np.count_nonzero(funnel_void)
    )
    geometry.funnel_trap_throat_voxel_count = int(
        np.count_nonzero(throat_void)
    )

    # Useful nondimensional trapping descriptors.
    geometry.funnel_trap_throat_to_cavity_ratio = float(
        funnel_throat_radius_m
        / cavity_radius_m
    )
    geometry.funnel_trap_mouth_to_throat_ratio = float(
        funnel_mouth_radius_m
        / funnel_throat_radius_m
    )
    geometry.funnel_trap_throat_aspect_ratio = float(
        throat_length_m
        / funnel_throat_radius_m
        if funnel_throat_radius_m > 0
        else np.nan
    )

    return geometry




def make_funnel_trap_array_geometry(
    P: Params,
    surface_z_m: float,
    cavity_radius_m: float,
    funnel_depth_m: float,
    funnel_mouth_radius_m: float,
    funnel_throat_radius_m: float,
    pitch_m: float,
    throat_length_m: float = 0.0,
    cavity_overlap_m: Optional[float] = None,
    layout: str = "square",
    edge_margin_m: Optional[float] = None,
    reactive_region: str = "all",
    name: str = "funnel_trap_array",
) -> SensorGeometry:
    """
    Construct an array of funnel traps beneath a planar sensor surface.

    Each trap consists of a conical funnel connected through an optional
    cylindrical throat to a spherical cavity. The traps are carved as fluid
    voids into a common solid slab.

    This constructor is designed to be directly comparable with
    ``make_nanopore_array_geometry``. In particular, ``pitch_m`` and ``layout``
    have the same meaning, and ``2 * funnel_mouth_radius_m`` can be matched to
    ``pore_diameter_m`` when comparing projected openings.

    Parameters
    ----------
    P : Params
        Simulation parameters.

    surface_z_m : float
        Height of the planar sensor surface. The solid slab occupies lattice
        sites at or below this height.

    cavity_radius_m : float
        Radius of each spherical cavity.

    funnel_depth_m : float
        Vertical depth of each conical funnel from the planar surface to its
        narrow end.

    funnel_mouth_radius_m : float
        Radius of each funnel opening at the planar surface.

    funnel_throat_radius_m : float
        Radius at the narrow end of each funnel.

    pitch_m : float
        Center-to-center spacing between neighboring traps.

    throat_length_m : float
        Length of the optional cylindrical narrow throat between funnel and
        spherical cavity.

    cavity_overlap_m : float or None
        Overlap between the spherical cavity and the lower end of the throat.
        If None, defaults to one lattice spacing to preserve voxel connectivity.

    layout : {"square", "hexagonal", "hex"}
        Lateral arrangement of trap centers.

    edge_margin_m : float or None
        Minimum center-to-edge margin. If None, defaults to the largest lateral
        trap radius so cavities/openings remain inside the side boundaries.

    reactive_region : {"all", "trap_only"}
        ``"all"`` makes every exposed gold face reactive.
        ``"trap_only"`` restricts receptor placement to solid voxels adjacent
        to funnel/throat/cavity voids.

    name : str
        Geometry name.

    Returns
    -------
    SensorGeometry
        Voxelized funnel-trap array geometry.

    Notes
    -----
    With the well-mixed reservoir enabled, the reservoir remains above the
    global planar sensor envelope. Transport inside every funnel, throat, and
    cavity remains explicitly diffusion dominated.

    For a matched comparison to a nanopore array, useful choices are

        pore_diameter_m = 2 * funnel_mouth_radius_m
        nanopore pitch_m = funnel-trap pitch_m

    while varying the hidden cavity/throat geometry independently.
    """
    Nx, Ny, Nz = _grid_counts(P)
    shape = (Nx, Ny, Nz + 1)

    surface_z_m = float(surface_z_m)
    cavity_radius_m = float(cavity_radius_m)
    funnel_depth_m = float(funnel_depth_m)
    funnel_mouth_radius_m = float(funnel_mouth_radius_m)
    funnel_throat_radius_m = float(funnel_throat_radius_m)
    pitch_m = float(pitch_m)
    throat_length_m = float(throat_length_m)

    if not (0.0 <= surface_z_m < P.H_m):
        raise ValueError(
            "surface_z_m must satisfy 0 <= surface_z_m < P.H_m."
        )

    for parameter_name, value in {
        "cavity_radius_m": cavity_radius_m,
        "funnel_depth_m": funnel_depth_m,
        "funnel_mouth_radius_m": funnel_mouth_radius_m,
        "funnel_throat_radius_m": funnel_throat_radius_m,
        "pitch_m": pitch_m,
    }.items():
        if value <= 0:
            raise ValueError(f"{parameter_name} must be positive.")

    if throat_length_m < 0:
        raise ValueError("throat_length_m cannot be negative.")

    if funnel_throat_radius_m > funnel_mouth_radius_m:
        raise ValueError(
            "funnel_throat_radius_m must be <= funnel_mouth_radius_m."
        )

    if funnel_depth_m >= surface_z_m:
        raise ValueError(
            "funnel_depth_m must be smaller than surface_z_m."
        )

    if reactive_region not in {"all", "trap_only"}:
        raise ValueError(
            "reactive_region must be either 'all' or 'trap_only'."
        )

    if cavity_overlap_m is None:
        cavity_overlap_m = P.a_m
    cavity_overlap_m = float(cavity_overlap_m)

    if cavity_overlap_m < 0:
        raise ValueError("cavity_overlap_m cannot be negative.")

    if edge_margin_m is None:
        edge_margin_m = max(
            cavity_radius_m,
            funnel_mouth_radius_m,
        )
    edge_margin_m = float(edge_margin_m)

    funnel_bottom_z_m = surface_z_m - funnel_depth_m
    throat_bottom_z_m = funnel_bottom_z_m - throat_length_m
    cavity_center_z_m = (
        throat_bottom_z_m
        - cavity_radius_m
        + cavity_overlap_m
    )
    cavity_bottom_z_m = cavity_center_z_m - cavity_radius_m

    if cavity_bottom_z_m < 0:
        raise ValueError(
            "The spherical cavities would extend below z=0. Increase "
            "surface_z_m or reduce funnel/cavity/throat dimensions."
        )

    # Generate trap centers using the same square/hexagonal convention as
    # make_nanopore_array_geometry().
    layout = str(layout).lower()

    if layout not in {"square", "hexagonal", "hex"}:
        raise ValueError(
            "layout must be 'square', 'hexagonal', or 'hex'."
        )

    x_max_m = (Nx - 1) * P.a_m
    y_max_m = (Ny - 1) * P.a_m

    x_min_center = edge_margin_m
    x_max_center = x_max_m - edge_margin_m
    y_min_center = edge_margin_m
    y_max_center = y_max_m - edge_margin_m

    if (
        x_min_center > x_max_center
        or y_min_center > y_max_center
    ):
        raise ValueError(
            "The requested edge margin leaves no room for funnel-trap centers."
        )

    center_list = []

    if layout == "square":
        x_centers = np.arange(
            x_min_center,
            x_max_center + 0.5 * pitch_m,
            pitch_m,
        )
        y_centers = np.arange(
            y_min_center,
            y_max_center + 0.5 * pitch_m,
            pitch_m,
        )

        for xc in x_centers:
            for yc in y_centers:
                center_list.append((xc, yc))

    else:
        row_spacing_m = pitch_m * np.sqrt(3.0) / 2.0

        y_centers = np.arange(
            y_min_center,
            y_max_center + 0.5 * row_spacing_m,
            row_spacing_m,
        )

        for row_index, yc in enumerate(y_centers):
            x_offset = 0.5 * pitch_m if row_index % 2 else 0.0

            x_centers = np.arange(
                x_min_center + x_offset,
                x_max_center + 0.5 * pitch_m,
                pitch_m,
            )

            for xc in x_centers:
                if xc <= x_max_center + tolerance:
                    center_list.append((xc, yc))

    centers_xy_m = np.asarray(
        center_list,
        dtype=float,
    ).reshape(-1, 2)

    if centers_xy_m.size == 0:
        raise ValueError(
            "No funnel-trap centers fit inside the requested domain."
        )

    # Keep the hidden spherical cavities distinct. Allowing cavity overlap
    # would create a subsurface fluid network and change the intended trap
    # topology.
    if pitch_m < 2.0 * cavity_radius_m:
        raise ValueError(
            "pitch_m must be at least the spherical cavity diameter "
            "(2 * cavity_radius_m) so neighboring funnel traps do not merge."
        )

    x_m = np.arange(Nx, dtype=float) * P.a_m
    y_m = np.arange(Ny, dtype=float) * P.a_m
    z_m = np.arange(Nz + 1, dtype=float) * P.a_m

    X, Y, Z = np.meshgrid(
        x_m,
        y_m,
        z_m,
        indexing="ij",
    )

    tolerance = 1e-12 * P.a_m

    solid_mask = (
        Z <= surface_z_m + tolerance
    )

    all_trap_void = np.zeros(shape, dtype=bool)
    all_funnel_void = np.zeros(shape, dtype=bool)
    all_throat_void = np.zeros(shape, dtype=bool)
    all_cavity_void = np.zeros(shape, dtype=bool)

    in_funnel_z = (
        (Z <= surface_z_m + tolerance)
        & (Z >= funnel_bottom_z_m - tolerance)
    )

    u = np.clip(
        (surface_z_m - Z) / funnel_depth_m,
        0.0,
        1.0,
    )

    funnel_radius_at_z_m = (
        funnel_mouth_radius_m
        + u * (
            funnel_throat_radius_m
            - funnel_mouth_radius_m
        )
    )

    for center_x_m, center_y_m in centers_xy_m:
        radial_distance_m = np.sqrt(
            (X - center_x_m) ** 2
            + (Y - center_y_m) ** 2
        )

        funnel_void = (
            in_funnel_z
            & (
                radial_distance_m
                <= funnel_radius_at_z_m + tolerance
            )
        )

        if throat_length_m > 0:
            throat_void = (
                (Z <= funnel_bottom_z_m + tolerance)
                & (Z >= throat_bottom_z_m - tolerance)
                & (
                    radial_distance_m
                    <= funnel_throat_radius_m + tolerance
                )
            )
        else:
            throat_void = np.zeros(
                shape,
                dtype=bool,
            )

        cavity_distance_squared_m2 = (
            (X - center_x_m) ** 2
            + (Y - center_y_m) ** 2
            + (Z - cavity_center_z_m) ** 2
        )

        cavity_void = (
            cavity_distance_squared_m2
            <= cavity_radius_m**2 + tolerance
        )

        trap_void = (
            funnel_void
            | throat_void
            | cavity_void
        )

        all_funnel_void |= funnel_void
        all_throat_void |= throat_void
        all_cavity_void |= cavity_void
        all_trap_void |= trap_void

    solid_mask = (
        solid_mask
        & ~all_trap_void
    )

    if reactive_region == "all":
        reactive_solid_mask = None

    else:
        trap_neighbor = np.zeros(
            shape,
            dtype=bool,
        )

        for direction in FACE_DIRECTIONS:
            dx, dy, dz = direction

            src_x = slice(
                max(0, -dx),
                min(Nx, Nx - dx),
            )
            src_y = slice(
                max(0, -dy),
                min(Ny, Ny - dy),
            )
            src_z = slice(
                max(0, -dz),
                min(Nz + 1, Nz + 1 - dz),
            )

            dst_x = slice(
                max(0, dx),
                min(Nx, Nx + dx),
            )
            dst_y = slice(
                max(0, dy),
                min(Ny, Ny + dy),
            )
            dst_z = slice(
                max(0, dz),
                min(Nz + 1, Nz + 1 + dz),
            )

            trap_neighbor[
                dst_x,
                dst_y,
                dst_z,
            ] |= all_trap_void[
                src_x,
                src_y,
                src_z,
            ]

        reactive_solid_mask = (
            solid_mask
            & trap_neighbor
        )

    geometry = geometry_from_solid_mask(
        P,
        solid_mask,
        name=name,
        reactive_solid_mask=reactive_solid_mask,
    )

    geometry.funnel_trap_array_centers_xy_m = np.asarray(
        centers_xy_m,
        dtype=float,
    )
    geometry.funnel_trap_array_n_traps = int(
        len(centers_xy_m)
    )
    geometry.funnel_trap_array_pitch_m = pitch_m
    geometry.funnel_trap_array_layout = str(layout)
    geometry.funnel_trap_array_edge_margin_m = edge_margin_m

    geometry.funnel_trap_surface_z_m = surface_z_m
    geometry.funnel_trap_cavity_radius_m = cavity_radius_m
    geometry.funnel_trap_cavity_center_z_m = float(
        cavity_center_z_m
    )
    geometry.funnel_trap_cavity_bottom_z_m = float(
        cavity_bottom_z_m
    )
    geometry.funnel_trap_funnel_depth_m = funnel_depth_m
    geometry.funnel_trap_funnel_mouth_radius_m = (
        funnel_mouth_radius_m
    )
    geometry.funnel_trap_funnel_throat_radius_m = (
        funnel_throat_radius_m
    )
    geometry.funnel_trap_funnel_bottom_z_m = float(
        funnel_bottom_z_m
    )
    geometry.funnel_trap_throat_length_m = throat_length_m
    geometry.funnel_trap_throat_bottom_z_m = float(
        throat_bottom_z_m
    )
    geometry.funnel_trap_cavity_overlap_m = cavity_overlap_m
    geometry.funnel_trap_reactive_region = reactive_region

    geometry.funnel_trap_void_voxel_count = int(
        np.count_nonzero(all_trap_void)
    )
    geometry.funnel_trap_cavity_voxel_count = int(
        np.count_nonzero(all_cavity_void)
    )
    geometry.funnel_trap_funnel_voxel_count = int(
        np.count_nonzero(all_funnel_void)
    )
    geometry.funnel_trap_throat_voxel_count = int(
        np.count_nonzero(all_throat_void)
    )

    geometry.funnel_trap_throat_to_cavity_ratio = float(
        funnel_throat_radius_m
        / cavity_radius_m
    )
    geometry.funnel_trap_mouth_to_throat_ratio = float(
        funnel_mouth_radius_m
        / funnel_throat_radius_m
    )
    geometry.funnel_trap_throat_aspect_ratio = float(
        throat_length_m
        / funnel_throat_radius_m
        if funnel_throat_radius_m > 0
        else np.nan
    )

    geometry.funnel_trap_array_total_mouth_area_m2 = float(
        len(centers_xy_m)
        * np.pi
        * funnel_mouth_radius_m**2
    )

    geometry.funnel_trap_array_projected_area_fraction = float(
        geometry.funnel_trap_array_total_mouth_area_m2
        / (P.Lx_m * P.Ly_m)
    )

    return geometry



def make_nanopore_array_geometry(
    P: Params,
    pore_diameter_m: float,
    pore_depth_m: float,
    pitch_m: float,
    rim_z_m: float,
    layout: str = "square",
    edge_margin_m: Optional[float] = None,
    pore_centers_xy_m: Optional[np.ndarray] = None,
    name: str = "nanopore_array",
) -> SensorGeometry:
    """
    Construct a planar slab containing an array of cylindrical nanopores.

    The sensor occupies all lattice sites satisfying z <= h(x, y). Outside
    the pores, h(x, y) equals ``rim_z_m``. Inside each pore, h(x, y) equals

        rim_z_m - pore_depth_m

    so the exposed pore consists of a recessed bottom and voxelized sidewalls.

    Parameters
    ----------
    P : Params
        Simulation parameters.

    pore_diameter_m : float
        Physical pore diameter in meters.

    pore_depth_m : float
        Pore depth measured downward from the upper slab surface.

    pitch_m : float
        Center-to-center pore spacing. Used only when pore centers are
        generated automatically.

    rim_z_m : float
        Height of the upper slab surface above z = 0.

    layout : {"square", "hexagonal"}
        Arrangement used to generate pore centers automatically.

    edge_margin_m : float or None
        Minimum distance from each pore center to the lateral domain edge.
        If None, defaults to one pore radius plus half a pitch.

    pore_centers_xy_m : array-like or None
        Optional explicit pore centers with shape (N_pores, 2), in meters.
        If provided, ``layout`` and automatic center generation are ignored.

    name : str
        Geometry name.

    Returns
    -------
    SensorGeometry
        Voxelized nanopore-array geometry.

    Notes
    -----
    All exposed surfaces are reactive by default, including:

    - the upper planar surface,
    - pore sidewalls,
    - pore bottoms.

    The physical pore shape is voxelized on the same lattice used for ligand
    diffusion.
    """
    if pore_diameter_m <= 0:
        raise ValueError("pore_diameter_m must be positive.")

    if pore_depth_m <= 0:
        raise ValueError("pore_depth_m must be positive.")

    if pitch_m <= 0:
        raise ValueError("pitch_m must be positive.")

    if rim_z_m < 0:
        raise ValueError("rim_z_m must be nonnegative.")

    if rim_z_m > P.H_m:
        raise ValueError(
            "rim_z_m must not exceed the simulation height P.H_m."
        )

    if pore_depth_m > rim_z_m:
        raise ValueError(
            "pore_depth_m must not exceed rim_z_m because the pore bottom "
            "cannot lie below z = 0."
        )

    pore_radius_m = 0.5 * pore_diameter_m
    pore_bottom_z_m = rim_z_m - pore_depth_m

    if pitch_m < pore_diameter_m:
        raise ValueError(
            "pitch_m must be at least as large as pore_diameter_m to prevent "
            "overlapping pores."
        )

    Nx, Ny, _ = _grid_counts(P)

    # Coordinates used by make_height_field_geometry run from 0 to
    # (N - 1) * a rather than exactly to L.
    x_max_m = (Nx - 1) * P.a_m
    y_max_m = (Ny - 1) * P.a_m

    if edge_margin_m is None:
        edge_margin_m = pore_radius_m + 0.5 * pitch_m

    if edge_margin_m < pore_radius_m:
        raise ValueError(
            "edge_margin_m must be at least one pore radius."
        )

    # ------------------------------------------------------------------
    # Generate pore centers
    # ------------------------------------------------------------------

    if pore_centers_xy_m is not None:
        centers = np.asarray(pore_centers_xy_m, dtype=float)

        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(
                "pore_centers_xy_m must have shape (N_pores, 2)."
            )

        if centers.shape[0] == 0:
            raise ValueError(
                "pore_centers_xy_m must contain at least one pore center."
            )

    else:
        layout = str(layout).lower()

        if layout not in {"square", "hexagonal", "hex"}:
            raise ValueError(
                "layout must be 'square' or 'hexagonal'."
            )

        x_min_center = edge_margin_m
        x_max_center = x_max_m - edge_margin_m
        y_min_center = edge_margin_m
        y_max_center = y_max_m - edge_margin_m

        if (
            x_min_center > x_max_center
            or y_min_center > y_max_center
        ):
            raise ValueError(
                "The requested edge margin leaves no room for pore centers."
            )

        center_list = []

        if layout == "square":
            x_centers = np.arange(
                x_min_center,
                x_max_center + 0.5 * pitch_m,
                pitch_m,
            )

            y_centers = np.arange(
                y_min_center,
                y_max_center + 0.5 * pitch_m,
                pitch_m,
            )

            for xc in x_centers:
                for yc in y_centers:
                    center_list.append((xc, yc))

        else:
            # Nearest-neighbor spacing is pitch_m.
            row_spacing_m = pitch_m * np.sqrt(3.0) / 2.0

            y_centers = np.arange(
                y_min_center,
                y_max_center + 0.5 * row_spacing_m,
                row_spacing_m,
            )

            for row_index, yc in enumerate(y_centers):
                x_offset = 0.5 * pitch_m if row_index % 2 else 0.0

                x_centers = np.arange(
                    x_min_center + x_offset,
                    x_max_center + 0.5 * pitch_m,
                    pitch_m,
                )

                for xc in x_centers:
                    if xc <= x_max_center:
                        center_list.append((xc, yc))

        if not center_list:
            raise ValueError(
                "No pores fit within the requested domain and edge margin."
            )

        centers = np.asarray(center_list, dtype=float)

    # Ensure explicit centers keep the complete circular pore in the domain.
    invalid = (
        (centers[:, 0] - pore_radius_m < 0.0)
        | (centers[:, 0] + pore_radius_m > x_max_m)
        | (centers[:, 1] - pore_radius_m < 0.0)
        | (centers[:, 1] + pore_radius_m > y_max_m)
    )

    if np.any(invalid):
        bad_indices = np.flatnonzero(invalid)

        raise ValueError(
            "Some pore centers place part of a pore outside the lateral "
            f"domain. Invalid center indices: {bad_indices.tolist()}."
        )

    # ------------------------------------------------------------------
    # Define the height field
    # ------------------------------------------------------------------

    radius_squared_m2 = pore_radius_m**2

    def nanopore_height(
        X_m: np.ndarray,
        Y_m: np.ndarray,
    ) -> np.ndarray:
        height = np.full(
            X_m.shape,
            rim_z_m,
            dtype=float,
        )

        inside_any_pore = np.zeros(
            X_m.shape,
            dtype=bool,
        )

        for xc_m, yc_m in centers:
            radial_distance_squared = (
                (X_m - xc_m) ** 2
                + (Y_m - yc_m) ** 2
            )

            inside_any_pore |= (
                radial_distance_squared <= radius_squared_m2
            )

        height[inside_any_pore] = pore_bottom_z_m

        return height

    geometry = make_height_field_geometry(
        P,
        height_m=nanopore_height,
        name=name,
    )

    # Store convenient metadata directly on the geometry object. These fields
    # are optional and do not affect the simulation.
    geometry.pore_centers_xy_m = centers
    geometry.pore_diameter_m = float(pore_diameter_m)
    geometry.pore_depth_m = float(pore_depth_m)
    geometry.pore_pitch_m = float(pitch_m)
    geometry.rim_z_m = float(rim_z_m)
    geometry.layout = str(layout)

    return geometry