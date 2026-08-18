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