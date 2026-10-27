"""Steady-state finite-element capture calculations with FEniCSx.

This module solves diffusion through a periodic fluid unit cell above a sensor
described by a single-valued height field ``z = h(x, y)``.  The top boundary is
held at ``c_bulk``.  The sensor is either perfectly absorbing (the
diffusion-limited accessibility calculation) or partially reactive through a
Robin boundary condition.

FEniCSx and dolfinx_mpc are imported lazily so that the rest of the
SensorModeling package remains importable in environments where the FEM stack
is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np


HeightFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
SurfaceDensity = Union[
    float,
    Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
]
SurfaceMode = Literal["perfect_absorber", "finite_reactivity"]

AVOGADRO_PER_MOL = 6.02214076e23

SURFACE_TAG = 1
TOP_TAG = 2
X_MIN_TAG = 3
X_MAX_TAG = 4
Y_MIN_TAG = 5
Y_MAX_TAG = 6


@dataclass(frozen=True)
class FEMeshConfig:
    """Dimensions and resolution of the mapped periodic fluid unit cell."""

    Lx_m: float
    Ly_m: float
    top_z_m: float
    nx: int = 32
    ny: int = 32
    nz: int = 24
    periodic_x: bool = True
    periodic_y: bool = True

    def validate(self) -> None:
        if self.Lx_m <= 0 or self.Ly_m <= 0 or self.top_z_m <= 0:
            raise ValueError("Lx_m, Ly_m, and top_z_m must be positive.")
        if min(self.nx, self.ny, self.nz) < 2:
            raise ValueError("nx, ny, and nz must each be at least 2.")


@dataclass(frozen=True)
class DiffusionCaptureConfig:
    """Physical and numerical parameters for one steady capture solve."""

    D_m2_s: float
    c_bulk_mol_m3: float = 1.0
    surface_mode: SurfaceMode = "finite_reactivity"
    kappa_m_s: float = 0.0
    polynomial_degree: int = 1
    linear_solver: str = "cg"
    preconditioner: str = "hypre"

    def validate(self) -> None:
        if self.D_m2_s <= 0:
            raise ValueError("D_m2_s must be positive.")
        if self.c_bulk_mol_m3 <= 0:
            raise ValueError("c_bulk_mol_m3 must be positive.")
        if self.surface_mode not in {"perfect_absorber", "finite_reactivity"}:
            raise ValueError("Unknown surface_mode.")
        if self.surface_mode == "finite_reactivity" and self.kappa_m_s < 0:
            raise ValueError("kappa_m_s must be nonnegative.")
        if self.polynomial_degree < 1:
            raise ValueError("polynomial_degree must be at least 1.")


@dataclass
class FEMeshBundle:
    """Mapped DOLFINx mesh and its exterior-facet tags."""

    domain: object
    facet_tags: object
    height_function: HeightFunction
    config: FEMeshConfig


@dataclass
class FEACaptureResult:
    """Solution fields and integrated observables from one FEM solve."""

    concentration: object
    surface_flux_density: object
    surface_boundary_measure: object
    mesh_bundle: FEMeshBundle
    config: DiffusionCaptureConfig
    capture_flux_mol_s: float
    supply_flux_mol_s: float
    capture_rate_m3_s: float
    surface_area_m2: float
    mass_balance_relative_error: float
    mpc: object


@dataclass(frozen=True)
class PlanarResistanceFit:
    """Linear calibration of areal transport resistance versus fluid height.

    ``A / k_capture`` has units s/m and is fitted as ``intercept + slope*gap``.
    For an exactly resolved planar Robin boundary, the expected coefficients
    are ``intercept = 1/kappa`` and ``slope = 1/D``.
    """

    intercept_s_m: float
    slope_s_m2: float
    r_squared: float
    gaps_m: np.ndarray
    areal_resistances_s_m: np.ndarray

    def predict_areal_resistance(self, gap_m) -> np.ndarray:
        gap = np.asarray(gap_m, dtype=float)
        return self.intercept_s_m + self.slope_s_m2 * gap


@dataclass(frozen=True)
class EquilibriumSurfaceDelivery:
    """Gross one-way association rates on a triangulated height-field surface."""

    triangle_centers_m: np.ndarray
    triangle_areas_m2: np.ndarray
    receptor_density_m2: np.ndarray
    free_receptor_density_m2: np.ndarray
    association_rate_density_m2_s: np.ndarray
    association_rate_per_triangle_s: np.ndarray
    surface_area_m2: float
    projected_area_m2: float
    total_association_rate_s: float
    total_association_rate_mol_s: float
    c_bulk_M: float
    free_receptor_fraction: float


def calculate_equilibrium_surface_delivery(
    height_function: HeightFunction,
    *,
    Lx_m: float,
    Ly_m: float,
    c_bulk_mol_m3: float,
    k_on_M_inv_s: float,
    receptor_density_m2: SurfaceDensity,
    k_off_s: Optional[float] = None,
    nx: int = 150,
    ny: int = 150,
) -> EquilibriumSurfaceDelivery:
    """Calculate gross equilibrium association events over a sensor surface.

    This is not a net diffusive flux: the net equilibrium flux is zero.  It
    evaluates ``j_on = k_on * C_bulk * rho_R,free`` on true surface-area
    elements.  With ``k_off_s=None``, all receptors are treated as available
    (the low-occupancy limit).  Supplying ``k_off_s`` applies independent-site
    Langmuir availability, ``rho_R,free/rho_R = KD/(KD + C_bulk)``.

    ``receptor_density_m2`` may be a scalar or a function of the triangle
    center coordinates ``(x, y, z)`` and is interpreted per true surface area.
    """

    if Lx_m <= 0 or Ly_m <= 0 or c_bulk_mol_m3 < 0:
        raise ValueError("Lx_m and Ly_m must be positive and concentration nonnegative.")
    if k_on_M_inv_s < 0:
        raise ValueError("k_on_M_inv_s must be nonnegative.")
    if k_off_s is not None and k_off_s < 0:
        raise ValueError("k_off_s must be nonnegative when supplied.")
    if min(nx, ny) < 1:
        raise ValueError("nx and ny must each be at least 1.")

    x = np.linspace(0.0, float(Lx_m), int(nx) + 1)
    y = np.linspace(0.0, float(Ly_m), int(ny) + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.asarray(height_function(X, Y), dtype=float)
    if Z.shape != X.shape or np.any(~np.isfinite(Z)):
        raise ValueError("height_function must return finite values matching x and y.")
    points = np.stack((X, Y, Z), axis=-1)
    p00 = points[:-1, :-1].reshape(-1, 3)
    p10 = points[1:, :-1].reshape(-1, 3)
    p11 = points[1:, 1:].reshape(-1, 3)
    p01 = points[:-1, 1:].reshape(-1, 3)
    triangles = np.concatenate(
        (np.stack((p00, p10, p11), axis=1), np.stack((p00, p11, p01), axis=1)),
        axis=0,
    )
    centers = np.mean(triangles, axis=1)
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )

    if callable(receptor_density_m2):
        density = np.asarray(
            receptor_density_m2(centers[:, 0], centers[:, 1], centers[:, 2]),
            dtype=float,
        )
        density = np.broadcast_to(density, areas.shape).copy()
    else:
        density = np.full_like(areas, float(receptor_density_m2), dtype=float)
    if np.any(~np.isfinite(density)) or np.any(density < 0):
        raise ValueError("receptor_density_m2 must be finite and nonnegative.")

    c_bulk_M = float(c_bulk_mol_m3) / 1000.0
    if k_off_s is None:
        free_fraction = 1.0
    elif k_on_M_inv_s == 0:
        free_fraction = 1.0
    else:
        kd_M = float(k_off_s) / float(k_on_M_inv_s)
        denominator = kd_M + c_bulk_M
        free_fraction = 1.0 if denominator == 0 else kd_M / denominator
    free_density = density * free_fraction
    rate_density = float(k_on_M_inv_s) * c_bulk_M * free_density
    element_rates = rate_density * areas
    total_rate = float(np.sum(element_rates))
    return EquilibriumSurfaceDelivery(
        triangle_centers_m=centers,
        triangle_areas_m2=areas,
        receptor_density_m2=density,
        free_receptor_density_m2=free_density,
        association_rate_density_m2_s=rate_density,
        association_rate_per_triangle_s=element_rates,
        surface_area_m2=float(np.sum(areas)),
        projected_area_m2=float(Lx_m) * float(Ly_m),
        total_association_rate_s=total_rate,
        total_association_rate_mol_s=total_rate / AVOGADRO_PER_MOL,
        c_bulk_M=c_bulk_M,
        free_receptor_fraction=float(free_fraction),
    )


def fit_planar_bulk_resistance(
    gaps_m,
    capture_rates_m3_s,
    projected_area_m2: float,
) -> PlanarResistanceFit:
    """Fit the planar series-resistance relation ``A/k = R0 + H/D``."""

    gaps = np.asarray(gaps_m, dtype=float)
    rates = np.asarray(capture_rates_m3_s, dtype=float)
    if gaps.ndim != 1 or rates.ndim != 1 or gaps.size != rates.size:
        raise ValueError("gaps_m and capture_rates_m3_s must be equal-length 1D arrays.")
    if gaps.size < 2:
        raise ValueError("At least two planar heights are required for the fit.")
    if np.any(~np.isfinite(gaps)) or np.any(gaps <= 0):
        raise ValueError("All planar gaps must be finite and positive.")
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError("All capture rates must be finite and positive.")
    if not np.isfinite(projected_area_m2) or projected_area_m2 <= 0:
        raise ValueError("projected_area_m2 must be finite and positive.")

    resistance = float(projected_area_m2) / rates
    slope, intercept = np.polyfit(gaps, resistance, deg=1)
    fitted = intercept + slope * gaps
    residual_ss = float(np.sum((resistance - fitted) ** 2))
    total_ss = float(np.sum((resistance - np.mean(resistance)) ** 2))
    r_squared = 1.0 if total_ss == 0 and residual_ss == 0 else 1.0 - residual_ss / total_ss
    return PlanarResistanceFit(
        intercept_s_m=float(intercept),
        slope_s_m2=float(slope),
        r_squared=float(r_squared),
        gaps_m=gaps.copy(),
        areal_resistances_s_m=resistance,
    )


def height_corrected_surface_accessibility(
    *,
    capture_rate_m3_s: float,
    projected_area_m2: float,
    top_z_m: float,
    mean_surface_z_m: float,
    planar_fit: PlanarResistanceFit,
) -> dict[str, float]:
    """Subtract fitted bulk-slab resistance from a geometry's total resistance.

    The bulk distance is the volume-equivalent mean gap ``top_z - mean(h)``.
    A nonpositive corrected resistance is retained as a fit diagnostic but is
    not inverted into a surface velocity.
    """

    if capture_rate_m3_s <= 0 or projected_area_m2 <= 0:
        raise ValueError("capture_rate_m3_s and projected_area_m2 must be positive.")
    mean_gap = float(top_z_m) - float(mean_surface_z_m)
    if mean_gap <= 0:
        raise ValueError("top_z_m must exceed mean_surface_z_m.")
    total_resistance = float(projected_area_m2) / float(capture_rate_m3_s)
    bulk_resistance = planar_fit.slope_s_m2 * mean_gap
    surface_resistance = total_resistance - bulk_resistance
    velocity = np.nan if surface_resistance <= 0 else 1.0 / surface_resistance
    corrected_capture_rate = (
        np.nan if not np.isfinite(velocity) else float(projected_area_m2) * velocity
    )
    correction_factor = (
        np.nan
        if not np.isfinite(corrected_capture_rate)
        else corrected_capture_rate / float(capture_rate_m3_s)
    )
    enhancement = (
        np.nan
        if surface_resistance <= 0 or planar_fit.intercept_s_m <= 0
        else planar_fit.intercept_s_m / surface_resistance
    )
    return {
        "mean_fluid_gap_m": mean_gap,
        "total_areal_resistance_s_m": total_resistance,
        "fitted_bulk_resistance_s_m": bulk_resistance,
        "height_corrected_surface_resistance_s_m": surface_resistance,
        "height_independent_surface_velocity_m_s": float(velocity),
        "height_corrected_capture_rate_m3_s": float(corrected_capture_rate),
        "bulk_resistance_correction_factor": float(correction_factor),
        "surface_accessibility_vs_planar": float(enhancement),
    }


def require_fenicsx() -> None:
    """Raise an actionable error if the optional FEM dependencies are absent."""

    missing = []
    for module in ("dolfinx", "ufl", "petsc4py", "mpi4py", "dolfinx_mpc"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise ImportError(
            "The FEniCSx capture workflow requires the optional packages "
            f"{', '.join(missing)}. Install matching FEniCSx and dolfinx_mpc "
            "builds, preferably in a dedicated conda environment or the "
            "dolfinx_mpc container."
        )


def robin_velocity_from_kon_density(
    k_on_M_inv_s: float,
    receptor_density_m2: float,
) -> float:
    """Convert bimolecular ``k_on`` and receptor density to Robin velocity.

    The homogenized boundary law is ``j = kappa*c`` with concentration in
    mol/m^3.  ``k_on_M_inv_s`` is converted to m^3/(mol s), and a receptor
    number density is converted to mol/m^2.
    """

    if k_on_M_inv_s < 0 or receptor_density_m2 < 0:
        raise ValueError("k_on_M_inv_s and receptor_density_m2 must be nonnegative.")
    return float(
        (k_on_M_inv_s / 1000.0)
        * (receptor_density_m2 / AVOGADRO_PER_MOL)
    )


def planar_height(z_m: float = 0.0) -> HeightFunction:
    """Return a planar height field."""

    z_m = float(z_m)

    def height(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        del y_m
        return np.full_like(np.asarray(x_m, dtype=float), z_m)

    return height


def sinusoidal_height(
    *,
    amplitude_m: float,
    wavelength_x_m: float,
    wavelength_y_m: Optional[float] = None,
    mean_z_m: Optional[float] = None,
    phase_x_rad: float = 0.0,
    phase_y_rad: float = 0.0,
) -> HeightFunction:
    """Return the sinusoidal height field used by the lattice geometries."""

    if amplitude_m <= 0 or wavelength_x_m <= 0:
        raise ValueError("amplitude_m and wavelength_x_m must be positive.")
    if wavelength_y_m is not None and wavelength_y_m <= 0:
        raise ValueError("wavelength_y_m must be positive when supplied.")
    if mean_z_m is None:
        mean_z_m = amplitude_m
    kx = 2.0 * np.pi / float(wavelength_x_m)
    ky = None if wavelength_y_m is None else 2.0 * np.pi / float(wavelength_y_m)

    def height(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        value = np.cos(kx * x_m + phase_x_rad)
        if ky is not None:
            value = value * np.cos(ky * y_m + phase_y_rad)
        return float(mean_z_m) + float(amplitude_m) * value

    return height


def spherical_bowl_height(
    *,
    radius_m: float,
    depth_m: float,
    center_xy_m: tuple[float, float],
    rim_z_m: Optional[float] = None,
) -> HeightFunction:
    """Return a spherical bowl recessed below a planar rim."""

    if radius_m <= 0 or depth_m <= 0 or depth_m > radius_m:
        raise ValueError("Require radius_m > 0 and 0 < depth_m <= radius_m.")
    if rim_z_m is None:
        rim_z_m = depth_m
    xc, yc = map(float, center_xy_m)
    footprint_r2 = 2.0 * radius_m * depth_m - depth_m**2
    center_z_m = float(rim_z_m) + radius_m - depth_m

    def height(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        r2 = (x_m - xc) ** 2 + (y_m - yc) ** 2
        values = np.full_like(r2, float(rim_z_m), dtype=float)
        inside = r2 <= footprint_r2
        values[inside] = center_z_m - np.sqrt(
            np.maximum(radius_m**2 - r2[inside], 0.0)
        )
        return values

    return height


def cylindrically_curved_sheet_height(
    *,
    radius_m: float,
    center_m: float,
    axis: Literal["x", "y"] = "y",
    vertex_z_m: float = 0.0,
) -> HeightFunction:
    """Return a concave cylindrical sheet, planar when radius is infinite."""

    if radius_m <= 0 or axis not in {"x", "y"}:
        raise ValueError("radius_m must be positive and axis must be 'x' or 'y'.")

    def height(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        coordinate = y_m - center_m if axis == "x" else x_m - center_m
        if np.isinf(radius_m):
            return np.full_like(coordinate, float(vertex_z_m), dtype=float)
        if np.any(np.abs(coordinate) > radius_m * (1.0 + 1e-12)):
            raise ValueError("The requested cylindrical radius does not span the domain.")
        root = np.sqrt(np.maximum(radius_m**2 - coordinate**2, 0.0))
        return float(vertex_z_m) + coordinate**2 / (radius_m + root)

    return height


def _validate_height_field(height: HeightFunction, config: FEMeshConfig) -> None:
    sample_count = max(config.nx, config.ny) + 1
    x = np.linspace(0.0, config.Lx_m, sample_count)
    y = np.linspace(0.0, config.Ly_m, sample_count)
    X, Y = np.meshgrid(x, y, indexing="ij")
    h = np.asarray(height(X, Y), dtype=float)
    if h.shape != X.shape:
        h = np.broadcast_to(h, X.shape)
    if not np.all(np.isfinite(h)):
        raise ValueError("The sensor height field contains non-finite values.")
    if np.min(h) < 0:
        raise ValueError("The sensor height field must be nonnegative.")
    if np.max(h) >= config.top_z_m:
        raise ValueError("top_z_m must lie strictly above the complete sensor surface.")

    scale = max(config.Lx_m, config.Ly_m, config.top_z_m)
    tolerance = 1e-10 * scale
    if config.periodic_x and not np.allclose(
        height(np.zeros_like(y), y),
        height(np.full_like(y, config.Lx_m), y),
        rtol=1e-9,
        atol=tolerance,
    ):
        raise ValueError("The height field does not match across periodic x faces.")
    if config.periodic_y and not np.allclose(
        height(x, np.zeros_like(x)),
        height(x, np.full_like(x, config.Ly_m)),
        rtol=1e-9,
        atol=tolerance,
    ):
        raise ValueError("The height field does not match across periodic y faces.")


def create_height_field_mesh(
    height_function: HeightFunction,
    config: FEMeshConfig,
    *,
    comm=None,
) -> FEMeshBundle:
    """Create a tetrahedral fluid mesh mapped onto ``z=h(x,y)``.

    A unit-height structured box is generated first.  Its coordinates are then
    mapped by ``z = h + zeta*(top_z-h)``.  This preserves node-to-node matching
    of opposite periodic faces.
    """

    require_fenicsx()
    config.validate()
    _validate_height_field(height_function, config)

    from dolfinx import mesh
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD
    domain = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([config.Lx_m, config.Ly_m, 1.0])],
        [config.nx, config.ny, config.nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    coordinates = domain.geometry.x
    zeta = coordinates[:, 2].copy()
    bottom = np.asarray(height_function(coordinates[:, 0], coordinates[:, 1]), dtype=float)
    coordinates[:, 2] = bottom + zeta * (config.top_z_m - bottom)

    tdim = domain.topology.dim
    fdim = tdim - 1
    tolerance = 1e-8 * max(config.Lx_m, config.Ly_m, config.top_z_m)

    def on_surface(x):
        return np.isclose(
            x[2], height_function(x[0], x[1]), rtol=1e-8, atol=tolerance
        )

    locations = {
        SURFACE_TAG: on_surface,
        TOP_TAG: lambda x: np.isclose(x[2], config.top_z_m, atol=tolerance),
        X_MIN_TAG: lambda x: np.isclose(x[0], 0.0, atol=tolerance),
        X_MAX_TAG: lambda x: np.isclose(x[0], config.Lx_m, atol=tolerance),
        Y_MIN_TAG: lambda x: np.isclose(x[1], 0.0, atol=tolerance),
        Y_MAX_TAG: lambda x: np.isclose(x[1], config.Ly_m, atol=tolerance),
    }
    facet_indices = []
    facet_values = []
    for tag, locator in locations.items():
        facets = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(facets)
        facet_values.append(np.full(facets.size, tag, dtype=np.int32))
    indices = np.concatenate(facet_indices).astype(np.int32)
    values = np.concatenate(facet_values).astype(np.int32)
    order = np.argsort(indices)
    facet_tags = mesh.meshtags(domain, fdim, indices[order], values[order])

    return FEMeshBundle(domain, facet_tags, height_function, config)


def _create_periodic_mpc(V, bcs, config: FEMeshConfig):
    """Constrain matching P1/Pk degrees of freedom on opposite x/y faces."""

    import dolfinx_mpc

    scale = max(config.Lx_m, config.Ly_m, config.top_z_m)
    tolerance = 1e-8 * scale
    mpc = dolfinx_mpc.MultiPointConstraint(V)

    if config.periodic_x:
        def x_slave(x):
            return np.logical_and(
                np.isclose(x[0], config.Lx_m, atol=tolerance),
                ~np.isclose(x[1], config.Ly_m, atol=tolerance),
            )

        def x_relation(x):
            mapped = x.copy()
            mapped[0] -= config.Lx_m
            return mapped

        mpc.create_periodic_constraint_geometrical(V, x_slave, x_relation, bcs)

    if config.periodic_y:
        def y_slave(x):
            return np.logical_and(
                np.isclose(x[1], config.Ly_m, atol=tolerance),
                ~np.isclose(x[0], config.Lx_m, atol=tolerance),
            )

        def y_relation(x):
            mapped = x.copy()
            mapped[1] -= config.Ly_m
            return mapped

        mpc.create_periodic_constraint_geometrical(V, y_slave, y_relation, bcs)

    if config.periodic_x and config.periodic_y:
        def corner_slave(x):
            return np.logical_and(
                np.isclose(x[0], config.Lx_m, atol=tolerance),
                np.isclose(x[1], config.Ly_m, atol=tolerance),
            )

        def corner_relation(x):
            mapped = x.copy()
            mapped[0] -= config.Lx_m
            mapped[1] -= config.Ly_m
            return mapped

        mpc.create_periodic_constraint_geometrical(
            V, corner_slave, corner_relation, bcs
        )

    mpc.finalize()
    return mpc


def solve_steady_capture(
    mesh_bundle: FEMeshBundle,
    config: DiffusionCaptureConfig,
) -> FEACaptureResult:
    """Solve steady diffusion and calculate local and integrated capture flux."""

    require_fenicsx()
    config.validate()

    import dolfinx_mpc
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    domain = mesh_bundle.domain
    facet_tags = mesh_bundle.facet_tags
    mesh_config = mesh_bundle.config
    fdim = domain.topology.dim - 1
    V = fem.functionspace(domain, ("Lagrange", config.polynomial_degree))

    top_facets = facet_tags.find(TOP_TAG)
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    top_bc = fem.dirichletbc(
        PETSc.ScalarType(config.c_bulk_mol_m3), top_dofs, V
    )
    bcs = [top_bc]

    if config.surface_mode == "perfect_absorber":
        surface_dofs = fem.locate_dofs_topological(
            V, fdim, facet_tags.find(SURFACE_TAG)
        )
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), surface_dofs, V))

    mpc = _create_periodic_mpc(V, bcs, mesh_config)

    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    D = fem.Constant(domain, PETSc.ScalarType(config.D_m2_s))
    zero = fem.Constant(domain, PETSc.ScalarType(0.0))

    a = D * ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
    if config.surface_mode == "finite_reactivity":
        kappa = fem.Constant(domain, PETSc.ScalarType(config.kappa_m_s))
        a += kappa * trial * test * ds(SURFACE_TAG)
    else:
        kappa = None
    L = zero * test * dx

    # dolfinx_mpc augments the original space with the master-DOF ghost map.
    # Its LinearProblem therefore requires the solution Function to live on
    # the finalized MPC space, not on the pre-constraint FunctionSpace ``V``.
    concentration = fem.Function(
        mpc.function_space, name="concentration_mol_m3"
    )
    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        mpc,
        bcs=bcs,
        u=concentration,
        petsc_options={
            "ksp_type": config.linear_solver,
            "pc_type": config.preconditioner,
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
        },
    )
    concentration = problem.solve()
    concentration.x.scatter_forward()

    normal = ufl.FacetNormal(domain)
    if config.surface_mode == "finite_reactivity":
        capture_flux_density = kappa * concentration
    else:
        capture_flux_density = -D * ufl.dot(ufl.grad(concentration), normal)
    supply_flux_density = D * ufl.dot(ufl.grad(concentration), normal)

    local_capture = fem.assemble_scalar(
        fem.form(capture_flux_density * ds(SURFACE_TAG))
    )
    local_supply = fem.assemble_scalar(
        fem.form(supply_flux_density * ds(TOP_TAG))
    )
    local_area = fem.assemble_scalar(fem.form(1.0 * ds(SURFACE_TAG)))
    comm = domain.comm
    capture_flux = float(comm.allreduce(local_capture, op=MPI.SUM).real)
    supply_flux = float(comm.allreduce(local_supply, op=MPI.SUM).real)
    surface_area = float(comm.allreduce(local_area, op=MPI.SUM).real)

    Q = fem.functionspace(domain, ("DG", 0))
    q = ufl.TestFunction(Q)
    integrated_flux = fem_petsc.assemble_vector(
        fem.form(capture_flux_density * q * ds(SURFACE_TAG))
    )
    integrated_area = fem_petsc.assemble_vector(
        fem.form(1.0 * q * ds(SURFACE_TAG))
    )
    integrated_flux.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
    )
    integrated_area.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
    )
    surface_flux = fem.Function(Q, name="surface_capture_flux_mol_m2_s")
    surface_measure = fem.Function(Q, name="surface_boundary_area_m2")
    flux_array = integrated_flux.array_r
    area_array = integrated_area.array_r
    owned = area_array.size
    surface_measure.x.array[:owned] = area_array[:owned]
    surface_flux.x.array[:owned] = np.divide(
        flux_array[:owned],
        area_array[:owned],
        out=np.zeros(owned, dtype=surface_flux.x.array.dtype),
        where=area_array[:owned] > 0,
    )
    surface_flux.x.scatter_forward()
    surface_measure.x.scatter_forward()

    denominator = max(abs(capture_flux), abs(supply_flux), np.finfo(float).tiny)
    mass_balance_error = abs(capture_flux - supply_flux) / denominator
    return FEACaptureResult(
        concentration=concentration,
        surface_flux_density=surface_flux,
        surface_boundary_measure=surface_measure,
        mesh_bundle=mesh_bundle,
        config=config,
        capture_flux_mol_s=capture_flux,
        supply_flux_mol_s=supply_flux,
        capture_rate_m3_s=capture_flux / config.c_bulk_mol_m3,
        surface_area_m2=surface_area,
        mass_balance_relative_error=mass_balance_error,
        mpc=mpc,
    )


def result_summary(result: FEACaptureResult) -> dict[str, float | str]:
    """Return a serializable one-row summary."""

    return {
        "surface_mode": result.config.surface_mode,
        "D_m2_s": result.config.D_m2_s,
        "c_bulk_mol_m3": result.config.c_bulk_mol_m3,
        "kappa_m_s": result.config.kappa_m_s,
        "capture_flux_mol_s": result.capture_flux_mol_s,
        "supply_flux_mol_s": result.supply_flux_mol_s,
        "capture_rate_m3_s": result.capture_rate_m3_s,
        "surface_area_m2": result.surface_area_m2,
        "mass_balance_relative_error": result.mass_balance_relative_error,
    }


def write_fea_result(result: FEACaptureResult, output_directory: str | Path) -> Path:
    """Write concentration/flux fields and a JSON summary."""

    require_fenicsx()
    import json

    from dolfinx import io

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    domain = result.mesh_bundle.domain
    with io.XDMFFile(
        domain.comm, str(output_directory / "concentration.xdmf"), "w"
    ) as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(result.concentration)
    with io.XDMFFile(
        domain.comm, str(output_directory / "surface_flux.xdmf"), "w"
    ) as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(result.surface_flux_density)
        xdmf.write_function(result.surface_boundary_measure)
    if domain.comm.rank == 0:
        with (output_directory / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(result_summary(result), stream, indent=2)
    return output_directory


__all__ = [
    "DiffusionCaptureConfig",
    "EquilibriumSurfaceDelivery",
    "FEACaptureResult",
    "FEMeshBundle",
    "FEMeshConfig",
    "PlanarResistanceFit",
    "cylindrically_curved_sheet_height",
    "calculate_equilibrium_surface_delivery",
    "create_height_field_mesh",
    "fit_planar_bulk_resistance",
    "height_corrected_surface_accessibility",
    "planar_height",
    "require_fenicsx",
    "result_summary",
    "robin_velocity_from_kon_density",
    "sinusoidal_height",
    "solve_steady_capture",
    "spherical_bowl_height",
    "write_fea_result",
]
