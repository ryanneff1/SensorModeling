"""Minimal two-dimensional escape/rebinding assay.

This module is a proof-of-concept companion to :mod:`utils.escape_assay`.
It follows one tagged molecule at a time on an ``x-z`` lattice above a
periodic height profile.  The reduced model preserves diffusion, first-order
dissociation, rebinding, self/cross classification, censoring, and escape to
an absorbing bulk plane.

Association in two dimensions does not have the same dimensional conversion
as a three-dimensional molar ``k_on``.  The model therefore uses an explicit
``p_bind_contact``: the probability of binding to one available receptor
after a diffusion step ends in that receptor's reaction lattice site.  It is
intended for qualitative comparisons made with identical numerical settings,
not direct quantitative prediction from a 3D association constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EscapeAssay2DParams:
    """Physical and numerical parameters for the reduced assay."""

    width_m: float = 100e-9
    height_m: float = 120e-9
    lattice_spacing_m: float = 2e-9
    diffusion_m2_s: float = 1e-11
    k_off_s: float = 1e-2
    receptor_line_density_m_inv: float = 1e8
    p_bind_contact: float = 0.20
    escape_clearance_m: float = 30e-9
    max_time_s: float = 1000.0
    n_trials_per_receptor: int = 100
    seed: int = 20261019

    def __post_init__(self) -> None:
        positive = {
            "width_m": self.width_m,
            "height_m": self.height_m,
            "lattice_spacing_m": self.lattice_spacing_m,
            "diffusion_m2_s": self.diffusion_m2_s,
            "receptor_line_density_m_inv": self.receptor_line_density_m_inv,
            "escape_clearance_m": self.escape_clearance_m,
            "max_time_s": self.max_time_s,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Parameters must be positive: {invalid}")
        if self.k_off_s < 0:
            raise ValueError("k_off_s must be nonnegative.")
        if not 0 <= self.p_bind_contact <= 1:
            raise ValueError("p_bind_contact must lie in [0, 1].")
        if self.n_trials_per_receptor < 1:
            raise ValueError("n_trials_per_receptor must be at least one.")

    @property
    def nx(self) -> int:
        return int(round(self.width_m / self.lattice_spacing_m))

    @property
    def nz(self) -> int:
        return int(round(self.height_m / self.lattice_spacing_m)) + 1

    @property
    def dt_s(self) -> float:
        # A small stay probability avoids sitting exactly on the 2D FTCS limit.
        return 0.95 * self.lattice_spacing_m**2 / (4.0 * self.diffusion_m2_s)


@dataclass(frozen=True)
class EscapeGeometry2D:
    """Periodic height profile and a sampled receptor layout."""

    name: str
    x_m: np.ndarray
    height_m: np.ndarray
    height_index: np.ndarray
    receptor_x_index: np.ndarray
    receptor_z_index: np.ndarray
    receptor_arc_weight_m: np.ndarray
    escape_z_m: float
    metadata: Mapping[str, float | str]

    @property
    def n_receptors(self) -> int:
        return int(self.receptor_x_index.size)


@dataclass
class EscapeAssay2DResult:
    """Output tables from one two-dimensional assay."""

    params: EscapeAssay2DParams
    geometry: EscapeGeometry2D
    trajectories: pd.DataFrame
    survival: pd.DataFrame
    receptors: pd.DataFrame


def _surface_profile(
    params: EscapeAssay2DParams,
    kind: str,
    **geometry_parameters: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Return the continuous height profile for a named geometry."""

    x_m = np.arange(params.nx, dtype=float) * params.lattice_spacing_m
    center_m = 0.5 * (params.nx - 1) * params.lattice_spacing_m
    u_m = x_m - center_m
    kind = str(kind).lower()

    if kind == "planar":
        height_m = np.zeros_like(x_m)
        metadata: dict[str, float | str] = {"geometry_type": kind}
    elif kind in {"curved", "semicircular"}:
        half_width_m = max(abs(u_m.min()), abs(u_m.max()))
        if kind == "semicircular":
            radius_m = half_width_m
        else:
            radius_m = float(geometry_parameters.get("radius_m", 100e-9))
        if radius_m < half_width_m:
            raise ValueError("radius_m must span the complete periodic cell.")
        height_m = radius_m - np.sqrt(
            np.maximum(radius_m**2 - u_m**2, 0.0)
        )
        metadata = {
            "geometry_type": kind,
            "radius_m": radius_m,
            "curvature_m_inv": 1.0 / radius_m,
        }
    elif kind == "sinusoidal":
        amplitude_m = float(geometry_parameters.get("amplitude_m", 15e-9))
        wavelength_m = float(geometry_parameters.get("wavelength_m", 50e-9))
        if amplitude_m < 0 or wavelength_m <= 0:
            raise ValueError("Amplitude must be nonnegative and wavelength positive.")
        # The minimum is zero and the peak-to-valley height is 2A.
        height_m = amplitude_m * (
            1.0 + np.cos(2.0 * np.pi * x_m / wavelength_m)
        )
        metadata = {
            "geometry_type": kind,
            "amplitude_m": amplitude_m,
            "wavelength_m": wavelength_m,
        }
    else:
        raise ValueError(
            "kind must be planar, curved, semicircular, or sinusoidal."
        )
    return height_m, metadata


def make_escape_geometry_2d(
    params: EscapeAssay2DParams,
    kind: str,
    *,
    seed: Optional[int] = None,
    name: Optional[str] = None,
    **geometry_parameters: float,
) -> EscapeGeometry2D:
    """Construct a height profile and sample receptors by surface arc length."""

    continuous_height_m, metadata = _surface_profile(
        params, kind, **geometry_parameters
    )
    a_m = params.lattice_spacing_m
    height_index = np.rint(continuous_height_m / a_m).astype(np.int32)
    height_m = height_index.astype(float) * a_m
    x_m = np.arange(params.nx, dtype=float) * a_m

    # Periodic central differences provide a local arc-length weight for every
    # graph location. Sampling without replacement places at most one receptor
    # in each top-adjacent reaction site.
    previous = np.roll(height_m, 1)
    following = np.roll(height_m, -1)
    slope = (following - previous) / (2.0 * a_m)
    arc_weight_m = a_m * np.sqrt(1.0 + slope**2)
    n_receptors = max(
        1, int(round(params.receptor_line_density_m_inv * arc_weight_m.sum()))
    )
    if n_receptors > params.nx:
        raise ValueError(
            "The requested line density exceeds one receptor per x site."
        )
    rng = np.random.default_rng(params.seed if seed is None else seed)
    receptor_x = np.sort(
        rng.choice(
            params.nx,
            size=n_receptors,
            replace=False,
            p=arc_weight_m / arc_weight_m.sum(),
        )
    ).astype(np.int32)
    receptor_z = height_index[receptor_x] + 1
    escape_z_m = float(height_m.max() + params.escape_clearance_m)
    if escape_z_m >= params.height_m:
        raise ValueError(
            "The escape plane does not fit in the domain; increase height_m."
        )
    return EscapeGeometry2D(
        name=str(name or kind),
        x_m=x_m,
        height_m=height_m,
        height_index=height_index,
        receptor_x_index=receptor_x,
        receptor_z_index=receptor_z,
        receptor_arc_weight_m=arc_weight_m[receptor_x],
        escape_z_m=escape_z_m,
        metadata=metadata,
    )


def _kaplan_meier_2d(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Calculate the escape-time Kaplan--Meier estimate."""

    rows: list[dict[str, float | int | str]] = []
    for geometry_name, group in trajectories.groupby("geometry", sort=False):
        times = group["observation_time_s"].to_numpy(float)
        events = group["escaped"].to_numpy(bool)
        rows.append(
            {
                "geometry": geometry_name,
                "time_s": 0.0,
                "survival_probability": 1.0,
                "n_at_risk": int(times.size),
                "n_events": 0,
                "n_censored": 0,
            }
        )
        survival = 1.0
        for time_s in np.unique(times):
            at_time = times == time_s
            n_at_risk = int(np.sum(times >= time_s))
            n_events = int(np.sum(events & at_time))
            n_censored = int(np.sum((~events) & at_time))
            if n_events:
                survival *= 1.0 - n_events / n_at_risk
            rows.append(
                {
                    "geometry": geometry_name,
                    "time_s": float(time_s),
                    "survival_probability": float(survival),
                    "n_at_risk": n_at_risk,
                    "n_events": n_events,
                    "n_censored": n_censored,
                }
            )
    return pd.DataFrame(rows)


def run_escape_assay_2d(
    params: EscapeAssay2DParams,
    geometry: EscapeGeometry2D,
    *,
    seed: Optional[int] = None,
) -> EscapeAssay2DResult:
    """Release from every receptor and follow diffusion until escape/censoring."""

    if geometry.n_receptors == 0:
        raise ValueError("At least one receptor is required.")
    rng = np.random.default_rng(params.seed + 1 if seed is None else seed)
    dt_s = params.dt_s
    max_steps = int(np.ceil(params.max_time_s / dt_s))
    escape_index = int(np.ceil(geometry.escape_z_m / params.lattice_spacing_m))
    p_axis = params.diffusion_m2_s * dt_s / params.lattice_spacing_m**2
    move_probabilities = np.array(
        [1.0 - 4.0 * p_axis, p_axis, p_axis, p_axis, p_axis], dtype=float
    )
    moves = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    p_off = 1.0 - np.exp(-params.k_off_s * dt_s)

    site_to_receptors: dict[tuple[int, int], list[int]] = {}
    for receptor_id, (x_index, z_index) in enumerate(
        zip(geometry.receptor_x_index, geometry.receptor_z_index)
    ):
        site_to_receptors.setdefault((int(x_index), int(z_index)), []).append(
            receptor_id
        )

    rows: list[dict[str, float | int | bool | str]] = []
    for source_receptor in range(geometry.n_receptors):
        source_x = int(geometry.receptor_x_index[source_receptor])
        source_z = int(geometry.receptor_z_index[source_receptor])
        for trial_id in range(params.n_trials_per_receptor):
            x_index, z_index = source_x, source_z
            elapsed_steps = 0
            total_bound_steps = 0
            n_rebindings = 0
            n_self = 0
            n_cross = 0
            bound_receptor = source_receptor
            last_released_receptor = source_receptor
            escaped = False

            while elapsed_steps < max_steps and not escaped:
                if bound_receptor >= 0:
                    if p_off <= 0:
                        wait_steps = max_steps - elapsed_steps
                    else:
                        wait_steps = int(rng.geometric(p_off))
                    actual_wait = min(wait_steps, max_steps - elapsed_steps)
                    elapsed_steps += actual_wait
                    total_bound_steps += actual_wait
                    if wait_steps > actual_wait or elapsed_steps >= max_steps:
                        break
                    last_released_receptor = bound_receptor
                    x_index = int(geometry.receptor_x_index[bound_receptor])
                    z_index = int(geometry.receptor_z_index[bound_receptor])
                    bound_receptor = -1
                    continue

                move_index = int(rng.choice(5, p=move_probabilities))
                proposed_x = (x_index + int(moves[move_index, 0])) % params.nx
                proposed_z = z_index + int(moves[move_index, 1])
                # The profile and the upper non-escape domain boundary reflect.
                if (
                    proposed_z <= int(geometry.height_index[proposed_x])
                    or proposed_z < 0
                    or proposed_z >= params.nz
                ):
                    proposed_x, proposed_z = x_index, z_index
                x_index, z_index = proposed_x, proposed_z
                elapsed_steps += 1

                if z_index >= escape_index:
                    escaped = True
                    break

                candidates = site_to_receptors.get((x_index, z_index), [])
                if candidates:
                    combined_probability = 1.0 - (
                        1.0 - params.p_bind_contact
                    ) ** len(candidates)
                    if rng.random() < combined_probability:
                        selected = int(rng.choice(candidates))
                        n_rebindings += 1
                        if selected == last_released_receptor:
                            n_self += 1
                        else:
                            n_cross += 1
                        bound_receptor = selected

            observation_time_s = min(elapsed_steps * dt_s, params.max_time_s)
            rows.append(
                {
                    "geometry": geometry.name,
                    "source_receptor_id": source_receptor,
                    "source_x_m": source_x * params.lattice_spacing_m,
                    "source_z_m": source_z * params.lattice_spacing_m,
                    "trial_id": trial_id,
                    "escaped": escaped,
                    "censored": not escaped,
                    "observation_time_s": observation_time_s,
                    "escape_time_s": observation_time_s if escaped else np.nan,
                    "total_bound_time_s": total_bound_steps * dt_s,
                    "n_rebindings": n_rebindings,
                    "n_self_rebindings": n_self,
                    "n_cross_rebindings": n_cross,
                }
            )

    trajectories = pd.DataFrame(rows)
    receptors = pd.DataFrame(
        {
            "geometry": geometry.name,
            "receptor_id": np.arange(geometry.n_receptors),
            "x_m": geometry.receptor_x_index * params.lattice_spacing_m,
            "z_m": geometry.receptor_z_index * params.lattice_spacing_m,
            "arc_weight_m": geometry.receptor_arc_weight_m,
        }
    )
    return EscapeAssay2DResult(
        params=params,
        geometry=geometry,
        trajectories=trajectories,
        survival=_kaplan_meier_2d(trajectories),
        receptors=receptors,
    )


def run_geometry_comparison_2d(
    params: Optional[EscapeAssay2DParams] = None,
    *,
    geometry_specs: Optional[Iterable[Mapping[str, object]]] = None,
) -> tuple[dict[str, EscapeAssay2DResult], pd.DataFrame]:
    """Run the standard planar/curved/semicircular/sinusoidal comparison."""

    if params is None:
        params = EscapeAssay2DParams()
    if geometry_specs is None:
        geometry_specs = [
            {"kind": "planar", "name": "Planar"},
            {"kind": "curved", "name": "Curved", "radius_m": 100e-9},
            {"kind": "semicircular", "name": "Semicircular"},
            {
                "kind": "sinusoidal",
                "name": "Sinusoidal",
                "amplitude_m": 15e-9,
                "wavelength_m": 50e-9,
            },
        ]

    results: dict[str, EscapeAssay2DResult] = {}
    seed_sequence = np.random.SeedSequence(params.seed)
    specifications = [dict(specification) for specification in geometry_specs]
    child_seeds = seed_sequence.spawn(2 * len(specifications))
    for index, specification in enumerate(specifications):
        kind = str(specification.pop("kind"))
        name = str(specification.pop("name", kind.title()))
        receptor_seed = int(child_seeds[2 * index].generate_state(1)[0])
        trajectory_seed = int(child_seeds[2 * index + 1].generate_state(1)[0])
        geometry = make_escape_geometry_2d(
            params,
            kind,
            seed=receptor_seed,
            name=name,
            **specification,
        )
        results[name] = run_escape_assay_2d(
            params, geometry, seed=trajectory_seed
        )
    trajectories = pd.concat(
        [result.trajectories for result in results.values()], ignore_index=True
    )
    return results, trajectories


def summarize_geometry_comparison_2d(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """Return one summary row per geometry."""

    return (
        trajectories.groupby("geometry", sort=False)
        .agg(
            n_trajectories=("trial_id", "size"),
            escape_probability=("escaped", "mean"),
            mean_observation_time_s=("observation_time_s", "mean"),
            median_observation_time_s=("observation_time_s", "median"),
            mean_bound_time_s=("total_bound_time_s", "mean"),
            mean_rebindings=("n_rebindings", "mean"),
            mean_self_rebindings=("n_self_rebindings", "mean"),
            mean_cross_rebindings=("n_cross_rebindings", "mean"),
        )
        .reset_index()
    )


def plot_geometries_2d(
    results: Mapping[str, EscapeAssay2DResult],
    *,
    axes=None,
):
    """Plot surface profiles, receptor positions, and escape planes."""

    import matplotlib.pyplot as plt

    if axes is None:
        figure, axes = plt.subplots(
            1, len(results), figsize=(3.2 * len(results), 3.0), sharey=True
        )
    axes_array = np.atleast_1d(axes)
    for axis, (name, result) in zip(axes_array, results.items()):
        geometry = result.geometry
        x_nm = geometry.x_m * 1e9
        axis.fill_between(x_nm, 0, geometry.height_m * 1e9, color="0.75")
        axis.plot(x_nm, geometry.height_m * 1e9, color="black", lw=1.5)
        axis.scatter(
            geometry.receptor_x_index * result.params.lattice_spacing_m * 1e9,
            geometry.receptor_z_index * result.params.lattice_spacing_m * 1e9,
            s=12,
            color="#B2182B",
            zorder=3,
        )
        axis.axhline(geometry.escape_z_m * 1e9, color="#2166AC", ls="--")
        axis.set(title=name, xlabel="x (nm)", xlim=(x_nm.min(), x_nm.max()))
    axes_array[0].set_ylabel("z (nm)")
    return axes_array


def plot_survival_comparison_2d(
    results: Mapping[str, EscapeAssay2DResult],
    *,
    include_classical: bool = True,
    ax=None,
):
    """Plot geometry-level Kaplan--Meier retention curves.

    When ``include_classical`` is true, the plot also shows
    ``exp(-k_off*t)``.  This is the survival function for one initially bound
    molecule whose first dissociation causes immediate, irreversible loss;
    it is therefore the no-rebinding classical kinetic reference.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.0))
    for name, result in results.items():
        curve = result.survival
        ax.step(
            curve["time_s"],
            curve["survival_probability"],
            where="post",
            label=name,
        )
    if include_classical and results:
        first_result = next(iter(results.values()))
        k_off_s = float(first_result.params.k_off_s)
        maximum_time_s = max(
            float(result.params.max_time_s) for result in results.values()
        )
        positive_start_s = max(
            min(result.params.dt_s for result in results.values()),
            maximum_time_s * 1e-8,
        )
        classical_time_s = np.concatenate(
            ([0.0], np.geomspace(positive_start_s, maximum_time_s, 500))
        )
        lifetime_label = "infinite" if k_off_s == 0 else f"{1 / k_off_s:g} s"
        ax.plot(
            classical_time_s,
            np.exp(-k_off_s * classical_time_s),
            color="black",
            linestyle=":",
            linewidth=2.0,
            label=rf"Classical ($1/k_{{\mathrm{{off}}}}={lifetime_label}$)",
        )
    ax.set(
        xlabel="Time after initial capture (s)",
        ylabel="Probability not yet escaped",
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False)
    return ax


__all__ = [
    "EscapeAssay2DParams",
    "EscapeAssay2DResult",
    "EscapeGeometry2D",
    "make_escape_geometry_2d",
    "plot_geometries_2d",
    "plot_survival_comparison_2d",
    "run_escape_assay_2d",
    "run_geometry_comparison_2d",
    "summarize_geometry_comparison_2d",
]
