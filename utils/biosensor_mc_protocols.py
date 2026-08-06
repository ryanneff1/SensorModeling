"""
biosensor_mc_protocols.py

Protocol-level simulations and validation analyses for biosensor_mc.py.

The first protocol validates the lattice diffusion kernel against free,
unbounded three-dimensional Brownian diffusion with the same diffusion
coefficient. It intentionally excludes receptors, reactions, bulk exchange,
and finite-domain boundaries so that the diffusion rule can be tested in
isolation.

Expected free-diffusion relationships
-------------------------------------
For isotropic diffusion in three dimensions:

    <dx^2> = <dy^2> = <dz^2> = 2 D t
    <r^2>  = 6 D t

The Cartesian displacement distribution is Gaussian:

    p(x, t) = 1 / sqrt(4 pi D t) * exp[-x^2 / (4 D t)]

and the radial displacement distribution is:

    p(r, t) = 4 pi r^2 / (4 pi D t)^(3/2) * exp[-r^2 / (4 D t)]

The lattice distribution is discrete and non-Gaussian at very short times,
but approaches the Brownian Gaussian propagator after many independent steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.biosensor_mc import Params, SensorGeometry, derive, run_simulation
from utils.generate_geometries import (
    make_flat_geometry,
    make_nanopore_array_geometry,
)

PathLike = Union[str, Path]
@dataclass
class FreeDiffusionResult:
    """Results from the unbounded free-diffusion validation protocol."""

    params: Params
    derived: Derived
    history: pd.DataFrame
    final_displacements_m: np.ndarray
    n_particles: int
    n_steps: int
    D_fit_m2_s: float
    D_kernel_m2_s: float
    relative_D_error: float


def _validate_diffusion_inputs(
    seconds: float,
    n_particles: int,
    n_records: int,
) -> None:
    if seconds <= 0:
        raise ValueError("seconds must be greater than zero.")
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2.")
    if n_records < 2:
        raise ValueError("n_records must be at least 2.")


def _record_diffusion_statistics(
    positions_lattice: np.ndarray,
    a_m: float,
    t_s: float,
    D_m2_s: float,
) -> dict:
    """Calculate ensemble displacement statistics at one timepoint."""

    displacement_m = positions_lattice.astype(np.float64, copy=False) * a_m
    squared_m2 = displacement_m * displacement_m

    msd_x_m2, msd_y_m2, msd_z_m2 = np.mean(squared_m2, axis=0)
    r2_m2 = np.sum(squared_m2, axis=1)
    msd_r_m2 = float(np.mean(r2_m2))

    if msd_r_m2 > 0:
        r4_m4 = float(np.mean(r2_m2 * r2_m2))
        # Three-dimensional non-Gaussian parameter. It is zero for an
        # isotropic Gaussian displacement distribution.
        alpha2_3d = 3.0 * r4_m4 / (5.0 * msd_r_m2**2) - 1.0
    else:
        alpha2_3d = np.nan

    theory_axis_m2 = 2.0 * D_m2_s * t_s
    theory_r_m2 = 6.0 * D_m2_s * t_s

    if t_s > 0:
        D_running_m2_s = msd_r_m2 / (6.0 * t_s)
        relative_msd_error = (msd_r_m2 - theory_r_m2) / theory_r_m2
    else:
        D_running_m2_s = np.nan
        relative_msd_error = np.nan

    return {
        "t_s": float(t_s),
        "msd_x_m2": float(msd_x_m2),
        "msd_y_m2": float(msd_y_m2),
        "msd_z_m2": float(msd_z_m2),
        "msd_r_m2": msd_r_m2,
        "theory_axis_m2": float(theory_axis_m2),
        "theory_r_m2": float(theory_r_m2),
        "D_running_m2_s": float(D_running_m2_s),
        "relative_msd_error": float(relative_msd_error),
        "alpha2_3d": float(alpha2_3d),
    }


def _fit_diffusion_coefficient(history: pd.DataFrame) -> float:
    """Fit <r^2> = 6 D t through the physical origin."""

    positive = history["t_s"] > 0
    t_s = history.loc[positive, "t_s"].to_numpy(dtype=float)
    msd_m2 = history.loc[positive, "msd_r_m2"].to_numpy(dtype=float)

    denominator = float(np.dot(t_s, t_s))
    if denominator <= 0:
        return np.nan

    slope_m2_s = float(np.dot(t_s, msd_m2) / denominator)
    return slope_m2_s / 6.0


def run_free_diffusion_protocol(
    P: Params,
    seconds: float,
    n_particles: int = 100_000,
    n_records: int = 101,
    seed: Optional[int] = None,
    save_history_csv: Optional[PathLike] = None,
    verbose: bool = True,
) -> FreeDiffusionResult:
    """
    Validate the simulation's lattice diffusion rule in unbounded 3D space.

    This protocol uses the exact ``dt_s`` and ``move_probs`` returned by
    ``biosensor_mc.derive(P)``. All particles begin at the origin. At each
    timestep, every particle independently either stays in place or moves by
    one lattice spacing along one Cartesian axis, exactly as in the main
    simulation.

    Receptors, binding, dissociation, bulk entry, ligand loss, and domain
    boundaries are deliberately omitted. This is necessary to compare the
    movement kernel directly with free Brownian diffusion.

    Parameters
    ----------
    P : Params
        Parameters from biosensor_mc.py. The diffusion-relevant fields are
        ``D_m2_s``, ``a_m``, ``dt_s``, and ``seed``. Other fields must still
        define a valid main simulation because ``derive(P)`` is used.
    seconds : float
        Simulated duration in seconds.
    n_particles : int, default 100000
        Number of independent particles in the ensemble.
    n_records : int, default 101
        Approximate number of recorded timepoints, including time zero and the
        final time.
    seed : int or None
        Random seed for this protocol. If None, ``P.seed`` is used.
    save_history_csv : path-like or None
        Optional output path for the summary statistics CSV.
    verbose : bool, default True
        Print a concise validation summary.

    Returns
    -------
    FreeDiffusionResult
        History, final displacement ensemble, and fitted diffusion metrics.
    """

    _validate_diffusion_inputs(seconds, n_particles, n_records)

    G = derive(P)
    rng = np.random.default_rng(P.seed if seed is None else seed)

    n_steps = int(np.ceil(seconds / G.dt_s))
    record_steps = np.unique(
        np.linspace(0, n_steps, min(n_records, n_steps + 1), dtype=np.int64)
    )
    record_steps_set = set(record_steps.tolist())

    # Unbounded integer coordinates relative to the common starting point.
    positions = np.zeros((n_particles, 3), dtype=np.int64)

    rows = [
        _record_diffusion_statistics(
            positions_lattice=positions,
            a_m=P.a_m,
            t_s=0.0,
            D_m2_s=P.D_m2_s,
        )
    ]

    for step_index in range(1, n_steps + 1):
        moves = rng.choice(7, size=n_particles, p=G.move_probs)

        positions[moves == 1, 0] += 1
        positions[moves == 2, 0] -= 1
        positions[moves == 3, 1] += 1
        positions[moves == 4, 1] -= 1
        positions[moves == 5, 2] += 1
        positions[moves == 6, 2] -= 1

        if step_index in record_steps_set:
            rows.append(
                _record_diffusion_statistics(
                    positions_lattice=positions,
                    a_m=P.a_m,
                    t_s=step_index * G.dt_s,
                    D_m2_s=P.D_m2_s,
                )
            )

    history = pd.DataFrame(rows)
    history.insert(0, "step", record_steps[: len(history)])

    D_fit_m2_s = _fit_diffusion_coefficient(history)

    # For this seven-outcome kernel, one particular positive direction has
    # probability p = D*dt/a^2. Thus p*a^2/dt recovers the represented D.
    D_kernel_m2_s = float(G.move_probs[1] * P.a_m**2 / G.dt_s)

    relative_D_error = (
        (D_fit_m2_s - P.D_m2_s) / P.D_m2_s
        if P.D_m2_s > 0
        else np.nan
    )

    final_displacements_m = positions.astype(np.float64) * P.a_m

    if save_history_csv is not None:
        output_path = Path(save_history_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(output_path, index=False)

    if verbose:
        rms_theory_m = np.sqrt(6.0 * P.D_m2_s * n_steps * G.dt_s)
        print("=" * 64)
        print("Free-diffusion validation")
        print("=" * 64)
        print(f"Particles                 : {n_particles:,}")
        print(f"Steps                     : {n_steps:,}")
        print(f"Requested duration        : {seconds:.6e} s")
        print(f"Actual duration           : {n_steps * G.dt_s:.6e} s")
        print(f"Lattice spacing           : {P.a_m:.6e} m")
        print(f"Timestep                  : {G.dt_s:.6e} s")
        print(f"Input D                   : {P.D_m2_s:.6e} m^2/s")
        print(f"Kernel D                  : {D_kernel_m2_s:.6e} m^2/s")
        print(f"MSD-fit D                 : {D_fit_m2_s:.6e} m^2/s")
        print(f"Relative fitted-D error   : {relative_D_error:.3%}")
        print(f"Theoretical final RMS     : {rms_theory_m:.6e} m")
        print("=" * 64)

    return FreeDiffusionResult(
        params=P,
        derived=G,
        history=history,
        final_displacements_m=final_displacements_m,
        n_particles=n_particles,
        n_steps=n_steps,
        D_fit_m2_s=D_fit_m2_s,
        D_kernel_m2_s=D_kernel_m2_s,
        relative_D_error=relative_D_error,
    )


def plot_msd_validation(
    result: FreeDiffusionResult,
    figsize: Tuple[float, float] = (7.5, 4.8),
    log_axes: bool = False,
):
    """Plot simulated MSD against the Brownian predictions."""

    h = result.history
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(h["t_s"], h["msd_r_m2"], lw=2, label="Simulation: total MSD")
    ax.plot(h["t_s"], h["theory_r_m2"], "--", lw=2, label=r"Theory: $6Dt$")

    ax.plot(h["t_s"], h["msd_x_m2"], alpha=0.75, label=r"Simulation: $\langle x^2\rangle$")
    ax.plot(h["t_s"], h["msd_y_m2"], alpha=0.75, label=r"Simulation: $\langle y^2\rangle$")
    ax.plot(h["t_s"], h["msd_z_m2"], alpha=0.75, label=r"Simulation: $\langle z^2\rangle$")
    ax.plot(h["t_s"], h["theory_axis_m2"], ":", lw=2, label=r"Theory: $2Dt$")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Mean-squared displacement (m$^2$)")
    ax.set_title("Free-diffusion validation")
    ax.grid(True)

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig, ax


def plot_diffusion_coefficient_convergence(
    result: FreeDiffusionResult,
    figsize: Tuple[float, float] = (7.0, 4.2),
):
    """Plot the running MSD estimate D(t) = <r^2>/(6t)."""

    h = result.history
    positive = h["t_s"] > 0

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        h.loc[positive, "t_s"],
        h.loc[positive, "D_running_m2_s"],
        lw=2,
        label="Running MSD estimate",
    )
    ax.axhline(
        result.params.D_m2_s,
        ls="--",
        lw=2,
        label="Input diffusion coefficient",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Estimated $D$ (m$^2$/s)")
    ax.set_title("Diffusion-coefficient convergence")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    return fig, ax


def plot_cartesian_displacement_distribution(
    result: FreeDiffusionResult,
    axis: str = "x",
    bins: int = 80,
    figsize: Tuple[float, float] = (7.0, 4.5),
):
    """Compare one final Cartesian displacement distribution with theory."""

    axis_lookup = {"x": 0, "y": 1, "z": 2}
    axis_key = axis.lower()
    if axis_key not in axis_lookup:
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    values_m = result.final_displacements_m[:, axis_lookup[axis_key]]
    t_final_s = result.n_steps * result.derived.dt_s
    D_m2_s = result.params.D_m2_s

    sigma_m = np.sqrt(2.0 * D_m2_s * t_final_s)
    x_theory_m = np.linspace(values_m.min(), values_m.max(), 500)
    p_theory_m_inv = (
        np.exp(-(x_theory_m**2) / (4.0 * D_m2_s * t_final_s))
        / np.sqrt(4.0 * np.pi * D_m2_s * t_final_s)
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        values_m,
        bins=bins,
        density=True,
        alpha=0.65,
        label=f"Simulated {axis_key}-displacements",
    )
    ax.plot(
        x_theory_m,
        p_theory_m_inv,
        lw=2,
        label=rf"Gaussian theory, $\sigma=\sqrt{{2Dt}}={sigma_m:.2e}$ m",
    )

    ax.set_xlabel(f"{axis_key}-displacement (m)")
    ax.set_ylabel(r"Probability density (m$^{-1}$)")
    ax.set_title(f"Final {axis_key}-displacement distribution")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    return fig, ax


def plot_radial_displacement_distribution(
    result: FreeDiffusionResult,
    bins: int = 80,
    figsize: Tuple[float, float] = (7.0, 4.5),
):
    """Compare the final radial-displacement distribution with 3D theory."""

    radii_m = np.linalg.norm(result.final_displacements_m, axis=1)
    t_final_s = result.n_steps * result.derived.dt_s
    D_m2_s = result.params.D_m2_s

    r_theory_m = np.linspace(0.0, radii_m.max(), 500)
    p_theory_m_inv = (
        4.0
        * np.pi
        * r_theory_m**2
        * np.exp(-(r_theory_m**2) / (4.0 * D_m2_s * t_final_s))
        / (4.0 * np.pi * D_m2_s * t_final_s) ** 1.5
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        radii_m,
        bins=bins,
        density=True,
        alpha=0.65,
        label="Simulated radial displacement",
    )
    ax.plot(
        r_theory_m,
        p_theory_m_inv,
        lw=2,
        label="3D Brownian radial theory",
    )

    ax.set_xlabel("Radial displacement (m)")
    ax.set_ylabel(r"Probability density (m$^{-1}$)")
    ax.set_title("Final radial-displacement distribution")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    return fig, ax


def summarize_diffusion_validation(result: FreeDiffusionResult) -> pd.DataFrame:
    """Return a one-row table of the primary validation metrics."""

    final = result.history.iloc[-1]
    return pd.DataFrame(
        [
            {
                "n_particles": result.n_particles,
                "n_steps": result.n_steps,
                "t_final_s": float(final["t_s"]),
                "D_input_m2_s": result.params.D_m2_s,
                "D_kernel_m2_s": result.D_kernel_m2_s,
                "D_fit_m2_s": result.D_fit_m2_s,
                "relative_D_error": result.relative_D_error,
                "final_msd_simulated_m2": float(final["msd_r_m2"]),
                "final_msd_theory_m2": float(final["theory_r_m2"]),
                "final_relative_msd_error": float(final["relative_msd_error"]),
                "final_alpha2_3d": float(final["alpha2_3d"]),
            }
        ]
    )


if __name__ == "__main__":
    # A practical first validation with the default 1 nm lattice spacing and
    # D = 1e-10 m^2/s. The default timestep gives roughly 632 steps in 1 us.
    parameters = Params()

    validation = run_free_diffusion_protocol(
        parameters,
        seconds=1e-6,
        n_particles=100_000,
        n_records=101,
        seed=parameters.seed,
        save_history_csv="free_diffusion_validation.csv",
    )

    print(summarize_diffusion_validation(validation).to_string(index=False))

    plot_msd_validation(validation)
    plot_diffusion_coefficient_convergence(validation)
    plot_cartesian_displacement_distribution(validation, axis="x")
    plot_radial_displacement_distribution(validation)
    plt.show()