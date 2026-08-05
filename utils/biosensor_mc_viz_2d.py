"""
biosensor_mc_viz_2d.py

Visualization utilities for ``biosensor_mc_2d.py``.

The spatial state is shown as an x-z cross-section:

* receptors lie along the sensing surface at z = 0,
* unbound receptors are circles,
* bound receptors are triangles, and
* free ligands diffuse above the surface.

Typical use
-----------
from biosensor_mc_2d import Params, run_simulation
from biosensor_mc_viz_2d import (
    plot_occupancy,
    plot_state_2d,
    animate_state_frames,
)

P = Params(...)
history, frames = run_simulation(
    P,
    seconds=1e-5,
    save_state_frames=True,
    n_state_frames=30,
)

plot_occupancy(history, P=P)
plot_state_2d(frames[-1])
animate_state_frames(frames)
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def predicted_occupancy_curve(
    t_s,
    ligand_conc_M: float,
    k_on_M_inv_s: float,
    k_off_s: float,
    theta0: float = 0.0,
) -> np.ndarray:
    """Return the well-mixed Langmuir binding prediction.

    The governing equation is

        dtheta/dt = k_on*C*(1 - theta) - k_off*theta

    Parameters
    ----------
    t_s : array-like
        Time in seconds.
    ligand_conc_M : float
        Ligand concentration in M.
    k_on_M_inv_s : float
        Association rate constant in M^-1 s^-1.
    k_off_s : float
        Dissociation rate constant in s^-1.
    theta0 : float, default 0
        Initial fractional receptor occupancy.

    Returns
    -------
    numpy.ndarray
        Predicted fractional receptor occupancy at each time.
    """

    t_s = np.asarray(t_s, dtype=float)
    k_obs = k_on_M_inv_s * ligand_conc_M + k_off_s

    if k_obs <= 0:
        return np.full_like(t_s, theta0, dtype=float)

    theta_eq = (k_on_M_inv_s * ligand_conc_M) / k_obs
    return theta_eq + (theta0 - theta_eq) * np.exp(-k_obs * t_s)


def plot_occupancy(
    history,
    P=None,
    figsize: Tuple[float, float] = (6, 4),
    auto_ylim: bool = False,
    show_bound_number: bool = False,
    show_langmuir: bool = True,
    theta0: float = 0.0,
):
    """Plot simulated receptor occupancy over time.

    If ``P`` is supplied, the well-mixed Langmuir prediction is calculated
    using ``P.ligand_conc_M``, ``P.k_on_M_inv_s``, and ``P.k_off_s``.
    """

    required = {"t_s", "theta", "B"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"history is missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        history["t_s"],
        history["theta"],
        lw=2,
        label="Monte Carlo",
    )

    if show_langmuir and P is not None:
        theta_predicted = predicted_occupancy_curve(
            history["t_s"],
            ligand_conc_M=P.ligand_conc_M,
            k_on_M_inv_s=P.k_on_M_inv_s,
            k_off_s=P.k_off_s,
            theta0=theta0,
        )

        ax.plot(
            history["t_s"],
            theta_predicted,
            "--",
            lw=2,
            label="Well-mixed prediction",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction of receptors bound")

    if not auto_ylim:
        ax.set_ylim(0, 1.05)

    ax.grid(True)
    ax.set_title("Sensor occupancy vs time")

    if show_bound_number:
        ax2 = ax.twinx()
        ax2.plot(
            history["t_s"],
            history["B"],
            ":",
            alpha=0.7,
            label="Bound receptors",
        )
        ax2.set_ylabel("Number of bound receptors")

        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="lower right")
    else:
        ax.legend(loc="lower right")

    fig.tight_layout()
    return fig, ax


def _validate_frame(frame: Dict) -> None:
    required = {
        "receptor_x_nm",
        "receptor_bound",
        "ligand_xz_nm",
        "Lx_nm",
        "H_nm",
        "a_nm",
        "t_s",
        "B",
        "theta",
        "N_free",
    }
    missing = required.difference(frame)
    if missing:
        raise KeyError(f"state frame is missing required fields: {sorted(missing)}")


def _draw_state_2d(
    ax,
    frame: Dict,
    ligand_size: float,
    receptor_size: float,
    show_surface: bool,
    equal_aspect: bool,
    title_prefix: Optional[str] = None,
):
    """Draw one 2D state frame on an existing Matplotlib axis."""

    _validate_frame(frame)

    receptor_x = np.asarray(frame["receptor_x_nm"], dtype=float)
    bound = np.asarray(frame["receptor_bound"], dtype=bool)
    ligands = np.asarray(frame["ligand_xz_nm"], dtype=float)

    if receptor_x.ndim != 1:
        raise ValueError("frame['receptor_x_nm'] must be one-dimensional.")
    if bound.shape != receptor_x.shape:
        raise ValueError(
            "frame['receptor_bound'] must have the same shape as "
            "frame['receptor_x_nm']."
        )
    if ligands.size and (ligands.ndim != 2 or ligands.shape[1] != 2):
        raise ValueError("frame['ligand_xz_nm'] must have shape (N, 2).")

    if show_surface:
        ax.axhline(0, lw=1.5, label="sensor surface")

    if np.any(~bound):
        ax.scatter(
            receptor_x[~bound],
            np.zeros(np.sum(~bound)),
            s=receptor_size,
            alpha=0.35,
            label="unbound receptors",
            zorder=3,
        )

    if np.any(bound):
        ax.scatter(
            receptor_x[bound],
            np.zeros(np.sum(bound)),
            s=receptor_size,
            marker="^",
            alpha=0.9,
            label="bound receptors",
            zorder=4,
        )

    if ligands.size > 0:
        ax.scatter(
            ligands[:, 0],
            ligands[:, 1],
            s=ligand_size,
            alpha=0.8,
            label="free ligands",
            zorder=2,
        )

    a_nm = float(frame["a_nm"])
    Lx_nm = float(frame["Lx_nm"])
    H_nm = float(frame["H_nm"])

    # Lattice-site centers extend from x = 0 to Lx - a. Half-cell margins
    # keep receptors on the first and last sites fully visible.
    ax.set_xlim(-0.5 * a_nm, Lx_nm - 0.5 * a_nm)
    ax.set_ylim(-0.05 * max(H_nm, a_nm), H_nm + 0.5 * a_nm)

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.grid(True)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    state_title = (
        f"t = {frame['t_s']:.3e} s | "
        f"B = {frame['B']} | "
        f"theta = {frame['theta']:.3f} | "
        f"N_free = {frame['N_free']}"
    )
    ax.set_title(f"{title_prefix} | {state_title}" if title_prefix else state_title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")


def plot_state_2d(
    frame: Dict,
    figsize: Tuple[float, float] = (7, 6),
    ligand_size: float = 30,
    receptor_size: float = 60,
    show_surface: bool = True,
    equal_aspect: bool = True,
):
    """Plot a single saved 2D Monte Carlo state.

    Parameters
    ----------
    frame : dict
        One frame returned by ``biosensor_mc_2d.capture_state_frame`` or by
        ``run_simulation(..., save_state_frames=True)``.
    figsize : tuple, default (7, 6)
        Figure size in inches.
    ligand_size, receptor_size : float
        Matplotlib scatter-marker areas.
    show_surface : bool, default True
        Draw a horizontal line at z = 0.
    equal_aspect : bool, default True
        Use equal physical scaling on the x and z axes.
    """

    fig, ax = plt.subplots(figsize=figsize)
    _draw_state_2d(
        ax=ax,
        frame=frame,
        ligand_size=ligand_size,
        receptor_size=receptor_size,
        show_surface=show_surface,
        equal_aspect=equal_aspect,
    )
    fig.tight_layout()
    return fig, ax


def animate_state_frames(
    state_frames: Sequence[Dict],
    interval: int = 300,
    figsize: Tuple[float, float] = (7, 6),
    ligand_size: float = 30,
    receptor_size: float = 60,
    show_surface: bool = True,
    equal_aspect: bool = True,
    repeat: bool = True,
    return_html: bool = True,
):
    """Animate saved 2D state frames.

    Parameters
    ----------
    state_frames : sequence of dict
        Frames returned by ``run_simulation(..., save_state_frames=True)``.
    interval : int, default 300
        Delay between frames in milliseconds.
    return_html : bool, default True
        When True, return an ``IPython.display.HTML`` object suitable for a
        Jupyter notebook. When False, return the raw ``FuncAnimation`` object.
    """

    if len(state_frames) == 0:
        raise ValueError("state_frames cannot be empty.")
    if interval <= 0:
        raise ValueError("interval must be positive.")

    for frame in state_frames:
        _validate_frame(frame)

    fig, ax = plt.subplots(figsize=figsize)

    def update(i):
        ax.clear()
        _draw_state_2d(
            ax=ax,
            frame=state_frames[i],
            ligand_size=ligand_size,
            receptor_size=receptor_size,
            show_surface=show_surface,
            equal_aspect=equal_aspect,
            title_prefix=f"Frame {i + 1}/{len(state_frames)}",
        )
        return ax.get_children()

    ani = FuncAnimation(
        fig,
        update,
        frames=len(state_frames),
        interval=interval,
        repeat=repeat,
        blit=False,
    )

    plt.close(fig)

    if return_html:
        try:
            from IPython.display import HTML
        except ImportError as exc:
            raise ImportError(
                "IPython is required when return_html=True. Set "
                "return_html=False to receive the FuncAnimation object."
            ) from exc

        return HTML(ani.to_jshtml())

    return ani


def plot_ligand_count(
    history,
    figsize: Tuple[float, float] = (7, 4),
):
    """Plot active, free, and bound ligand counts over time."""

    required = {"t_s", "N_active_ligands", "N_free", "N_bound"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"history is missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        history["t_s"],
        history["N_active_ligands"],
        label="active ligands",
    )
    ax.plot(
        history["t_s"],
        history["N_free"],
        label="free ligands",
    )
    ax.plot(
        history["t_s"],
        history["N_bound"],
        label="bound ligands",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ligand count")
    ax.set_title("Ligands in represented simulation volume")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig, ax


def plot_exchange_balance(
    history,
    figsize: Tuple[float, float] = (7, 4),
):
    """Plot ligand entry and loss during each recorded interval."""

    required = {"t_s", "entered_from_bulk_total", "lost_to_bulk_total"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"history is missing required columns: {sorted(missing)}")

    h = history.copy()
    h["entered_increment"] = h["entered_from_bulk_total"].diff().fillna(0)
    h["lost_increment"] = h["lost_to_bulk_total"].diff().fillna(0)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        h["t_s"],
        h["entered_increment"],
        label="entered per record",
    )
    ax.plot(
        h["t_s"],
        h["lost_increment"],
        label="lost per record",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ligand count per record interval")
    ax.set_title("Bulk exchange balance")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig, ax


# Backward-friendly alias for users who prefer a more explicit name.
animate_state_frames_2d = animate_state_frames


__all__ = [
    "predicted_occupancy_curve",
    "plot_occupancy",
    "plot_state_2d",
    "animate_state_frames",
    "animate_state_frames_2d",
    "plot_ligand_count",
    "plot_exchange_balance",
]