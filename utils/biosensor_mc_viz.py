import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def predicted_occupancy_curve(
    t_s,
    ligand_conc_M,
    k_on_M_inv_s,
    k_off_s,
    theta0=0.0,
):
    """
    Well-mixed binding prediction.

    dtheta/dt = k_on*C*(1 - theta) - k_off*theta

    Parameters
    ----------
    t_s : array-like
        Time in seconds.
    ligand_conc_M : float
        Ligand concentration in mol/L (M).
    k_on_M_inv_s : float
        Association rate in M^-1 s^-1.
    k_off_s : float
        Dissociation rate in s^-1.
    theta0 : float, default=0.0
        Initial fractional occupancy.

    Returns
    -------
    np.ndarray
        Predicted fractional occupancy.
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
    figsize=(6, 4),
    auto_ylim=False,
    show_bound_number=False,
    show_langmuir=True,
    theta0=0.0,
):
    """
    Plot simulated receptor occupancy over time.

    If P is provided, overlay the well-mixed Langmuir prediction using:
        P.ligand_conc_M
        P.k_on_M_inv_s
        P.k_off_s

    Notes
    -----
    M in variable names means molar concentration (mol/L), not meters.
    Therefore, the concentration and kinetic parameters do not require
    conversion when the spatial coordinates are changed to meters.
    """
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
            label="Predicted binding",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction of receptors bound")

    if not auto_ylim:
        ax.set_ylim(0, 1.05)

    ax.grid(True)

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

    ax.set_title("Sensor occupancy vs time")

    plt.tight_layout()
    return fig, ax


def plot_state_3d(
    frame,
    figsize=(7, 6),
    ligand_size=30,
    receptor_size=60,
):
    """
    Plot a single simulation-state frame using spatial coordinates in meters.

    Required frame keys
    -------------------
    receptor_xy_m : array-like
        Receptor x-y coordinates in meters.
    ligand_xyz_m : array-like
        Free-ligand x-y-z coordinates in meters.
    Lx_m, Ly_m, H_m : float
        Simulation dimensions in meters.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    receptor_xy = frame["receptor_xy_m"]
    bound = frame["receptor_bound"]
    ligands = frame["ligand_xyz_m"]

    if np.any(~bound):
        ax.scatter(
            receptor_xy[~bound, 0],
            receptor_xy[~bound, 1],
            np.zeros(np.sum(~bound)),
            s=receptor_size,
            alpha=0.35,
            label="unbound receptors",
        )

    if np.any(bound):
        ax.scatter(
            receptor_xy[bound, 0],
            receptor_xy[bound, 1],
            np.zeros(np.sum(bound)),
            s=receptor_size,
            marker="^",
            alpha=0.9,
            label="bound receptors",
        )

    if ligands.size > 0:
        ax.scatter(
            ligands[:, 0],
            ligands[:, 1],
            ligands[:, 2],
            s=ligand_size,
            alpha=0.8,
            label="free ligands",
        )

    ax.set_xlim(0, frame["Lx_m"])
    ax.set_ylim(0, frame["Ly_m"])
    ax.set_zlim(0, frame["H_m"])

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    ax.set_title(
        f"t = {frame['t_s']:.3e} s | "
        f"B = {frame['B']} | "
        f"theta = {frame['theta']:.3f} | "
        f"N_free = {frame['N_free']}"
    )

    ax.legend(loc="upper left")
    plt.tight_layout()

    return fig, ax


def _extract_surface_face_centers(
    solid_mask,
    a_m,
    max_surface_points=None,
    seed=0,
):
    """
    Extract centers of solid-fluid interface faces from a voxelized geometry.

    Parameters
    ----------
    solid_mask : np.ndarray of bool, shape (Nx, Ny, Nz)
        True for solid sensor voxels and False for fluid voxels.
    a_m : float
        Lattice spacing in meters.
    max_surface_points : int or None
        Maximum number of surface-face centers returned. If the geometry
        contains more faces, a reproducible random subset is returned.
    seed : int
        Seed used for surface-point subsampling.

    Returns
    -------
    surface_xyz_m : np.ndarray, shape (N_surface_faces, 3)
        Physical coordinates of exposed solid-fluid face centers.
    """
    solid_mask = np.asarray(solid_mask, dtype=bool)

    if solid_mask.ndim != 3:
        raise ValueError("solid_mask must be a three-dimensional Boolean array.")

    surface_parts = []

    # Each tuple is:
    # (solid source slice, neighboring destination slice, direction vector)
    neighbor_definitions = [
        (
            (slice(0, -1), slice(None), slice(None)),
            (slice(1, None), slice(None), slice(None)),
            np.array([1.0, 0.0, 0.0]),
        ),
        (
            (slice(1, None), slice(None), slice(None)),
            (slice(0, -1), slice(None), slice(None)),
            np.array([-1.0, 0.0, 0.0]),
        ),
        (
            (slice(None), slice(0, -1), slice(None)),
            (slice(None), slice(1, None), slice(None)),
            np.array([0.0, 1.0, 0.0]),
        ),
        (
            (slice(None), slice(1, None), slice(None)),
            (slice(None), slice(0, -1), slice(None)),
            np.array([0.0, -1.0, 0.0]),
        ),
        (
            (slice(None), slice(None), slice(0, -1)),
            (slice(None), slice(None), slice(1, None)),
            np.array([0.0, 0.0, 1.0]),
        ),
        (
            (slice(None), slice(None), slice(1, None)),
            (slice(None), slice(None), slice(0, -1)),
            np.array([0.0, 0.0, -1.0]),
        ),
    ]

    for source_slice, destination_slice, direction in neighbor_definitions:
        # A surface face occurs where the source voxel is solid and the
        # neighboring voxel is fluid.
        interface = (
            solid_mask[source_slice]
            & ~solid_mask[destination_slice]
        )

        local_xyz = np.argwhere(interface)

        if local_xyz.size == 0:
            continue

        # Convert local indices back to full-array indices because some source
        # slices begin at index 1.
        source_offsets = np.array([
            0 if source_slice[0].start is None else source_slice[0].start,
            0 if source_slice[1].start is None else source_slice[1].start,
            0 if source_slice[2].start is None else source_slice[2].start,
        ])

        solid_xyz = local_xyz + source_offsets

        # The face center lies halfway between the solid voxel center and the
        # neighboring fluid voxel center.
        face_xyz_m = (
            solid_xyz.astype(float)
            + 0.5 * direction[None, :]
        ) * a_m

        surface_parts.append(face_xyz_m)

    if not surface_parts:
        return np.empty((0, 3), dtype=float)

    surface_xyz_m = np.vstack(surface_parts)

    if (
        max_surface_points is not None
        and surface_xyz_m.shape[0] > max_surface_points
    ):
        rng = np.random.default_rng(seed)

        selected = rng.choice(
            surface_xyz_m.shape[0],
            size=int(max_surface_points),
            replace=False,
        )

        surface_xyz_m = surface_xyz_m[selected]

    return surface_xyz_m


def animate_state_frames(
    state_frames,
    interval=300,
    figsize=(8, 7),
    ligand_size=30,
    receptor_size=60,
    unbound_receptor_alpha=0.05,
    bound_receptor_alpha=0.95,
    surface_size=4,
    surface_alpha=0.12,
    max_surface_points=50_000,
    show_surface=True,
    show_release_sites=False,
    release_site_size=16,
    release_site_alpha=0.45,
    elev=25,
    azim=-55,
    legend_outside=True,
    return_html=True,
):
    """
    Animate generalized biosensor Monte Carlo state frames.

    This version supports flat and nonplanar voxelized sensor geometries,
    including spherical caps, bowls, posts, wells, and implicit geometries.

    Parameters
    ----------
    state_frames : list of dict
        Frames returned by the generalized ``capture_state_frame`` function.
        Expected fields include:

        - ``solid_mask``
        - ``a_m``
        - ``receptor_surface_center_m``
        - ``receptor_release_xyz_m``
        - ``receptor_bound``
        - ``ligand_xyz_m``
        - ``Lx_m``
        - ``Ly_m``
        - ``H_m``
        - ``t_s``
        - ``B``
        - ``theta``
        - ``N_free``

    interval : int
        Delay between animation frames in milliseconds.
    figsize : tuple
        Matplotlib figure size.
    ligand_size : float
        Marker size for free ligands.
    receptor_size : float
        Marker size for receptors.
    surface_size : float
        Marker size for voxelized surface-face centers.
    surface_alpha : float
        Opacity of the sensor surface.
    max_surface_points : int or None
        Maximum number of surface points displayed. This only changes the
        visualization and does not alter the simulation.
    show_surface : bool
        Whether to display the solid-fluid interface.
    show_release_sites : bool
        Whether to display receptor-associated fluid reaction/release sites.
    release_site_size : float
        Marker size for reaction/release sites.
    elev : float
        Initial 3D elevation viewing angle.
    azim : float
        Initial 3D azimuth viewing angle.
    legend_outside : bool
        Place the legend outside the plotting axes.
    return_html : bool
        If True, return an IPython HTML animation. If False, return the raw
        Matplotlib FuncAnimation object.

    Returns
    -------
    IPython.display.HTML or matplotlib.animation.FuncAnimation
        Notebook-ready HTML animation or raw animation object.
    """
    if state_frames is None or len(state_frames) == 0:
        raise ValueError("state_frames must contain at least one frame.")

    first_frame = state_frames[0]

    required_fields = [
        "receptor_bound",
        "ligand_xyz_m",
        "Lx_m",
        "Ly_m",
        "H_m",
        "t_s",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in first_frame
    ]

    if missing_fields:
        raise KeyError(
            "State frames are missing required fields: "
            + ", ".join(missing_fields)
        )

    # ------------------------------------------------------------------
    # Extract the static geometry only once.
    # ------------------------------------------------------------------

    surface_xyz_m = np.empty((0, 3), dtype=float)

    if show_surface:
        if "solid_mask" not in first_frame:
            raise KeyError(
                "show_surface=True requires each frame to contain "
                "'solid_mask'. Regenerate the frames using the generalized "
                "capture_state_frame function."
            )

        a_m = float(first_frame["a_m"])

        surface_xyz_m = _extract_surface_face_centers(
            solid_mask=first_frame["solid_mask"],
            a_m=a_m,
            max_surface_points=max_surface_points,
            seed=0,
        )

    # Use the physical centers of receptor-bearing surface faces.
    if "receptor_surface_center_m" in first_frame:
        static_receptor_xyz_m = np.asarray(
            first_frame["receptor_surface_center_m"],
            dtype=float,
        )
    elif "receptor_xyz_m" in first_frame:
        static_receptor_xyz_m = np.asarray(
            first_frame["receptor_xyz_m"],
            dtype=float,
        )
    elif "receptor_xy_m" in first_frame:
        # Backward-compatible fallback for old planar frames.
        receptor_xy_m = np.asarray(
            first_frame["receptor_xy_m"],
            dtype=float,
        )

        static_receptor_xyz_m = np.column_stack([
            receptor_xy_m,
            np.zeros(receptor_xy_m.shape[0]),
        ])
    else:
        raise KeyError(
            "Frames must contain 'receptor_surface_center_m', "
            "'receptor_xyz_m', or the older 'receptor_xy_m' field."
        )

    release_xyz_m = np.asarray(
        first_frame.get(
            "receptor_release_xyz_m",
            np.empty((0, 3)),
        ),
        dtype=float,
    )

    Lx_m = float(first_frame["Lx_m"])
    Ly_m = float(first_frame["Ly_m"])
    H_m = float(first_frame["H_m"])

    # Matplotlib's 3D axes should use the physical domain dimensions rather
    # than visually stretching every axis to the same screen length.
    box_aspect = (
        max(Lx_m, np.finfo(float).eps),
        max(Ly_m, np.finfo(float).eps),
        max(H_m, np.finfo(float).eps),
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()

        frame = state_frames[i]

        receptor_bound = np.asarray(
            frame["receptor_bound"],
            dtype=bool,
        )

        ligand_xyz_m = np.asarray(
            frame["ligand_xyz_m"],
            dtype=float,
        )

        # Prefer receptor positions stored in the current frame, although
        # receptor positions should normally be static.
        if "receptor_surface_center_m" in frame:
            receptor_xyz_m = np.asarray(
                frame["receptor_surface_center_m"],
                dtype=float,
            )
        elif "receptor_xyz_m" in frame:
            receptor_xyz_m = np.asarray(
                frame["receptor_xyz_m"],
                dtype=float,
            )
        else:
            receptor_xyz_m = static_receptor_xyz_m

        if receptor_xyz_m.shape[0] != receptor_bound.shape[0]:
            raise ValueError(
                "The number of receptor positions does not match the number "
                "of receptor-bound state values."
            )

        # --------------------------------------------------------------
        # Static sensor geometry
        # --------------------------------------------------------------

        if show_surface and surface_xyz_m.size > 0:
            ax.scatter(
                surface_xyz_m[:, 0],
                surface_xyz_m[:, 1],
                surface_xyz_m[:, 2],
                s=surface_size,
                alpha=surface_alpha,
                marker="s",
                linewidths=0,
                label="sensor surface",
            )

        # --------------------------------------------------------------
        # Receptors
        # --------------------------------------------------------------

        unbound_mask = ~receptor_bound

        if np.any(unbound_mask):
            ax.scatter(
                receptor_xyz_m[unbound_mask, 0],
                receptor_xyz_m[unbound_mask, 1],
                receptor_xyz_m[unbound_mask, 2],
                s=receptor_size,
                alpha=unbound_receptor_alpha,
                marker="o",
                edgecolors="none",
                label="unbound receptors",
            )

        if np.any(receptor_bound):
            ax.scatter(
                receptor_xyz_m[receptor_bound, 0],
                receptor_xyz_m[receptor_bound, 1],
                receptor_xyz_m[receptor_bound, 2],
                s=receptor_size,
                alpha=bound_receptor_alpha,
                marker="^",
                edgecolors="none",
                label="bound receptors",
            )

        # --------------------------------------------------------------
        # Receptor reaction/release sites
        # --------------------------------------------------------------

        if show_release_sites and release_xyz_m.size > 0:
            ax.scatter(
                release_xyz_m[:, 0],
                release_xyz_m[:, 1],
                release_xyz_m[:, 2],
                s=release_site_size,
                alpha=release_site_alpha,
                marker="x",
                label="reaction/release sites",
            )

        # --------------------------------------------------------------
        # Free ligands
        # --------------------------------------------------------------

        if ligand_xyz_m.size > 0:
            if ligand_xyz_m.ndim != 2 or ligand_xyz_m.shape[1] != 3:
                raise ValueError(
                    "frame['ligand_xyz_m'] must have shape (N, 3)."
                )

            ax.scatter(
                ligand_xyz_m[:, 0],
                ligand_xyz_m[:, 1],
                ligand_xyz_m[:, 2],
                s=ligand_size,
                alpha=0.8,
                marker=".",
                label="free ligands",
            )

        # --------------------------------------------------------------
        # Axes
        # --------------------------------------------------------------

        ax.set_xlim(0.0, Lx_m)
        ax.set_ylim(0.0, Ly_m)
        ax.set_zlim(0.0, H_m)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        ax.set_box_aspect(box_aspect)
        ax.view_init(elev=elev, azim=azim)

        # --------------------------------------------------------------
        # Frame title
        # --------------------------------------------------------------

        geometry_name = frame.get("geometry_name", "sensor geometry")
        bound_count = int(
            frame.get(
                "B",
                np.count_nonzero(receptor_bound),
            )
        )

        if "theta" in frame:
            theta = float(frame["theta"])
        elif receptor_bound.size > 0:
            theta = float(np.mean(receptor_bound))
        else:
            theta = np.nan

        free_count = int(
            frame.get(
                "N_free",
                ligand_xyz_m.shape[0],
            )
        )

        open_watches = int(
            frame.get(
                "N_open_rebinding_watches",
                0,
            )
        )

        if np.isfinite(theta):
            theta_text = f"{theta:.3f}"
        else:
            theta_text = "NA"

        ax.set_title(
            f"{geometry_name} | frame {i + 1}/{len(state_frames)}\n"
            f"t = {frame['t_s']:.3e} s | "
            f"B = {bound_count} | "
            f"theta = {theta_text} | "
            f"N_free = {free_count} | "
            f"rebinding watches = {open_watches}"
        )

        handles, labels = ax.get_legend_handles_labels()

        if handles:
            if legend_outside:
                ax.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                )
            else:
                ax.legend(
                    handles,
                    labels,
                    loc="upper left",
                )

        return ax.collections

    ani = FuncAnimation(
        fig,
        update,
        frames=len(state_frames),
        interval=interval,
        blit=False,
        repeat=True,
    )

    if legend_outside:
        fig.subplots_adjust(right=0.76)
    else:
        fig.tight_layout()

    # Draw the first frame before closing the static notebook figure.
    update(0)
    plt.close(fig)

    if return_html:
        return HTML(ani.to_jshtml())

    return ani

def plot_ligand_count(history, figsize=(7, 4)):
    """Plot active, free, and bound ligand counts over time."""
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

    ax.set_xlabel("time (s)")
    ax.set_ylabel("ligand count")
    ax.set_title("Ligands in simulation volume")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return fig, ax


def plot_exchange_balance(history, figsize=(7, 4)):
    """Plot ligand entries and losses per recorded time interval."""
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

    ax.set_xlabel("time (s)")
    ax.set_ylabel("ligand count per record interval")
    ax.set_title("Bulk exchange balance")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return fig, ax

def plot_rebinding_events_over_time(
    history,
    ax=None,
    show_total=True,
    show_self=True,
    show_cross=True,
    normalize=False,
    time_scale="linear",
    title="Cumulative rebinding events",
    figsize=(7, 5),
):
    """
    Plot cumulative rebinding events over simulation time.

    Parameters
    ----------
    history : pandas.DataFrame
        Simulation history returned by ``run_simulation``. Expected columns
        include:

        - ``t_s``
        - ``rebinding_events_total``
        - ``self_rebindings_total``
        - ``cross_rebindings_total``
        - ``unbinding_events_total`` when normalize=True

    ax : matplotlib.axes.Axes or None
        Existing axes on which to draw. If None, a new figure and axes are
        created.

    show_total : bool
        Plot all rebinding events.

    show_self : bool
        Plot self-rebinding events, where the ligand returns to the receptor
        from which it dissociated.

    show_cross : bool
        Plot cross-rebinding events, where the ligand binds a different
        receptor.

    normalize : bool
        If False, plot cumulative event counts.

        If True, divide each cumulative rebinding count by the cumulative
        number of unbinding events. The result is then a cumulative rebinding
        fraction rather than a raw count.

    time_scale : {"linear", "log"}
        Scale used for the time axis.

    title : str or None
        Plot title. Use None to omit the title.

    figsize : tuple
        Figure size used when ax is not supplied.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure.

    ax : matplotlib.axes.Axes
        Matplotlib axes.
    """
    required_columns = {"t_s"}

    if show_total:
        required_columns.add("rebinding_events_total")

    if show_self:
        required_columns.add("self_rebindings_total")

    if show_cross:
        required_columns.add("cross_rebindings_total")

    if normalize:
        required_columns.add("unbinding_events_total")

    missing_columns = required_columns.difference(history.columns)

    if missing_columns:
        raise KeyError(
            "The history DataFrame is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if not any([show_total, show_self, show_cross]):
        raise ValueError(
            "At least one of show_total, show_self, or show_cross must be True."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    time_s = history["t_s"].to_numpy(dtype=float)

    if normalize:
        unbinding_count = history[
            "unbinding_events_total"
        ].to_numpy(dtype=float)

        # A fraction is undefined before the first unbinding event. Use NaN
        # there so Matplotlib does not display a misleading value.
        denominator = np.where(
            unbinding_count > 0,
            unbinding_count,
            np.nan,
        )

        y_label = "Cumulative rebinding fraction"
    else:
        denominator = 1.0
        y_label = "Cumulative event count"

    if show_total:
        values = (
            history["rebinding_events_total"].to_numpy(dtype=float)
            / denominator
        )

        ax.step(
            time_s,
            values,
            where="post",
            linewidth=2,
            label="all rebindings",
        )

    if show_self:
        values = (
            history["self_rebindings_total"].to_numpy(dtype=float)
            / denominator
        )

        ax.step(
            time_s,
            values,
            where="post",
            linewidth=1.7,
            label="self-rebindings",
        )

    if show_cross:
        values = (
            history["cross_rebindings_total"].to_numpy(dtype=float)
            / denominator
        )

        ax.step(
            time_s,
            values,
            where="post",
            linewidth=1.7,
            label="cross-rebindings",
        )

    ax.set_xscale(time_scale)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    ax.grid(alpha=0.25)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()

    return fig, ax

# -----------------------------------------------------------------------------
# Generalized geometry and receptor inspection
# -----------------------------------------------------------------------------


def prepare_geometry_inspection(P, geometry=None):
    """Derive a geometry and initialize its receptor distribution without running.

    Parameters
    ----------
    P : biosensor_mc.Params
        Simulation parameters. ``P.seed`` controls receptor placement.
    geometry : biosensor_mc.SensorGeometry or None
        Generalized geometry. ``None`` uses the default flat geometry.

    Returns
    -------
    G : biosensor_mc.Derived
        Derived geometry and simulation quantities.
    S : biosensor_mc.State
        Initialized state containing the exact receptor distribution that a
        subsequent simulation initialized with the same ``P`` and ``geometry``
        will use.
    """
    from utils.biosensor_mc import derive, initialize

    G = derive(P, geometry=geometry)
    S = initialize(P, G)
    return G, S


def geometry_receptor_summary(P, geometry=None, G=None, S=None):
    """Return a one-row summary of geometry and receptor initialization."""
    import pandas as pd

    if G is None or S is None:
        G, S = prepare_geometry_inspection(P, geometry=geometry)

    occupied_reaction_sites = 0
    if S.receptor_release_xyz.size:
        occupied_reaction_sites = int(
            np.unique(S.receptor_release_xyz, axis=0).shape[0]
        )

    receptors_per_site = []
    if S.receptor_release_xyz.size:
        _, counts = np.unique(
            S.receptor_release_xyz,
            axis=0,
            return_counts=True,
        )
        receptors_per_site = counts

    return pd.DataFrame(
        [
            {
                "geometry": G.geometry.name,
                "grid_Nx": G.Nx,
                "grid_Ny": G.Ny,
                "grid_Nz": G.Nz,
                "lattice_spacing_m": G.a_m,
                "accessible_volume_m3": G.volume_m3,
                "reactive_surface_area_m2": G.sensing_area_m2,
                "exposed_surface_faces": G.geometry.n_surface_faces,
                "active_reactive_faces": int(G.reactive_face_ids.size),
                "receptors": G.NR,
                "occupied_reaction_sites": occupied_reaction_sites,
                "mean_receptors_per_occupied_site": (
                    float(np.mean(receptors_per_site))
                    if len(receptors_per_site)
                    else 0.0
                ),
                "max_receptors_per_occupied_site": (
                    int(np.max(receptors_per_site))
                    if len(receptors_per_site)
                    else 0
                ),
                "receptor_density_actual_m2": (
                    G.NR / G.sensing_area_m2
                    if G.sensing_area_m2 > 0
                    else np.nan
                ),
            }
        ]
    )


def _inspection_objects(P, geometry=None, G=None, S=None):
    if G is None or S is None:
        G, S = prepare_geometry_inspection(P, geometry=geometry)
    return G, S


def _set_equal_3d_axes(ax, limits):
    """Set a 3D axis to equal physical scaling when supported by Matplotlib."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    try:
        ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    except AttributeError:
        pass


def _subsample_indices(n, max_points, rng):
    if max_points is None or n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=int(max_points), replace=False))


def plot_geometry_receptors_3d(
    P,
    geometry=None,
    G=None,
    S=None,
    figsize=(8, 7),
    show_all_surface=True,
    show_reactive_surface=False,
    show_receptors=True,
    show_release_sites=False,
    show_normals=False,
    surface_size=8,
    receptor_size=42,
    release_size=18,
    normal_length_m=None,
    max_surface_points=100_000,
    max_normal_arrows=500,
    receptor_alpha=0.95,
    surface_alpha=0.12,
    reactive_alpha=0.35,
    elev=24,
    azim=-55,
):
    """Plot the generalized sensor surface and initialized receptors in 3D.

    Surface geometry is displayed using solid--fluid face centers rather than
    rendering every solid voxel. This shows the actual interfaces used for
    receptor placement and ligand reflection.

    Parameters
    ----------
    P : biosensor_mc.Params
        Simulation parameters.
    geometry : biosensor_mc.SensorGeometry or None
        Geometry to inspect. Ignored when both ``G`` and ``S`` are supplied.
    G, S : biosensor_mc.Derived, biosensor_mc.State, optional
        Precomputed objects. Supplying these guarantees inspection of an
        already initialized receptor distribution.
    show_all_surface : bool
        Plot all exposed solid--fluid faces.
    show_reactive_surface : bool
        Highlight active reactive faces adjacent to bulk-accessible fluid.
    show_receptors : bool
        Plot initialized receptors at their surface-face centers.
    show_release_sites : bool
        Plot the adjacent fluid sites used for binding and dissociation release.
    show_normals : bool
        Plot a subsample of receptor outward-normal vectors.
    max_surface_points : int or None
        Randomly subsample very large surfaces for plotting only.

    Returns
    -------
    fig, ax
    """
    G, S = _inspection_objects(P, geometry=geometry, G=G, S=S)
    geom = G.geometry
    rng = np.random.default_rng(P.seed + 10_001)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if show_all_surface and geom.n_surface_faces:
        ids = _subsample_indices(
            geom.n_surface_faces,
            max_surface_points,
            rng,
        )
        xyz = geom.surface_centers_m[ids]
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=surface_size,
            alpha=surface_alpha,
            marker="s",
            label="exposed surface",
            rasterized=True,
        )

    if show_reactive_surface and G.reactive_face_ids.size:
        ids = G.reactive_face_ids
        if max_surface_points is not None and ids.size > max_surface_points:
            ids = ids[
                _subsample_indices(ids.size, max_surface_points, rng)
            ]
        xyz = geom.surface_centers_m[ids]
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=surface_size + 2,
            alpha=reactive_alpha,
            marker="s",
            label="reactive surface",
            rasterized=True,
        )

    receptor_centers = (
        geom.surface_centers_m[S.receptor_face_id]
        if S.receptor_face_id.size
        else np.empty((0, 3), dtype=float)
    )

    if show_receptors and receptor_centers.size:
        ax.scatter(
            receptor_centers[:, 0],
            receptor_centers[:, 1],
            receptor_centers[:, 2],
            s=receptor_size,
            marker="o",
            alpha=receptor_alpha,
            label=f"receptors (N={G.NR})",
            depthshade=False,
        )

    if show_release_sites and S.receptor_release_xyz.size:
        release_m = S.receptor_release_xyz * G.a_m
        ax.scatter(
            release_m[:, 0],
            release_m[:, 1],
            release_m[:, 2],
            s=release_size,
            marker="x",
            alpha=0.8,
            label="reaction/release sites",
            depthshade=False,
        )

    if show_normals and receptor_centers.size:
        n_show = min(int(max_normal_arrows), receptor_centers.shape[0])
        ids = _subsample_indices(
            receptor_centers.shape[0],
            n_show,
            rng,
        )
        centers = receptor_centers[ids]
        normals = S.receptor_normal[ids].astype(float)
        length = G.a_m if normal_length_m is None else float(normal_length_m)
        ax.quiver(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            length=length,
            normalize=True,
            alpha=0.75,
            label="outward normals",
        )

    limits = (
        (-0.5 * G.a_m, (G.Nx - 0.5) * G.a_m),
        (-0.5 * G.a_m, (G.Ny - 0.5) * G.a_m),
        (-0.5 * G.a_m, (G.Nz + 0.5) * G.a_m),
    )
    _set_equal_3d_axes(ax, limits)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(
        f"{geom.name}: surface geometry and receptor distribution\n"
        f"reactive area = {G.sensing_area_m2:.3e} m$^2$, "
        f"receptors = {G.NR:,}"
    )
    ax.view_init(elev=elev, azim=azim)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
        )

    fig.subplots_adjust(right=0.78)
    return fig, ax


def plot_geometry_slice(
    P,
    geometry=None,
    G=None,
    S=None,
    axis="y",
    index=None,
    coordinate_m=None,
    figsize=(8, 6),
    show_receptors=True,
    show_reaction_sites=True,
    receptor_tolerance_layers=0.5,
    receptor_size=45,
    title=None,
):
    """Inspect one orthogonal lattice slice with receptor overlays.

    Exactly one of ``index`` or ``coordinate_m`` may be supplied. If neither
    is supplied, the center slice is used. Receptors are projected onto the
    slice when their face centers fall within ``receptor_tolerance_layers``
    lattice spacings of it.

    ``axis='x'`` plots y versus z, ``axis='y'`` plots x versus z, and
    ``axis='z'`` plots x versus y.
    """
    G, S = _inspection_objects(P, geometry=geometry, G=G, S=S)

    axis = str(axis).lower()
    axis_to_dim = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_dim:
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    if index is not None and coordinate_m is not None:
        raise ValueError("Supply only one of index or coordinate_m.")

    dim = axis_to_dim[axis]
    n_axis = G.grid_shape[dim]

    if coordinate_m is not None:
        index = int(round(float(coordinate_m) / G.a_m))
    elif index is None:
        index = n_axis // 2

    index = int(index)
    if not 0 <= index < n_axis:
        raise ValueError(
            f"Slice index {index} is outside axis '{axis}' range 0..{n_axis - 1}."
        )

    solid = G.geometry.solid_mask
    reactive_solid = np.zeros(G.grid_shape, dtype=bool)
    if G.reactive_face_ids.size:
        xyz = G.geometry.surface_solid_xyz[G.reactive_face_ids]
        reactive_solid[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True

    reaction_sites = G.reaction_site_mask

    if axis == "x":
        solid_slice = solid[index, :, :].T
        reactive_slice = reactive_solid[index, :, :].T
        reaction_slice = reaction_sites[index, :, :].T
        extent = (-0.5 * G.a_m, (G.Ny - 0.5) * G.a_m,
                  -0.5 * G.a_m, (G.Nz + 0.5) * G.a_m)
        xlabel, ylabel = "y (m)", "z (m)"
        plot_dims = (1, 2)
    elif axis == "y":
        solid_slice = solid[:, index, :].T
        reactive_slice = reactive_solid[:, index, :].T
        reaction_slice = reaction_sites[:, index, :].T
        extent = (-0.5 * G.a_m, (G.Nx - 0.5) * G.a_m,
                  -0.5 * G.a_m, (G.Nz + 0.5) * G.a_m)
        xlabel, ylabel = "x (m)", "z (m)"
        plot_dims = (0, 2)
    else:
        solid_slice = solid[:, :, index].T
        reactive_slice = reactive_solid[:, :, index].T
        reaction_slice = reaction_sites[:, :, index].T
        extent = (-0.5 * G.a_m, (G.Nx - 0.5) * G.a_m,
                  -0.5 * G.a_m, (G.Ny - 0.5) * G.a_m)
        xlabel, ylabel = "x (m)", "y (m)"
        plot_dims = (0, 1)

    category = np.zeros_like(solid_slice, dtype=np.int8)
    category[solid_slice] = 1
    category[reactive_slice] = 2
    if show_reaction_sites:
        category[reaction_slice] = 3

    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap([
        "white",
        "0.55",
        "0.25",
        "0.82",
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        category,
        origin="lower",
        interpolation="nearest",
        extent=extent,
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )

    legend_handles = [
        Patch(facecolor="white", edgecolor="0.7", label="fluid"),
        Patch(facecolor="0.55", label="solid"),
        Patch(facecolor="0.25", label="reactive solid"),
    ]
    if show_reaction_sites:
        legend_handles.append(Patch(facecolor="0.82", label="reaction site"))

    if show_receptors and S.receptor_face_id.size:
        centers = G.geometry.surface_centers_m[S.receptor_face_id]
        slice_coordinate = index * G.a_m
        tolerance = float(receptor_tolerance_layers) * G.a_m
        keep = np.abs(centers[:, dim] - slice_coordinate) <= tolerance
        points = centers[keep]

        if points.size:
            ax.scatter(
                points[:, plot_dims[0]],
                points[:, plot_dims[1]],
                s=receptor_size,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                label=f"receptors near slice (N={points.shape[0]})",
            )
            handles, labels = ax.get_legend_handles_labels()
            legend_handles.extend(handles)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        title
        if title is not None
        else (
            f"{G.geometry.name}: {axis} slice at index {index} "
            f"({index * G.a_m:.3e} m)"
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    fig.subplots_adjust(right=0.75)
    return fig, ax


def plot_receptor_projections(
    P,
    geometry=None,
    G=None,
    S=None,
    figsize=(15, 4.5),
    receptor_size=18,
    show_surface_outline=True,
    surface_size=2,
    surface_alpha=0.12,
):
    """Plot receptor distributions in the x--y, x--z, and y--z projections."""
    G, S = _inspection_objects(P, geometry=geometry, G=G, S=S)
    geom = G.geometry

    receptor_centers = (
        geom.surface_centers_m[S.receptor_face_id]
        if S.receptor_face_id.size
        else np.empty((0, 3), dtype=float)
    )
    surface = geom.surface_centers_m[G.reactive_face_ids]

    projections = (
        (0, 1, "x (m)", "y (m)", "x--y projection"),
        (0, 2, "x (m)", "z (m)", "x--z projection"),
        (1, 2, "y (m)", "z (m)", "y--z projection"),
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (d1, d2, xlabel, ylabel, subtitle) in zip(axes, projections):
        if show_surface_outline and surface.size:
            ax.scatter(
                surface[:, d1],
                surface[:, d2],
                s=surface_size,
                alpha=surface_alpha,
                marker="s",
                label="reactive surface",
                rasterized=True,
            )

        if receptor_centers.size:
            ax.scatter(
                receptor_centers[:, d1],
                receptor_centers[:, d2],
                s=receptor_size,
                alpha=0.85,
                label="receptors",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    axes[-1].legend(
        loc="upper left",
        bbox_to_anchor=(1.04, 1.0),
        borderaxespad=0,
    )
    fig.suptitle(
        f"{geom.name}: receptor projections (N={G.NR:,})",
        y=1.02,
    )
    fig.subplots_adjust(right=0.87, wspace=0.35)
    return fig, axes


def plot_receptor_density_by_height(
    P,
    geometry=None,
    G=None,
    S=None,
    bins=20,
    figsize=(7, 4),
):
    """Plot receptor count and reactive surface area as functions of height.

    Receptor counts and surface area are binned using surface-face-center z
    coordinates. The secondary curve shows the resulting receptor density in
    each height bin, which is useful for checking receptor placement on curved
    geometries.
    """
    G, S = _inspection_objects(P, geometry=geometry, G=G, S=S)
    geom = G.geometry

    reactive_ids = G.reactive_face_ids
    z_surface = geom.surface_centers_m[reactive_ids, 2]
    area = geom.surface_area_m2[reactive_ids]
    z_receptors = (
        geom.surface_centers_m[S.receptor_face_id, 2]
        if S.receptor_face_id.size
        else np.empty(0, dtype=float)
    )

    if np.isscalar(bins):
        n_bins = int(bins)
        if n_bins < 1:
            raise ValueError("bins must be at least 1.")
        z_min = -0.5 * G.a_m
        z_max = (G.Nz + 0.5) * G.a_m
        edges = np.linspace(z_min, z_max, n_bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)

    receptor_count, _ = np.histogram(z_receptors, bins=edges)
    surface_area, _ = np.histogram(z_surface, bins=edges, weights=area)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    density = np.divide(
        receptor_count,
        surface_area,
        out=np.full(surface_area.shape, np.nan, dtype=float),
        where=surface_area > 0,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        centers,
        receptor_count,
        width=0.9 * widths,
        alpha=0.65,
        label="receptor count",
    )
    ax.set_xlabel("Surface height z (m)")
    ax.set_ylabel("Receptors per height bin")
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        centers,
        density,
        marker="o",
        lw=1.5,
        label="local receptor density",
    )
    ax2.set_ylabel("Receptors per m$^2$")

    lines = ax.get_lines() + ax2.get_lines()
    handles = list(ax.containers) + lines
    labels = ["receptor count"] + [line.get_label() for line in lines]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.15, 1.0),
        borderaxespad=0,
    )
    ax.set_title(f"{geom.name}: receptor distribution by height")
    fig.subplots_adjust(right=0.72)
    return fig, (ax, ax2)