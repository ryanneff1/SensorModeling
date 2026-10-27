#!/usr/bin/env python3
"""Rebuild manuscript Figures 2–4 around occupancy-aware C50 sweeps."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import nbformat as nbf


HERE = Path(__file__).resolve().parent


def lines(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


def code(value: str):
    return nbf.v4.new_code_cell(lines(value))


def markdown(value: str):
    return nbf.v4.new_markdown_cell(lines(value))


def formatting_from(filename: str) -> str:
    """Preserve the user's first-cell formatting verbatim."""

    with (HERE / filename).open(encoding="utf-8") as stream:
        notebook = json.load(stream)
    return "".join(notebook["cells"][0]["source"])


def notebook(filename: str, title: str, introduction: str, body: list):
    nb = nbf.v4.new_notebook(
        cells=[
            nbf.v4.new_code_cell(formatting_from(filename)),
            markdown(f"# {title}\n\n{introduction}"),
            *body,
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbf.write(nb, HERE / filename)


SETUP = r"""
from pathlib import Path
import sys
import importlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "analysis_tools":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import analysis_tools.manuscript_escape_analysis as manuscript_escape_analysis
importlib.reload(manuscript_escape_analysis)
from analysis_tools.manuscript_escape_analysis import (
    condition_statistics, discover_runs, load_receptor_summaries,
    replicate_survival_curves, save_figure, size_figure_for_axes,
    summarize_replicates,
)

SAVE_FIGURES = False
REFRESH_CACHE = False
SHOW_PANEL_TITLES = False
FIGURE_ROOT = PROJECT_ROOT / "analysis_tools" / "manuscript_panels"
CACHE_ROOT = PROJECT_ROOT / "analysis_tools" / ".analysis_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
ONEDRIVE_RESULTS_ROOT = Path(
    "/Users/rynon/Library/CloudStorage/OneDrive-NorthwesternUniversity/PhD/Data/"
    "Agent-based simulations"
)
RESULTS_ROOT = ONEDRIVE_RESULTS_ROOT if ONEDRIVE_RESULTS_ROOT.exists() else PROJECT_ROOT / "results"
print(f"Simulation results root: {RESULTS_ROOT}")

# JACS double-column figures are approximately 7 inches wide. These plot-area
# footprints are designed for panels tiled two or three across that width.
TWO_PANEL_PLOT_AREA_IN = (2.55, 2.15)
THREE_PANEL_PLOT_AREA_IN = (1.75, 1.65)
THREE_PANEL_TALL_PLOT_AREA_IN = (1.75, 2.15)
MAP_PAIR_SUBPLOT_AREA_IN = (2.40, 2.15)
PLOT_AREA_SIZE_IN = TWO_PANEL_PLOT_AREA_IN
MAP_PLOT_AREA_SIZE_IN = THREE_PANEL_PLOT_AREA_IN
SINGLE_PANEL_SIZE_IN = (3.4, 2.8)  # provisional canvas before measurement
MAP_PANEL_SIZE_IN = (2.6, 2.35)

def finish_panel(fig, ax, plot_area=PLOT_AREA_SIZE_IN, padding_in=0.08):
    if not SHOW_PANEL_TITLES:
        for current_ax in fig.axes:
            current_ax.set_title("")
    size_figure_for_axes(fig, ax, *plot_area, padding_in=padding_in)

def save_panel(fig, output_directory, stem, dpi=None):
    save_figure(fig, output_directory, stem, dpi=dpi, exact_size=True)
"""


RESPONSE_HELPERS = r"""
def add_occupancy_response_coordinates(frame):
    # Map imposed occupancy to normalized concentration for each run.
    result = frame.copy()
    phi = result["background_occupancy_fraction"].to_numpy(float)
    denominator = (
        (1 - phi)
        * result["k_off_s"].to_numpy(float)
        * result["mean_bound_time_s"].to_numpy(float)
    )
    result["C_over_KD"] = np.divide(
        phi, denominator,
        out=np.full(len(result), np.nan),
        where=denominator > 0,
    )
    return result


def interpolate_replicate_responses(frame, group_columns, x_grid):
    rows = []
    for keys, group in frame.groupby([*group_columns, "replicate"], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        usable = (group.loc[np.isfinite(group.C_over_KD) & group.C_over_KD.gt(0),
                            ["C_over_KD", "background_occupancy_fraction"]]
                  .sort_values("C_over_KD").drop_duplicates("C_over_KD"))
        if len(usable) < 2:
            continue
        source_x = np.log10(usable.C_over_KD.to_numpy(float))
        target_x = np.log10(np.asarray(x_grid, float))
        theta = np.interp(
            target_x, source_x,
            usable.background_occupancy_fraction.to_numpy(float),
        )
        theta[(target_x < source_x.min()) | (target_x > source_x.max())] = np.nan
        values = dict(zip([*group_columns, "replicate"], keys))
        rows.append(pd.DataFrame({**values, "C_over_KD": x_grid, "theta": theta}))
    if not rows:
        raise ValueError("No replicate response curves could be interpolated.")
    return pd.concat(rows, ignore_index=True)


def add_c50_metrics(frame):
    # Calculate operational C50 from simulations performed at phi=0.5.
    result = frame.copy()
    result["C50_M"] = 1 / (
        result["k_on_M_inv_s"] * result["mean_bound_time_s"]
    )
    result["C50_over_KD"] = 1 / (
        result["k_off_s"] * result["mean_bound_time_s"]
    )
    result["C50_reduction_fold"] = 1 / result["C50_over_KD"]
    return result
"""


fig2_name = "analyze_manuscript_figure2_rebinding_affinity.ipynb"
fig2_body = [
    code(SETUP + "\n" + RESPONSE_HELPERS),
    markdown(
        """
        ## 1. Load the dilute rebinding control and occupancy-aware sweep

        The earlier on/off experiment establishes the residence-time mechanism. The
        new occupancy grid then determines how that mechanism changes the complete
        concentration response and operational C50.
        """
    ),
    code(
        r"""
        ON_OFF_ROOT = RESULTS_ROOT / "261005_manuscript_fig2_rebinding_on_off"
        DATA_ROOT = RESULTS_ROOT / "261006_manuscript_fig2_occupancy_grid"
        onoff_index = discover_runs(
            ON_OFF_ROOT,
            cache_path=CACHE_ROOT / "figure2_onoff_index.csv",
            refresh_cache=REFRESH_CACHE,
        )
        run_index = discover_runs(
            DATA_ROOT,
            cache_path=CACHE_ROOT / "figure2_occupancy_index.csv",
            refresh_cache=REFRESH_CACHE,
        )
        onoff = summarize_replicates(
            onoff_index,
            cache_path=CACHE_ROOT / "figure2_onoff_replicates.csv",
            refresh_cache=REFRESH_CACHE,
        )
        replicates = summarize_replicates(
            run_index,
            cache_path=CACHE_ROOT / "figure2_occupancy_replicates.csv",
            refresh_cache=REFRESH_CACHE,
        )
        onoff["geometry"] = onoff.condition_label.str.split("__").str[0]
        onoff["rebinding_state"] = np.where(
            onoff.rebinding_multiplier.gt(0), "Rebinding on", "Rebinding off"
        )
        replicates["geometry"] = replicates.condition_label.str.split("__").str[0]
        print(f"On/off control: {len(onoff_index)} / 48 expected runs")
        print(f"Occupancy grid: {len(run_index)} / 504 expected runs")
        if len(onoff_index) != 48 or len(run_index) != 504:
            warnings.warn("Figure 2 is incomplete or the discovery cache is stale.")
        """
    ),
    markdown("## 2. Quality control and response reconstruction"),
    code(
        r"""
        response_runs = add_occupancy_response_coordinates(replicates)
        half = add_c50_metrics(
            response_runs.loc[np.isclose(
                response_runs.background_occupancy_fraction, 0.5
            )].copy()
        )
        assert half.groupby("geometry").size().eq(8).all()
        geometry_order = ["planar", "R100nm", "R49p5nm"]
        geometry_names = {
            "planar": "Planar",
            "R100nm": "Curved sheet",
            "R49p5nm": "Semicircular sheet",
        }
        onoff_geometry_names = {
            "planar": "Planar",
            "moderate_R100nm": "Curved sheet",
            "semicircular_R49p5nm": "Semicircular sheet",
        }
        colors = dict(zip(
            geometry_order,
            plt.cm.viridis(np.linspace(0.12, 0.88, len(geometry_order))),
        ))
        onoff_colors = {
            "planar": colors["planar"], "moderate_R100nm": colors["R100nm"],
            "semicircular_R49p5nm": colors["R49p5nm"],
        }
        concentration_grid = np.logspace(-2, 1, 241)
        curves = interpolate_replicate_responses(
            response_runs, ["geometry"], concentration_grid
        )
        onoff_diagnostic = (onoff.groupby(["geometry","rebinding_state"])
                            .agg(mean_bound_time_s_mean=("mean_bound_time_s","mean"),
                                 mean_bound_time_s_sem=("mean_bound_time_s","sem"),
                                 mean_rebindings_mean=("mean_rebindings","mean"),
                                 mean_rebindings_sem=("mean_rebindings","sem"))
                            .reset_index())
        display(onoff_diagnostic)
        """
    ),
    markdown(
        r"""
        ## Geometry reference panels

        These three panels show the sensor surfaces used throughout Figure 2 at a
        common scale and viewing angle. The curved-sheet radius is documented in
        the figure caption rather than in the panel text.
        """
    ),
    code(
        r"""
        # Smooth analytical renderings of the same height fields used to create
        # the voxelized simulation geometries. Keeping the limits and camera fixed
        # makes curvature directly comparable between panels.
        sheet_x_nm = np.linspace(-49.5, 49.5, 121)
        sheet_y_nm = np.linspace(-49.5, 49.5, 81)
        SHEET_X_NM, SHEET_Y_NM = np.meshgrid(sheet_x_nm, sheet_y_nm)

        def cylindrical_sheet_height_nm(radius_nm, vertex_nm):
            if np.isinf(radius_nm):
                return np.full_like(SHEET_X_NM, vertex_nm)
            return vertex_nm + radius_nm - np.sqrt(
                np.maximum(radius_nm**2 - SHEET_X_NM**2, 0.0)
            )

        def plot_geometry_reference(height_nm, title, color, stem):
            fig = plt.figure(figsize=MAP_PANEL_SIZE_IN, constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(
                SHEET_X_NM, SHEET_Y_NM, height_nm,
                color=color, alpha=0.96, linewidth=0, antialiased=True,
                shade=True, rcount=61, ccount=81,
            )
            # Sparse contours clarify the bend without making the rendering look
            # like a different lattice resolution from the simulation.
            ax.plot_wireframe(
                SHEET_X_NM, SHEET_Y_NM, height_nm,
                color="white", alpha=0.28, linewidth=0.35,
                rstride=16, cstride=20,
            )
            ax.set(xlim=(-52, 52), ylim=(-52, 52), zlim=(0, 55), title=title)
            ax.set_box_aspect((104, 104, 55))
            ax.view_init(elev=25, azim=-58)
            ax.set_axis_off()
            finish_panel(fig, ax, plot_area=MAP_PLOT_AREA_SIZE_IN)
            if SAVE_FIGURES:
                save_panel(fig, FIGURE_ROOT, stem)
            plt.show()
        """
    ),
    markdown("### Planar"),
    code(
        r"""
        plot_geometry_reference(
            cylindrical_sheet_height_nm(np.inf, 50.0),
            "Planar", colors["planar"], "figure2_geometry_planar",
        )
        """
    ),
    markdown("### Curved sheet"),
    code(
        r"""
        plot_geometry_reference(
            cylindrical_sheet_height_nm(100.0, 36.88929738466067),
            "Curved sheet", colors["R100nm"], "figure2_geometry_curved_sheet",
        )
        """
    ),
    markdown("### Semicircular sheet"),
    code(
        r"""
        plot_geometry_reference(
            cylindrical_sheet_height_nm(49.5, 0.5),
            "Semicircular sheet", colors["R49p5nm"],
            "figure2_geometry_semicircular_sheet",
        )
        """
    ),
    markdown(
        r"""
        ## 3. Figure 2A — Geometry-dependent retention survival

        Kaplan–Meier curves show the probability that an initially bound analyte has
        not yet escaped. Lines are replicate means and shading is SEM across receptor
        layouts. Solid and dotted lines distinguish rebinding-on and rebinding-off.
        """
    ),
    code(
        r"""
        survival_max_time=float(onoff.max_time_s.max())
        survival_grid=np.geomspace(1.0,survival_max_time,300)
        survival=replicate_survival_curves(onoff_index,survival_grid)
        survival_metadata=onoff[["condition_label","geometry","rebinding_state"]].drop_duplicates()
        survival=survival.merge(survival_metadata,on="condition_label",validate="many_to_one")
        survival_stats=(survival.groupby(["geometry","rebinding_state","time_s"])
                        .survival.agg(["mean","sem"]).reset_index())

        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        for geometry in ["planar","moderate_R100nm","semicircular_R49p5nm"]:
            for state,linestyle,alpha in [
                ("Rebinding on","-",.16),("Rebinding off",(0,(2,1.4)),.07)
            ]:
                frame=survival_stats.loc[
                    survival_stats.geometry.eq(geometry)
                    & survival_stats.rebinding_state.eq(state)
                ].sort_values("time_s")
                ax.step(frame.time_s,frame["mean"],where="post",
                        color=onoff_colors[geometry],linestyle=linestyle)
                ax.fill_between(frame.time_s,
                                np.clip(frame["mean"]-frame["sem"],0,1),
                                np.clip(frame["mean"]+frame["sem"],0,1),
                                step="post",color=onoff_colors[geometry],alpha=alpha,
                                linewidth=0)
        ax.set(xscale="log",xlabel="Time after initial binding (s)",
               ylabel="Probability not escaped",
               title="A  Nanostructure geometry extends retention survival",
               xlim=(1.0,survival_max_time),ylim=(-.02,1.02))
        geometry_legend=ax.legend(
            handles=[Line2D([0],[0],color=onoff_colors[g],label=onoff_geometry_names[g])
                     for g in ["planar","moderate_R100nm","semicircular_R49p5nm"]],
            title="Geometry",frameon=False,loc="lower left")
        ax.add_artist(geometry_legend)
        ax.legend(handles=[
            Line2D([0],[0],color="0.2",linestyle="-",label="Rebinding on"),
            Line2D([0],[0],color="0.2",linestyle=(0,(2,1.4)),label="Rebinding off"),
        ],title="Post-release binding",frameon=False,loc="upper right")
        finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,FIGURE_ROOT,"figure2_A_retention_survival")
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 4. Figure 2B — Rebinding increases surface residence time

        Rebinding-on and rebinding-off conditions use matched replicate seeds. Points
        and error bars show replicate mean ± SEM; the dashed line is the isolated
        receptor expectation $1/k_{off}$.
        """
    ),
    code(
        r"""
        order = ["planar","moderate_R100nm","semicircular_R49p5nm"]
        positions = np.arange(len(order)); offsets={"Rebinding off":-.12,"Rebinding on":.12}
        fig, ax = plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        for state, marker, fill in [("Rebinding off","o",False),("Rebinding on","o",True)]:
            for position, geometry in zip(positions,order):
                values=onoff.loc[onoff.geometry.eq(geometry)&onoff.rebinding_state.eq(state),
                                 "mean_bound_time_s"]
                ax.errorbar(position+offsets[state],values.mean(),yerr=values.sem(),marker=marker,
                            markerfacecolor=onoff_colors[geometry] if fill else "white",
                            markeredgecolor=onoff_colors[geometry],color=onoff_colors[geometry],
                            capsize=4,linestyle="none")
        ax.axhline(1/onoff.k_off_s.iloc[0],color="black",linestyle="--",
                   label=r"Isolated receptor: $1/k_{off}$")
        ax.set_xticks(positions,[onoff_geometry_names[g] for g in order])
        plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
        ax.set(yscale="log",ylabel="Mean cumulative bound time (s)",
               title="B  Rebinding increases analyte residence time")
        handles=[Line2D([0],[0],marker="o",color="0.2",markerfacecolor="white",linestyle="none",label="Rebinding off"),
                 Line2D([0],[0],marker="o",color="0.2",markerfacecolor="0.2",linestyle="none",label="Rebinding on"),
                 Line2D([0],[0],color="black",linestyle="--",label=r"$1/k_{off}$")]
        ax.legend(handles=handles,frameon=False)
        finish_panel(fig,ax)
        if SAVE_FIGURES:
            save_panel(fig,FIGURE_ROOT,"figure2_B_rebinding_residence")
        plt.show()
        """
    ),
    markdown("## 5. Figure 2C — Geometry controls rebinding and residence time"),
    code(
        r"""
        on = onoff.loc[onoff.rebinding_state.eq("Rebinding on")].copy()
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        for geometry in ["planar","moderate_R100nm","semicircular_R49p5nm"]:
            frame=on.loc[on.geometry.eq(geometry)]
            ax.scatter(frame.mean_rebindings,frame.mean_bound_time_s,
                       color=onoff_colors[geometry],alpha=.45,
                       label=onoff_geometry_names[geometry])
            ax.errorbar(frame.mean_rebindings.mean(),frame.mean_bound_time_s.mean(),
                        xerr=frame.mean_rebindings.sem(),yerr=frame.mean_bound_time_s.sem(),
                        color=onoff_colors[geometry],marker="o",capsize=4)
        usable=on[["mean_rebindings","mean_bound_time_s"]].dropna()
        slope,intercept=np.polyfit(usable.mean_rebindings,usable.mean_bound_time_s,1)
        xx=np.linspace(usable.mean_rebindings.min(),usable.mean_rebindings.max(),100)
        ax.plot(xx,intercept+slope*xx,color="black",linestyle="--")
        correlation=usable.corr().iloc[0,1]
        ax.text(.04,.96,rf"$r={correlation:.2f}$",transform=ax.transAxes,va="top")
        ax.set(xlabel="Mean rebindings per trajectory",ylabel="Mean cumulative bound time (s)",
               title="C  Geometry-dependent rebinding extends residence")
        ax.legend(frameon=False)
        finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,FIGURE_ROOT,"figure2_C_geometry_rebinding_residence")
        plt.show()
        """
    ),
    markdown("## 6. Figure 2D — Increased residence shifts occupancy curves leftward"),
    code(
        r"""
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        for geometry in geometry_order:
            stats=(curves.loc[curves.geometry.eq(geometry)].groupby("C_over_KD").theta
                   .agg(["mean","sem"]).reset_index())
            ax.plot(stats.C_over_KD,stats["mean"],color=colors[geometry],
                    label=geometry_names[geometry])
            ax.fill_between(stats.C_over_KD,np.clip(stats["mean"]-stats["sem"],0,1),
                            np.clip(stats["mean"]+stats["sem"],0,1),
                            color=colors[geometry],alpha=.16,linewidth=0)
            simulated=(response_runs.loc[
                response_runs.geometry.eq(geometry)
                & np.isfinite(response_runs.C_over_KD)
                & response_runs.C_over_KD.gt(0)
            ].groupby("background_occupancy_fraction").C_over_KD
              .agg(["mean","sem"]).reset_index())
            simulated=simulated.loc[simulated["mean"].between(
                concentration_grid.min(),concentration_grid.max())]
            ax.errorbar(simulated["mean"],simulated.background_occupancy_fraction,
                        xerr=simulated["sem"],color=colors[geometry],marker="o",
                        markeredgecolor="white",markeredgewidth=.7,markersize=5,
                        linestyle="none",capsize=2,alpha=.9,
                        label="Simulated occupancy states" if geometry==geometry_order[0]
                        else "_nolegend_")
        ax.plot(concentration_grid,concentration_grid/(1+concentration_grid),
                color="black",linestyle="--",label="Classical kinetics")
        ax.axhline(.5,color=".5",linestyle=":")
        ax.set(xscale="log",xlabel=r"Normalized concentration $C/K_D$",
               ylabel=r"Equilibrium occupancy $\theta$",ylim=(-.02,1.02),
               title="D  Rebinding shifts concentration responses leftward")
        ax.legend(frameon=False)
        finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,FIGURE_ROOT,"figure2_D_occupancy_curves")
        plt.show()
        """
    ),
    markdown("## 7. Figure 2E — Left-shifted responses reduce operational C50"),
    code(
        r"""
        positions=np.arange(len(geometry_order))
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        c50_means=np.array([
            half.loc[half.geometry.eq(geometry),"C50_over_KD"].mean()
            for geometry in geometry_order
        ])
        c50_sems=np.array([
            half.loc[half.geometry.eq(geometry),"C50_over_KD"].sem()
            for geometry in geometry_order
        ])
        ax.bar(positions,c50_means,yerr=c50_sems,
               color=[colors[geometry] for geometry in geometry_order],
               edgecolor="0.2",linewidth=1,capsize=4)
        ax.axhline(1,color="black",linestyle="--",label="Classical C50")
        ax.set_xticks(positions,[geometry_names[g] for g in geometry_order])
        plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
        ax.set(yscale="log",ylabel=r"Operational $C_{50}/K_D$",
               title="E  Rebinding improves operational sensor sensitivity")
        ax.legend(frameon=False)
        finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,FIGURE_ROOT,"figure2_E_operational_C50")
        plt.show()
        """
    ),
    markdown("## 8. Export figure-level values"),
    code(
        r"""
        c50_summary = (half.groupby("geometry")
                       .agg(C50_M=("C50_M", "mean"), C50_M_sem=("C50_M", "sem"),
                            C50_over_KD=("C50_over_KD", "mean"),
                            C50_over_KD_sem=("C50_over_KD", "sem"),
                            C50_reduction_fold=("C50_reduction_fold", "mean"),
                            C50_reduction_fold_sem=("C50_reduction_fold", "sem"))
                       .reset_index())
        display(c50_summary)
        if SAVE_FIGURES:
            output = FIGURE_ROOT / "figure2_tables"; output.mkdir(parents=True, exist_ok=True)
            half.to_csv(output / "figure2_C50_replicates.csv", index=False)
            c50_summary.to_csv(output / "figure2_C50_summary.csv", index=False)
            curves.to_csv(output / "figure2_occupancy_curves.csv", index=False)
            onoff.to_csv(output / "figure2_rebinding_on_off_replicates.csv", index=False)
        """
    ),
]


fig3_name = "analyze_manuscript_figure3_curvature_density.ipynb"
fig3_body = [
    code(SETUP + "\n" + RESPONSE_HELPERS),
    markdown("## 1. Load the C50 factorial and representative response grids"),
    code(
        r"""
        FACTORIAL_ROOT = RESULTS_ROOT / "261006_manuscript_fig3_c50_factorial"
        RESPONSE_ROOT = RESULTS_ROOT / "261006_manuscript_fig3_response_grid"
        factorial_index = discover_runs(
            FACTORIAL_ROOT, cache_path=CACHE_ROOT/"figure3_factorial_index.csv",
            refresh_cache=REFRESH_CACHE,
        )
        response_index = discover_runs(
            RESPONSE_ROOT, cache_path=CACHE_ROOT/"figure3_response_index.csv",
            refresh_cache=REFRESH_CACHE,
        )
        factorial = summarize_replicates(
            factorial_index, cache_path=CACHE_ROOT/"figure3_factorial_replicates.csv",
            refresh_cache=REFRESH_CACHE,
        )
        response_runs = summarize_replicates(
            response_index, cache_path=CACHE_ROOT/"figure3_response_replicates.csv",
            refresh_cache=REFRESH_CACHE,
        )
        print(f"Factorial: {len(factorial_index)} / 200 runs")
        print(f"Response grid: {len(response_index)} / 840 runs")
        if len(factorial_index) != 200 or len(response_index) != 840:
            warnings.warn("Figure 3 is incomplete or a discovery cache is stale.")
        """
    ),
    markdown("## 2. Calculate operational C50 and inspect quality control"),
    code(
        r"""
        factorial = add_c50_metrics(factorial)
        factorial["curvature_nm_inv"] = factorial.curvature_m_inv * 1e-9
        factorial["density_1e16_m2"] = factorial.receptor_density_m2 / 1e16
        response_runs["condition"] = response_runs.condition_label.str.split("__").str[0]
        response_runs = add_occupancy_response_coordinates(response_runs)
        display(factorial.groupby(["curvature_nm_inv", "density_1e16_m2"])
                .agg(runs=("replicate", "size"), receptors=("n_receptors", "mean"),
                     censoring=("censoring_fraction", "max")))
        """
    ),
    markdown("## 3. Figure 3A: curvature–density C50 surface"),
    code(
        r"""
        def plot_factorial_heatmap(letter, metric, title, cmap, stem):
            table = (factorial.groupby(["curvature_nm_inv", "density_1e16_m2"])[metric]
                     .mean().unstack())
            finite = table.to_numpy(float)[np.isfinite(table.to_numpy(float)) & (table.to_numpy(float)>0)]
            fig, ax = plt.subplots(figsize=SINGLE_PANEL_SIZE_IN, constrained_layout=True)
            image = ax.imshow(table.to_numpy(float), origin="lower", aspect="auto",
                              cmap=cmap, norm=LogNorm(vmin=finite.min(), vmax=finite.max()))
            ax.set_xticks(np.arange(len(table.columns)), [f"{v:g}" for v in table.columns])
            ax.set_yticks(np.arange(len(table.index)), [f"{v:.3f}" for v in table.index])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.set(xlabel=r"Receptor density ($10^{16}$ m$^{-2}$)",
                   ylabel=r"Concave curvature (nm$^{-1}$)")
            colorbar = fig.colorbar(image, ax=ax, pad=.03)
            colorbar.set_label(title)
            ax.set_title(title)
            ax.text(-.18, 1.06, letter, transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top")
            finish_panel(fig,ax,THREE_PANEL_PLOT_AREA_IN)
            if SAVE_FIGURES:
                save_panel(fig, FIGURE_ROOT, stem)
            plt.show()

        plot_factorial_heatmap(
            "A", "C50_over_KD", r"Operational $C_{50}/K_D$", "viridis_r",
            "figure3_A_C50_over_KD",
        )
        """
    ),
    markdown("## 4. Figure 3C: curvature–density rebinding surface"),
    code(
        r"""
        plot_factorial_heatmap(
            "C", "mean_rebindings", "Mean rebindings at 50% occupancy",
            "cividis", "figure3_C_mean_rebindings",
        )
        """
    ),
    markdown("## 5. Figure 3D: representative occupancy-aware response curves"),
    code(
        r"""
        condition_order = [
            "planar_low_density", "planar_high_density", "R100nm_mid_density",
            "semicircular_low_density", "semicircular_high_density",
        ]
        condition_names = {
            "planar_low_density": "Planar, low density",
            "planar_high_density": "Planar, high density",
            "R100nm_mid_density": "R = 100 nm, intermediate density",
            "semicircular_low_density": "Semicircular, low density",
            "semicircular_high_density": "Semicircular, high density",
        }
        colors = dict(zip(condition_order, plt.cm.plasma(np.linspace(.08, .92, len(condition_order)))))
        concentration_grid = np.logspace(-2, 1, 241)
        response_curves = interpolate_replicate_responses(
            response_runs, ["condition"], concentration_grid
        )
        fig, ax = plt.subplots(figsize=SINGLE_PANEL_SIZE_IN, constrained_layout=True)
        for condition in condition_order:
            stats = (response_curves.loc[response_curves.condition.eq(condition)]
                     .groupby("C_over_KD").theta.agg(["mean", "sem"]).reset_index())
            ax.plot(stats.C_over_KD, stats["mean"], color=colors[condition],
                    label=condition_names[condition])
            ax.fill_between(stats.C_over_KD,
                            np.clip(stats["mean"]-stats["sem"],0,1),
                            np.clip(stats["mean"]+stats["sem"],0,1),
                            color=colors[condition], alpha=.14, linewidth=0)
            simulated=(response_runs.loc[
                response_runs.condition.eq(condition)
                & np.isfinite(response_runs.C_over_KD)
                & response_runs.C_over_KD.gt(0)
            ].groupby("background_occupancy_fraction").C_over_KD
              .agg(["mean","sem"]).reset_index())
            simulated=simulated.loc[simulated["mean"].between(
                concentration_grid.min(),concentration_grid.max())]
            ax.errorbar(simulated["mean"],simulated.background_occupancy_fraction,
                        xerr=simulated["sem"],color=colors[condition],marker="o",
                        markeredgecolor="white",markeredgewidth=.7,markersize=5,
                        linestyle="none",capsize=2,alpha=.9,
                        label="Simulated occupancy states" if condition==condition_order[0]
                        else "_nolegend_")
        ax.plot(concentration_grid, concentration_grid/(1+concentration_grid),
                color="black", linestyle="--", label="Classical kinetics")
        ax.axhline(.5, color=".5", linestyle=":")
        ax.set(xscale="log", xlabel=r"Normalized concentration $C/K_D$",
               ylabel=r"Equilibrium occupancy $\theta$",
               title="D  Curvature and density jointly shift the response",
               ylim=(-.02,1.02))
        ax.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left")
        finish_panel(fig,ax)
        if SAVE_FIGURES:
            save_panel(fig, FIGURE_ROOT, "figure3_D_representative_responses")
        plt.show()
        """
    ),
    markdown("## 6. Export main-figure source tables"),
    code(
        r"""
        if SAVE_FIGURES:
            output=FIGURE_ROOT/"figure3_tables"; output.mkdir(parents=True,exist_ok=True)
            factorial.to_csv(output/"figure3_C50_factorial_replicates.csv",index=False)
            response_curves.to_csv(output/"figure3_response_curves.csv",index=False)
        """
    ),
    markdown(
        r"""
        ## Supplemental: nonlinear curvature × receptor-density interaction

        The main figure reports the simulated response surfaces directly. This
        supplemental analysis fits a quadratic response surface to
        $\log(C_{50}/K_D)$ and compares it with the corresponding nonlinear
        additive model lacking curvature–density cross terms.
        """
    ),
    code(
        r"""
        from scipy.stats import f as f_distribution

        quadratic_model = factorial.dropna(subset=[
            "C50_over_KD", "curvature_nm_inv", "receptor_density_m2"
        ]).copy()
        curvature_center = quadratic_model.curvature_nm_inv.mean()
        curvature_scale = quadratic_model.curvature_nm_inv.std()
        density_log = np.log10(quadratic_model.receptor_density_m2)
        density_center = density_log.mean()
        density_scale = density_log.std()
        curvature_z = (
            quadratic_model.curvature_nm_inv-curvature_center
        )/curvature_scale
        density_z = (density_log-density_center)/density_scale
        response_y = np.log(quadratic_model.C50_over_KD.to_numpy(float))

        additive_matrix = np.column_stack([
            np.ones(len(quadratic_model)), curvature_z, curvature_z**2,
            density_z, density_z**2,
        ])
        interaction_matrix = np.column_stack([
            additive_matrix, curvature_z*density_z,
            curvature_z**2*density_z, curvature_z*density_z**2,
        ])
        quadratic_terms = [
            "intercept", "curvature", "curvature_squared", "log_density",
            "log_density_squared", "curvature_x_log_density",
            "curvature_squared_x_log_density",
            "curvature_x_log_density_squared",
        ]

        additive_beta, *_ = np.linalg.lstsq(
            additive_matrix, response_y, rcond=None
        )
        quadratic_beta, *_ = np.linalg.lstsq(
            interaction_matrix, response_y, rcond=None
        )
        additive_residual = response_y-additive_matrix@additive_beta
        quadratic_residual = response_y-interaction_matrix@quadratic_beta
        additive_rss = float(additive_residual@additive_residual)
        quadratic_rss = float(quadratic_residual@quadratic_residual)
        numerator_df = interaction_matrix.shape[1]-additive_matrix.shape[1]
        denominator_df = len(response_y)-interaction_matrix.shape[1]
        interaction_f = (
            (additive_rss-quadratic_rss)/numerator_df
        )/(quadratic_rss/denominator_df)
        interaction_p = f_distribution.sf(
            interaction_f, numerator_df, denominator_df
        )
        quadratic_r2 = 1-quadratic_rss/np.sum(
            (response_y-response_y.mean())**2
        )
        quadratic_covariance = np.linalg.pinv(
            interaction_matrix.T@interaction_matrix
        )*(quadratic_rss/denominator_df)
        quadratic_interaction_table = pd.DataFrame({
            "term": quadratic_terms,
            "coefficient": quadratic_beta,
            "standard_error": np.sqrt(np.diag(quadratic_covariance)),
        })
        quadratic_interaction_table["t_value"] = (
            quadratic_interaction_table.coefficient
            / quadratic_interaction_table.standard_error
        )
        quadratic_model_comparison = pd.DataFrame([{
            "interaction_F": interaction_f,
            "numerator_df": numerator_df,
            "denominator_df": denominator_df,
            "interaction_p": interaction_p,
            "quadratic_R2": quadratic_r2,
            "quadratic_log_RMSE": np.sqrt(quadratic_rss/len(response_y)),
        }])
        display(quadratic_model_comparison)
        display(quadratic_interaction_table)

        curvature_grid = np.linspace(
            quadratic_model.curvature_nm_inv.min(),
            quadratic_model.curvature_nm_inv.max(), 200
        )
        densities = np.sort(quadratic_model.receptor_density_m2.unique())
        density_colors = plt.cm.viridis(np.linspace(.08,.92,len(densities)))
        fig, ax = plt.subplots(
            figsize=SINGLE_PANEL_SIZE_IN, constrained_layout=True
        )
        for color, density in zip(density_colors, densities):
            cz = (curvature_grid-curvature_center)/curvature_scale
            dz_value = (np.log10(density)-density_center)/density_scale
            dz = np.full_like(cz, dz_value)
            prediction_matrix = np.column_stack([
                np.ones_like(cz), cz, cz**2, dz, dz**2, cz*dz,
                cz**2*dz, cz*dz**2,
            ])
            prediction = prediction_matrix@quadratic_beta
            prediction_se = np.sqrt(np.einsum(
                "ij,jk,ik->i", prediction_matrix,
                quadratic_covariance, prediction_matrix,
            ))
            label = rf"{density/1e16:g} $\times 10^{{16}}$ m$^{{-2}}$"
            ax.plot(
                curvature_grid, np.exp(prediction), color=color, label=label
            )
            ax.fill_between(
                curvature_grid, np.exp(prediction-1.96*prediction_se),
                np.exp(prediction+1.96*prediction_se), color=color,
                alpha=.14, linewidth=0,
            )
            observed = quadratic_model.loc[np.isclose(
                quadratic_model.receptor_density_m2, density
            )]
            observed_stats = (
                observed.groupby("curvature_nm_inv").C50_over_KD
                .agg(["mean", "sem"]).reset_index()
            )
            ax.errorbar(
                observed_stats.curvature_nm_inv, observed_stats["mean"],
                yerr=observed_stats["sem"], color=color, marker="o",
                linestyle="none", capsize=3,
            )
        ax.axhline(1,color="black",linestyle="--")
        ax.set(
            xlabel=r"Concave curvature (nm$^{-1}$)",
            ylabel=r"Operational $C_{50}/K_D$", yscale="log",
            title="Nonlinear curvature × receptor-density interaction",
        )
        ax.legend(
            title="Receptor density", bbox_to_anchor=(1.02,1),
            loc="upper left"
        )
        finish_panel(fig,ax)
        if SAVE_FIGURES:
            save_panel(
                fig, FIGURE_ROOT,
                "supp_figure3_quadratic_curvature_density_interaction",
            )
            output=FIGURE_ROOT/"figure3_tables"
            output.mkdir(parents=True,exist_ok=True)
            quadratic_interaction_table.to_csv(
                output/"supp_figure3_quadratic_interaction_coefficients.csv",
                index=False,
            )
            quadratic_model_comparison.to_csv(
                output/"supp_figure3_quadratic_interaction_comparison.csv",
                index=False,
            )
        plt.show()
        """
    ),
]


fig4_name = "analyze_manuscript_figure4_local_curvature.ipynb"
fig4_body = [
    code(
        SETUP + "\n" + RESPONSE_HELPERS + r"""
import importlib
import utils.escape_assay_viz as escape_assay_viz
importlib.reload(escape_assay_viz)
from utils.escape_assay_viz import (
    load_escape_assay_run, plot_surface_metric_3d, plot_surface_metric_xy,
)
"""
    ),
    markdown("## 1. Load the local occupancy grid and select the half-occupancy runs"),
    code(
        r"""
        DATA_ROOT = RESULTS_ROOT / "261006_manuscript_fig4_local_c50_grid"
        run_index = discover_runs(
            DATA_ROOT, cache_path=CACHE_ROOT/"figure4_local_C50_index.csv",
            refresh_cache=REFRESH_CACHE,
        )
        replicates = summarize_replicates(
            run_index, cache_path=CACHE_ROOT/"figure4_local_C50_replicates.csv",
            refresh_cache=REFRESH_CACHE,
        )
        print(f"Found {len(run_index)} / 420 expected runs")
        if len(run_index) != 420:
            warnings.warn("Figure 4 is incomplete or a discovery cache is stale.")

        half_index = run_index.loc[np.isclose(
            run_index.background_occupancy_fraction, 0.5
        )].copy()
        assert len(half_index) == 20, "Expected one phi=0.5 run per replicate."
        local = load_receptor_summaries(half_index)
        local["C50_M"] = 1 / (local.k_on_M_inv_s * local.mean_bound_time_s)
        local["C50_over_KD"] = 1 / (local.k_off_s * local.mean_bound_time_s)
        local["C50_reduction_fold"] = 1 / local.C50_over_KD
        local["mean_curvature_nm_inv"] = local.mean_curvature_m_inv * 1e-9
        print(f"{len(local):,} receptor-resolved C50 estimates")
        display(local[["replicate","source_face_id","mean_curvature_nm_inv",
                       "mean_bound_time_s","mean_rebindings","C50_over_KD"]].head())
        """
    ),
    markdown(
        "## 2. Sinusoidal geometry and receptor-resolved Figure 4A–C maps"
    ),
    code(
        r"""
        SELECTED_REPLICATE = int(half_index.replicate.min())
        selected = half_index.loc[half_index.replicate.eq(SELECTED_REPLICATE)].iloc[0]
        archive = load_escape_assay_run(selected.run_directory)
        map_summary = archive.receptor_summary.copy()
        map_summary["C50_over_KD"] = 1 / (
            archive.params.k_off_s * map_summary.mean_bound_time_s
        )
        map_summary["C50_reduction_fold"] = 1 / map_summary.C50_over_KD
        map_summary["mean_curvature_nm_inv"] = (
            map_summary.mean_curvature_m_inv * 1e-9
        )

        # Geometry reference: use symmetric limits so the diverging color scale
        # carries a consistent sign and zero is visually centered.
        curvature_limit = float(np.nanmax(np.abs(
            map_summary.mean_curvature_nm_inv.to_numpy(float)
        )))
        def plot_surface_map_pair(
            metric, cmap, colorbar_label, letter, stem, *, vmin=None, vmax=None
        ):
            # Each manuscript panel is one full-width figure containing the
            # oblique and overhead views. The colorbar is drawn only once.
            fig = plt.figure(figsize=(7.0, 3.0), constrained_layout=True)
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            plot_surface_metric_3d(
                archive.geometry,map_summary,metric,a_m=archive.params.a_m,
                cmap=cmap,vmin=vmin,vmax=vmax,show_receptors=False,
                colorbar_label=colorbar_label,
                show_colorbar=False,rasterize_surface=True,
                view_zoom=.88,ax=ax,
            )
            ax.view_init(elev=35,azim=-60)
            ax.zaxis.set_major_locator(MultipleLocator(10))
            ax.set_title(f"{letter}  {colorbar_label}")

            ax2 = fig.add_subplot(1, 2, 2)
            plot_surface_metric_xy(
                archive.geometry,map_summary,metric,a_m=archive.params.a_m,
                cmap=cmap,vmin=vmin,vmax=vmax,show_receptors=False,
                rasterize_surface=True,colorbar_label=colorbar_label,ax=ax2,
            )
            ax2.set_title("Overhead projection")
            finish_panel(fig,ax,MAP_PAIR_SUBPLOT_AREA_IN)
            if SAVE_FIGURES:
                save_panel(fig,FIGURE_ROOT,stem,dpi=300)
            plt.show()

        plot_surface_map_pair(
            "mean_curvature_nm_inv", "coolwarm",
            r"Signed mean curvature (nm$^{-1}$)", "A",
            "figure4_A_signed_mean_curvature_maps",
            vmin=-curvature_limit, vmax=curvature_limit,
        )
        plot_surface_map_pair(
            "C50_over_KD", "viridis_r", r"Local $C_{50}/K_D$", "B",
            "figure4_B_local_C50_maps",
        )
        plot_surface_map_pair(
            "mean_rebindings", "magma", "Mean rebindings at 50% occupancy", "C",
            "figure4_C_rebinding_maps",
        )
        """
    ),
    markdown("## 3. Figure 4D–E: signed curvature predicts local C50 and rebinding"),
    code(
        r"""
        local["curvature_bin"] = pd.qcut(
            local.mean_curvature_nm_inv, 15, duplicates="drop"
        )
        per_rep_bin = (local.groupby(["replicate","curvature_bin"],observed=True)
                       .agg(curvature=("mean_curvature_nm_inv","mean"),
                            C50_over_KD=("C50_over_KD","mean"),
                            rebindings=("mean_rebindings","mean"),
                            reduction=("C50_reduction_fold","mean"))
                       .reset_index())
        binned = (per_rep_bin.groupby("curvature_bin",observed=True)
                  .agg(curvature=("curvature","mean"),
                       C50_mean=("C50_over_KD","mean"),
                       C50_sem=("C50_over_KD","sem"),
                       rebind_mean=("rebindings","mean"),
                       rebind_sem=("rebindings","sem"),
                       reduction_mean=("reduction","mean"),
                       reduction_sem=("reduction","sem"))
                  .reset_index())
        fig,ax=plt.subplots(figsize=MAP_PANEL_SIZE_IN,constrained_layout=True)
        ax.errorbar(binned.curvature,binned.C50_mean,yerr=binned.C50_sem,
                    marker="o",capsize=3,color="tab:blue")
        ax.set(xlabel=r"Signed mean curvature (nm$^{-1}$)",
               ylabel=r"Local operational $C_{50}/K_D$",yscale="log",
               title="D  Curvature predicts local C50")
        finish_panel(fig,ax,THREE_PANEL_TALL_PLOT_AREA_IN)
        if SAVE_FIGURES: save_panel(fig,FIGURE_ROOT,"figure4_D_curvature_local_C50")
        plt.show()

        fig,ax=plt.subplots(figsize=MAP_PANEL_SIZE_IN,constrained_layout=True)
        ax.errorbar(binned.curvature,binned.rebind_mean,yerr=binned.rebind_sem,
                    marker="o",capsize=3,color="tab:orange")
        ax.set(xlabel=r"Signed mean curvature (nm$^{-1}$)",
               ylabel="Mean rebindings at 50% occupancy",
               title="E  Curvature predicts local rebinding")
        finish_panel(fig,ax,THREE_PANEL_TALL_PLOT_AREA_IN)
        if SAVE_FIGURES: save_panel(fig,FIGURE_ROOT,"figure4_E_curvature_rebinding")
        plt.show()
        """
    ),
    markdown("## 4. Figure 4F: population concentration response"),
    code(
        r"""
        response_runs = add_occupancy_response_coordinates(replicates)
        concentration_grid = np.logspace(-2,1,241)
        response_runs["surface"] = "Sinusoidal surface"
        population_curves = interpolate_replicate_responses(
            response_runs,["surface"],concentration_grid
        )
        stats=(population_curves.groupby("C_over_KD").theta
               .agg(["mean","sem"]).reset_index())
        fig,ax=plt.subplots(figsize=MAP_PANEL_SIZE_IN,constrained_layout=True)
        ax.plot(stats.C_over_KD,stats["mean"],label="Sinusoidal surface")
        ax.fill_between(stats.C_over_KD,np.clip(stats["mean"]-stats["sem"],0,1),
                        np.clip(stats["mean"]+stats["sem"],0,1),alpha=.16)
        simulated=(response_runs.loc[
            np.isfinite(response_runs.C_over_KD) & response_runs.C_over_KD.gt(0)
        ].groupby("background_occupancy_fraction").C_over_KD
          .agg(["mean","sem"]).reset_index())
        simulated=simulated.loc[simulated["mean"].between(
            concentration_grid.min(),concentration_grid.max())]
        ax.errorbar(simulated["mean"],simulated.background_occupancy_fraction,
                    xerr=simulated["sem"],color="tab:blue",marker="o",
                    markeredgecolor="white",markeredgewidth=.7,markersize=5,
                    linestyle="none",capsize=2,label="_nolegend_")
        ax.plot(concentration_grid,concentration_grid/(1+concentration_grid),
                color="black",linestyle="--",label="Classical kinetics")
        ax.set(xscale="log",xlabel=r"Normalized concentration $C/K_D$",
               ylabel=r"Mean occupancy $\theta$",
               title="F  Population concentration response",ylim=(-.02,1.02))
        ax.legend(frameon=False,loc="upper left")
        finish_panel(fig,ax,THREE_PANEL_TALL_PLOT_AREA_IN)
        if SAVE_FIGURES: save_panel(fig,FIGURE_ROOT,"figure4_F_population_response")
        plt.show()
        """
    ),
    markdown("## 5. Receptor-level curvature model"),
    code(
        r"""
        predictors=["mean_curvature_m_inv","gaussian_curvature_m_inv2","surface_z_m"]
        data=local.dropna(subset=predictors+["C50_over_KD"]).copy()
        Xparts=[np.ones(len(data))]; names=["intercept"]
        for name in predictors:
            values=data[name].to_numpy(float)
            values=(values-values.mean())/values.std()
            Xparts.append(values); names.append(name)
        replicate_dummies=pd.get_dummies(data.replicate,drop_first=True,dtype=float)
        Xparts.extend([replicate_dummies[col].to_numpy() for col in replicate_dummies])
        names.extend([f"replicate_{col}" for col in replicate_dummies])
        X=np.column_stack(Xparts)
        y=np.log(np.clip(data.C50_over_KD.to_numpy(float),1e-12,None))
        beta,*_=np.linalg.lstsq(X,y,rcond=None); residual=y-X@beta
        covariance=np.linalg.pinv(X.T@X)*(residual@residual)/(len(y)-X.shape[1])
        coefficients=pd.DataFrame({"term":names,"coefficient":beta,
                                   "standard_error":np.sqrt(np.diag(covariance))})
        coefficients["t_value"]=coefficients.coefficient/coefficients.standard_error
        display(coefficients.iloc[:len(predictors)+1])
        if SAVE_FIGURES:
            output=FIGURE_ROOT/"figure4_tables"; output.mkdir(parents=True,exist_ok=True)
            local.to_csv(output/"figure4_local_C50_receptors.csv",index=False)
            binned.to_csv(output/"figure4_curvature_C50_bins.csv",index=False)
            coefficients.to_csv(output/"figure4_local_C50_model.csv",index=False)
            population_curves.to_csv(output/"figure4_population_response.csv",index=False)
        """
    ),
]


notebook(
    fig2_name,
    "Manuscript Figure 2 — Rebinding and geometry shift operational C50",
    "Combines the dilute rebinding on/off control with the occupancy-aware sweep to establish the sequence: rebinding extends residence, geometry controls rebinding, occupancy curves shift left, and operational C50 decreases.",
    fig2_body,
)
notebook(
    fig3_name,
    "Manuscript Figure 3 — Curvature and receptor density synergistically decrease C50",
    "Combines the half-occupancy factorial experiment with representative full response grids to quantify the curvature × receptor-density interaction.",
    fig3_body,
)
notebook(
    fig4_name,
    "Manuscript Figure 4 — Local curvature produces heterogeneous retention and C50",
    "Calculates receptor-resolved operational C50 on a sinusoidal surface and tests whether signed local curvature predicts rebinding and spatial occupancy heterogeneity.",
    fig4_body,
)
print("Updated Figures 2–4 occupancy-aware analysis notebooks.")
