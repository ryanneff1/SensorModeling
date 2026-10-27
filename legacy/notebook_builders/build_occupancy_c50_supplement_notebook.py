#!/usr/bin/env python3
"""Build the occupancy-aware C50 supplemental-controls notebook."""

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


with (HERE / "analyze_manuscript_figure2_rebinding_affinity.ipynb").open(
    encoding="utf-8"
) as stream:
    formatting = "".join(json.load(stream)["cells"][0]["source"])


cells = [
    nbf.v4.new_code_cell(formatting),
    markdown(
        r"""
        # Occupancy-aware supplemental controls

        This notebook analyzes the fixed-receptor-number, numerical-convergence,
        kinetic-robustness, and escape-height controls using operational C50 measured
        at background occupancy $\phi=0.5$. Every panel is plotted independently and
        uses the same fixed 5 × 4 inch axes area as the main manuscript figures.
        """
    ),
    code(
        r"""
        from pathlib import Path
        import sys
        import importlib
        import warnings
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        PROJECT_ROOT = Path.cwd().resolve()
        if PROJECT_ROOT.name == "analysis_tools":
            PROJECT_ROOT = PROJECT_ROOT.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        import analysis_tools.manuscript_escape_analysis as manuscript_escape_analysis
        importlib.reload(manuscript_escape_analysis)
        from analysis_tools.manuscript_escape_analysis import (
            discover_runs, save_figure, size_figure_for_axes, summarize_replicates,
        )

        SAVE_FIGURES = False
        REFRESH_CACHE = False
        FIGURE_ROOT = PROJECT_ROOT / "analysis_tools" / "manuscript_panels"
        CACHE_ROOT = PROJECT_ROOT / "analysis_tools" / ".analysis_cache"
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)

        ONEDRIVE_RESULTS_ROOT = Path(
            "/Users/rynon/Library/CloudStorage/OneDrive-NorthwesternUniversity/PhD/Data/"
            "Agent-based simulations"
        )
        RESULTS_ROOT = ONEDRIVE_RESULTS_ROOT if ONEDRIVE_RESULTS_ROOT.exists() else PROJECT_ROOT / "results"

        TWO_PANEL_PLOT_AREA_IN = (2.55, 2.15)
        THREE_PANEL_PLOT_AREA_IN = (1.75, 1.65)
        PLOT_AREA_SIZE_IN = TWO_PANEL_PLOT_AREA_IN
        MAP_PLOT_AREA_SIZE_IN = THREE_PANEL_PLOT_AREA_IN
        SINGLE_PANEL_SIZE_IN = (3.4, 2.8)
        MAP_PANEL_SIZE_IN = (2.6, 2.35)

        def finish_panel(fig, ax):
            size_figure_for_axes(fig, ax, *PLOT_AREA_SIZE_IN)

        def save_panel(fig, stem):
            save_figure(fig, FIGURE_ROOT, stem, exact_size=True)

        def add_c50_metrics(frame):
            result = frame.copy()
            result["C50_M"] = 1 / (result.k_on_M_inv_s * result.mean_bound_time_s)
            result["C50_over_KD"] = 1 / (result.k_off_s * result.mean_bound_time_s)
            result["C50_reduction_fold"] = 1 / result.C50_over_KD
            return result

        print(f"Simulation results root: {RESULTS_ROOT}")
        """
    ),
    markdown("## 1. Load the four supplemental datasets"),
    code(
        r"""
        roots = {
            "fixed": RESULTS_ROOT / "261006_manuscript_supp_fixed_receptor_c50",
            "numerical": RESULTS_ROOT / "261006_manuscript_supp_numerical_c50",
            "kinetic": RESULTS_ROOT / "261006_manuscript_supp_kinetic_c50",
            "height": RESULTS_ROOT / "261006_manuscript_supp_height_c50",
        }
        expected = {"fixed": 40, "numerical": 36, "kinetic": 50, "height": 36}
        indexes, datasets = {}, {}
        for name, root in roots.items():
            indexes[name] = discover_runs(
                root, cache_path=CACHE_ROOT/f"supp_C50_{name}_index.csv",
                refresh_cache=REFRESH_CACHE,
            )
            datasets[name] = add_c50_metrics(summarize_replicates(
                indexes[name], cache_path=CACHE_ROOT/f"supp_C50_{name}_replicates.csv",
                refresh_cache=REFRESH_CACHE,
            ))
        counts = pd.DataFrame({"expected": expected,
                               "observed": {name: len(frame) for name,frame in indexes.items()}})
        display(counts)
        if not counts.expected.eq(counts.observed).all():
            warnings.warn("One or more supplemental sweeps are incomplete or use a stale cache.")
        """
    ),
    markdown("## S1A — Fixed receptor number: curvature-dependent C50"),
    code(
        r"""
        fixed=datasets["fixed"].copy()
        fixed["curvature_nm_inv"]=fixed.curvature_m_inv*1e-9
        fixed_stats=(fixed.groupby("curvature_nm_inv").C50_over_KD
                     .agg(["mean","sem"]).reset_index())
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        ax.errorbar(fixed_stats.curvature_nm_inv,fixed_stats["mean"],
                    yerr=fixed_stats["sem"],marker="o",capsize=4)
        ax.axhline(1,color="black",linestyle="--",label="Classical C50")
        ax.set(xlabel=r"Concave curvature (nm$^{-1}$)",
               ylabel=r"Operational $C_{50}/K_D$",yscale="log",
               title="S1A  Curvature lowers C50 at fixed receptor number")
        ax.legend(frameon=False); finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,"supp_S1A_fixed_receptor_C50")
        plt.show()
        """
    ),
    markdown("## S1B — Fixed receptor number: curvature-dependent rebinding"),
    code(
        r"""
        fixed_rebinding=(fixed.groupby("curvature_nm_inv").mean_rebindings
                         .agg(["mean","sem"]).reset_index())
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        ax.errorbar(fixed_rebinding.curvature_nm_inv,fixed_rebinding["mean"],
                    yerr=fixed_rebinding["sem"],marker="o",capsize=4,color="tab:orange")
        ax.set(xlabel=r"Concave curvature (nm$^{-1}$)",
               ylabel="Mean rebindings at 50% occupancy",
               title="S1B  Shape increases rebinding at fixed receptor number")
        finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,"supp_S1B_fixed_receptor_rebinding")
        plt.show()
        """
    ),
    markdown("## S2A — Numerical convergence of operational C50"),
    code(
        r"""
        numerical=datasets["numerical"].copy()
        baseline=(numerical.loc[numerical.test_parameter.eq("baseline"),
                   ["replicate","C50_over_KD","mean_rebindings","censoring_fraction"]]
                  .set_index("replicate").add_suffix("_baseline"))
        numerical=numerical.join(baseline,on="replicate")
        numerical["C50_ratio_to_baseline"]=numerical.C50_over_KD/numerical.C50_over_KD_baseline
        numerical["rebindings_ratio_to_baseline"]=numerical.mean_rebindings/numerical.mean_rebindings_baseline
        order=list(numerical.groupby("condition_index").condition_label.first().sort_index())
        summary=(numerical.groupby("condition_label").C50_ratio_to_baseline
                 .agg(["mean","sem"]).reindex(order).reset_index())
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        x=np.arange(len(summary))
        ax.errorbar(x,summary["mean"],yerr=summary["sem"],marker="o",capsize=4)
        ax.axhline(1,color="black",linestyle="--")
        ax.set_xticks(x,summary.condition_label)
        plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
        ax.set(ylabel="C50 ratio to paired baseline",
               title="S2A  Operational C50 is numerically converged")
        finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,"supp_S2A_numerical_C50")
        plt.show()
        """
    ),
    markdown("## S2B — Numerical convergence of rebinding"),
    code(
        r"""
        summary_rebinding=(numerical.groupby("condition_label").rebindings_ratio_to_baseline
                           .agg(["mean","sem"]).reindex(order).reset_index())
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        x=np.arange(len(summary_rebinding))
        ax.errorbar(x,summary_rebinding["mean"],yerr=summary_rebinding["sem"],
                    marker="o",capsize=4,color="tab:orange")
        ax.axhline(1,color="black",linestyle="--")
        ax.set_xticks(x,summary_rebinding.condition_label)
        plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
        ax.set(ylabel="Rebinding ratio to paired baseline",
               title="S2B  Rebinding is numerically converged")
        finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,"supp_S2B_numerical_rebinding")
        plt.show()
        """
    ),
    markdown("## S3A — Kinetic robustness of geometry-dependent C50"),
    code(
        r"""
        kinetic=datasets["kinetic"].copy()
        kinetic["geometry"]=np.where(kinetic.curvature_m_inv.fillna(0).eq(0),
                                      "Planar","Semicircular")
        kinetic["kinetic_label"]=[rf"$10^{{{int(np.log10(kon))}}},10^{{{int(np.log10(koff))}}}$"
                                   for kon,koff in zip(kinetic.k_on_M_inv_s,kinetic.k_off_s)]
        kinetic_order=[r"$10^{6},10^{-2}$",r"$10^{7},10^{-2}$",r"$10^{8},10^{-2}$",
                       r"$10^{7},10^{-3}$",r"$10^{7},10^{-1}$"]
        kinetic_stats=(kinetic.groupby(["kinetic_label","geometry"]).C50_over_KD
                       .agg(["mean","sem"]).reset_index())
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        x=np.arange(len(kinetic_order)); offsets={"Planar":-.08,"Semicircular":.08}
        colors={"Planar":"tab:blue","Semicircular":"tab:orange"}
        for geometry in ["Planar","Semicircular"]:
            frame=kinetic_stats.loc[kinetic_stats.geometry.eq(geometry)].set_index("kinetic_label").reindex(kinetic_order)
            ax.errorbar(x+offsets[geometry],frame["mean"],yerr=frame["sem"],
                        marker="o",capsize=4,color=colors[geometry],label=geometry)
        ax.axhline(1,color="black",linestyle="--")
        ax.set_xticks(x,kinetic_order)
        plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
        ax.set(yscale="log",xlabel=r"$(k_{on},k_{off})$",
               ylabel=r"Operational $C_{50}/K_D$",
               title="S3A  Geometry-dependent C50 is kinetically robust")
        ax.legend(frameon=False); finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,"supp_S3A_kinetic_C50")
        plt.show()
        """
    ),
    markdown("## S3B — Geometry benefit across kinetic conditions"),
    code(
        r"""
        geometry_paired=(kinetic.pivot(index=["kinetic_label","replicate"],
                                       columns="geometry",values="C50_over_KD").reset_index())
        geometry_paired["geometry_C50_ratio"]=geometry_paired.Semicircular/geometry_paired.Planar
        geometry_stats=(geometry_paired.groupby("kinetic_label").geometry_C50_ratio
                        .agg(["mean","sem"]).reindex(kinetic_order).reset_index())
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        x=np.arange(len(geometry_stats))
        ax.errorbar(x,geometry_stats["mean"],yerr=geometry_stats["sem"],
                    marker="o",capsize=4,color="tab:purple")
        ax.axhline(1,color="black",linestyle="--")
        ax.set_xticks(x,geometry_stats.kinetic_label)
        plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
        ax.set(xlabel=r"$(k_{on},k_{off})$",
               ylabel="Semicircular/planar C50 ratio",
               title="S3B  Confinement benefit persists across kinetics")
        finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,"supp_S3B_kinetic_geometry_ratio")
        plt.show()
        """
    ),
    markdown("## S4A — Escape-boundary robustness of operational C50"),
    code(
        r"""
        height=datasets["height"].copy(); height["mean_z_nm"]=height.mean_z_m*1e9
        boundary_names={"fixed_absolute_plane":"Fixed absolute escape plane",
                        "fixed_40nm_above_peak":"Fixed offset above surface peak"}
        height_stats=(height.groupby(["boundary_series","mean_z_nm"]).C50_over_KD
                      .agg(["mean","sem"]).reset_index())
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        for series,frame in height_stats.groupby("boundary_series"):
            frame=frame.sort_values("mean_z_nm")
            ax.errorbar(frame.mean_z_nm,frame["mean"],yerr=frame["sem"],
                        marker="o",capsize=4,label=boundary_names.get(series,series))
        ax.axhline(1,color="black",linestyle="--")
        ax.set(xlabel="Mean surface height (nm)",ylabel=r"Operational $C_{50}/K_D$",
               yscale="log",title="S4A  C50 is robust to escape-plane placement")
        ax.legend(frameon=False); finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,"supp_S4A_height_C50")
        plt.show()
        """
    ),
    markdown("## S4B — Escape-boundary effects on rebinding"),
    code(
        r"""
        height_rebinding=(height.groupby(["boundary_series","mean_z_nm"]).mean_rebindings
                          .agg(["mean","sem"]).reset_index())
        fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)
        for series,frame in height_rebinding.groupby("boundary_series"):
            frame=frame.sort_values("mean_z_nm")
            ax.errorbar(frame.mean_z_nm,frame["mean"],yerr=frame["sem"],
                        marker="o",capsize=4,label=boundary_names.get(series,series))
        ax.set(xlabel="Mean surface height (nm)",
               ylabel="Mean rebindings at 50% occupancy",
               title="S4B  Rebinding is robust to escape-plane placement")
        ax.legend(frameon=False); finish_panel(fig,ax)
        if SAVE_FIGURES: save_panel(fig,"supp_S4B_height_rebinding")
        plt.show()
        """
    ),
    markdown("## Export supplemental source tables"),
    code(
        r"""
        if SAVE_FIGURES:
            output=FIGURE_ROOT/"occupancy_C50_supplement_tables"
            output.mkdir(parents=True,exist_ok=True)
            fixed.to_csv(output/"fixed_receptor_C50.csv",index=False)
            numerical.to_csv(output/"numerical_C50.csv",index=False)
            kinetic.to_csv(output/"kinetic_C50.csv",index=False)
            height.to_csv(output/"height_C50.csv",index=False)
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
destination = HERE / "analyze_manuscript_supp_occupancy_c50_controls.ipynb"
nbf.write(notebook, destination)
print(destination)
