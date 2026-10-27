#!/usr/bin/env python3
"""Generate the manuscript escape-assay analysis notebooks."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


HERE = Path(__file__).resolve().parent


FORMATTING = textwrap.dedent(
    '''
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper", palette="deep")
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 14,
        "figure.titlesize": 34, "figure.titleweight": "bold",
        "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 16, "legend.title_fontsize": 18,
        "lines.linewidth": 3, "lines.markersize": 8,
        "axes.linewidth": 2,
        "xtick.major.size": 8, "ytick.major.size": 8,
        "xtick.major.width": 2, "ytick.major.width": 2,
        "xtick.minor.size": 4, "ytick.minor.size": 4,
        "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
        "grid.linewidth": 1, "grid.alpha": 0.3,
        "figure.dpi": 150, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    '''
).strip() + "\n"


def lines(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def notebook(title: str, introduction: str, cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [
        nbf.v4.new_code_cell(FORMATTING),
        nbf.v4.new_markdown_cell(lines(f"# {title}\n\n{introduction}")),
        *cells,
    ]
    return nb


SETUP = lines(
    """
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "analysis_tools":
        PROJECT_ROOT = PROJECT_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from analysis_tools.manuscript_escape_analysis import (
        classical_occupancy, condition_statistics, discover_runs,
        load_receptor_summaries, occupancy_from_bound_time,
        replicate_survival_curves, save_figure, summarize_replicates,
    )

    SAVE_FIGURES = False
    FIGURE_ROOT = PROJECT_ROOT / "analysis_tools" / "manuscript_panels"
    """
)


def md(text: str):
    return nbf.v4.new_markdown_cell(lines(text))


def code(text: str):
    return nbf.v4.new_code_cell(lines(text))


fig2 = notebook(
    "Manuscript Figure 2 — Rebinding shifts apparent affinity",
    "Loads the rebinding on/off and rebinding-dose datasets. The replicate receptor layout is the statistical unit; shaded regions and error bars represent uncertainty across independent replicates.",
    [
        code(SETUP),
        md("## 1. Select data folders\n\nChange only these paths after copying the cluster results."),
        code(
            """
            ON_OFF_ROOT = PROJECT_ROOT / "results/261005_manuscript_fig2_rebinding_on_off"
            DOSE_ROOT = PROJECT_ROOT / "results/261005_manuscript_fig2_rebinding_dose_response"

            onoff_index = discover_runs(ON_OFF_ROOT)
            dose_index = discover_runs(DOSE_ROOT)
            display(onoff_index.groupby("condition_label").size().rename("replicates"))
            display(dose_index.groupby("condition_label").size().rename("replicates"))
            """
        ),
        md("## 2. Replicate summaries and quality control"),
        code(
            """
            onoff = summarize_replicates(onoff_index)
            dose = summarize_replicates(dose_index)
            qc_columns = ["condition_label", "replicate", "n_receptors", "n_trajectories", "censoring_fraction", "mean_bound_time_s", "mean_rebindings", "affinity_shift_fold"]
            display(onoff[qc_columns])
            print("Maximum censoring:", onoff.censoring_fraction.max())
            """
        ),
        md("## 3. Figure 2A–B: survival and effective residence time"),
        code(
            """
            grid = np.unique(np.r_[0, np.geomspace(1e-2, onoff.max_time_s.max(), 250)])
            survival = replicate_survival_curves(onoff_index, grid)
            survival_summary = survival.groupby(["condition_label", "time_s"]).survival.agg(["mean", "sem"]).reset_index()

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
            geometry_labels = list(dict.fromkeys(label.split("__")[0] for label in onoff.condition_label))
            geometry_colors = dict(zip(geometry_labels, plt.cm.viridis(np.linspace(0.08, 0.92, len(geometry_labels)))))
            for label, frame in survival_summary.groupby("condition_label", sort=False):
                frame = frame.sort_values("time_s")
                geometry_label = label.split("__")[0]
                rebinding_on = bool(onoff.loc[onoff.condition_label.eq(label), "rebinding_multiplier"].iloc[0] > 0)
                color = geometry_colors[geometry_label]
                linestyle = "-" if rebinding_on else (0, (2.0, 1.3))
                axes[0].step(frame.time_s, frame["mean"], where="post", color=color, linestyle=linestyle, alpha=1.0 if rebinding_on else 0.8)
                axes[0].fill_between(frame.time_s, np.clip(frame["mean"]-frame["sem"],0,1), np.clip(frame["mean"]+frame["sem"],0,1), step="post", color=color, alpha=.14 if rebinding_on else .06)
            axes[0].set(xscale="log", xlabel="Time after initial binding (s)", ylabel="Probability not escaped", title="A  Effective retention survival")
            geometry_legend = axes[0].legend(
                handles=[Line2D([0],[0], color=geometry_colors[label], label=label) for label in geometry_labels],
                title="Geometry", loc="lower left",
            )
            axes[0].add_artist(geometry_legend)
            axes[0].legend(
                handles=[
                    Line2D([0],[0], color="0.15", linestyle="-", label="Rebinding on"),
                    Line2D([0],[0], color="0.15", linestyle=(0,(2.0,1.3)), label="Rebinding off"),
                ],
                title="Post-release binding", loc="upper right",
            )

            order = list(onoff.groupby("condition_index").condition_label.first().sort_index())
            values = [onoff.loc[onoff.condition_label.eq(label), "mean_bound_time_s"] for label in order]
            positions = np.arange(1, len(order)+1)
            for position, label, data in zip(positions, order, values):
                geometry_label = label.split("__")[0]
                rebinding_on = bool(onoff.loc[onoff.condition_label.eq(label), "rebinding_multiplier"].iloc[0] > 0)
                color = geometry_colors[geometry_label]
                axes[1].errorbar(position, data.mean(), yerr=data.sem(), marker="o", markerfacecolor=color if rebinding_on else "white", markeredgecolor=color, color=color, linestyle="none", capsize=4)
            axes[1].set_xticks(positions, order)
            axes[1].axhline(1/onoff.k_off_s.iloc[0], color="black", ls="--", label=r"$1/k_{off}$")
            axes[1].set(yscale="log", ylabel="Mean cumulative bound time (s)", title="B  Rebinding extends bound residence")
            plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            axes[1].legend()
            if SAVE_FIGURES: save_figure(fig, FIGURE_ROOT, "figure2_AB_survival_residence")
            plt.show()
            """
        ),
        md("## 4. Figure 2C–D: occupancy curves and apparent affinity shift"),
        code(
            """
            concentrations = np.logspace(-13, -6, 240)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
            geometry_labels = list(dict.fromkeys(label.split("__")[0] for label in onoff.condition_label))
            geometry_colors = dict(zip(geometry_labels, plt.cm.viridis(np.linspace(0.08, 0.92, len(geometry_labels)))))
            for label, frame in onoff.groupby("condition_label", sort=False):
                geometry_label = label.split("__")[0]
                rebinding_on = bool(frame["rebinding_multiplier"].iloc[0] > 0)
                color = geometry_colors[geometry_label]
                linestyle = "-" if rebinding_on else ":"
                curves = np.vstack([occupancy_from_bound_time(row.mean_bound_time_s, row.k_on_M_inv_s, concentrations) for row in frame.itertuples()])
                axes[0].plot(concentrations, curves.mean(0), color=color, linestyle=linestyle)
                axes[0].fill_between(concentrations, np.clip(curves.mean(0)-curves.std(0,ddof=1)/np.sqrt(len(curves)),0,1), np.clip(curves.mean(0)+curves.std(0,ddof=1)/np.sqrt(len(curves)),0,1), color=color, alpha=.14 if rebinding_on else .06)
            classical = classical_occupancy(onoff.k_on_M_inv_s.iloc[0], onoff.k_off_s.iloc[0], concentrations)
            axes[0].plot(concentrations, classical, color="black", linestyle="-.")
            axes[0].set(xscale="log", xlabel=r"$C_{bulk}$ (M)", ylabel=r"Estimated occupancy $\theta$", ylim=(-.02,1.02), title="C  Rebinding shifts apparent affinity")
            geometry_legend = axes[0].legend(
                handles=[Line2D([0],[0], color=geometry_colors[label], label=label) for label in geometry_labels],
                title="Geometry", loc="upper left",
            )
            axes[0].add_artist(geometry_legend)
            axes[0].legend(
                handles=[
                    Line2D([0],[0], color="0.15", linestyle="-", label="Rebinding on"),
                    Line2D([0],[0], color="0.15", linestyle=":", label="Rebinding off"),
                    Line2D([0],[0], color="black", linestyle="-.", label="Classical Langmuir"),
                ],
                title="Model", loc="lower right",
            )

            order = list(onoff.groupby("condition_index").condition_label.first().sort_index())
            values = [onoff.loc[onoff.condition_label.eq(label), "affinity_shift_fold"] for label in order]
            positions = np.arange(1, len(order)+1)
            for position, label, data in zip(positions, order, values):
                geometry_label = label.split("__")[0]
                rebinding_on = bool(onoff.loc[onoff.condition_label.eq(label), "rebinding_multiplier"].iloc[0] > 0)
                color = geometry_colors[geometry_label]
                axes[1].errorbar(position, data.mean(), yerr=data.sem(), marker="o", markerfacecolor=color if rebinding_on else "white", markeredgecolor=color, color=color, linestyle="none", capsize=4)
            axes[1].set_xticks(positions, order)
            axes[1].axhline(1, color="black", ls="--")
            axes[1].set(yscale="log", ylabel=r"Affinity shift $K_D/K_{D,app}$", title="D  Apparent affinity enhancement")
            plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            if SAVE_FIGURES: save_figure(fig, FIGURE_ROOT, "figure2_CD_occupancy_affinity")
            plt.show()
            """
        ),
        md("## 5. Rebinding-strength dose response"),
        code(
            """
            dose_stats = condition_statistics(dose, ["rebinding_multiplier"], ["mean_rebindings", "mean_bound_time_s", "affinity_shift_fold"]).sort_values("rebinding_multiplier")
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            for ax, metric, ylabel in zip(axes, ["mean_rebindings", "mean_bound_time_s", "affinity_shift_fold"], ["Mean rebindings", "Cumulative bound time (s)", r"$K_D/K_{D,app}$"]):
                ax.errorbar(dose_stats.rebinding_multiplier, dose_stats[f"{metric}_mean"], yerr=dose_stats[f"{metric}_sem"], marker="o", capsize=3)
                ax.set(xlabel=r"Post-release $k_{on}$ multiplier", ylabel=ylabel)
            axes[1].set_yscale("log"); axes[2].set_yscale("log"); axes[2].axhline(1,color="k",ls="--")
            if SAVE_FIGURES: save_figure(fig, FIGURE_ROOT, "figure2_rebinding_dose_response")
            plt.show()
            display(dose_stats)
            """
        ),
        code("""OUTPUT = FIGURE_ROOT / "figure2_tables"\nif SAVE_FIGURES:\n    OUTPUT.mkdir(parents=True, exist_ok=True)\n    onoff.to_csv(OUTPUT / "figure2_onoff_replicates.csv", index=False)\n    dose.to_csv(OUTPUT / "figure2_dose_replicates.csv", index=False)"""),
    ],
)


fig3 = notebook(
    "Manuscript Figure 3 — Curvature and receptor density act synergistically",
    "Analyzes the 5×5 cylindrical-sheet curvature/receptor-density factorial experiment and explicitly estimates the curvature × density interaction.",
    [
        code(SETUP),
        code("""DATA_ROOT = PROJECT_ROOT / "results/261005_manuscript_fig3_curvature_density_factorial"\nrun_index = discover_runs(DATA_ROOT)\nreplicates = summarize_replicates(run_index)\nreplicates["curvature_nm_inv"] = replicates["curvature_m_inv"] * 1e-9\nreplicates["density_1e16_m2"] = replicates["receptor_density_m2"] / 1e16\ndisplay(replicates.groupby(["curvature_nm_inv", "density_1e16_m2"]).size().unstack())"""),
        md("## 1. Quality control"),
        code("""display(replicates.groupby(["curvature_nm_inv", "density_1e16_m2"])[["n_receptors", "n_trajectories", "censoring_fraction"]].agg(["mean", "min", "max"]))"""),
        md("## 2. Figure 3A–C: factorial heatmaps"),
        code(
            """
            metrics = [("mean_bound_time_s", "Mean cumulative bound time (s)"), ("mean_rebindings", "Mean rebindings"), ("affinity_shift_fold", r"$K_D/K_{D,app}$")]
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
            for ax, (metric, title) in zip(axes, metrics):
                matrix = replicates.pivot_table(index="curvature_nm_inv", columns="density_1e16_m2", values=metric, aggfunc="mean").sort_index().sort_index(axis=1)
                values = matrix.to_numpy(float)
                positive_values = values[np.isfinite(values) & (values > 0)]
                if positive_values.size == 0:
                    raise ValueError(f"{metric} has no positive values and cannot use a log color scale.")
                vmin, vmax = positive_values.min(), positive_values.max()
                if np.isclose(vmin, vmax):
                    vmax = vmin * (1 + 1e-6)
                log_values = np.ma.masked_less_equal(values, 0)
                cmap = mpl.colormaps["magma"].copy()
                cmap.set_bad("0.9")
                image = ax.imshow(log_values, origin="lower", aspect="auto", cmap=cmap, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
                ax.set_xticks(range(matrix.shape[1]), [f"{x:g}" for x in matrix.columns])
                ax.set_yticks(range(matrix.shape[0]), [f"{x:.4f}" for x in matrix.index])
                ax.set(xlabel=r"Receptor density ($10^{16}$ m$^{-2}$)", ylabel=r"Curvature (nm$^{-1}$)", title=title)
                colorbar = fig.colorbar(image, ax=ax, shrink=.85)
                colorbar.set_label(title)
            if SAVE_FIGURES: save_figure(fig, FIGURE_ROOT, "figure3_ABC_factorial_heatmaps")
            plt.show()
            """
        ),
        md("## 3. Figure 3D: curvature response at each density"),
        code(
            """
            stats = condition_statistics(replicates, ["curvature_nm_inv", "density_1e16_m2"], ["affinity_shift_fold", "mean_bound_time_s", "mean_rebindings"])
            fig, ax = plt.subplots(figsize=(7,5), constrained_layout=True)
            for density, frame in stats.groupby("density_1e16_m2"):
                frame=frame.sort_values("curvature_nm_inv")
                ax.errorbar(frame.curvature_nm_inv, frame.affinity_shift_fold_mean, yerr=frame.affinity_shift_fold_sem, marker="o", capsize=2, label=f"{density:g}")
            ax.axhline(1,color="k",ls="--")
            ax.set(xlabel=r"Concave curvature (nm$^{-1}$)", ylabel=r"Affinity shift $K_D/K_{D,app}$", yscale="log", title="Curvature response strengthens with receptor density")
            ax.legend(title=r"Density ($10^{16}$ m$^{-2}$)")
            if SAVE_FIGURES: save_figure(fig, FIGURE_ROOT, "figure3_D_synergy_lines")
            plt.show()
            """
        ),
        md("## 4. Interaction model\n\nThe response and predictors are log/standardized so the interaction coefficient tests whether the curvature response depends on receptor density."),
        code(
            """
            model_data = replicates.replace([np.inf,-np.inf],np.nan).dropna(subset=["affinity_shift_fold", "curvature_nm_inv", "receptor_density_m2"]).copy()
            model_data["response"] = np.log(model_data.affinity_shift_fold)
            curvature_center = model_data.curvature_nm_inv.mean()
            curvature_scale = model_data.curvature_nm_inv.std()
            log_density = np.log10(model_data.receptor_density_m2)
            log_density_center = log_density.mean()
            log_density_scale = log_density.std()
            model_data["curvature_z"] = (model_data.curvature_nm_inv-curvature_center)/curvature_scale
            model_data["log_density_z"] = (log_density-log_density_center)/log_density_scale
            X = np.column_stack([np.ones(len(model_data)), model_data.curvature_z, model_data.log_density_z, model_data.curvature_z*model_data.log_density_z])
            names = ["intercept", "curvature", "log_density", "curvature_x_log_density"]
            beta, *_ = np.linalg.lstsq(X, model_data.response, rcond=None)
            residual = model_data.response.to_numpy()-X@beta
            covariance = np.linalg.inv(X.T@X)*(residual@residual)/(len(X)-X.shape[1])
            interaction_table = pd.DataFrame({"term":names,"coefficient":beta,"standard_error":np.sqrt(np.diag(covariance))})
            interaction_table["t_value"] = interaction_table.coefficient/interaction_table.standard_error
            display(interaction_table)
            """
        ),
        md("## 5. Interaction-model visualization\n\nLines show the fitted curvature response at each receptor density. Shading is the model-based 95% confidence interval; points and error bars are the observed replicate mean ± SEM."),
        code(
            """
            curvature_grid = np.linspace(model_data.curvature_nm_inv.min(), model_data.curvature_nm_inv.max(), 200)
            density_values = np.sort(model_data.receptor_density_m2.unique())
            density_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(density_values)))
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            for color, density in zip(density_colors, density_values):
                curvature_z = (curvature_grid-curvature_center)/curvature_scale
                density_z = (np.log10(density)-log_density_center)/log_density_scale
                prediction_matrix = np.column_stack([
                    np.ones_like(curvature_z), curvature_z,
                    np.full_like(curvature_z, density_z), curvature_z*density_z,
                ])
                predicted_log = prediction_matrix @ beta
                predicted_se = np.sqrt(np.einsum("ij,jk,ik->i", prediction_matrix, covariance, prediction_matrix))
                label = rf"{density/1e16:g} $\\times 10^{{16}}$ m$^{{-2}}$"
                ax.plot(curvature_grid, np.exp(predicted_log), color=color, label=label)
                ax.fill_between(curvature_grid, np.exp(predicted_log-1.96*predicted_se), np.exp(predicted_log+1.96*predicted_se), color=color, alpha=0.15)
                observed = stats.loc[np.isclose(stats.density_1e16_m2, density/1e16)].sort_values("curvature_nm_inv")
                ax.errorbar(observed.curvature_nm_inv, observed.affinity_shift_fold_mean, yerr=observed.affinity_shift_fold_sem, color=color, marker="o", linestyle="none", capsize=3)
            ax.axhline(1, color="black", linestyle="--")
            ax.set(xlabel=r"Concave curvature (nm$^{-1}$)", ylabel=r"Affinity shift $K_D/K_{D,app}$", yscale="log", title="Fitted curvature × receptor-density interaction")
            ax.legend(title="Receptor density", bbox_to_anchor=(1.02, 1), loc="upper left")
            if SAVE_FIGURES: save_figure(fig, FIGURE_ROOT, "figure3_interaction_model")
            plt.show()
            """
        ),
        code("""if SAVE_FIGURES:\n    out=FIGURE_ROOT/"figure3_tables"; out.mkdir(parents=True,exist_ok=True)\n    replicates.to_csv(out/"figure3_replicates.csv",index=False)\n    stats.to_csv(out/"figure3_condition_statistics.csv",index=False)\n    interaction_table.to_csv(out/"figure3_interaction_model.csv",index=False)"""),
    ],
)


fig4 = notebook(
    "Manuscript Figure 4 — Local curvature predicts heterogeneous retention and occupancy",
    "Builds receptor-resolved maps and tests local curvature while accounting for surface height and local receptor spacing.",
    [
        code(SETUP + "\nimport importlib\nfrom scipy.spatial import cKDTree\nimport utils.escape_assay_viz as escape_assay_viz\nimportlib.reload(escape_assay_viz)\nfrom utils.escape_assay_viz import load_escape_assay_run, plot_surface_metric_3d, plot_surface_metric_xy"),
        code("""DATA_ROOT = PROJECT_ROOT / "results/261005_manuscript_fig4_sinusoidal_local_curvature"\nrun_index = discover_runs(DATA_ROOT)\nreplicates = summarize_replicates(run_index)\nreceptors = load_receptor_summaries(run_index)\nprint(f"{len(run_index)} replicates, {len(receptors):,} receptor locations")\ndisplay(replicates[["replicate","n_receptors","n_trajectories","censoring_fraction","mean_bound_time_s","mean_rebindings"]])"""),
        md("## 1. Add local receptor-neighborhood covariates"),
        code(
            """
            enriched=[]
            for run_dir, frame in receptors.groupby("run_directory", sort=False):
                frame=frame.copy()
                xyz=frame[["surface_x_m","surface_y_m","surface_z_m"]].to_numpy(float)
                tree=cKDTree(xyz)
                distances,_=tree.query(xyz,k=min(2,len(xyz)))
                frame["nearest_receptor_distance_m"] = distances[:,-1] if len(xyz)>1 else np.nan
                frame["neighbors_within_10nm"] = np.array([len(v)-1 for v in tree.query_ball_point(xyz,10e-9)])
                enriched.append(frame)
            receptors=pd.concat(enriched,ignore_index=True)
            curvature_column="mean_curvature_m_inv"
            receptors["mean_curvature_nm_inv"]=receptors[curvature_column]*1e-9
            """
        ),
        md("## 2. Figure 4A–C: representative spatial maps"),
        code(
            """
            SELECTED_REPLICATE = int(run_index.replicate.min())
            MAP_CONCENTRATION_M = 1e-9
            selected_row=run_index.loc[run_index.replicate.eq(SELECTED_REPLICATE)].iloc[0]
            archive=load_escape_assay_run(selected_row.run_directory)
            map_summary=archive.receptor_summary.copy()
            map_summary["theta_effective"] = occupancy_from_bound_time(map_summary.mean_bound_time_s, archive.params.k_on_M_inv_s, MAP_CONCENTRATION_M)
            panels=[("mean_bound_time_s","viridis","Cumulative bound time (s)"),("mean_rebindings","magma","Mean rebindings"),("theta_effective","cividis",rf"Occupancy at {MAP_CONCENTRATION_M:.0e} M")]
            fig=plt.figure(figsize=(18,10),constrained_layout=True)
            for j,(metric,cmap,title) in enumerate(panels):
                ax=fig.add_subplot(2,3,j+1,projection="3d")
                plot_surface_metric_3d(archive.geometry,map_summary,metric,a_m=archive.params.a_m,cmap=cmap,show_receptors=False,colorbar_pad=0.18,ax=ax)
                ax.view_init(elev=35,azim=-60); ax.set_title(title)
                ax2=fig.add_subplot(2,3,j+4)
                plot_surface_metric_xy(archive.geometry,map_summary,metric,a_m=archive.params.a_m,cmap=cmap,show_receptors=False,ax=ax2)
            if SAVE_FIGURES: save_figure(fig,FIGURE_ROOT,"figure4_ABC_spatial_maps")
            plt.show()
            """
        ),
        md("## 3. Figure 4D–E: curvature relationships"),
        code(
            """
            receptors["curvature_bin"] = pd.qcut(receptors.mean_curvature_nm_inv, 15, duplicates="drop")
            per_rep_bin=(receptors.groupby(["replicate","curvature_bin"],observed=True).agg(curvature=("mean_curvature_nm_inv","mean"),bound_time=("mean_bound_time_s","mean"),rebindings=("mean_rebindings","mean")).reset_index())
            binned=(per_rep_bin.groupby("curvature_bin",observed=True).agg(curvature=("curvature","mean"),bound_mean=("bound_time","mean"),bound_sem=("bound_time","sem"),rebind_mean=("rebindings","mean"),rebind_sem=("rebindings","sem")).reset_index())
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
            axes[0].errorbar(binned.curvature,binned.bound_mean,yerr=binned.bound_sem,marker="o",capsize=2)
            axes[1].errorbar(binned.curvature,binned.rebind_mean,yerr=binned.rebind_sem,marker="o",capsize=2,color="tab:orange")
            axes[0].set(xlabel=r"Signed mean curvature (nm$^{-1}$)",ylabel="Mean cumulative bound time (s)",title="D  Local retention")
            axes[1].set(xlabel=r"Signed mean curvature (nm$^{-1}$)",ylabel="Mean rebindings",title="E  Local rebinding")
            if SAVE_FIGURES: save_figure(fig,FIGURE_ROOT,"figure4_DE_curvature_relationships")
            plt.show()
            """
        ),
        md("## 4. Multivariable receptor-level model\n\nReplicate indicator terms account for layout-to-layout shifts. Predictors are standardized before fitting."),
        code(
            """
            predictors=["mean_curvature_m_inv","gaussian_curvature_m_inv2","surface_z_m","nearest_receptor_distance_m","neighbors_within_10nm"]
            data=receptors.dropna(subset=predictors+["mean_bound_time_s"]).copy()
            Xparts=[np.ones(len(data))]; names=["intercept"]
            for name in predictors:
                values=data[name].to_numpy(float); values=(values-values.mean())/values.std()
                Xparts.append(values); names.append(name)
            replicate_dummies=pd.get_dummies(data.replicate,drop_first=True,dtype=float)
            Xparts.extend([replicate_dummies[col].to_numpy() for col in replicate_dummies]); names.extend([f"replicate_{col}" for col in replicate_dummies])
            X=np.column_stack(Xparts); y=np.log(np.clip(data.mean_bound_time_s.to_numpy(float),1e-12,None))
            beta,*_=np.linalg.lstsq(X,y,rcond=None); residual=y-X@beta
            cov=np.linalg.pinv(X.T@X)*(residual@residual)/(len(y)-X.shape[1])
            coefficients=pd.DataFrame({"term":names,"coefficient":beta,"standard_error":np.sqrt(np.diag(cov))})
            coefficients["t_value"]=coefficients.coefficient/coefficients.standard_error
            display(coefficients.iloc[:len(predictors)+1])
            """
        ),
        code("""if SAVE_FIGURES:\n    out=FIGURE_ROOT/"figure4_tables"; out.mkdir(parents=True,exist_ok=True)\n    binned.to_csv(out/"figure4_curvature_bins.csv",index=False)\n    coefficients.to_csv(out/"figure4_multivariable_model.csv",index=False)"""),
    ],
)


supp_num = notebook(
    "Supplement — Numerical and boundary convergence",
    "Tests escape height, lattice spacing, lateral domain size, and observation horizon. Each condition is paired to its same-seed baseline replicate.",
    [
        code(SETUP),
        code("""DATA_ROOT=PROJECT_ROOT/"results/261005_manuscript_supp_numerical_convergence"\nindex=discover_runs(DATA_ROOT)\nreplicates=summarize_replicates(index)\nbaseline=replicates.loc[replicates.test_parameter.eq("baseline"),["replicate","mean_bound_time_s","mean_rebindings","affinity_shift_fold","censoring_fraction"]].set_index("replicate").add_suffix("_baseline")\npaired=replicates.join(baseline,on="replicate")\nfor metric in ["mean_bound_time_s","mean_rebindings","affinity_shift_fold"]:\n    paired[f"{metric}_ratio"] = paired[metric]/paired[f"{metric}_baseline"]\ndisplay(paired[["condition_label","replicate","test_parameter","test_value","censoring_fraction","mean_bound_time_s_ratio","mean_rebindings_ratio","affinity_shift_fold_ratio"]])"""),
        code(
            """
            parameters=["escape_z_m","a_m","lateral_domain_m","max_time_s"]
            fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
            for ax,parameter in zip(axes.flat,parameters):
                frame=paired.loc[paired.test_parameter.isin([parameter,"baseline"])].copy()
                if parameter=="escape_z_m": frame.loc[frame.test_parameter.eq("baseline"),"test_value"]=75e-9
                elif parameter=="a_m": frame.loc[frame.test_parameter.eq("baseline"),"test_value"]=1e-9
                elif parameter=="lateral_domain_m": frame.loc[frame.test_parameter.eq("baseline"),"test_value"]=100e-9
                elif parameter=="max_time_s": frame.loc[frame.test_parameter.eq("baseline"),"test_value"]=10000
                summary=frame.groupby("test_value").affinity_shift_fold_ratio.agg(["mean","sem"]).reset_index().sort_values("test_value")
                scale=1e9 if parameter in {"escape_z_m","a_m","lateral_domain_m"} else 1
                ax.errorbar(summary.test_value*scale,summary["mean"],yerr=summary["sem"],marker="o",capsize=3)
                ax.axhline(1,color="k",ls="--"); ax.set(xlabel=parameter+(" (nm)" if scale==1e9 else " (s)"),ylabel="Affinity-shift ratio to baseline",title=parameter)
                if parameter=="max_time_s": ax.set_xscale("log")
            if SAVE_FIGURES: save_figure(fig,FIGURE_ROOT,"supp_numerical_convergence")
            plt.show()
            """
        ),
        code("""fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)\norder=paired.groupby("condition_index").condition_label.first().sort_index().tolist()\nax.boxplot([paired.loc[paired.condition_label.eq(x),"censoring_fraction"] for x in order],tick_labels=order)\nax.set(ylabel="Censoring fraction",title="Censoring diagnostic"); plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")\nplt.show()"""),
    ],
)


supp_controls = notebook(
    "Supplement — Fixed receptor number, kinetic robustness, and height controls",
    "Analyzes the three mechanistic-control datasets that test receptor-number confounding, kinetic generality, and correlation with distance to the escape plane.",
    [
        code(SETUP),
        code("""FIXED_ROOT=PROJECT_ROOT/"results/261005_manuscript_supp_fixed_receptor_count"\nKINETIC_ROOT=PROJECT_ROOT/"results/261005_manuscript_supp_kinetic_robustness"\nHEIGHT_ROOT=PROJECT_ROOT/"results/261005_manuscript_supp_sinusoidal_height_control"\nfixed=summarize_replicates(discover_runs(FIXED_ROOT)); kinetic=summarize_replicates(discover_runs(KINETIC_ROOT)); height=summarize_replicates(discover_runs(HEIGHT_ROOT))\nfixed["curvature_nm_inv"]=fixed.curvature_m_inv*1e-9\ndisplay(fixed.groupby("condition_label").n_receptors.agg(["min","mean","max"]))"""),
        md("## 1. Fixed total receptor number"),
        code("""stats=condition_statistics(fixed,["curvature_nm_inv"],["mean_bound_time_s","mean_rebindings","affinity_shift_fold"]).sort_values("curvature_nm_inv")\nfig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)\nfor ax,metric,label in zip(axes,["mean_bound_time_s","mean_rebindings","affinity_shift_fold"],["Bound time (s)","Mean rebindings",r"$K_D/K_{D,app}$"]):\n ax.errorbar(stats.curvature_nm_inv,stats[f"{metric}_mean"],yerr=stats[f"{metric}_sem"],marker="o",capsize=3); ax.set(xlabel=r"Curvature (nm$^{-1}$)",ylabel=label)\nif SAVE_FIGURES: save_figure(fig,FIGURE_ROOT,"supp_fixed_receptor_number")\nplt.show()"""),
        md("## 2. Kinetic robustness"),
        code("""kinetic["geometry"] = np.where(kinetic.curvature_m_inv.fillna(0).eq(0),"planar","semicircular")\nsummary=condition_statistics(kinetic,["k_on_M_inv_s","k_off_s","geometry"],["affinity_shift_fold"])\nfig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)\nfor geometry,frame in summary.groupby("geometry"):\n labels=[rf"$10^{{{int(np.log10(kon))}}},10^{{{int(np.log10(koff))}}}$" for kon,koff in zip(frame.k_on_M_inv_s,frame.k_off_s)]\n x=np.arange(len(frame)); ax.errorbar(x+(0 if geometry=="planar" else .08),frame.affinity_shift_fold_mean,yerr=frame.affinity_shift_fold_sem,marker="o",ls="none",capsize=3,label=geometry)\nax.axhline(1,color="k",ls="--"); ax.set_xticks(np.arange(len(labels)),labels,rotation=45,ha="right",rotation_mode="anchor"); ax.set(yscale="log",xlabel=r"$(k_{on}, k_{off})$",ylabel=r"$K_D/K_{D,app}$")\nax.legend();\nif SAVE_FIGURES: save_figure(fig,FIGURE_ROOT,"supp_kinetic_robustness")\nplt.show()"""),
        md("## 3. Sinusoidal height and escape-plane control"),
        code("""height["mean_z_nm"]=height.mean_z_m*1e9\nstats_h=condition_statistics(height,["boundary_series","mean_z_nm"],["mean_bound_time_s","mean_rebindings","affinity_shift_fold"])\nfig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)\nfor series,frame in stats_h.groupby("boundary_series"):\n frame=frame.sort_values("mean_z_nm")\n for ax,metric in zip(axes,["mean_bound_time_s","mean_rebindings","affinity_shift_fold"]): ax.errorbar(frame.mean_z_nm,frame[f"{metric}_mean"],yerr=frame[f"{metric}_sem"],marker="o",capsize=3,label=series)\nfor ax,label in zip(axes,["Bound time (s)","Mean rebindings",r"$K_D/K_{D,app}$"]): ax.set(xlabel="Mean surface height (nm)",ylabel=label)\naxes[0].legend()\nif SAVE_FIGURES: save_figure(fig,FIGURE_ROOT,"supp_sinusoidal_height_control")\nplt.show()"""),
    ],
)


OUTPUTS = {
    "analyze_manuscript_figure2_rebinding_affinity.ipynb": fig2,
    "analyze_manuscript_figure3_curvature_density.ipynb": fig3,
    "analyze_manuscript_figure4_local_curvature.ipynb": fig4,
    "analyze_manuscript_supp_numerical_convergence.ipynb": supp_num,
    "analyze_manuscript_supp_mechanistic_controls.ipynb": supp_controls,
}

for filename, nb in OUTPUTS.items():
    destination = HERE / filename
    nbf.write(nb, destination)
    print(destination)
