#!/usr/bin/env python3
"""Build the analysis notebook for the final low-concentration simulations."""

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


def manuscript_formatting() -> str:
    """Copy the user's established formatting cell without modifying it."""

    source = HERE / "analyze_manuscript_figure2_rebinding_affinity.ipynb"
    with source.open(encoding="utf-8") as stream:
        notebook = json.load(stream)
    return "".join(notebook["cells"][0]["source"])


cells = [
    nbf.v4.new_code_cell(manuscript_formatting()),
    markdown(
        r"""
        # Final computational figure — Low-concentration occupancy gain from cross-rebinding

        This notebook combines the full-rebinding, self-only, and no-rebinding
        controls. It maps imposed background occupancy to normalized concentration,

        $$
        \frac{C}{K_D} =
        \frac{\phi}{(1-\phi)k_{off}\tau_{eff}(\phi)},
        $$

        and compares all mechanisms at the matched concentration
        $C/K_D=10^{-3}$. Replicate receptor layouts are the statistical unit.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys
        import warnings
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
            discover_runs, save_figure, summarize_replicates,
        )

        SAVE_FIGURES = False
        REFRESH_CACHE = False
        FIGURE_ROOT = PROJECT_ROOT / "analysis_tools" / "manuscript_panels"
        CACHE_ROOT = PROJECT_ROOT / "analysis_tools" / ".analysis_cache"
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        """
    ),
    markdown(
        """
        ## 1. Select and discover the three result folders

        Change these paths after copying the cluster results. The first pass creates
        local index and summary caches; subsequent runs avoid rescanning and reparsing
        all 1,032 trajectory archives. Set `REFRESH_CACHE = True` after adding or
        replacing result files.
        """
    ),
    code(
        """
        FULL_ROOT = PROJECT_ROOT / "results/261006_manuscript_final_low_concentration_full_rebinding"
        SELF_ROOT = PROJECT_ROOT / "results/261006_manuscript_final_low_concentration_self_only"
        NONE_ROOT = PROJECT_ROOT / "results/261006_manuscript_final_low_concentration_no_rebinding"

        datasets = {
            "Full rebinding": (FULL_ROOT, "final_lowC_full"),
            "Self only": (SELF_ROOT, "final_lowC_self"),
            "No rebinding": (NONE_ROOT, "final_lowC_none"),
        }

        indexes = {}
        summaries = {}
        for mode, (root, cache_stem) in datasets.items():
            indexes[mode] = discover_runs(
                root,
                cache_path=CACHE_ROOT / f"{cache_stem}_index.csv",
                refresh_cache=REFRESH_CACHE,
            )
            summaries[mode] = summarize_replicates(
                indexes[mode],
                cache_path=CACHE_ROOT / f"{cache_stem}_replicates.csv",
                refresh_cache=REFRESH_CACHE,
            )
            summaries[mode]["simulation_mode"] = mode

        runs = pd.concat(indexes, names=["simulation_mode", "row"]).reset_index(level=0)
        replicates = pd.concat(summaries.values(), ignore_index=True)
        replicates["geometry"] = replicates["condition_label"].str.split("__").str[0]

        expected = {"Full rebinding": 504, "Self only": 504, "No rebinding": 24}
        observed = {mode: len(frame) for mode, frame in indexes.items()}
        display(pd.DataFrame({"expected_runs": expected, "observed_runs": observed}))
        if observed != expected:
            warnings.warn(
                "The sweep is incomplete or a stale cache is being used. "
                "Set REFRESH_CACHE=True after all files are present."
            )
        """
    ),
    markdown("## 2. Quality control and self/cross classification"),
    code(
        """
        qc = (replicates.groupby(["simulation_mode", "geometry"], sort=False)
              .agg(runs=("replicate", "size"),
                   receptors_mean=("n_receptors", "mean"),
                   trajectories=("n_trajectories", "sum"),
                   censoring_max=("censoring_fraction", "max"),
                   mean_bound_time_s=("mean_bound_time_s", "mean"),
                   mean_self_rebindings=("mean_self_rebindings", "mean"),
                   mean_cross_rebindings=("mean_cross_rebindings", "mean"))
              .reset_index())
        display(qc)

        self_cross_check = np.allclose(
            replicates["mean_rebindings"],
            replicates["mean_self_rebindings"] + replicates["mean_cross_rebindings"],
            rtol=1e-10, atol=1e-12,
        )
        assert self_cross_check, "Self + cross counts do not equal total rebindings."
        assert np.allclose(
            replicates.loc[replicates.simulation_mode.eq("Self only"), "mean_cross_rebindings"],
            0,
        ), "The self-only control contains cross-rebinding events."
        print("Self/cross accounting passed; self-only cross-rebindings are zero.")
        """
    ),
    markdown(
        r"""
        ## 3. Convert the occupancy grid to $C/K_D$

        Each replicate is converted separately before averaging. This preserves
        receptor-layout uncertainty and avoids treating trajectories or occupancy-grid
        points as independent biological replicates.
        """
    ),
    code(
        """
        grid_runs = replicates.loc[
            replicates.simulation_mode.isin(["Full rebinding", "Self only"])
        ].copy()
        phi = grid_runs["background_occupancy_fraction"].to_numpy(float)
        resistance = (
            (1 - phi)
            * grid_runs["k_off_s"].to_numpy(float)
            * grid_runs["mean_bound_time_s"].to_numpy(float)
        )
        grid_runs["C_over_KD"] = np.divide(
            phi, resistance,
            out=np.full(len(grid_runs), np.nan),
            where=resistance > 0,
        )

        def interpolate_group(frame, x_values, column):
            usable = frame[["C_over_KD", column]].replace([np.inf, -np.inf], np.nan).dropna()
            usable = usable.loc[(usable.C_over_KD > 0) & (usable[column] >= 0)]
            usable = usable.sort_values("C_over_KD").drop_duplicates("C_over_KD")
            if len(usable) < 2:
                return np.full_like(np.asarray(x_values, float), np.nan)
            log_x = np.log10(np.asarray(x_values, float))
            source_x = np.log10(usable.C_over_KD.to_numpy(float))
            values = np.interp(log_x, source_x, usable[column].to_numpy(float))
            values[(log_x < source_x.min()) | (log_x > source_x.max())] = np.nan
            return values

        RESPONSE_X = np.logspace(-5, -1, 241)
        TARGET_C_OVER_KD = 1e-3
        response_rows = []
        target_rows = []
        metrics = [
            "background_occupancy_fraction", "mean_bound_time_s",
            "mean_rebindings", "mean_self_rebindings", "mean_cross_rebindings",
        ]
        for (mode, geometry, replicate), frame in grid_runs.groupby(
            ["simulation_mode", "geometry", "replicate"], sort=False
        ):
            values = {
                metric: interpolate_group(frame, RESPONSE_X, metric)
                for metric in metrics
            }
            response_rows.append(pd.DataFrame({
                "simulation_mode": mode, "geometry": geometry,
                "replicate": replicate, "C_over_KD": RESPONSE_X,
                **values,
            }))
            target_rows.append({
                "simulation_mode": mode, "geometry": geometry,
                "replicate": replicate,
                **{
                    metric: interpolate_group(frame, [TARGET_C_OVER_KD], metric)[0]
                    for metric in metrics
                },
            })

        response = pd.concat(response_rows, ignore_index=True)
        target = pd.DataFrame(target_rows).rename(
            columns={"background_occupancy_fraction": "theta"}
        )
        missing_target = target.loc[target.theta.isna(),
                                    ["simulation_mode", "geometry", "replicate"]]
        if not missing_target.empty:
            display(missing_target)
            raise ValueError(
                "C/KD=1e-3 falls outside the simulated concentration range for "
                "one or more replicates. Inspect C_over_KD before plotting."
            )

        # The no-rebinding control has occupancy-independent residence time, so
        # its matched-concentration occupancy follows directly from tau_eff.
        no_rebinding = replicates.loc[replicates.simulation_mode.eq("No rebinding")].copy()
        q = TARGET_C_OVER_KD * no_rebinding.k_off_s * no_rebinding.mean_bound_time_s
        no_rebinding["theta"] = q / (1 + q)
        target = pd.concat([
            target,
            no_rebinding[["simulation_mode", "geometry", "replicate", "theta",
                          "mean_bound_time_s", "mean_rebindings",
                          "mean_self_rebindings", "mean_cross_rebindings"]],
        ], ignore_index=True)

        classical_theta = TARGET_C_OVER_KD / (1 + TARGET_C_OVER_KD)
        display(target.groupby(["simulation_mode", "geometry"])
                .theta.agg(["mean", "sem", "count"]))
        """
    ),
    markdown(
        r"""
        ## 4. Final figure: confinement increases low-concentration occupancy through cross-rebinding

        Panel A shows the low-concentration response without unstable division by the
        very small classical occupancy. Panel B compares mechanisms at exactly
        $C/K_D=10^{-3}$. Panel C separates self- and cross-rebinding, and panel D
        relates the absolute cross-rebinding occupancy contribution to the number of
        cross-rebinding events.
        """
    ),
    code(
        """
        geometry_order = ["planar", "R100nm", "R49p5nm"]
        geometry_names = {
            "planar": "Planar",
            "R100nm": "Curved sheet (R = 100 nm)",
            "R49p5nm": "Semicircular sheet",
        }
        colors = dict(zip(
            geometry_order,
            plt.cm.viridis(np.linspace(0.12, 0.88, len(geometry_order))),
        ))

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)

        # A: occupancy response in the dilute regime.
        ax = axes[0, 0]
        for geometry in geometry_order:
            frame = response.loc[
                response.geometry.eq(geometry)
                & response.simulation_mode.eq("Full rebinding")
            ]
            stats = (frame.groupby("C_over_KD").background_occupancy_fraction
                     .agg(["mean", "sem"]).reset_index())
            ax.plot(stats.C_over_KD, stats["mean"], color=colors[geometry],
                    label=geometry_names[geometry])
            ax.fill_between(stats.C_over_KD,
                            np.clip(stats["mean"] - stats["sem"], 1e-12, 1),
                            np.clip(stats["mean"] + stats["sem"], 1e-12, 1),
                            color=colors[geometry], alpha=0.16, linewidth=0)
        ax.plot(RESPONSE_X, RESPONSE_X / (1 + RESPONSE_X), color="black",
                linestyle="--", label="Classical")
        ax.axvline(TARGET_C_OVER_KD, color="0.35", linestyle=":", linewidth=2)
        ax.set(xscale="log", yscale="log", xlabel=r"Normalized concentration $C/K_D$",
               ylabel=r"Predicted occupancy $\theta$",
               title="A  Confinement amplifies dilute occupancy")
        ax.legend(frameon=False)

        # B: classical -> no rebinding -> self only -> full at matched C/KD.
        ax = axes[0, 1]
        mode_order = ["Classical", "No rebinding", "Self only", "Full rebinding"]
        x_positions = np.arange(len(mode_order))
        for geometry in geometry_order:
            rows = [{"simulation_mode": "Classical", "replicate": replicate,
                     "theta": classical_theta} for replicate in range(1, 9)]
            rows.extend(target.loc[target.geometry.eq(geometry)].to_dict("records"))
            frame = pd.DataFrame(rows)
            stats = frame.groupby("simulation_mode").theta.agg(["mean", "sem"])
            means = np.array([stats.loc[mode, "mean"] for mode in mode_order])
            sems = np.array([stats.loc[mode, "sem"] for mode in mode_order])
            ax.errorbar(x_positions, means, yerr=sems, color=colors[geometry],
                        marker="o", capsize=4, label=geometry_names[geometry])
        ax.set_xticks(x_positions, mode_order)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set(yscale="log", ylabel=rf"Occupancy at $C/K_D={TARGET_C_OVER_KD:.0e}$",
               title="B  Cross-rebinding supplies the additional gain")
        ax.legend(frameon=False)

        # C: full-rebinding event composition at matched concentration.
        ax = axes[1, 0]
        full_target = target.loc[target.simulation_mode.eq("Full rebinding")].copy()
        event_stats = (full_target.groupby("geometry")
                       [["mean_self_rebindings", "mean_cross_rebindings"]]
                       .agg(["mean", "sem"]))
        x = np.arange(len(geometry_order))
        width = 0.36
        self_mean = event_stats.loc[geometry_order, ("mean_self_rebindings", "mean")].to_numpy()
        self_sem = event_stats.loc[geometry_order, ("mean_self_rebindings", "sem")].to_numpy()
        cross_mean = event_stats.loc[geometry_order, ("mean_cross_rebindings", "mean")].to_numpy()
        cross_sem = event_stats.loc[geometry_order, ("mean_cross_rebindings", "sem")].to_numpy()
        ax.bar(x - width/2, self_mean, width, yerr=self_sem, capsize=4,
               color="0.70", edgecolor="0.2", label="Self-rebinding")
        ax.bar(x + width/2, cross_mean, width, yerr=cross_sem, capsize=4,
               color=[colors[g] for g in geometry_order], edgecolor="0.2",
               label="Cross-rebinding")
        ax.set_xticks(x, [geometry_names[g] for g in geometry_order])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set(ylabel="Mean rebindings per trajectory",
               title="C  Confinement preferentially increases cross-rebinding")
        ax.legend(frameon=False)

        # D: paired cross contribution to occupancy versus cross events.
        ax = axes[1, 1]
        full_values = target.loc[
            target.simulation_mode.eq("Full rebinding"),
            ["geometry", "replicate", "theta", "mean_cross_rebindings"],
        ].rename(columns={"theta": "theta_full",
                          "mean_cross_rebindings": "cross_full"})
        self_values = target.loc[
            target.simulation_mode.eq("Self only"),
            ["geometry", "replicate", "theta", "mean_cross_rebindings"],
        ].rename(columns={"theta": "theta_self",
                          "mean_cross_rebindings": "cross_self"})
        paired = full_values.merge(
            self_values, on=["geometry", "replicate"], validate="one_to_one"
        )
        paired["cross_occupancy_gain"] = paired.theta_full - paired.theta_self
        for geometry in geometry_order:
            frame = paired.loc[paired.geometry.eq(geometry)]
            ax.scatter(frame.cross_full, frame.cross_occupancy_gain,
                       color=colors[geometry], alpha=0.75,
                       label=geometry_names[geometry])
        usable = paired[["cross_full", "cross_occupancy_gain"]].dropna()
        if len(usable) >= 3 and usable.cross_full.nunique() > 1:
            slope, intercept = np.polyfit(usable.cross_full, usable.cross_occupancy_gain, 1)
            xx = np.linspace(usable.cross_full.min(), usable.cross_full.max(), 100)
            correlation = usable.corr().iloc[0, 1]
            ax.plot(xx, intercept + slope*xx, color="black", linestyle="--")
            ax.text(0.04, 0.96, rf"$r={correlation:.2f}$", transform=ax.transAxes,
                    va="top", ha="left")
        ax.axhline(0, color="0.4", linewidth=1.5)
        ax.set(xlabel="Mean cross-rebindings per trajectory",
               ylabel=r"Cross-rebinding occupancy gain $\theta_{full}-\theta_{self}$",
               title="D  Cross-rebinding predicts absolute occupancy gain")
        ax.legend(frameon=False)

        for ax in axes.flat:
            ax.text(-0.14, 1.08, ax.get_title().split()[0], transform=ax.transAxes,
                    fontsize=16, fontweight="bold", va="top")
            ax.set_title(" ".join(ax.get_title().split()[1:]))

        if SAVE_FIGURES:
            save_figure(fig, FIGURE_ROOT, "final_low_concentration_cross_rebinding")
        plt.show()
        """
    ),
    markdown("## 5. Numerical values and exportable figure tables"),
    code(
        """
        target_summary = (target.groupby(["simulation_mode", "geometry"])
                          .theta.agg(["mean", "std", "sem", "count"])
                          .reset_index())
        classical_rows = pd.DataFrame({
            "simulation_mode": ["Classical"], "geometry": ["all"],
            "mean": [classical_theta], "std": [0.0], "sem": [0.0], "count": [8],
        })
        target_summary = pd.concat([classical_rows, target_summary], ignore_index=True)

        gain_table = (paired.groupby("geometry")
                      .agg(theta_full=("theta_full", "mean"),
                           theta_self=("theta_self", "mean"),
                           cross_occupancy_gain=("cross_occupancy_gain", "mean"),
                           cross_occupancy_gain_sem=("cross_occupancy_gain", "sem"),
                           cross_rebindings=("cross_full", "mean"),
                           cross_rebindings_sem=("cross_full", "sem"))
                      .reset_index())
        gain_table["theta_classical"] = classical_theta
        gain_table["total_gain_over_classical"] = (
            gain_table.theta_full - gain_table.theta_classical
        )
        display(target_summary)
        display(gain_table)

        if SAVE_FIGURES:
            output = FIGURE_ROOT / "final_low_concentration_tables"
            output.mkdir(parents=True, exist_ok=True)
            target.to_csv(output / "matched_concentration_replicates.csv", index=False)
            target_summary.to_csv(output / "matched_concentration_summary.csv", index=False)
            gain_table.to_csv(output / "cross_rebinding_occupancy_gain.csv", index=False)
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
destination = HERE / "analyze_manuscript_final_low_concentration.ipynb"
nbf.write(notebook, destination)
print(destination)
