#!/usr/bin/env python3
"""Apply physical panel dimensions to the manuscript supplement notebooks."""

from pathlib import Path

import nbformat


HERE = Path(__file__).resolve().parent
NOTEBOOKS = [
    "analyze_manuscript_supp_numerical_convergence.ipynb",
    "analyze_manuscript_supp_mechanistic_controls.ipynb",
]
SIZE_SETUP = '''

# JACS double-column layout: larger panels tile two across, compact panels
# tile three across a roughly 7-inch-wide assembled figure.
TWO_PANEL_PLOT_AREA_IN = (2.55, 2.15)
THREE_PANEL_PLOT_AREA_IN = (1.75, 1.65)
PLOT_AREA_SIZE_IN = TWO_PANEL_PLOT_AREA_IN
MAP_PLOT_AREA_SIZE_IN = THREE_PANEL_PLOT_AREA_IN
SINGLE_PANEL_SIZE_IN = (3.4, 2.8)
MAP_PANEL_SIZE_IN = (2.6, 2.35)
WIDE_PANEL_SIZE_IN = (7.0, 2.8)
THREE_PANEL_SIZE_IN = (7.0, 2.5)
FOUR_PANEL_SIZE_IN = (7.0, 5.6)

def save_panel(fig, output_directory, stem, dpi=None):
    size_figure_for_axes(fig, fig.axes[0], *PLOT_AREA_SIZE_IN)
    save_figure(fig, output_directory, stem, dpi=dpi, exact_size=True)
'''

REPLACEMENTS = {
    "# Physical size of each axes data rectangle in inches. The canvas grows around\n"
    "# labels, titles, legends, and colorbars without shrinking this plot area.\n"
    "PLOT_AREA_SIZE_IN = (5.0, 4.0)\nMAP_PLOT_AREA_SIZE_IN = (5.5, 4.0)\n"
    "SINGLE_PANEL_SIZE_IN = (5.0, 4.0)\nMAP_PANEL_SIZE_IN = (5.5, 4.0)\n"
    "WIDE_PANEL_SIZE_IN = (10.0, 4.0)\nTHREE_PANEL_SIZE_IN = (15.0, 4.0)\n"
    "FOUR_PANEL_SIZE_IN = (10.0, 8.0)\n\ndef save_panel(fig, output_directory, stem, dpi=None):\n"
    "    size_figure_for_axes(fig, fig.axes[0], *PLOT_AREA_SIZE_IN)\n"
    "    save_figure(fig, output_directory, stem, dpi=dpi, exact_size=True)":
        SIZE_SETUP.strip(),
    "from analysis_tools.manuscript_escape_analysis import (":
        "import importlib\nimport analysis_tools.manuscript_escape_analysis as manuscript_escape_analysis\n"
        "importlib.reload(manuscript_escape_analysis)\n"
        "from analysis_tools.manuscript_escape_analysis import (",
    "replicate_survival_curves, save_figure, summarize_replicates,":
        "replicate_survival_curves, save_figure, size_figure_for_axes, summarize_replicates,",
    "# Physical manuscript export dimensions in inches.\nSINGLE_PANEL_SIZE_IN = (3.35, 3.05)\nWIDE_PANEL_SIZE_IN = (7.0, 3.05)\nTHREE_PANEL_SIZE_IN = (7.0, 2.75)\nFOUR_PANEL_SIZE_IN = (7.0, 6.3)\n\ndef save_panel(fig, output_directory, stem, dpi=None):\n    save_figure(fig, output_directory, stem, dpi=dpi, exact_size=True)":
        SIZE_SETUP.strip(),
    "# Physical manuscript export dimensions in inches.\nSINGLE_PANEL_SIZE_IN = (3.35, 3.05)\nWIDE_PANEL_SIZE_IN = (7.0, 3.05)\nTHREE_PANEL_SIZE_IN = (7.0, 2.75)\nFOUR_PANEL_SIZE_IN = (7.0, 6.3)\n\ndef save_panel(fig, output_directory, stem, dpi=None):\n    save_panel(fig, output_directory, stem, dpi=dpi, exact_size=True)":
        SIZE_SETUP.strip(),
    "fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)":
        "fig,axes=plt.subplots(2,2,figsize=FOUR_PANEL_SIZE_IN,constrained_layout=True)",
    "fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)":
        "fig,ax=plt.subplots(figsize=WIDE_PANEL_SIZE_IN,constrained_layout=True)",
    "fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)":
        "fig,axes=plt.subplots(1,3,figsize=THREE_PANEL_SIZE_IN,constrained_layout=True)",
    "fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)":
        "fig,ax=plt.subplots(figsize=SINGLE_PANEL_SIZE_IN,constrained_layout=True)",
    "save_figure(fig,": "save_panel(fig,",
}

for filename in NOTEBOOKS:
    path = HERE / filename
    notebook = nbformat.read(path, as_version=4)
    setup_cell = next(
        cell for cell in notebook.cells
        if cell.cell_type == "code" and "FIGURE_ROOT" in cell.source
    )
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for old, new in REPLACEMENTS.items():
            if old.startswith("from analysis_tools") and "importlib.reload(manuscript_escape_analysis)" in cell.source:
                continue
            if old == "save_figure(fig," and cell is setup_cell:
                continue
            cell.source = cell.source.replace(old, new)
    if "SINGLE_PANEL_SIZE_IN" not in setup_cell.source:
        setup_cell.source += SIZE_SETUP
    nbformat.write(notebook, path)
    print(path)
