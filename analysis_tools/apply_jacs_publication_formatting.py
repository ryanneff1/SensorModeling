#!/usr/bin/env python3
"""Apply compact JACS-oriented Matplotlib styling to manuscript notebooks."""

from pathlib import Path

import nbformat


HERE = Path(__file__).resolve().parent
NOTEBOOKS = sorted(HERE.glob("analyze_manuscript*.ipynb"))
FORMATTING = '''import matplotlib as mpl
import seaborn as sns

sns.set_theme(style="ticks", context="paper", palette="deep")
mpl.rcParams.update({
    # Typography sized for final 7-inch double-column figure assembly.
    "font.family": "Arial",
    "font.size": 8,
    "figure.titlesize": 9,
    "figure.titleweight": "bold",
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.title_fontsize": 7,

    # Compact lines, markers, axes, and ticks.
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "patch.linewidth": 0.8,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "legend.frameon": False,

    # Subtle grids and publication export settings.
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
'''

for path in NOTEBOOKS:
    notebook = nbformat.read(path, as_version=4)
    if not notebook.cells or notebook.cells[0].cell_type != "code":
        raise RuntimeError(f"{path.name} does not begin with a formatting code cell.")
    notebook.cells[0].source = FORMATTING
    nbformat.write(notebook, path)
    print(path)
