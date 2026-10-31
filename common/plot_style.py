"""
Shared plotting style for all figures in the project, so every graph looks the same.

Colors are a fixed categorical order (never cycled): series 1 is always blue,
series 2 orange, series 3 aqua. Analytical curves are drawn in dark ink and
simulation/numerical results in color, so "theory vs. us" reads the same way
in every figure.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # Write PNGs without needing a display
import matplotlib.pyplot as plt

# Figures go in figures/ next to the script being run (e.g. "Project 2/figures")
_main = getattr(sys.modules.get("__main__"), "__file__", None)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(_main)) if _main else os.getcwd(), "figures")

# Fixed categorical order
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"
SERIES = [BLUE, ORANGE, AQUA]
# One fixed color per planet index 0-7, so a planet has the same color in every figure
PLANET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

INK = "#0b0b0b"  # Analytical / reference curves
MUTED = "#52514e"  # Secondary text, guide lines
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "figure.dpi": 110,
})


def save(fig, name, rect=(0, 0, 1, 1)):
    """Save a figure as figures/<name>.png and close it. rect leaves room for figure-level legends."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{name}.png")
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"      -> figures/{name}.png")
