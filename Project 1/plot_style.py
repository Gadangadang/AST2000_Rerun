"""
Shared plotting style for all Part 1 figures, so every graph looks the same.

Colors are a fixed categorical order (never cycled): series 1 is always blue,
series 2 orange, series 3 aqua. Analytical curves are drawn in dark ink and
simulation/numerical results in color, so "theory vs. us" reads the same way
in every figure.
"""
import os

import matplotlib

matplotlib.use("Agg")  # Write PNGs without needing a display
import matplotlib.pyplot as plt

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Fixed categorical order
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"
SERIES = [BLUE, ORANGE, AQUA]

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


def save(fig, name):
    """Save a figure as figures/<name>.png and close it."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"      -> figures/{name}.png")
