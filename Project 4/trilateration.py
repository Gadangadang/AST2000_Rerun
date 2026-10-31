"""
Part 4, challenge D: the spacecraft's position from distances (trilateration).

At time t we know where every planet is (Kepler's equation, Part 2) and the
star sits at the origin. The radar gives the distance d_i to planet i and d_s
to the star, so the spacecraft lies on a circle around each of them:
    |x - p_i|^2 = d_i^2,      |x|^2 = d_s^2.

Clever bit: subtracting the star's equation from each planet's removes |x|^2,
the only non-linear term:
    2 p_i . x = |p_i|^2 + d_s^2 - d_i^2.
That is a linear system with one equation per planet (8 for 2 unknowns), solved
by least squares in one step: no initial guess, no iteration. With exact
distances two planets would do; the extra ones average down measurement noise.

Run:  ../.venv/bin/python trilateration.py
Figures go to figures/d*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(2)

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np

from common.config import get_mission
from common.plot_style import BLUE, INK, MUTED, ORANGE, PLANET_COLORS, save
from orbits import kepler_state


def trilaterate(system, time, distances):
    """
    Position (x, y) [AU] from distances [AU] to planets 0..n-1 and the star (last), measured at `time` [yr].

    distances: (n + 1,) for one measurement, or (..., n + 1) for many at the same time. Every
    measurement shares the matrix A (the planet positions), so one least-squares call solves them all.
    Returns the position(s) (..., 2) and the residuals |x - p_i| - d_i (..., n + 1) [AU].
    """
    distances = np.asarray(distances, dtype=float)
    planets = kepler_state(system, [time])[0][:, :, 0].T  # (n, 2)
    d_planets, d_star = distances[..., :-1], distances[..., -1:]
    A = 2 * planets
    b = np.sum(planets ** 2, axis=1) + d_star ** 2 - d_planets ** 2  # (..., n)
    flat = b.reshape(-1, b.shape[-1]).T  # One column per measurement
    position = np.linalg.lstsq(A, flat, rcond=None)[0].T.reshape(*b.shape[:-1], 2)
    to_planets = np.linalg.norm(planets - position[..., None, :], axis=-1) - d_planets
    residuals = np.concatenate([to_planets, np.linalg.norm(position, axis=-1, keepdims=True) - d_star], axis=-1)
    return position, residuals


def distances_from(system, time, position):
    """What the radar would measure at `position` (for testing)."""
    planets = kepler_state(system, [time])[0][:, :, 0].T
    return np.append(np.linalg.norm(planets - position, axis=1), np.linalg.norm(position))


def noise_study(system, time, position, noise_levels, n_trials=2000, seed=8):
    """Median position error vs. relative noise on the measured distances (all trials solved at once)."""
    rng = np.random.default_rng(seed)
    exact = distances_from(system, time, position)
    noisy = exact * (1 + rng.normal(0, 1, (len(noise_levels), n_trials, exact.size)) * np.asarray(noise_levels)[:, None, None])
    found, _ = trilaterate(system, time, noisy)
    return np.median(np.linalg.norm(found - position, axis=-1), axis=1)


# ===================================================================== figures
def plot_trilateration(system, time, distances, position, noise_levels, noise_errors):
    planets = kepler_state(system, [time])[0][:, :, 0].T
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    theta = np.linspace(0, 2 * np.pi, 600)
    for ax, lim, title in ((axes[0], 9, "Every distance is a circle"),
                           (axes[1], 0.002, "Zoom: they all cross at one point")):
        for p in range(system.number_of_planets):
            cx, cy = planets[p]
            ax.plot(cx + distances[p] * np.cos(theta), cy + distances[p] * np.sin(theta), color=PLANET_COLORS[p], lw=1.2,
                    label=f"planet {p}")
            ax.plot(cx, cy, "o", color=PLANET_COLORS[p], ms=7, mec=INK, mew=0.5)
        ax.plot(distances[-1] * np.cos(theta), distances[-1] * np.sin(theta), color=INK, lw=1.2, ls="--", label="star")
        ax.plot(0, 0, "*", color="#eda100", ms=14, mec=INK, mew=0.5)
        ax.plot(*position, "x", color=INK, ms=12, mew=2.5, label="spacecraft (solution)")
        centre = (0, 0) if lim > 1 else position
        ax.set_xlim(centre[0] - lim, centre[0] + lim)
        ax.set_ylim(centre[1] - lim, centre[1] + lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title(title, fontsize=10)
    axes[1].ticklabel_format(useOffset=False)
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    ax = axes[2]
    ax.loglog(noise_levels, noise_errors * const.AU / 1e3, "o-", color=BLUE, label="median position error")
    ax.axhline(0.01 * const.AU / 1e3, color=INK, ls="--", lw=1.2)
    ax.text(noise_levels[0], 0.01 * const.AU / 1e3, " API limit: 0.01 AU", va="bottom", fontsize=9)
    ax.set_xlabel("relative noise on every distance")
    ax.set_ylabel("position error [km]")
    ax.set_title("Robustness: error grows linearly with the noise", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"D  Trilateration at t = {time:.6f} yr", fontsize=12)
    save(fig, "d_trilateration")


def main(mission=None):
    """mission: a launched, verified SpaceMission. Without one, a made-up position is used as the test."""
    system = (mission or get_mission()).system
    # Test with a position we choose
    time, true_position = 1.234, np.array([0.3, -1.1])
    found, _ = trilaterate(system, time, distances_from(system, time, true_position))
    print(f"D     test: true position {true_position}, found {found} "
          f"(error {np.linalg.norm(found - true_position) * const.AU:.1e} m)")
    if mission is None:
        return None
    time, distances = mission.time_after_launch, mission.measure_distances()
    position, residuals = trilaterate(system, time, distances)
    print(f"D     measured distances at t = {time:.6e} yr: " + ", ".join(f"{d:.4f}" for d in distances) + " AU")
    print(f"      position = ({position[0]:.8f}, {position[1]:.8f}) AU; "
          f"residuals at most {np.abs(residuals).max() * const.AU / 1e3:.2f} km")
    noise_levels = np.logspace(-8, -2, 7)
    noise_errors = noise_study(system, time, position, noise_levels)
    plot_trilateration(system, time, distances, position, noise_levels, noise_errors)
    return position


if __name__ == "__main__":
    main()
