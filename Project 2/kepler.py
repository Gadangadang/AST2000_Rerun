"""
Part 2, challenge B: Comparing our orbits to Kepler's laws.

B1 (2nd law): the line star-planet sweeps equal areas in equal times.
    We measure the area swept in the same time interval near perihelion and
    near aphelion, the distance travelled and the mean speed. Our home planet
    is almost circular (e = 0.0013), so we also do the most eccentric planet,
    where the effect is visible.
    Why it holds: dA/dt = |r x v| / 2 = h / 2, and the angular momentum h is
    conserved for a central force. So we also plot dA/dt over time.

B2 (3rd law): we measure every planet's period from the simulation and compare
      Kepler (our star):  P^2 = a^3 / M_star                (same for all planets)
      Newton:             P^2 = a^3 / (M_star + m_planet)
    Our simulation keeps the star fixed, which is exactly Kepler's assumption,
    so it follows Kepler's version. The difference to Newton is m/(2M) in P:
    challenge C (star + planet both moving) measures it.

Run:  ../.venv/bin/python kepler.py
Figures go to figures/b*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import matplotlib.pyplot as plt
import numpy as np

from common.config import get_mission
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, save
from orbits import G, simulate_orbits

N_LONGEST_ORBITS = 2.2  # Simulate long enough for the slowest planet to go around twice
STEPS_PER_YEAR = 20000  # Plenty for measuring periods and areas
AREA_FRACTION = 1 / 20  # Time interval for the swept areas, as a fraction of the period


def swept_area(x, y):
    """Area swept by the radius vector along a sampled path: sum of the small triangles 1/2 |r_i x r_i+1|."""
    return 0.5 * np.sum(np.abs(x[:-1] * y[1:] - x[1:] * y[:-1]))


def path_length(x, y):
    return np.sum(np.hypot(np.diff(x), np.diff(y)))


def measure_periods(times, positions):
    """Period of each planet: average time between successive full turns of its polar angle."""
    angle = np.unwrap(np.arctan2(positions[1], positions[0]), axis=1)
    turns = (angle - angle[:, :1]) / (2 * np.pi)
    periods = []
    for p in range(positions.shape[1]):
        n_turns = int(np.floor(turns[p, -1]))
        # Time at which turns == k, by linear interpolation (angle grows monotonically)
        crossings = np.interp(np.arange(1, n_turns + 1), turns[p], times)
        periods.append(crossings[-1] / n_turns)
    return np.array(periods)


def kepler_second_law(system, times, positions, velocities, planet):
    """Swept area, distance and mean speed near perihelion and aphelion for one planet."""
    P = 2 * np.pi * np.sqrt(system.semi_major_axes[planet] ** 3 / (G * system.star_mass))
    dt_window = AREA_FRACTION * P
    # One full orbit, starting late enough that a window centred on any point fits inside the data
    first_orbit = (times >= dt_window) & (times <= dt_window + P)
    r = np.linalg.norm(positions[:, planet], axis=0)
    results = {}
    for name, idx in (("perihelion", np.argmin(np.where(first_orbit, r, np.inf))),
                      ("aphelion", np.argmax(np.where(first_orbit, r, -np.inf)))):
        window = np.abs(times - times[idx]) <= dt_window / 2
        x, y = positions[0, planet, window], positions[1, planet, window]
        duration = times[window][-1] - times[window][0]
        results[name] = {"area": swept_area(x, y), "distance": path_length(x, y), "duration": duration,
                         "speed": path_length(x, y) / duration, "x": x, "y": y, "r": r[idx]}
    return results


def plot_second_law(system, times, positions, velocities, planets, results):
    fig, axes = plt.subplots(2, len(planets), figsize=(7 * len(planets), 10))
    for col, p in enumerate(planets):
        ax = axes[0, col]
        P = 2 * np.pi * np.sqrt(system.semi_major_axes[p] ** 3 / (G * system.star_mass))
        orbit = times <= P
        ax.plot(positions[0, p, orbit], positions[1, p, orbit], color=PLANET_COLORS[p], lw=1.5)
        ax.plot(0, 0, "*", color="#eda100", ms=14, mec=INK, mew=0.5)
        for name, color in (("perihelion", BLUE), ("aphelion", ORANGE)):
            res = results[p][name]
            wedge_x = np.concatenate([[0], res["x"], [0]])
            wedge_y = np.concatenate([[0], res["y"], [0]])
            ax.fill(wedge_x, wedge_y, color=color, alpha=0.45, lw=0,
                    label=f"{name}: A = {res['area']:.5f} AU$^2$")
        diff = results[p]["perihelion"]["area"] / results[p]["aphelion"]["area"] - 1
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title(f"Planet {p} (e = {system.eccentricities[p]:.4f}): areas in {AREA_FRACTION * P:.3f} yr\n"
                     f"differ by {diff:.1e}")
        ax.legend(loc="upper left", fontsize=8)

        # dA/dt = h/2 stays constant while the speed changes
        ax = axes[1, col]
        h = positions[0, p] * velocities[1, p] - positions[1, p] * velocities[0, p]
        speed = np.linalg.norm(velocities[:, p], axis=0)
        span = times <= 2 * P
        ax.plot(times[span] / P, (h[span] / 2) / (h[0] / 2) - 1, color=INK, lw=2, label=r"$dA/dt = |\vec r\times\vec v|/2$")
        ax.plot(times[span] / P, speed[span] / speed[span].mean() - 1, color=PLANET_COLORS[p], lw=1.5,
                label="orbital speed")
        ax.set_xlabel("time [orbits]")
        ax.set_ylabel("relative change")
        ax.set_title("Speed varies, areal velocity does not")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("B1  Kepler's 2nd law: equal areas in equal times", fontsize=12)
    save(fig, "b1_kepler_second_law")


def plot_third_law(system, periods):
    a, m, M = system.semi_major_axes, system.masses, system.star_mass
    kepler = np.sqrt(a ** 3 / M)
    newton = np.sqrt(a ** 3 / (M + m))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    a_line = np.logspace(np.log10(a.min()) - 0.2, np.log10(a.max()) + 0.2, 100)
    ax1.loglog(a_line ** 3, a_line ** 3 / M, color=INK, lw=1.2, label=r"$P^2 = a^3/M_\star$")
    ax1.loglog(a_line ** 3, a_line ** 3, color=MUTED, lw=1, ls=":", label=r"$P^2 = a^3$ (our Sun's constant)")
    for p in range(len(a)):
        ax1.loglog(a[p] ** 3, periods[p] ** 2, "o", color=PLANET_COLORS[p], ms=9, mec=INK, mew=0.5)
        ax1.annotate(f" {p}", (a[p] ** 3, periods[p] ** 2), fontsize=9, va="top")
    ax1.set_xlabel(r"$a^3$ [AU$^3$]")
    ax1.set_ylabel(r"$P^2$ [yr$^2$]")
    ax1.set_title("All planets on one line")
    ax1.legend(fontsize=8)

    x = np.arange(len(a))
    ax2.bar(x - 0.2, (periods / kepler - 1) * 100, width=0.4, color=BLUE, label="simulation vs. Kepler")
    ax2.bar(x + 0.2, (newton / kepler - 1) * 100, width=0.4, color=ORANGE, label="Newton vs. Kepler")
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p}\n{system.types[p]}" for p in x])
    ax2.set_xlabel("planet")
    ax2.set_ylabel("period difference [%]")
    ax2.set_title("Newton's correction $-m/2M$: visible only for the gas giants")
    ax2.legend(fontsize=8, loc="lower left")
    fig.suptitle("B2  Kepler's 3rd law", fontsize=12)
    save(fig, "b2_kepler_third_law")


def main():
    system = get_mission().system
    P = 2 * np.pi * np.sqrt(system.semi_major_axes ** 3 / (G * system.star_mass))
    duration = N_LONGEST_ORBITS * P.max()
    print(f"B     simulating {duration:.1f} yr so every planet completes at least two orbits ...")
    times, positions, velocities = simulate_orbits(system, duration, steps_per_year=STEPS_PER_YEAR, save_every=10)

    most_eccentric = int(np.argmax(system.eccentricities))
    planets = [0, most_eccentric]
    results = {p: kepler_second_law(system, times, positions, velocities, p) for p in planets}
    print(f"B1    areas swept in {AREA_FRACTION:.2f} of an orbit:")
    for p in planets:
        peri, aph = results[p]["perihelion"], results[p]["aphelion"]
        print(f"      planet {p} (e = {system.eccentricities[p]:.4f})")
        for name, res in (("perihelion", peri), ("aphelion", aph)):
            print(f"        near {name:<10} r = {res['r']:.4f} AU: area {res['area']:.6e} AU^2, "
                  f"distance {res['distance']:.5f} AU, mean speed {res['speed']:.4f} AU/yr")
        print(f"        area ratio perihelion/aphelion - 1 = {peri['area'] / aph['area'] - 1:.2e}, "
              f"speed ratio = {peri['speed'] / aph['speed']:.4f}")
    plot_second_law(system, times, positions, velocities, planets, results)

    periods = measure_periods(times, positions)
    kepler = np.sqrt(system.semi_major_axes ** 3 / system.star_mass)
    newton = np.sqrt(system.semi_major_axes ** 3 / (system.star_mass + system.masses))
    print("B2    planet   a [AU]   P_sim [yr]   P_Kepler [yr]   sim/Kepler-1   Newton/Kepler-1")
    for p in np.argsort(system.semi_major_axes):
        print(f"      {p:>6}  {system.semi_major_axes[p]:7.3f}  {periods[p]:11.5f}  {kepler[p]:14.5f}  "
              f"{periods[p] / kepler[p] - 1:13.1e}  {newton[p] / kepler[p] - 1:16.1e}")
    plot_third_law(system, periods)


if __name__ == "__main__":
    main()
