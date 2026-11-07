"""
Part 3, challenge 1: the habitable zone.

Model: the star and the planets are black bodies in equilibrium.

1. Star's luminosity L = 4 pi R*^2 sigma T*^4, spread over a sphere of radius r:
       F(r) = L / (4 pi r^2) = sigma T*^4 (R* / r)^2
2. Solar panels: power = efficiency x F x A, so the lander needs
       A = P_needed / (efficiency F(r))
3. A planet catches the light falling on its cross-section (a disk, pi R^2):
   parallel rays can't tell a sphere from a disk of the same radius.
       E_in = F(r) pi R^2   (per second)
4. It glows from its whole surface (4 pi R^2) at temperature T. Energy in = energy out:
       F pi R^2 = sigma T^4 4 pi R^2   ->   T = T* sqrt(R* / (2 r))
   The planet's own size cancels: every planet at distance r has the same T.
5. Inverting T(r) gives the distance for a given temperature:
       r(T) = (R* / 2) (T* / T)^2
   so liquid water (260-390 K) puts the habitable zone between r(390 K) and r(260 K).

Run:  ../.venv/bin/python habitable_zone.py
Figures go to figures/1*.png.
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
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, save
from common.report import record, record_many
from orbits import analytic_orbit, kepler_state

T_WATER = (260.0, 390.0)  # Liquid water at the surface [K]
T_MARGIN = 15.0  # The assignment's +-15 K uncertainty on those limits
PANEL_POWER = 40.0  # Power the lander needs [W]
PANEL_EFFICIENCY = 0.12


def flux(T_star, R_star, r):
    """Flux from the star at distance r [W/m^2]; R_star and r in the same unit."""
    return const.sigma * T_star ** 4 * (R_star / r) ** 2


def panel_area(F, power=PANEL_POWER, efficiency=PANEL_EFFICIENCY):
    """Smallest solar panel that gives `power` watts in flux F [m^2]."""
    return power / (efficiency * F)


def planet_temperature(T_star, R_star, r):
    """Equilibrium temperature of a black-body planet at distance r [K]."""
    return T_star * np.sqrt(R_star / (2 * r))


def distance_for_temperature(T_star, R_star, T):
    """Inverse of planet_temperature: where a planet would have temperature T (same unit as R_star)."""
    return R_star / 2 * (T_star / T) ** 2


class Star:
    def __init__(self, system):
        self.T = system.star_temperature  # [K]
        self.R = system.star_radius * 1e3 / const.AU  # [AU]
        self.L = 4 * np.pi * (system.star_radius * 1e3) ** 2 * const.sigma * self.T ** 4  # [W]


def habitable_zone(star, margin=0.0):
    """Inner and outer edge [AU]; a positive margin widens the zone (the +-15 K uncertainty)."""
    inner = distance_for_temperature(star.T, star.R, T_WATER[1] + margin)
    outer = distance_for_temperature(star.T, star.R, T_WATER[0] - margin)
    return inner, outer


def planet_table(system, star):
    """Temperature, flux and panel area for every planet, at its mean distance and at its extremes."""
    a, e = system.semi_major_axes, system.eccentricities
    return {
        "a": a,
        "T": planet_temperature(star.T, star.R, a),
        "T_hot": planet_temperature(star.T, star.R, a * (1 - e)),  # At perihelion
        "T_cold": planet_temperature(star.T, star.R, a * (1 + e)),  # At aphelion
        "flux": flux(star.T, star.R, a),
        "panel": panel_area(flux(star.T, star.R, a)),
    }


# ===================================================================== figures
def plot_temperature(system, star, table):
    hz, hz_wide = habitable_zone(star), habitable_zone(star, T_MARGIN)
    hz_narrow = habitable_zone(star, -T_MARGIN)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    r = np.logspace(np.log10(0.08), np.log10(12), 400)
    ax.axvspan(*hz_wide, color=AQUA, alpha=0.12, lw=0, label=f"habitable zone $\\pm${T_MARGIN:.0f} K")
    ax.axvspan(*hz_narrow, color=AQUA, alpha=0.25, lw=0, label=f"habitable zone ({T_WATER[0]:.0f}-{T_WATER[1]:.0f} K)")
    ax.axhspan(*T_WATER, color=BLUE, alpha=0.06, lw=0)
    ax.plot(r, planet_temperature(star.T, star.R, r), color=INK, lw=1.5,
            label=r"$T = T_\star\sqrt{R_\star / 2r}$")
    for p in range(system.number_of_planets):
        ax.errorbar(table["a"][p], table["T"][p],
                    yerr=[[table["T"][p] - table["T_cold"][p]], [table["T_hot"][p] - table["T"][p]]],
                    fmt="o", color=PLANET_COLORS[p], ms=10 if system.types[p] == "gas" else 8, mec=INK, mew=0.5,
                    capsize=3, zorder=3)
        ax.annotate(f"{p} ({table['T'][p]:.0f} K)", (table["a"][p], table["T"][p]), xytext=(8, 6),
                    textcoords="offset points", fontsize=9)
    for T in T_WATER:
        ax.axhline(T, color=MUTED, lw=0.8, ls=":")
        ax.text(0.085, T, f" {T:.0f} K", color=MUTED, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("distance from the star [AU]")
    ax.set_ylabel("surface temperature [K]")
    ax.set_title(f"1  Black-body temperature of the planets (star: {star.T:.0f} K, "
                 f"{system.star_radius / 6.957e5:.2f} R$_\\odot$). "
                 f"Habitable zone {hz[0]:.2f}-{hz[1]:.2f} AU", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "1a_temperatures")


def plot_zone_map(system, star):
    """Top-down view of the system with the habitable zone as a ring."""
    hz, hz_wide = habitable_zone(star), habitable_zone(star, T_MARGIN)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6))
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax, lim, title in ((axes[0], 9, "Whole system"), (axes[1], 2.0, "Inner system")):
        for (r_in, r_out), alpha in ((hz_wide, 0.12), (hz, 0.3)):
            ax.fill(np.concatenate([r_out * np.cos(theta), r_in * np.cos(theta[::-1])]),
                    np.concatenate([r_out * np.sin(theta), r_in * np.sin(theta[::-1])]), color=AQUA, alpha=alpha, lw=0)
        for p in range(system.number_of_planets):
            x, y = analytic_orbit(system.semi_major_axes[p], system.eccentricities[p], system.aphelion_angles[p])
            ax.plot(x, y, color=PLANET_COLORS[p], lw=1.5)
            x0, y0 = system.initial_positions[:, p]
            ax.plot(x0, y0, "o", color=PLANET_COLORS[p], ms=11 if system.types[p] == "gas" else 8, mec=INK, mew=0.5)
            if np.hypot(x0, y0) < lim:
                ax.annotate(f" {p}", (x0, y0), fontsize=10, fontweight="bold")
        ax.plot(0, 0, "*", color="#eda100", ms=16, mec=INK, mew=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title(title)
    axes[0].fill([], [], color=AQUA, alpha=0.3, label=f"habitable zone {hz[0]:.2f}-{hz[1]:.2f} AU")
    axes[0].fill([], [], color=AQUA, alpha=0.12, label=f"with $\\pm${T_MARGIN:.0f} K: {hz_wide[0]:.2f}-{hz_wide[1]:.2f} AU")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("1  Where liquid water is possible (big dots: gas giants, positions at t = 0)", fontsize=12)
    save(fig, "1b_habitable_zone_map", rect=(0, 0, 1, 0.96))


def plot_solar_power(system, star, destination):
    """
    Power for the lander: panel area against distance, and how the destination's
    flux and temperature change over its year (its orbit is not quite circular).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    r = np.logspace(np.log10(0.1), np.log10(10), 300)
    hz = habitable_zone(star)
    ax1.loglog(r, panel_area(flux(star.T, star.R, r)), color=INK, lw=1.5,
               label=f"A = {PANEL_POWER:.0f} W / ({PANEL_EFFICIENCY:.0%} x F)")
    ax1.axvspan(*hz, color=AQUA, alpha=0.2, lw=0, label="habitable zone")
    for p in range(system.number_of_planets):
        a = system.semi_major_axes[p]
        ax1.plot(a, panel_area(flux(star.T, star.R, a)), "o", color=PLANET_COLORS[p], ms=9, mec=INK, mew=0.5)
        ax1.annotate(f" {p}", (a, panel_area(flux(star.T, star.R, a))), fontsize=9)
    ax1.set_xlabel("distance from the star [AU]")
    ax1.set_ylabel("solar panel area [m$^2$]")
    area_hz = panel_area(flux(star.T, star.R, np.array(hz)))
    ax1.set_title(f"Panel area grows as $r^2$: {area_hz[0]:.2f}-{area_hz[1]:.2f} m$^2$ across the zone")
    ax1.legend(fontsize=8)

    P = 2 * np.pi * np.sqrt(system.semi_major_axes[destination] ** 3 / (const.G_sol * system.star_mass))
    t = np.linspace(0, P, 400)
    d = np.linalg.norm(kepler_state(system, t)[0][:, destination], axis=0)
    F = flux(star.T, star.R, d)
    T = planet_temperature(star.T, star.R, d)
    ax2.plot(t / P, F, color=ORANGE)
    ax2.set_xlabel("time [destination years]")
    ax2.set_ylabel("flux at the planet [W/m$^2$]")
    # Energy balance F pi R^2 = sigma T^4 4 pi R^2 makes T a function of F alone: a second scale, not a second curve
    sigma = const.sigma
    right = ax2.secondary_yaxis("right", functions=(lambda F: (np.maximum(F, 1e-9) / (4 * sigma)) ** 0.25,
                                                     lambda T: 4 * sigma * T ** 4))
    right.set_ylabel("equilibrium temperature $(F/4\\sigma)^{1/4}$ [K]")
    i_peri = np.argmax(F)
    ax2.annotate("perihelion", (t[i_peri] / P, F[i_peri]), xytext=(0, -18), textcoords="offset points", ha="center",
                 fontsize=8, color=MUTED)
    ax2.set_title(f"Planet {destination} over one orbit: flux {F.min():.0f}-{F.max():.0f} W/m$^2$, "
                  f"T {T.min():.0f}-{T.max():.0f} K")
    fig.suptitle("1  Power and climate for the lander", fontsize=12)
    save(fig, "1c_solar_power")
    return F, T


def main():
    system = get_mission().system
    star = Star(system)
    table = planet_table(system, star)
    print(f"1     star: T = {star.T:.0f} K, R = {system.star_radius:,.0f} km, L = {star.L:.3e} W "
          f"({star.L / const.L_sun:.2f} L_sun)")
    hz, hz_wide = habitable_zone(star), habitable_zone(star, T_MARGIN)
    print(f"      habitable zone ({T_WATER[0]:.0f}-{T_WATER[1]:.0f} K): {hz[0]:.3f} - {hz[1]:.3f} AU; "
          f"with +-{T_MARGIN:.0f} K: {hz_wide[0]:.3f} - {hz_wide[1]:.3f} AU")
    print("      planet  type   a [AU]   T [K] (perihelion-aphelion)   flux [W/m^2]   panel [m^2]   habitable?")
    for p in np.argsort(system.semi_major_axes):
        inside = hz[0] <= table["a"][p] <= hz[1]
        edge = not inside and hz_wide[0] <= table["a"][p] <= hz_wide[1]
        print(f"      {p:>6}  {system.types[p]:<5} {table['a'][p]:7.3f}   {table['T'][p]:5.0f} "
              f"({table['T_hot'][p]:4.0f}-{table['T_cold'][p]:4.0f})           {table['flux'][p]:10.1f}   "
              f"{table['panel'][p]:10.2f}   {'YES' if inside else ('edge (within 15 K)' if edge else 'no')}")
    plot_temperature(system, star, table)
    plot_zone_map(system, star)
    record_many({"star-T": (star.T, "{:.0f}"), "star-R": (system.star_radius, "{:,.0f}"),
                 "star-Rsun": (system.star_radius / 6.957e5, "{:.2f}"), "star-L": (star.L / const.L_sun, "{:.2f}"),
                 "hz-in": (hz[0], "{:.2f}"), "hz-out": (hz[1], "{:.2f}"),
                 "hz-wide-in": (hz_wide[0], "{:.2f}"), "hz-wide-out": (hz_wide[1], "{:.2f}"),
                 "T-water-lo": (T_WATER[0], "{:.0f}"), "T-water-hi": (T_WATER[1], "{:.0f}"),
                 "T-margin": (T_MARGIN, "{:.0f}"), "panel-power": (PANEL_POWER, "{:.0f}"),
                 "panel-eff": (PANEL_EFFICIENCY * 100, "{:.0f}"),
                 "T-home": (table["T"][0], "{:.0f}")})
    inside = [p for p in np.argsort(system.semi_major_axes) if hz[0] <= table["a"][p] <= hz[1]]
    record("hz-planets", ", ".join(map(str, inside)) or "none")
    from destination import get_destination  # Here, not at the top: destination.py imports this module
    dest = get_destination(system)
    F, T = plot_solar_power(system, star, dest)
    record_many({"dest-T": (table["T"][dest], "{:.0f}"), "dest-T-min": (T.min(), "{:.0f}"),
                 "dest-T-max": (T.max(), "{:.0f}"),
                 "dest-F-min": (F.min(), "{:.0f}"), "dest-F-max": (F.max(), "{:.0f}"), "dest-flux": (table["flux"][dest], "{:.0f}"),
                 "dest-panel": (table["panel"][dest], "{:.2f}")})
    return star, table


if __name__ == "__main__":
    main()
