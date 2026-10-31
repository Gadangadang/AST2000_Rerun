"""
Part 3, challenges 2, 4 and 5: choosing the destination, and what "close" means.

2. Everything that matters for the choice, for every planet:
     - temperature, habitable zone and solar panel size (challenge 1)
     - landing: surface gravity, escape speed, atmosphere density, day length
     - getting there: a Hohmann transfer (the cheapest two-burn trip between
       circular orbits) gives the extra speed needed (delta-v), the travel time,
       and how often the planets line up for it (the synodic period), plus
       when the next such launch window opens.
   The Hohmann numbers are estimates (circular orbits, ignoring the planets'
   own gravity); Part 5 does the real trip.

4. The spacecraft is "close" when the planet pulls k times harder than the star:
       G M_p m / l^2 = k G M_s m / |r|^2   ->   l = |r| sqrt(M_p / (k M_s))
   (Part 5 uses k = 10.)

5. A camera with P x P pixels and field of view F x F radians gives each pixel
   an angle F / P. The planet (diameter 2R) covers an angle ~ 2R / L at distance
   L (small angles), so it spans more than one pixel when 2R / L > F / P, i.e.
       L < 2 R P / F   (the assignment's estimate L ~ R P / F, up to the factor 2).

Run:  ../.venv/bin/python destination.py
Figures go to figures/2*.png and figures/4*.png.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # Repo root, for common/
sys.path.insert(0, os.path.join(HERE, "..", "Project 2"))  # Orbits from Part 2

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np

from common.config import DESTINATION, SEED, get_mission
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, save
from habitable_zone import Star, T_WATER, habitable_zone, planet_table
from orbits import G, analytic_orbit, kepler_state

HOME = 0
K_DOMINANT = 10  # Planet's pull / star's pull that counts as "arrived" (challenge 4)
# Camera for challenge 5. ASSUMED values until Part 4 gives the real camera.
CAMERA_PIXELS = 640
CAMERA_FOV = np.radians(70)
AU_PER_YR = const.AU / const.yr  # [m/s] per [AU/yr]


def hohmann(GM, r1, r2):
    """Delta-v at departure and arrival [AU/yr] and transfer time [yr] between circular orbits r1 -> r2."""
    a_t = (r1 + r2) / 2
    dv1 = abs(np.sqrt(GM / r1) * (np.sqrt(r2 / a_t) - 1))
    dv2 = abs(np.sqrt(GM / r2) * (1 - np.sqrt(r1 / a_t)))
    return dv1, dv2, np.pi * np.sqrt(a_t ** 3 / GM)


def next_window(system, target, transfer_time, t_max=40.0, dt=1e-3):
    """
    First time > 0 when the target leads home by the Hohmann angle pi - n_target t_transfer,
    so that it arrives where the spacecraft does. Uses the exact orbits.
    """
    t = np.arange(0, t_max, dt)
    pos, _ = kepler_state(system, t)
    angle = np.arctan2(pos[1], pos[0])
    n_target = 2 * np.pi / np.sqrt(system.semi_major_axes[target] ** 3 / (G * system.star_mass))
    needed = np.pi - n_target * transfer_time
    miss = np.angle(np.exp(1j * (angle[target] - angle[HOME] - needed)))  # Wrapped to (-pi, pi]
    crossing = np.where((np.sign(miss[:-1]) != np.sign(miss[1:])) & (np.abs(miss[:-1]) < 1))[0]
    return t[crossing[0]] if len(crossing) else np.nan


def choose_destination(rows):
    """Our rule from Part 3: rocky, habitable if possible, then the smallest delta-v."""
    rocky = [r for r in rows if r["type"] == "rock"] or rows
    habitable = [r for r in rocky if r["habitable"]] or rocky
    return min(habitable, key=lambda r: r["dv"])["planet"]


def get_destination(system):
    """The destination planet: chosen by hand for this seed (common.config), otherwise by our rule."""
    if DESTINATION is not None:
        return DESTINATION
    return choose_destination(candidates(system, Star(system)))


def capture_distance(system, planet, r_star, k=K_DOMINANT):
    """Distance [AU] from the planet where its pull is k times the star's (challenge 4)."""
    return r_star * np.sqrt(system.masses[planet] / (k * system.star_mass))


def resolve_distance(radius, pixels=CAMERA_PIXELS, fov=CAMERA_FOV):
    """Farthest distance at which the planet covers more than one pixel (challenge 5); radius's unit."""
    return 2 * radius * pixels / fov


def candidates(system, star):
    """All the numbers for every planet except home."""
    table = planet_table(system, star)
    GM = G * system.star_mass
    hz = habitable_zone(star)
    rows = []
    for p in range(system.number_of_planets):
        if p == HOME:
            continue
        M = system.masses[p] * const.m_sun
        R = system.radii[p] * 1e3
        dv1, dv2, t_transfer = hohmann(GM, system.semi_major_axes[HOME], system.semi_major_axes[p])
        P_home, P_target = (2 * np.pi * np.sqrt(system.semi_major_axes[[HOME, p]] ** 3 / GM))
        rows.append({
            "planet": p, "type": system.types[p], "a": system.semi_major_axes[p],
            "T": table["T"][p], "habitable": hz[0] <= system.semi_major_axes[p] <= hz[1],
            "panel": table["panel"][p],
            "radius_km": system.radii[p], "g": const.G * M / R ** 2, "v_esc": np.sqrt(2 * const.G * M / R),
            "rho_atm": system.atmospheric_densities[p], "day_h": system.rotational_periods[p] * 24,
            "dv": (dv1 + dv2) * AU_PER_YR, "dv1": dv1 * AU_PER_YR, "dv2": dv2 * AU_PER_YR,
            "transfer_yr": t_transfer, "synodic_yr": 1 / abs(1 / P_home - 1 / P_target),
            "window_yr": next_window(system, p, t_transfer),
            "capture_km": capture_distance(system, p, system.semi_major_axes[p]) * const.AU / 1e3,
            "resolve_km": resolve_distance(system.radii[p]),
        })
    return rows


# ===================================================================== figures
def plot_comparison(system, star, rows):
    home_g = const.G * system.masses[HOME] * const.m_sun / (system.radii[HOME] * 1e3) ** 2
    planets = [r["planet"] for r in rows]
    x = np.arange(len(rows))
    gas = np.array([r["type"] == "gas" for r in rows])
    panels = [
        ("T", "surface temperature [K]", "Temperature (band: liquid water)", False),
        ("g", "surface gravity [m/s$^2$]", f"Gravity (dotted: home, {home_g:.1f} m/s$^2$)", False),
        ("rho_atm", "atmosphere density [kg/m$^3$]", "Atmosphere at the surface (thicker: harder landing)", False),
        ("dv", "delta-v [km/s]", "Fuel cost: Hohmann delta-v from home", False),
        ("transfer_yr", "travel time [yr]", "Travel time (Hohmann)", False),
        ("panel", "panel area [m$^2$]", "Solar panel for 40 W", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (key, ylabel, title, log) in zip(axes.flat, panels):
        values = np.array([r[key] for r in rows]) / (1e3 if key == "dv" else 1)
        bars = ax.bar(x, values, color=[PLANET_COLORS[p] for p in planets], width=0.65)
        for bar, is_gas in zip(bars, gas):
            if is_gas:
                bar.set_hatch("///")
                bar.set_alpha(0.45)
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.3g}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                        color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n{r['type']}" + ("\nHZ" if r["habitable"] else "") for p, r in zip(planets, rows)],
                           fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        if log:
            ax.set_yscale("log")
    axes[0, 0].axhspan(*T_WATER, color=AQUA, alpha=0.2, lw=0, zorder=0)
    axes[0, 1].axhline(home_g, color=INK, ls=":", lw=1.2)
    fig.suptitle("2  Choosing the destination (hatched: gas giants, not recommended; HZ: in the habitable zone)",
                 fontsize=12)
    save(fig, "2a_destination_comparison")


def plot_transfers(system, rows):
    """Hohmann transfer ellipses from home to every rocky candidate."""
    GM = G * system.star_mass
    r_home = system.semi_major_axes[HOME]
    rocky = [r for r in rows if r["type"] == "rock"]
    fig, ax = plt.subplots(figsize=(9, 9))
    theta = np.linspace(0, np.pi, 300)
    for r in rocky:
        p = r["planet"]
        x, y = analytic_orbit(system.semi_major_axes[p], system.eccentricities[p], system.aphelion_angles[p])
        ax.plot(x, y, color=PLANET_COLORS[p], lw=1.5)
        # Transfer half-ellipse: from home's orbit at angle 0 to the target's orbit at angle pi
        r1, r2 = r_home, r["a"]
        a_t, e_t = (r1 + r2) / 2, abs(r2 - r1) / (r1 + r2)
        rr = a_t * (1 - e_t ** 2) / (1 + e_t * np.cos(theta if r2 > r1 else theta + np.pi))
        ax.plot(rr * np.cos(theta), rr * np.sin(theta), color=PLANET_COLORS[p], lw=2.2, ls="--",
                label=f"to planet {p}: {r['dv'] / 1e3:.1f} km/s, {r['transfer_yr'] * 365.25:.0f} days, "
                      f"next window t = {r['window_yr']:.2f} yr")
    x, y = analytic_orbit(r_home, system.eccentricities[HOME], system.aphelion_angles[HOME])
    ax.plot(x, y, color=PLANET_COLORS[HOME], lw=2.5, label="home (planet 0)")
    ax.plot(0, 0, "*", color="#eda100", ms=16, mec=INK, mew=0.5)
    lim = max(r["a"] for r in rocky) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("2  Cheapest routes (Hohmann half-ellipses, dashed) from home to the rocky planets", fontsize=11)
    save(fig, "2b_transfer_routes")


def plot_closeness(rows):
    """Challenges 4 and 5: how close is 'arrived', and how close to see the planet."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(rows))
    for xi, r in zip(x, rows):
        color = PLANET_COLORS[r["planet"]]
        ax.semilogy([xi - 0.15], [r["capture_km"]], "o", color=color, ms=10, mec=INK, mew=0.5)
        ax.semilogy([xi + 0.15], [r["resolve_km"]], "s", color=color, ms=9, mec=INK, mew=0.5)
        ax.semilogy([xi - 0.3, xi + 0.3], [r["radius_km"]] * 2, color=INK, lw=2)
        ax.annotate(f"{r['capture_km']:.2g}", (xi - 0.15, r["capture_km"]), xytext=(-4, 8), textcoords="offset points",
                    ha="right", fontsize=8, color=MUTED)
        ax.annotate(f"{r['resolve_km']:.2g}", (xi + 0.15, r["resolve_km"]), xytext=(4, 8), textcoords="offset points",
                    fontsize=8, color=MUTED)
    ax.plot([], [], "o", color=MUTED, ms=9, label=f"4: planet pulls {K_DOMINANT}x harder than the star inside this "
                                                 r"distance, $l = |r|\sqrt{M_p / kM_\star}$")
    ax.plot([], [], "s", color=MUTED, ms=9, label=f"5: planet bigger than one pixel inside this distance, $L = 2RP/F$ "
                                                 f"(assumed {CAMERA_PIXELS} px, {np.degrees(CAMERA_FOV):.0f}$^\\circ$)")
    ax.plot([], [], color=INK, lw=2, label="planet radius")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['planet']}\n{r['type']}" for r in rows])
    ax.set_xlabel("planet")
    ax.set_ylabel("distance from the planet [km]")
    ax.set_title("4 + 5  How close is close? Captured vs. visible on camera", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 30)
    save(fig, "4_capture_and_camera")


def main():
    system = get_mission().system
    star = Star(system)
    rows = candidates(system, star)
    print(f"2     destinations (Hohmann estimates from home, planet {HOME}):")
    print("      planet type  T [K]  HZ   g [m/s2] v_esc   atm    day    dv     travel  windows  next     panel")
    print("                                       [km/s] [kg/m3] [h]  [km/s]  [days]  every    window   [m^2]")
    for r in sorted(rows, key=lambda r: r["a"]):
        print(f"      {r['planet']:>6} {r['type']:<4} {r['T']:5.0f}  {'yes' if r['habitable'] else 'no ':<4} "
              f"{r['g']:6.2f}  {r['v_esc'] / 1e3:5.2f}  {r['rho_atm']:5.2f} {r['day_h']:6.1f} "
              f"{r['dv'] / 1e3:6.2f}  {r['transfer_yr'] * 365.25:6.0f}  {r['synodic_yr']:5.2f} yr "
              f"{r['window_yr']:5.2f} yr {r['panel']:7.2f}")
    print(f"4+5   capture distance (k = {K_DOMINANT}) and camera distance "
          f"(ASSUMED {CAMERA_PIXELS} px, {np.degrees(CAMERA_FOV):.0f} deg field of view):")
    for r in sorted(rows, key=lambda r: r["a"]):
        print(f"      planet {r['planet']}: captured within {r['capture_km']:,.0f} km, "
              f"visible within {r['resolve_km']:,.0f} km (radius {r['radius_km']:,.0f} km)")
    auto = choose_destination(rows)
    chosen = DESTINATION if DESTINATION is not None else auto
    print(f"2     destination: planet {chosen} " + ("(chosen by hand for seed %d; our rule also picks %d)" % (SEED, auto)
                                                    if DESTINATION is not None else "(picked by our rule: rocky, "
                                                    "habitable if possible, cheapest)"))
    plot_comparison(system, star, rows)
    plot_transfers(system, rows)
    plot_closeness(rows)
    return rows


if __name__ == "__main__":
    main()
