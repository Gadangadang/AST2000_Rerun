"""
Part 6, challenge D: scouting for landing sites.

1. Coordinates on a spinning planet. Spherical coordinates as in Part 4:
   r, theta (from the +z axis) and phi (from +x, counterclockwise), in the
   landing frame (planet-centred, axes parallel to the solar-system frame).
   The planet spins about z with omega = 2 pi / P_rot, so a point fixed on the
   surface keeps r and theta, and
       phi(t0 + dt) = phi(t0) + omega dt          (new_coordinates below)

2. Before trusting that, we check the two things it rests on:
   * The frame: Part 5's last orientation (solar-system frame, relative to the
     planet) must equal the landing sequence's first one.
   * The spin's direction and rate: the air co-rotates with the planet (Part 7),
     so a retrograde orbit meets the air faster than a prograde one,
         decay_retro / decay_pro = ((v + omega r) / (v - omega r))^2,
     which we measure with two test orbits and solve for omega.

3. Scouting: one orbit at the low altitude, camera straight down, N_PICTURES
   pictures (MCAst XML files). For each we record the point below the
   spacecraft, its planet-fixed longitude (phi at t = 0), and how high the
   star stands there, from the planet's position on its orbit (Kepler, Part 2).

4. Choice: the lander will come down near a pass of the spacecraft overhead.
   For every pictured site we find the next pass after scouting and the
   star's elevation there at that time, and pick the best-lit one (closest to
   local noon). The pictures are for a human to veto bad terrain (MCAst).

Run:  ../.venv/bin/python landing_sites.py
Figures go to figures/d*.png; the chosen site to landing_site.json for Part 7.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(2)

import json
import shutil

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np

import low_orbit
from common.config import JOURNEY_STATE_FILE, MCAST_DATA_DIR, XML_DIR
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, VIOLET, save
from common.report import record, record_many
from orbits import kepler_state

HERE = os.path.dirname(os.path.abspath(__file__))
SITE_FILE = os.path.join(HERE, "landing_site.json")
N_PICTURES = 8  # Pictures during the scouting orbit
SPIN_TEST_ALTITUDE = 290e3  # Low enough for clear drag, high enough to survive 10 orbits [m]
SPIN_TEST_ORBITS = 10
SITE_CHOICE = None  # Picture index to land at instead of the automatic choice (after looking in MCAst)


# ===================================================================== coordinates
def to_spherical(xyz):
    """(..., 3) Cartesian -> (r, theta, phi) with theta from +z and phi in [0, 2 pi)."""
    xyz = np.asarray(xyz, dtype=float)
    r = np.linalg.norm(xyz, axis=-1)
    return np.stack([r, np.arccos(xyz[..., 2] / r), np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]), 2 * np.pi)], axis=-1)


def to_cartesian(coords):
    r, theta, phi = np.moveaxis(np.asarray(coords, dtype=float), -1, 0)
    return np.stack([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)], axis=-1)


def new_coordinates(coords, elapsed, rotation_period):
    """
    Where a point fixed on the surface is after `elapsed` seconds: (r, theta, phi) -> (r, theta, phi + omega dt).
    Works on any number of points and times at once (NumPy broadcasting).
    """
    coords = np.asarray(coords, dtype=float)
    omega = 2 * np.pi / rotation_period
    phi = np.mod(coords[..., 2] + omega * np.asarray(elapsed, dtype=float), 2 * np.pi)
    return np.stack(np.broadcast_arrays(coords[..., 0], coords[..., 1], phi), axis=-1)


def star_direction(system, planet_index, t0_years, t_seconds):
    """Unit vectors (..., 3) from the planet towards the star at landing times t [s] (t = 0 at t0_years)."""
    t = t0_years + np.atleast_1d(t_seconds) / const.yr
    p = kepler_state(system, t)[0][:, planet_index].T  # (n, 2) [AU]
    d = -p / np.linalg.norm(p, axis=1, keepdims=True)
    return np.column_stack([d, np.zeros(len(d))])


def solar_elevation(site_coords, star_dir):
    """Elevation [rad] of the star above the horizon at surface points (..., 3 coords) for directions (..., 3)."""
    normal = to_cartesian(np.concatenate([np.ones_like(site_coords[..., :1]), site_coords[..., 1:]], axis=-1))
    return np.arcsin(np.clip(np.sum(normal * star_dir, axis=-1), -1, 1))


# ===================================================================== checks
def frame_check(orbiter):
    """Distance [m] between Part 5's last state (solar-system frame) and the landing sequence's first."""
    with open(JOURNEY_STATE_FILE) as f:
        state = json.load(f)
    orbiter.restart()
    _, r, v = orbiter.orient()
    return (np.linalg.norm(r[:2] - np.array(state["position_relative_m"])),
            np.linalg.norm(v[:2] - np.array(state["velocity_relative_ms"])))


def spin_test(orbiter, planet, altitude=SPIN_TEST_ALTITUDE, n_orbits=SPIN_TEST_ORBITS):
    """
    Decay per orbit for a prograde and a retrograde circular orbit at the same height, and the
    rotation rate omega they imply. (Test runs of the landing simulator; the flight restarts after.)
    """
    decays = {}
    for sense in (+1, -1):
        orbiter.restart()
        low_orbit.descend(orbiter, planet, planet.R + altitude, samples=1)
        t, r, v = orbiter.orient()
        if sense < 0:  # Turn the orbit around: same speed, opposite direction
            low_orbit.quiet(orbiter.sequence.boost, -2 * v)
        start = np.linalg.norm(r)
        orbiter.fall(n_orbits * planet.period(start))
        decays[sense] = (start - np.linalg.norm(orbiter.samples[-1][1])) / n_orbits
    a = planet.R + altitude
    v = np.sqrt(planet.GM / a)
    k = np.sqrt(decays[-1] / decays[+1])  # = (v + omega a) / (v - omega a)
    omega = v * (k - 1) / (a * (k + 1))
    return decays, omega


# ===================================================================== scouting
def scout(orbiter, planet, t0_years, n=N_PICTURES):
    """One orbit with the camera straight down; a picture and the site below it every 1/n orbit."""
    period = planet.period(np.linalg.norm(orbiter.samples[-1][1]))
    low_orbit.quiet(orbiter.sequence.look_in_direction_of_planet)
    sites = []
    for k in range(n):
        t, r, v = orbiter.orient()
        name = f"site_{k}.xml"
        low_orbit.quiet(orbiter.sequence.take_picture, filename=name)
        copy_to_mcast(name)
        below = to_spherical(r)
        below[0] = planet.R  # The surface point straight below
        fixed = new_coordinates(below, -t, planet.rotation_period)  # Planet-fixed: where it was at t = 0
        star = star_direction(orbiter.mission.system, planet.index, t0_years, t)[0]
        sites.append({"index": k, "time": t, "coords": below, "fixed": fixed, "file": name,
                      "elevation_now": float(solar_elevation(below, star))})
        orbiter.fall(period / n)
    return sites


def next_pass(site, orbiter, planet, after):
    """First time after `after` [s] when the spacecraft is straight above the site (circular equatorial orbit)."""
    t, r, _ = orbiter.samples[-1]
    n_orbit = 2 * np.pi / planet.period(np.linalg.norm(r))
    omega = 2 * np.pi / planet.rotation_period
    phi_craft = to_spherical(r)[2] + n_orbit * (after - t)
    phi_site = new_coordinates(site["fixed"], after, planet.rotation_period)[2]
    gap = np.mod(phi_site - phi_craft, 2 * np.pi)  # The spacecraft gains on the site at n - omega
    return after + gap / (n_orbit - omega), 2 * np.pi / (n_orbit - omega)


def copy_to_mcast(name):
    path = os.path.join(XML_DIR, name)
    if os.path.exists(path) and os.path.isdir(MCAST_DATA_DIR):
        shutil.copy(path, MCAST_DATA_DIR)


# ===================================================================== figures
def plot_checks(frame_miss, decays, omega, planet):
    omega_api = 2 * np.pi / planet.rotation_period
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    a = planet.R + SPIN_TEST_ALTITUDE
    v = np.sqrt(planet.GM / a)
    ax1.bar(["prograde\n(with the spin)", "retrograde\n(against it)"], [decays[+1], decays[-1]], color=[BLUE, ORANGE],
            width=0.5)
    predicted = decays[+1] * ((v + omega_api * a) / (v - omega_api * a)) ** 2
    ax1.axhline(predicted, color=INK, ls="--", lw=1.2, label=f"predicted for retrograde with P$_{{rot}}$ = "
                                                            f"{planet.rotation_period / 3600:.2f} h")
    ax1.axhline(decays[+1], color=MUTED, ls=":", lw=1, label="no spin: both equal")
    ax1.set_ylabel(f"radius lost per orbit at {SPIN_TEST_ALTITUDE / 1e3:.0f} km [m]")
    ax1.set_title(f"Air co-rotates: retrograde decays {decays[-1] / decays[+1]:.2f}x faster", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax2.axis("off")
    text = (f"Frame check\n  Part 5 end vs. landing start: {frame_miss[0]:.2g} m, {frame_miss[1]:.2g} m/s\n\n"
            f"Spin from drag\n  omega = {omega:.4e} rad/s  ->  P = {2 * np.pi / omega / 3600:.2f} h\n"
            f"  ast2000tools: P = {planet.rotation_period / 3600:.2f} h "
            f"({omega / omega_api - 1:+.1%})\n  direction: counterclockwise about +z (as the orbits)")
    ax2.text(0, 0.5, text, family="monospace", fontsize=11, va="center")
    fig.suptitle("D  Checking the frame and the planet's spin before computing coordinates", fontsize=12)
    save(fig, "d1_frame_and_spin")


def plot_scouting(sites, chosen, planet, orbiter, t0_years, t_end, period_pass):
    """Planet-fixed view from the north pole: picture sites, day and night at the landing pass, and the sun at each site."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.3]})
    t_land = chosen["pass"]
    star = star_direction(orbiter.mission.system, planet.index, t0_years, t_land)[0]
    # Planet-fixed frame: rotate the star direction back by omega t_land
    omega = 2 * np.pi / planet.rotation_period
    angle = np.arctan2(star[1], star[0]) - omega * t_land
    theta = np.linspace(0, 2 * np.pi, 400)
    ax1.fill(np.cos(theta), np.sin(theta), color="#3b3a37", lw=0)  # Night
    day = np.linspace(angle - np.pi / 2, angle + np.pi / 2, 200)
    ax1.fill(np.cos(day), np.sin(day), color="#f3d37a", lw=0)  # Day: the half facing the star
    ax1.text(-0.45 * np.cos(angle), -0.45 * np.sin(angle), "night", color="white", ha="center", fontsize=10)
    ax1.text(0.45 * np.cos(angle), 0.45 * np.sin(angle), "day", color=INK, ha="center", fontsize=10)
    ax1.annotate("", xy=(1.45 * np.cos(angle), 1.45 * np.sin(angle)), xytext=(1.1 * np.cos(angle), 1.1 * np.sin(angle)),
                 arrowprops={"arrowstyle": "-|>", "color": "#eda100", "lw": 2})
    ax1.text(1.55 * np.cos(angle), 1.55 * np.sin(angle), "star", ha="center", va="center", fontsize=9)
    for s in sites:
        phi = s["fixed"][2]
        color = VIOLET if s is chosen else INK
        ax1.plot(np.cos(phi), np.sin(phi), "*" if s is chosen else "o", color=color, ms=18 if s is chosen else 8,
                 mec="white", mew=0.8)
        ax1.annotate(f"{s['index']}", (1.17 * np.cos(phi), 1.17 * np.sin(phi)), ha="center", va="center", fontsize=10)
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.7, 1.7)
    ax1.set_ylim(-1.7, 1.7)
    ax1.axis("off")
    ax1.set_title(f"Planet-fixed, from above the north pole: day side at the landing pass\n"
                  f"(t = {t_land / 3600:.1f} h); numbers: pictures taken on the equator", fontsize=10)

    t = np.linspace(0, planet.rotation_period, 600)
    stars = star_direction(orbiter.mission.system, planet.index, t0_years, t)
    for s in sites:
        coords = new_coordinates(s["fixed"], t, planet.rotation_period)
        elev = np.degrees(solar_elevation(coords, stars))
        ax2.plot(t / 3600, elev, color=VIOLET if s is chosen else MUTED, lw=2.2 if s is chosen else 0.9,
                 label=f"site {s['index']} (chosen)" if s is chosen else None)
        ax2.plot(s["pass"] / 3600, np.degrees(s["elevation_pass"]), "v", color=VIOLET if s is chosen else MUTED, ms=7)
    ax2.axhline(0, color=INK, lw=1)
    ax2.axvspan(sites[0]["time"] / 3600, t_end / 3600, color=AQUA, alpha=0.15, lw=0, label="scouting orbit")
    ax2.set_xlabel("hours since the start of the landing sequence (one planet day shown)")
    ax2.set_ylabel("star's elevation above the horizon [deg]")
    ax2.set_title(f"Sun at each site (triangles: next pass overhead after scouting; passes every "
                  f"{period_pass / 3600:.2f} h)", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")
    fig.suptitle("D  Scouting for a landing site", fontsize=12)
    save(fig, "d2_scouting")


def plot_rotation(chosen, planet, t_start, phi_start, n_orbit, t_scout, t_end, period_pass):
    """
    The coordinate function at work, from the start of the low orbit: the site's longitude in the landing
    frame against the spacecraft's (circular orbit, phi = phi_start + n t), and their separation.
    """
    t = np.linspace(t_start, chosen["pass"] + period_pass, 3000)
    site_phi = new_coordinates(chosen["fixed"], t, planet.rotation_period)[..., 2]
    craft_phi = np.mod(phi_start + n_orbit * (t - t_start), 2 * np.pi)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8))
    jump = lambda y: np.where(np.abs(np.diff(y, prepend=y[0])) > np.pi, np.nan, y)  # Don't draw the 360 -> 0 wraps
    ax1.plot(t / 3600, np.degrees(jump(site_phi)), color=VIOLET, lw=2.2,
             label=f"site {chosen['index']}: $\\phi$(t) = $\\phi_0$ + $\\omega$t")
    ax1.plot(t / 3600, np.degrees(jump(craft_phi)), color=MUTED, lw=1, label="spacecraft (circular low orbit)")
    ax1.set_xlabel("hours since the start of the landing sequence")
    ax1.set_ylabel("$\\phi$ in the landing frame [deg]")
    ax1.set_title("The site turns once per planet day, the spacecraft once per orbit", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    separation = np.degrees(np.abs(np.angle(np.exp(1j * (craft_phi - site_phi)))))
    ax2.plot(t / 3600, separation, color=INK)
    ax2.plot(chosen["time"] / 3600, 0, "o", color=VIOLET, ms=8, label="picture of the site")
    ax2.plot(chosen["pass"] / 3600, 0, "v", color=VIOLET, ms=11, label="predicted next pass overhead")
    ax2.axvspan(t_scout / 3600, t_end / 3600, color=AQUA, alpha=0.15, lw=0, label="scouting orbit")
    ax2.set_xlabel("hours since the start of the landing sequence")
    ax2.set_ylabel("angle between spacecraft and site [deg]")
    ax2.set_title(f"Overhead every {period_pass / 3600:.2f} h = 1 / (1/P$_{{orbit}}$ - 1/P$_{{rot}}$)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    fig.suptitle("D  Coordinates on a spinning planet", fontsize=12)
    save(fig, "d3_rotation")


# ===================================================================== main
def main():
    orbiter, planet, altitude, _ = low_orbit.enter_low_orbit(altitude=read_low_altitude())
    with open(JOURNEY_STATE_FILE) as f:
        t0_years = json.load(f)["time_in_orbit"]

    # Self-test of the coordinate function: a full planet day brings every point back
    test = np.array([[planet.R, 0.3, 1.0], [planet.R, 2.0, 5.5]])
    back = new_coordinates(test, planet.rotation_period, planet.rotation_period)
    roundtrip = np.max(np.abs(to_spherical(to_cartesian(test)) - test))
    print(f"D     coordinate function: after one planet day off by {np.max(np.abs(back - test)):.1e}; "
          f"spherical <-> Cartesian round trip {roundtrip:.1e}")

    frame_miss = frame_check(orbiter)
    decays, omega = spin_test(orbiter, planet)
    omega_api = 2 * np.pi / planet.rotation_period
    print(f"      frame: Part 5 end vs. landing start differ by {frame_miss[0]:.2g} m and {frame_miss[1]:.2g} m/s")
    print(f"      spin from drag: decay {decays[+1]:.2f} m/orbit prograde, {decays[-1]:.2f} retrograde -> omega "
          f"{omega:.4e} rad/s (P = {2 * np.pi / omega / 3600:.2f} h; ast2000tools {planet.rotation_period / 3600:.2f} h, "
          f"{omega / omega_api - 1:+.1%})")

    # Back to the real low orbit for the scouting
    orbiter, planet, altitude, _ = low_orbit.enter_low_orbit(altitude=altitude)
    t_start, r_start, _ = orbiter.samples[-1]  # First state in the circular low orbit
    n_orbit = 2 * np.pi / planet.period(np.linalg.norm(r_start))
    sites = scout(orbiter, planet, t0_years)
    t_end = orbiter.samples[-1][0]
    for s in sites:
        s["pass"], period_pass = next_pass(s, orbiter, planet, t_end)
        star = star_direction(orbiter.mission.system, planet.index, t0_years, s["pass"])[0]
        s["elevation_pass"] = float(solar_elevation(new_coordinates(s["fixed"], s["pass"], planet.rotation_period), star))
    chosen = sites[SITE_CHOICE] if SITE_CHOICE is not None else max(sites, key=lambda s: s["elevation_pass"])
    print(f"      scouting: {len(sites)} pictures (MCAst: site_0.xml ... site_{len(sites) - 1}.xml); passes overhead "
          f"every {period_pass / 3600:.2f} h")
    for s in sites:
        print(f"        site {s['index']}: t = {s['time'] / 60:6.1f} min, planet-fixed phi = {np.degrees(s['fixed'][2]):6.1f} deg, "
              f"sun {np.degrees(s['elevation_now']):+6.1f} deg now, next pass t = {s['pass'] / 3600:5.2f} h with sun "
              f"{np.degrees(s['elevation_pass']):+6.1f} deg" + ("   <- chosen" if s is chosen else ""))

    plot_checks(frame_miss, decays, omega, planet)
    plot_scouting(sites, chosen, planet, orbiter, t0_years, t_end, period_pass)
    plot_rotation(chosen, planet, t_start, to_spherical(r_start)[2], n_orbit, sites[0]["time"], t_end, period_pass)

    at_pass = new_coordinates(chosen["fixed"], chosen["pass"], planet.rotation_period)
    with open(SITE_FILE, "w") as f:
        json.dump({"index": chosen["index"], "planet_fixed_coords_at_t0": chosen["fixed"].tolist(), "picture": chosen["file"],
                   "picture_time_s": chosen["time"], "next_pass_s": chosen["pass"],
                   "coords_at_next_pass": at_pass.tolist(), "rotation_period_s": planet.rotation_period,
                   "low_orbit_altitude_m": altitude, "t0_years": t0_years}, f, indent=2)
    record_many({
        "frame-miss": (frame_miss[0], "{:.1g}"), "spin-alt": (SPIN_TEST_ALTITUDE / 1e3, "{:.0f}"),
        "decay-pro": (decays[+1], "{:.2f}"), "decay-retro": (decays[-1], "{:.2f}"),
        "decay-ratio": (decays[-1] / decays[+1], "{:.2f}"),
        "P-rot-drag": (2 * np.pi / omega / 3600, "{:.2f}"), "P-rot-api": (planet.rotation_period / 3600, "{:.2f}"),
        "spin-err": (abs(omega / omega_api - 1) * 100, "{:.1f}"),
        "n-pictures": (len(sites), "{:d}"), "pass-period": (period_pass / 3600, "{:.2f}"),
        "site-index": (chosen["index"], "{:d}"), "site-phi": (np.degrees(chosen["fixed"][2]), "{:.1f}"),
        "site-theta": (np.degrees(chosen["fixed"][1]), "{:.0f}"),
        "site-pass": (chosen["pass"] / 3600, "{:.2f}"), "site-sun": (np.degrees(chosen["elevation_pass"]), "{:.0f}"),
        "site-phi-pass": (np.degrees(at_pass[2]), "{:.1f}"),
        "n-lit": (sum(s["elevation_pass"] > 0 for s in sites), "{:d}"),
        "orbit-period-min": (planet.period(planet.R + altitude) / 60, "{:.1f}"),
    })
    return chosen


def read_low_altitude():
    """The low orbit chosen in challenge A (from its recorded results), so D does not rerun the drag probe."""
    with open(os.path.join(HERE, "paper", "results.json")) as f:
        return float(json.load(f)["low-alt"]) * 1e3


if __name__ == "__main__":
    main()
