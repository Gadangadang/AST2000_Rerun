"""
Part 6, challenge A: from the Part 5 orbit down to a low orbit.

From here on the spacecraft is commanded through ast2000tools' LandingSequence:
SI units, positions relative to the destination planet's centre, 3D boosts
(we keep z = 0, the planet's equatorial plane). It does not count fuel, so we
keep our own count with the rocket equation and the Part 1 exhaust speed.

1. Descent: a Hohmann transfer. At the current radius r1, make the velocity
   purely tangential with the speed of the transfer ellipse's apoapsis,
       v_apo = sqrt(GM (2/r1 - 1/a)),   a = (r1 + r2) / 2,
   fall half its period, pi sqrt(a^3 / GM), and circularise at r2 with
   v = sqrt(GM / r2). Setting the whole velocity (not adding a scalar) also
   removes any leftover radial motion from Part 5.

2. How low is safe? We measure it. For a ladder of altitudes we enter a
   circular orbit and watch it for a few orbits. Air drag takes energy, so the
   orbit sinks, and for a circular orbit
       decay per orbit = 2 pi a^2 rho(h) C_d A / m,
   proportional to the density at that height. The API itself stops the run
   ("orbital decay due to air resistance") when drag wins. The decay rate
   against altitude is therefore a direct measurement of how the atmosphere
   thins out: its slope is the density scale height H, an independent check
   of the atmosphere model in challenge C.

3. The low orbit: SAFETY_HEIGHTS scale heights above the "edge" where the fitted
   decay reaches DECAY_LIMIT per orbit, checked over CHECK_ORBITS orbits from the
   spacecraft's own orientations.

Run:  ../.venv/bin/python low_orbit.py
Figures go to figures/a*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import io
import json
import shutil
from contextlib import redirect_stdout

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission

from common.config import JOURNEY_MISSION_FILE, JOURNEY_STATE_FILE, MCAST_DATA_DIR, XML_DIR
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, VIOLET, save
from common.report import record, record_many

# Altitudes for the drag probe [m]: fine steps where drag switches on, coarse above
PROBE_ALTITUDES = np.r_[np.arange(150e3, 400e3, 10e3), np.geomspace(400e3, 2000e3, 6)]
PROBE_ORBITS = 10  # Orbits watched at each probe altitude
DECAY_LIMIT = 0.1  # Drag decay per orbit [m] that we call "the edge" of the atmosphere
SAFETY_HEIGHTS = 3  # The low orbit sits this many scale heights above that edge (drag e^3 = 20x weaker)
CHECK_ORBITS = 20  # Orbits watched in the final low orbit
SAMPLES_PER_ORBIT = 60


def quiet(function, *args, **kwargs):
    """Call an API function with its printing silenced."""
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class Planet:
    """The destination planet in SI units."""

    def __init__(self, system, index):
        self.index = index
        self.GM = const.G * system.masses[index] * const.m_sun
        self.R = system.radii[index] * 1e3
        self.rho0 = system.atmospheric_densities[index]
        self.rotation_period = system.rotational_periods[index] * const.day

    def period(self, a):
        return 2 * np.pi * np.sqrt(a ** 3 / self.GM)


def elements(r, v, GM):
    """Orbit from position and velocity (..., 3): semi-major axis, eccentricity, periapsis, apoapsis, energy."""
    r_norm = np.linalg.norm(r, axis=-1)
    energy = 0.5 * np.sum(v ** 2, axis=-1) - GM / r_norm
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / GM - r / r_norm[..., None]  # Points at the periapsis
    e = np.linalg.norm(e_vec, axis=-1)
    a = -GM / (2 * energy)
    return {"r": r_norm, "a": a, "e": e, "periapsis": a * (1 - e), "apoapsis": a * (1 + e), "energy": energy}


class Orbiter:
    """
    The spacecraft under LandingSequence control: quiet API calls, a log of
    orientations and boosts, and our own fuel count (rocket equation).
    """

    def __init__(self, mission, fuel, exhaust_speed):
        self.mission = mission
        self.sequence = quiet(mission.begin_landing_sequence)
        self.sequence.verbose = False
        self.fuel0, self.exhaust_speed = fuel, exhaust_speed
        self.restart()

    def restart(self):
        quiet(self.sequence.restart)
        self.fuel = self.fuel0
        self.samples = []  # (t, r, v)
        self.boosts = []  # (t, dv, name, fuel used)

    def orient(self):
        t, r, v = quiet(self.sequence.orient)
        self.samples.append((t, np.array(r), np.array(v)))
        return t, np.array(r), np.array(v)

    def boost(self, dv, name):
        t, _, _ = self.orient()
        mass = self.mission.spacecraft_mass + self.fuel
        used = mass * (1 - np.exp(-np.linalg.norm(dv) / self.exhaust_speed))
        self.fuel -= used
        quiet(self.sequence.boost, np.asarray(dv, dtype=float))
        self.boosts.append((t, np.array(dv), name, used))

    def fall(self, duration, n_samples=1):
        """Fall for `duration` seconds, orienting n_samples times on the way (for the figures)."""
        for _ in range(n_samples):
            quiet(self.sequence.fall, duration / n_samples)
            self.orient()
        return self.samples[-1]

    def log(self):
        t = np.array([s[0] for s in self.samples])
        return t, np.array([s[1] for s in self.samples]), np.array([s[2] for s in self.samples])


def tangential(r, v, speed):
    """Velocity of the given speed perpendicular to r in the orbital plane, in the current sense of rotation."""
    e_r = r / np.linalg.norm(r)
    sense = np.sign(np.cross(r, v)[2]) or 1.0
    return sense * speed * np.cross([0.0, 0.0, 1.0], e_r)


def record_video(orbiter, planet, filename="descent.xml", n_orbits=2):
    """Restart, descend again with the camera on the planet, and write an MCAst video (descent + n_orbits)."""
    altitude = np.linalg.norm(orbiter.samples[-1][1]) - planet.R
    orbiter.restart()
    quiet(orbiter.sequence.look_in_direction_of_planet)
    quiet(orbiter.sequence.start_video)
    descend(orbiter, planet, planet.R + altitude, samples=1)
    orbiter.fall(n_orbits * planet.period(planet.R + altitude))
    quiet(orbiter.sequence.finish_video, filename=filename, number_of_frames=1000)
    path = os.path.join(XML_DIR, filename)
    if os.path.exists(path) and os.path.isdir(MCAST_DATA_DIR):
        shutil.copy(path, MCAST_DATA_DIR)
        print(f"      MCAst video {filename} copied into {MCAST_DATA_DIR}")


def descend(orbiter, planet, r_target, samples=200):
    """Hohmann transfer from the current radius to a circular orbit of radius r_target."""
    t, r, v = orbiter.orient()
    r1 = np.linalg.norm(r)
    a = (r1 + r_target) / 2
    orbiter.boost(tangential(r, v, np.sqrt(planet.GM * (2 / r1 - 1 / a))) - v, "descent")
    orbiter.fall(np.pi * np.sqrt(a ** 3 / planet.GM), samples)
    t, r, v = orbiter.orient()
    orbiter.boost(tangential(r, v, np.sqrt(planet.GM / np.linalg.norm(r))) - v, "circularise")


def load(mission_file=JOURNEY_MISSION_FILE, state_file=JOURNEY_STATE_FILE):
    """The mission in orbit after Part 5, the destination, and the fuel and exhaust speed it left with."""
    mission = quiet(SpaceMission.load, mission_file)
    with open(state_file) as f:
        state = json.load(f)
    return mission, Planet(mission.system, state["destination"]), state["fuel_left"], state["exhaust_speed"]


# ===================================================================== drag probe
def decay_probe(orbiter, planet, altitudes=PROBE_ALTITUDES, n_orbits=PROBE_ORBITS):
    """
    Radius lost per orbit [m] at each altitude, from n_orbits of watching a
    circular orbit. NaN where the API stopped the run (drag won), with the time
    it survived.
    """
    decay, survived = np.full(len(altitudes), np.nan), np.full(len(altitudes), np.nan)
    for i, h in enumerate(altitudes):
        orbiter.restart()
        try:
            descend(orbiter, planet, planet.R + h, samples=1)
            t0, r, _ = orbiter.orient()
            start = np.linalg.norm(r)
            P = planet.period(planet.R + h)
            for _ in range(n_orbits):
                t, r, _ = orbiter.fall(P)
            decay[i] = (start - np.linalg.norm(r)) / n_orbits
        except RuntimeError:  # "Orbital decay of the spacecraft due to air resistance"
            t_stop = quiet(orbiter.sequence.orient)[0]
            survived[i] = t_stop - orbiter.boosts[-1][0] if orbiter.boosts else 0.0
    return decay, survived


def scale_height(altitudes, decay, floor):
    """Fit decay = C exp(-h / H) to the points clearly above the numerical floor; returns H [m] and the fit."""
    use = np.isfinite(decay) & (decay > 10 * floor)
    slope, intercept = np.polyfit(altitudes[use], np.log(decay[use]), 1)
    return -1 / slope, (slope, intercept), use


def choose_altitude(H, fit, limit=DECAY_LIMIT, margin=SAFETY_HEIGHTS):
    """
    Where the fitted decay falls to `limit` per orbit (the edge), plus `margin`
    scale heights, rounded up to 10 km. Returns (altitude, edge) [m].
    """
    slope, intercept = fit
    edge = (np.log(limit) - intercept) / slope
    return np.ceil((edge + margin * H) / 10e3) * 10e3, edge


# ===================================================================== figures
def plot_descent(orbiter, planet, altitude):
    t, r, v = orbiter.log()
    R = planet.R
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    theta = np.linspace(0, 2 * np.pi, 300)
    ax1.fill(np.cos(theta), np.sin(theta), color=PLANET_COLORS[planet.index], alpha=0.6, lw=0)
    r0 = np.linalg.norm(r[0])
    ax1.plot(r0 / R * np.cos(theta), r0 / R * np.sin(theta), color=MUTED, ls=":", lw=1.2, label="orbit after Part 5")
    t_circ = [b[0] for b in orbiter.boosts if b[1] is not None and b[2] == "circularise"][0]
    transfer = t <= t_circ
    ax1.plot(r[transfer, 0] / R, r[transfer, 1] / R, color=ORANGE, lw=2, label="Hohmann transfer")
    ax1.plot(r[~transfer, 0] / R, r[~transfer, 1] / R, color=VIOLET, lw=2,
             label=f"low orbit, {altitude / 1e3:.0f} km up")
    for tb, _, name, _ in orbiter.boosts:
        i = np.argmin(np.abs(t - tb))
        ax1.plot(r[i, 0] / R, r[i, 1] / R, "o", color=INK, ms=7)
        ax1.annotate(f" {name}", r[i, :2] / R, fontsize=9)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [planet radii]")
    ax1.set_ylabel("y [planet radii]")
    ax1.set_title("Planet frame, seen from above the north pole", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    hours = (t - t[0]) / 3600
    ax2.plot(hours, (np.linalg.norm(r, axis=1) - R) / 1e3, color=INK)
    for tb, dv, name, used in orbiter.boosts:
        ax2.axvline((tb - t[0]) / 3600, color=ORANGE, ls=":", lw=1.2)
        ax2.annotate(f"{name}\n{np.linalg.norm(dv):.0f} m/s, {used:.0f} kg", ((tb - t[0]) / 3600, 0.5),
                     xycoords=("data", "axes fraction"), fontsize=8, rotation=90, va="center", ha="right")
    ax2.set_yscale("log")
    ax2.set_xlim(-0.2, (t_circ - t[0]) / 3600 + 3 * planet.period(planet.R + altitude) / 3600)
    ax2.set_xlabel("hours since the start of Part 6")
    ax2.set_ylabel("altitude above the surface [km]")
    ax2.set_title(f"Down in {(t_circ - t[0]) / 3600:.1f} h, then circular", fontsize=10)
    fig.suptitle(f"A  Descent to a low orbit around planet {planet.index}", fontsize=12)
    save(fig, "a1_descent")


def plot_probe(altitudes, decay, survived, H, fit, used, floor, chosen, edge):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    near = altitudes <= 420e3  # Higher up, every orbit sits on the numerical floor
    altitudes, decay = altitudes[near], decay[near]
    used = used[near]
    ok = np.isfinite(decay)
    shown = np.maximum(decay, floor)
    ax.semilogy(altitudes[ok] / 1e3, shown[ok], "o", color=BLUE, ms=7, label=f"measured over {PROBE_ORBITS} orbits")
    ax.semilogy(altitudes[used] / 1e3, decay[used], "o", color=ORANGE, ms=7, label="used in the fit")
    h = np.linspace(altitudes[used].min() - 10e3, altitudes[ok].max(), 100)
    ax.semilogy(h / 1e3, np.exp(fit[1] + fit[0] * h), color=INK, lw=1.2, ls="--",
                label=f"$\\propto e^{{-h/H}}$, H = {H / 1e3:.1f} km")
    failed = ~ok
    if failed.any():
        top = shown[ok].max() * 30
        ax.plot(altitudes[failed] / 1e3, np.full(failed.sum(), top), "x", color=ORANGE, ms=11, mew=2.5,
                label="API stopped the run: drag won")
    ax.axhline(DECAY_LIMIT, color=AQUA, lw=1.2, label=f"edge: {DECAY_LIMIT} m per orbit, at {edge / 1e3:.0f} km")
    ax.axhline(floor, color=MUTED, lw=0.8, ls=":", label="numerical floor (no drag)")
    ax.axvline(chosen / 1e3, color=VIOLET, lw=1.5,
               label=f"chosen low orbit: {chosen / 1e3:.0f} km (edge + {SAFETY_HEIGHTS} H)")
    ax.set_ylim(floor / 3, shown[ok].max() * 300)
    ax.set_xlabel("altitude of the circular orbit [km] (higher orbits, up to 2000 km, all sit on the floor)")
    ax.set_ylabel("radius lost per orbit [m]")
    ax.set_title("Probing the atmosphere with the spacecraft itself: decay per orbit $\\propto \\rho(h)$", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("A  How low can we orbit?", fontsize=12)
    save(fig, "a2_drag_probe")


def plot_low_orbit(orbiter, planet, start):
    """start: index of the first orientation after the circularising burn."""
    t, r, v = orbiter.log()
    el = elements(r[start:], v[start:], planet.GM)
    hours = (t[start:] - t[start]) / 3600
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8))
    ax1.plot(hours, el["r"] - el["r"][0], color=INK, lw=1, label="distance r")
    ax1.plot(hours, el["a"] - el["a"][0], color=BLUE, lw=1.5, label="semi-major axis a")
    ax1.set_xlabel("hours in the low orbit")
    ax1.set_ylabel("change since the start [m]")
    ax1.set_title(f"{CHECK_ORBITS} orbits from the spacecraft's own orientations: no decay. The steady\n"
                  f"+{el['a'][-1] - el['a'][0]:.2f} m is the API integrator's numerical drift (the probe's floor)", fontsize=10)
    ax1.legend(fontsize=8)
    ax2.semilogy(hours, np.maximum(el["e"], 1e-12), color=VIOLET, lw=1.2)
    ax2.set_xlabel("hours in the low orbit")
    ax2.set_ylabel("eccentricity")
    ax2.set_title(f"Circular: e stays below {el['e'].max():.1e}", fontsize=10)
    fig.suptitle(f"A  The low orbit: {(el['a'].mean() - planet.R) / 1e3:.0f} km up, period "
                 f"{planet.period(el['a'].mean()) / 60:.1f} min", fontsize=12)
    save(fig, "a3_low_orbit")
    return el


# ===================================================================== main
def enter_low_orbit(altitude=None):
    """
    Load the Part 5 mission and descend to the low orbit (deterministic, ~1 s).
    Without an altitude, the drag probe chooses it. Returns (orbiter, planet, altitude, probe results).
    """
    mission, planet, fuel, u = load()
    orbiter = Orbiter(mission, fuel, u)
    probe = None
    if altitude is None:
        decay, survived = decay_probe(orbiter, planet)
        high = np.isfinite(decay) & (PROBE_ALTITUDES >= 400e3)  # Far above any drag: what is left is numerical
        floor = max(np.max(np.abs(decay[high])), 1e-6)
        H, fit, used = scale_height(PROBE_ALTITUDES, decay, floor)
        altitude, edge = choose_altitude(H, fit)
        probe = {"decay": decay, "survived": survived, "floor": floor, "H": H, "fit": fit, "used": used, "edge": edge}
        orbiter.restart()
    descend(orbiter, planet, planet.R + altitude)
    orbiter.orient()  # The first state in the low orbit
    return orbiter, planet, altitude, probe


def main():
    orbiter, planet, altitude, probe = enter_low_orbit()
    decay, survived, floor, H, fit, used, edge = (probe[k] for k in ("decay", "survived", "floor", "H", "fit", "used",
                                                                      "edge"))
    print(f"A     drag probe ({len(PROBE_ALTITUDES)} altitudes, {PROBE_ORBITS} orbits each):")
    for h, d, s in zip(PROBE_ALTITUDES, decay, survived):
        print(f"      {h / 1e3:7.1f} km: " + (f"{d:10.3g} m per orbit" if np.isfinite(d) else
                                             f"STOPPED by drag after {s / 3600:.2f} h"))
    print(f"      decay falls as exp(-h/H) with H = {H / 1e3:.2f} km (fit to {used.sum()} points); "
          f"edge ({DECAY_LIMIT} m per orbit) at {edge / 1e3:.0f} km; low orbit {altitude / 1e3:.0f} km "
          f"(edge + {SAFETY_HEIGHTS} H)")

    start = len(orbiter.samples) - 1
    orbiter.fall(CHECK_ORBITS * planet.period(planet.R + altitude), CHECK_ORBITS * SAMPLES_PER_ORBIT)
    plot_descent(orbiter, planet, altitude)
    plot_probe(PROBE_ALTITUDES, decay, survived, H, fit, used, floor, altitude, edge)
    el = plot_low_orbit(orbiter, planet, start)
    burns = {name: (np.linalg.norm(dv), used) for _, dv, name, used in orbiter.boosts}
    print(f"A     descent {burns['descent'][0]:.1f} m/s ({burns['descent'][1]:.1f} kg), circularise "
          f"{burns['circularise'][0]:.1f} m/s ({burns['circularise'][1]:.1f} kg); fuel left {orbiter.fuel:,.0f} kg")
    print(f"      low orbit over {CHECK_ORBITS} orbits: a changes {np.ptp(el['a']):.3g} m, e at most {el['e'].max():.1e}")

    lowest_ok = PROBE_ALTITUDES[np.isfinite(decay)].min()
    record_many({
        "start-radius": (np.linalg.norm(orbiter.samples[0][1]) / 1e3, "{:,.0f}"),
        "dest": (planet.index, "{:d}"), "R": (planet.R / 1e3, "{:,.0f}"), "rho0": (planet.rho0, "{:.2f}"),
        "low-alt": (altitude / 1e3, "{:.0f}"), "low-period": (planet.period(planet.R + altitude) / 60, "{:.1f}"),
        "dv-descent": (burns["descent"][0], "{:.1f}"), "dv-circ": (burns["circularise"][0], "{:.1f}"),
        "fuel-descent": (burns["descent"][1] + burns["circularise"][1], "{:.0f}"),
        "fuel-low": (orbiter.fuel, "{:,.0f}"), "fuel-start": (orbiter.fuel0, "{:,.0f}"),
        "probe-n": (len(PROBE_ALTITUDES), "{:d}"), "probe-orbits": (PROBE_ORBITS, "{:d}"),
        "probe-H": (H / 1e3, "{:.2f}"), "probe-fit-n": (int(used.sum()), "{:d}"),
        "probe-fit-lo": (PROBE_ALTITUDES[used].min() / 1e3, "{:.0f}"), "probe-fit-hi": (PROBE_ALTITUDES[used].max() / 1e3, "{:.0f}"),
        "probe-lowest-ok": (lowest_ok / 1e3, "{:.0f}"), "decay-limit": (DECAY_LIMIT, "{:g}"),
        "edge": (edge / 1e3, "{:.0f}"), "safety-heights": (SAFETY_HEIGHTS, "{:d}"), "probe-floor": (floor, "{:.2g}"),
        "check-orbits": (CHECK_ORBITS, "{:d}"), "low-a-change": (np.ptp(el["a"]), "{:.2g}"),
        "low-e-max": (el["e"].max(), "{:.1e}"),
    })
    record_video(orbiter, planet)
    return orbiter, planet, altitude, H


if __name__ == "__main__":
    main()
