"""
Part 2, challenge A: Planetary orbits.

Units: AU, yr and solar masses, so G = 4 pi^2 AU^3 / (yr^2 Msun).
The star is fixed at the origin and the planets only feel the star.

A1: The analytical orbit (Lecture Notes 1B) is the ellipse
        r(theta) = a (1 - e^2) / (1 + e cos(theta - omega)),
    where omega is the angle of the perihelion (= aphelion angle + pi).

A2: We integrate all planets at once (NumPy arrays) with the leapfrog
    scheme (kick-drift-kick), which conserves energy over long times.

    Clever bit: to know how accurate the simulation is, we don't just compare
    shapes. We solve Kepler's equation  M = E - e sin E  for the exact position
    of every planet at every time, and measure the error in POSITION ALONG THE
    ORBIT, which is what the API checks (max 1% deviation). That lets us choose
    the time step from a measured error curve instead of guessing.

Then: verify_planet_positions (the official check) and an XML video for SSView.

Run:  ../.venv/bin/python orbits.py
Figures go to figures/a*.png, our orbits to simulated_orbits.npz, the API's
exact orbits to planet_trajectories.npz, and the SSView video to XMLs/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import io
import shutil
import time
from contextlib import redirect_stdout

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np

from common.config import SSVIEW_DATA_DIR, TRAJECTORY_FILE, XML_DIR, get_mission
from common.plot_style import INK, MUTED, PLANET_COLORS, save

G = const.G_sol  # = 4 pi^2 [AU^3 yr^-2 Msun^-1]
N_HOME_ORBITS = 21  # Simulate a bit more than the 20 home-planet orbits the API checks
STEPS_PER_YEAR = 40000  # Chosen from the time-step study below (minimum allowed is 10 000)
SAVE_EVERY = 20  # Keep every 20th step -> 2000 positions per year
ORBIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulated_orbits.npz")


# ================================================================ A1: analytic
def analytic_orbit(a, e, aphelion_angle, n_points=1000):
    """Points (x, y) on the orbit ellipse, from r(theta) = a(1-e^2)/(1 + e cos(theta - omega_perihelion))."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    perihelion_angle = aphelion_angle + np.pi
    r = a * (1 - e ** 2) / (1 + e * np.cos(theta - perihelion_angle))
    return r * np.cos(theta), r * np.sin(theta)


def kepler_positions(system, times):
    """Exact positions from Kepler's equation, shape (2, n_planets, n_times)."""
    return kepler_state(system, times)[0]


def kepler_state(system, times):
    """
    Exact positions [AU] and velocities [AU/yr] from Kepler's equation, each (2, n_planets, n_times).

    Mean anomaly M grows linearly in time; solve M = E - e sin E for the
    eccentric anomaly E with Newton's method, then convert to the true
    anomaly f and the distance r = a (1 - e cos E). The velocity follows from
    the orbit equation: v = sqrt(GM/p) (-sin f, e + cos f) along (perihelion, 90 deg ahead),
    with p = a (1 - e^2).
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    a, e = system.semi_major_axes[:, None], system.eccentricities[:, None]
    perihelion = system.aphelion_angles[:, None] + np.pi
    n = np.sqrt(G * system.star_mass / a ** 3)  # Mean motion [rad/yr]

    theta0 = np.arctan2(system.initial_positions[1], system.initial_positions[0])[:, None]
    f0 = theta0 - perihelion
    E0 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f0 / 2))
    M = (E0 - e * np.sin(E0)) + n * times[None, :]

    E = M.copy()
    for _ in range(20):  # Newton converges in a handful of steps for e < 0.1
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E))
    theta = f + perihelion
    positions = np.array([r * np.cos(theta), r * np.sin(theta)])
    v_p = np.sqrt(G * system.star_mass / (a * (1 - e ** 2)))
    v_peri, v_ahead = -v_p * np.sin(f), v_p * (e + np.cos(f))  # Components along / 90 deg ahead of perihelion
    velocities = np.array([v_peri * np.cos(perihelion) - v_ahead * np.sin(perihelion),
                           v_peri * np.sin(perihelion) + v_ahead * np.cos(perihelion)])
    return positions, velocities


# ================================================================ A2: simulate
def simulate_orbits(system, duration, steps_per_year=STEPS_PER_YEAR, save_every=SAVE_EVERY, c=None):
    """
    Leapfrog (kick-drift-kick) for all planets at once around a fixed star.

    c: speed of light [AU/yr] to include general relativity's correction (see
    relativity.py). None (default) is pure Newton, which is what the API checks.

    Returns times (n_out,), positions and velocities (2, n_planets, n_out),
    sampled every `save_every` steps, uniformly from t = 0 to t = duration.
    """
    GM = G * system.star_mass
    n_out = int(np.ceil(duration * steps_per_year / save_every))
    n_steps = n_out * save_every
    dt = duration / n_steps  # So the last sample lands exactly on t = duration

    x = system.initial_positions.T.copy()  # (n_planets, 2)
    v = system.initial_velocities.T.copy()
    # GR multiplies the star's pull by 1 + 3 h^2 / (c^2 r^2); h = |r x v| is conserved, so compute it once
    h = x[:, :1] * v[:, 1:] - x[:, 1:] * v[:, :1]
    gr = 0.0 if c is None else 3 * h ** 2 / c ** 2

    def acceleration(x):
        r = np.linalg.norm(x, axis=1, keepdims=True)
        a = -GM * x / r ** 3
        return a if c is None else a * (1 + gr / r ** 2)

    times = np.linspace(0, duration, n_out + 1)
    positions = np.zeros((n_out + 1, *x.shape))
    velocities = np.zeros_like(positions)
    positions[0], velocities[0] = x, v

    a = acceleration(x)
    for i in range(1, n_out + 1):
        for _ in range(save_every):
            v += 0.5 * dt * a
            x += dt * v
            a = acceleration(x)
            v += 0.5 * dt * a
        positions[i], velocities[i] = x, v

    # (n_out, n_planets, 2) -> (2, n_planets, n_out), the layout ast2000tools uses
    return times, positions.transpose(2, 1, 0), velocities.transpose(2, 1, 0)


def relative_deviation(positions, exact):
    """|r_sim - r_exact| / |r_exact| for every planet and time: the API's accuracy measure."""
    return np.linalg.norm(positions - exact, axis=0) / np.linalg.norm(exact, axis=0)


# ===================================================================== figures
def plot_analytic(system):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    order = np.argsort(system.semi_major_axes)
    for ax, planets, title in ((ax1, order, "All planets"), (ax2, order[:3], "Inner system (zoom)")):
        ax.plot(0, 0, "*", color="#eda100", ms=16, mec=INK, mew=0.5)
        for p in planets:
            x, y = analytic_orbit(system.semi_major_axes[p], system.eccentricities[p], system.aphelion_angles[p])
            ax.plot(x, y, color=PLANET_COLORS[p], lw=1.8, label=f"planet {p} ({system.types[p]})")
            x0, y0 = system.initial_positions[:, p]
            ax.plot(x0, y0, "o", color=PLANET_COLORS[p], ms=7, mec=INK, mew=0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title(title)
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    lim = system.semi_major_axes[order[2]] * 1.15
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    fig.suptitle("A1  Analytical orbits, $r = a(1-e^2)/(1+e\\cos(\\theta-\\omega))$  (dots: positions at t = 0)",
                 fontsize=12)
    save(fig, "a1_analytic_orbits")


def plot_simulated(system, times, positions):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    order = np.argsort(system.semi_major_axes)
    for ax, planets, title in ((ax1, order, "All planets"), (ax2, order[:3], "Inner system (zoom)")):
        ax.plot(0, 0, "*", color="#eda100", ms=16, mec=INK, mew=0.5)
        for p in planets:
            x, y = analytic_orbit(system.semi_major_axes[p], system.eccentricities[p], system.aphelion_angles[p])
            ax.plot(x, y, color=INK, lw=0.8, ls="--")
            ax.plot(positions[0, p], positions[1, p], color=PLANET_COLORS[p], lw=2.2, alpha=0.6,
                    label=f"planet {p}")
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title(title)
    ax1.plot([], [], color=INK, lw=0.8, ls="--", label="analytical ellipse")
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    lim = system.semi_major_axes[order[2]] * 1.15
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    fig.suptitle(f"A2  Simulated orbits ({times[-1]:.1f} yr = {N_HOME_ORBITS} home orbits) on top of the "
                 "analytical ellipses", fontsize=12)
    save(fig, "a2_simulated_orbits")


def plot_conservation(system, times, positions, velocities):
    """Largest relative change in energy and angular momentum (per unit mass), per time bin."""
    GM = G * system.star_mass
    r = np.linalg.norm(positions, axis=0)
    energy = 0.5 * np.sum(velocities ** 2, axis=0) - GM / r
    h = positions[0] * velocities[1] - positions[1] * velocities[0]
    n_bins = 200
    edges = np.linspace(0, len(times), n_bins + 1).astype(int)
    t_bin = times[(edges[:-1] + edges[1:]) // 2]

    def envelope(q):
        dev = np.abs(q / q[:, :1] - 1)
        return np.array([dev[:, a:b].max(axis=1) for a, b in zip(edges[:-1], edges[1:])]).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, q in ((ax1, energy), (ax2, h)):
        env = envelope(q)
        for p in range(system.number_of_planets):
            ax.semilogy(t_bin, np.maximum(env[p], 1e-17), color=PLANET_COLORS[p], lw=1.5, label=f"planet {p}")
        ax.set_xlabel("time [yr]")
    ax1.set_ylabel(r"max $|E/E_0 - 1|$ per time bin")
    ax1.set_title("Energy: bounded oscillation, no drift (leapfrog)")
    ax2.set_ylabel(r"max $|h/h_0 - 1|$ per time bin")
    ax2.set_title("Angular momentum: conserved to round-off (~1e-13)")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8)
    fig.suptitle("A2  Conserved quantities in the simulation (faster planets: bigger, but constant, wiggle)",
                 fontsize=12)
    save(fig, "a2_conservation", rect=(0, 0.06, 1, 1))


def plot_position_error(system, times, deviation):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for p in range(system.number_of_planets):
        ax.plot(times, deviation[p] * 100, color=PLANET_COLORS[p], lw=1.2, label=f"planet {p}")
    ax.axhline(1, color=INK, ls="--", lw=1.2)
    ax.text(times[0], 1, " API limit: 1%", va="bottom", color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("relative position error [%]")
    ax.set_title("A2  Error vs. exact Kepler solution (grows linearly: a tiny phase lag that accumulates)")
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    save(fig, "a2_position_error")


def timestep_study(system, duration):
    """Max relative deviation over all planets for several time steps: picks STEPS_PER_YEAR."""
    print("A2    time-step study ...")
    steps = np.array([2500, 5000, 10000, 20000, 40000, 80000])
    worst = []
    for s in steps:
        t, pos, _ = simulate_orbits(system, duration, steps_per_year=s, save_every=max(1, s // 1000))
        worst.append(np.max(relative_deviation(pos, kepler_positions(system, t))))
        print(f"      {s:>6} steps/yr: worst deviation {worst[-1]:.2e}")
    worst = np.array(worst)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(steps, worst * 100, "o-", color=PLANET_COLORS[0], ms=6, label="simulation (worst planet)")
    ax.loglog(steps, worst[0] * 100 * (steps / steps[0]) ** -2.0, "--", color=MUTED, lw=1.2,
              label=r"$\propto \Delta t^2$ (leapfrog is 2nd order)")
    ax.axhline(1, color=INK, ls="--", lw=1.2, label="API limit: 1%")
    ax.axvline(10000, color=MUTED, ls=":", lw=1)
    ax.text(10000, worst.min() * 100, " minimum required", color=MUTED, fontsize=8, rotation=90, va="bottom")
    ax.axvline(STEPS_PER_YEAR, color=INK, lw=1.2)
    ax.text(STEPS_PER_YEAR, worst.max() * 100, " chosen", color=INK, fontsize=9, va="top")
    ax.set_xlabel("time steps per year")
    ax.set_ylabel("worst relative position error [%]")
    ax.set_title(f"A2  Choosing the time step ({N_HOME_ORBITS} home orbits, all planets)")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "a2_timestep_study")


# ================================================================ API checks
def verify_with_api(mission, times, positions):
    """
    Official check of our orbits; also saves the exact orbits for later parts.

    The API's own Kepler solver occasionally fails to converge at one particular
    sample time (and then crashes in its error message). If that happens we hand
    it every 2nd, 3rd, ... sample instead, which avoids that time: the samples
    stay uniform from t = 0 and still cover more than the 20 checked orbits.
    """
    mission.system.verbose = True  # The check runs on the SolarSystem, which has its own verbose flag
    min_duration = 20 * 2 * np.pi * np.sqrt(mission.system.semi_major_axes[0] ** 3 / (G * mission.system.star_mass))
    for k in range(1, 11):
        idx = np.arange(0, len(times), k)
        assert times[idx[-1]] >= min_duration and len(idx) >= 1000
        log = io.StringIO()
        try:
            with redirect_stdout(log):
                mission.verify_planet_positions(times[idx[-1]], positions[:, :, idx], filename=TRAJECTORY_FILE)
        except NameError:  # The API's crash after its solver failed to converge
            continue
        if k > 1:
            print(f"      (the API's solver failed on our full sample set; used 1 in {k} samples: {len(idx)} times)")
        print("      " + "\n      ".join(line for line in log.getvalue().splitlines() if line.strip()))
        return
    raise RuntimeError("the API's orbit solver kept failing")


def make_orbit_video(mission, times, positions):
    """XML for SSView, which lists the files in MCAst/ssv_data (MCAst itself uses data/)."""
    log = io.StringIO()
    with redirect_stdout(log):
        mission.generate_orbit_video(times, positions, filename="orbit_video.xml")
    path = os.path.join(XML_DIR, "orbit_video.xml")
    print(f"      SSView video written to {path}")
    if os.path.isdir(SSVIEW_DATA_DIR):
        shutil.copy(path, SSVIEW_DATA_DIR)
        print(f"      ... and copied into {SSVIEW_DATA_DIR}: run ./ssview from inside MCAst/ to watch it")


def main(study=True):
    """Run challenge A. study=False skips the ~20 s time-step study."""
    mission = get_mission(verbose=True)
    mission.verbose = True
    system = mission.system
    P = 2 * np.pi * np.sqrt(system.semi_major_axes ** 3 / (G * system.star_mass))
    print(f"A     star mass {system.star_mass:.4f} Msun, {system.number_of_planets} planets")
    for p in np.argsort(system.semi_major_axes):
        print(f"      planet {p} ({system.types[p]:<4}) a = {system.semi_major_axes[p]:6.3f} AU, "
              f"e = {system.eccentricities[p]:.4f}, period {P[p]:7.3f} yr, mass {system.masses[p]:.3e} Msun")

    plot_analytic(system)

    duration = N_HOME_ORBITS * P[0]
    if study:
        timestep_study(system, duration)
    t0 = time.perf_counter()
    times, positions, velocities = simulate_orbits(system, duration)
    print(f"A2    simulated {duration:.2f} yr ({N_HOME_ORBITS} home orbits) with {STEPS_PER_YEAR} steps/yr "
          f"({STEPS_PER_YEAR * P.min():.0f} steps per orbit of the fastest planet) in {time.perf_counter() - t0:.1f} s")
    np.savez_compressed(ORBIT_FILE, times=times, positions=positions, velocities=velocities)
    print(f"      orbits saved to {os.path.basename(ORBIT_FILE)} ({len(times)} samples)")

    deviation = relative_deviation(positions, kepler_positions(system, times))
    worst = np.unravel_index(np.argmax(deviation), deviation.shape)
    print(f"      worst deviation from Kepler's equation: {deviation[worst]:.2e} (planet {worst[0]})")
    plot_simulated(system, times, positions)
    plot_conservation(system, times, velocities=velocities, positions=positions)
    plot_position_error(system, times, deviation)

    print("API   verify_planet_positions")
    verify_with_api(mission, times, positions)
    make_orbit_video(mission, times, positions)
    return times, positions, velocities


if __name__ == "__main__":
    main()
