"""
Part 2, extra: general-relativistic corrections to the planet orbits.

This is not part of the assignment. The API's "true" orbits are Newtonian Kepler
ellipses, so the orbits we verify and save (orbits.py) stay Newtonian. Here we
measure what general relativity (GR) would change, and how that compares with
our numerical error and with the other physics a fixed-star model leaves out.

Physics: for a planet around a star of mass M, the Schwarzschild orbit equation
(Part 9) is Newton's orbit equation plus one term (u = 1/r):
        d^2u/dphi^2 + u = GM/h^2  +  3 GM u^2 / c^2
As a force per unit mass this is
        a = -(GM / r^2) (1 + 3 h^2 / (c^2 r^2)) r_hat,       h = |r x v|.
Written with the planet's proper time this is exact in Schwarzschild, not an
approximation. The extra pull grows close to the star, so the orbit no longer
closes: the perihelion advances by
        dphi = 6 pi GM / (c^2 a (1 - e^2))  per orbit
(Mercury: 43 arcsec per century).

The clever bits:
  1. The extra term is still a central force, so h stays constant. Leapfrog even
     keeps it constant exactly: kicks are along r, drifts are along v. So we
     compute h once, the force depends on position only, and leapfrog stays
     symplectic.
  2. GR rotates the orbit by about 1e-7 rad per orbit, which is smaller than
     leapfrog's own artificial precession. We therefore integrate each orbit
     twice with identical time steps, once with Newton and once with GR, and
     subtract. The integrator's error is the same in both and cancels; the
     difference is the GR effect.
  3. The perihelion direction comes from the Laplace-Runge-Lenz vector
     A = v x L - GM r_hat, which points at the perihelion and is constant
     under Newtonian gravity. So we can track it at every time step instead
     of waiting for perihelion passages; this matters because our orbits are
     nearly circular.
  4. Validation: the same code run on Mercury around the Sun must give 43"/century.

Run:  ../.venv/bin/python relativity.py
Figures go to figures/r*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np

from common.config import get_mission
from common.integrators import leapfrog
from common.plot_style import BLUE, INK, MUTED, ORANGE, PLANET_COLORS, save
from common.report import record
from orbits import G, N_HOME_ORBITS, STEPS_PER_YEAR, kepler_positions, relative_deviation, simulate_orbits
from star_wobble import simulate_nbody, star_and_planets

C = const.c_AU_pr_yr  # Speed of light [AU/yr]
ARCSEC = np.degrees(1) * 3600  # Arcseconds per radian
N_ORBITS = 300  # Orbits per body in the precession measurement
STEPS_PER_ORBIT = 1000
SAVE_EVERY = 20
MERCURY = {"a": 0.387098, "e": 0.205630, "M": 1.0}  # Validation case: 42.98"/century
ROSETTE_PRECESSION = np.radians(20)  # Precession per orbit in the exaggerated picture


def precession_per_orbit(GM, a, e, c=C):
    """First-order GR perihelion advance [rad per orbit]."""
    return 6 * np.pi * GM / (c ** 2 * a * (1 - e ** 2))


def leapfrog_central(GM, gr, x0, v0, dt, n_steps, save_every=SAVE_EVERY):
    """
    Leapfrog (common.integrators) for independent bodies around fixed stars, one body per row.

    GM, gr = 3 h^2 / c^2 (0 for Newton) and dt are per row, so bodies with very
    different periods can all do the same number of steps per orbit.
    x0, v0 (n, 2). Returns positions and velocities (2, n, n_out).
    """
    GM, gr, dt = (np.asarray(q, dtype=float)[:, None] for q in (GM, gr, dt))
    x, v = x0, v0

    def acceleration(x):
        r2 = np.sum(x ** 2, axis=1, keepdims=True)
        return -GM * x / r2 ** 1.5 * (1 + gr / r2)

    positions, velocities = leapfrog(acceleration, x, v, dt, n_steps // save_every, save_every)
    return positions.transpose(2, 1, 0), velocities.transpose(2, 1, 0)


def perihelion_angle(GM, positions, velocities):
    """Direction of the Laplace-Runge-Lenz vector A = v x L - GM r_hat, unwrapped in time. GM per row."""
    x, y = positions
    vx, vy = velocities
    h = x * vy - y * vx
    r = np.hypot(x, y)
    GM = np.asarray(GM)[:, None]
    return np.unwrap(np.arctan2(-vx * h - GM * y / r, vy * h - GM * x / r), axis=-1)


def bodies(system):
    """Initial state, GM, a, e and period for our planets plus Mercury (started at perihelion)."""
    n = system.number_of_planets
    GM = np.append(np.full(n, G * system.star_mass), G * MERCURY["M"])
    a = np.append(system.semi_major_axes, MERCURY["a"])
    e = np.append(system.eccentricities, MERCURY["e"])
    r_peri = MERCURY["a"] * (1 - MERCURY["e"])
    v_peri = np.sqrt(GM[-1] * (1 + MERCURY["e"]) / r_peri)
    x0 = np.vstack([system.initial_positions.T, [r_peri, 0.0]])
    v0 = np.vstack([system.initial_velocities.T, [0.0, v_peri]])
    P = 2 * np.pi * np.sqrt(a ** 3 / GM)
    names = [f"planet {p}" for p in range(n)] + ["Mercury"]
    return names, x0, v0, GM, a, e, P


# ===================================================== R1: perihelion precession
def measure_precession(system):
    """Newton and GR runs side by side for every body; GR - Newton gives the precession."""
    names, x0, v0, GM, a, e, P = bodies(system)
    h = x0[:, 0] * v0[:, 1] - x0[:, 1] * v0[:, 0]
    n = len(names)
    # Rows 0..n-1: Newton, rows n..2n-1: GR. Same time steps for the pair.
    pos, vel = leapfrog_central(np.tile(GM, 2), np.concatenate([np.zeros(n), 3 * h ** 2 / C ** 2]),
                        np.vstack([x0, x0]), np.vstack([v0, v0]), np.tile(P / STEPS_PER_ORBIT, 2),
                        N_ORBITS * STEPS_PER_ORBIT)
    angle = perihelion_angle(np.tile(GM, 2), pos, vel)
    orbits = np.arange(angle.shape[1]) * SAVE_EVERY / STEPS_PER_ORBIT  # Time in orbits (same for every row)
    newton, gr = angle[:n] - angle[:n, :1], angle[n:] - angle[n:, :1]
    shift = gr - newton
    measured = np.array([np.polyfit(orbits, s, 1)[0] for s in shift])  # [rad per orbit]
    spurious = np.array([np.polyfit(orbits, s, 1)[0] for s in newton])  # Leapfrog's own precession
    predicted = precession_per_orbit(GM, a, e)
    century = 100 / P  # Orbits per century
    return {"names": names, "orbits": orbits, "newton": newton, "gr": gr, "shift": shift,
            "measured": measured, "spurious": spurious, "predicted": predicted, "century": century, "e": e}


def plot_precession(res, fast):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5.5))
    t = res["orbits"]
    per_orbit = STEPS_PER_ORBIT // SAVE_EVERY
    n_full = len(t) // per_orbit * per_orbit
    orbit_mean = lambda q: q[:n_full].reshape(-1, per_orbit).mean(axis=1)  # Average out the wobble within each orbit
    t_mean = orbit_mean(t)

    # Raw: leapfrog's artificial precession dwarfs GR, and is the same in both runs
    ax0.plot(t_mean, orbit_mean(res["newton"][fast]) * ARCSEC, color=MUTED, lw=3, label="Newton run")
    ax0.plot(t_mean, orbit_mean(res["gr"][fast]) * ARCSEC, color=ORANGE, lw=1.2, label="GR run")
    ax0.set_xlabel("time [orbits]")
    ax0.set_ylabel("perihelion direction (orbit average) [arcsec]")
    ax0.set_title(f"{res['names'][fast]}: both runs drift with leapfrog's own error\n"
                  f"({abs(res['spurious'][fast] / res['predicted'][fast]):.0f}x bigger than GR, and backwards)")
    ax0.legend(fontsize=8, loc="lower left")

    ax1.plot(t, res["shift"][fast] * ARCSEC, color=BLUE, lw=1, alpha=0.5, label="GR $-$ Newton (wobbles within each orbit)")
    ax1.plot(t_mean, orbit_mean(res["shift"][fast]) * ARCSEC, color=BLUE, lw=2.5, label="  (orbit average)")
    ax1.plot(t, res["predicted"][fast] * t * ARCSEC, color=INK, ls="--", lw=1.2,
             label=r"$6\pi GM/(c^2a(1-e^2))$ per orbit")
    ax1.set_xlabel("time [orbits]")
    ax1.set_ylabel("GR perihelion advance [arcsec]")
    ax1.set_title("Subtracting the Newton run leaves the GR advance")
    ax1.legend(fontsize=8, loc="upper left")

    order = np.argsort(-res["predicted"] * res["century"])
    x = np.arange(len(order))
    pred = res["predicted"][order] * res["century"][order] * ARCSEC
    meas = res["measured"][order] * res["century"][order] * ARCSEC
    mercury = np.array([res["names"][i] == "Mercury" for i in order])
    colors = [PLANET_COLORS[i] for i in order[~mercury]]
    ax2.scatter(x, pred, marker="_", s=600, color=INK, lw=2, label="first-order GR prediction", zorder=2)
    ax2.scatter(x[~mercury], meas[~mercury], s=70, c=colors, edgecolor=INK, lw=0.5, label="measured: our planets",
                zorder=3)
    ax2.scatter(x[mercury], meas[mercury], s=80, marker="D", color=MUTED, edgecolor=INK, lw=0.5,
                label="measured: Mercury (check, 43\")", zorder=3)
    for xi, m, p in zip(x, meas, pred):
        ax2.annotate(f"{m:.3g}\"\n({m / p - 1:+.0e})", (xi, m), xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=7, color=MUTED)
    ax2.set_yscale("log")
    ax2.set_ylim(pred.min() / 3, pred.max() * 6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([res["names"][i].replace("planet ", "") for i in order])
    ax2.set_xlabel("planet (Mercury around the Sun as the check)")
    ax2.set_ylabel("perihelion advance [arcsec per century]")
    ax2.set_title("Measured vs. predicted precession (relative difference in brackets)")
    ax2.legend(fontsize=8, loc="upper right")
    fig.suptitle("R1  General relativity: the perihelion advances a little every orbit", fontsize=12)
    save(fig, "r1_perihelion_precession")


# =============================================== R2: what it does to our orbits
def compare_corrections(system):
    """Over the 21 home orbits the API checks: GR shift, N-body shift and our numerical error."""
    duration = N_HOME_ORBITS * 2 * np.pi * np.sqrt(system.semi_major_axes[0] ** 3 / (G * system.star_mass))
    times, newton, _ = simulate_orbits(system, duration)
    _, gr, _ = simulate_orbits(system, duration, c=C)
    numerical = relative_deviation(newton, kepler_positions(system, times))
    gr_shift = relative_deviation(gr, newton)

    # Full N-body (star moves, planets pull on each other), positions relative to the star
    planets = np.arange(system.number_of_planets)
    masses, x0, v0 = star_and_planets(system, planets)
    _, pos, _, _, _ = simulate_nbody(masses, x0, v0, duration, steps_per_year=STEPS_PER_YEAR, save_every=20)
    helio = (pos[:, 1:] - pos[:, :1]).transpose(2, 1, 0)
    nbody_shift = relative_deviation(helio, newton)
    return times, numerical, gr_shift, nbody_shift, helio, newton


def closest_encounter(helio):
    """Closest approach between any two planets in the N-body run: (distance, i, j, time index)."""
    d = np.linalg.norm(helio[:, :, None, :] - helio[:, None, :, :], axis=0)  # (n, n, n_out)
    d[np.arange(len(d)), np.arange(len(d))] = np.inf
    i, j, k = np.unravel_index(np.argmin(d), d.shape)
    return d[i, j, k], min(i, j), max(i, j), k


def plot_corrections(system, times, numerical, gr_shift, nbody_shift, helio, newton):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))
    for p in range(system.number_of_planets):
        ax1.semilogy(times[1:], gr_shift[p, 1:] * 100, color=PLANET_COLORS[p], lw=1.3, label=f"planet {p}")
    ax1.set_ylim(bottom=1e-9)
    ax1.set_xlabel("time [yr]")
    ax1.set_ylabel("|r$_{GR}$ $-$ r$_{Newton}$| / r  [%]")
    ax1.set_title("GR shift of each planet (mostly a tiny lead/lag along the orbit)")
    ax1.legend(fontsize=7, ncol=4, loc="lower right")

    worst = lambda q: np.maximum.accumulate(q.max(axis=0))[1:] * 100
    ax2.semilogy(times[1:], worst(nbody_shift), color=ORANGE, label="planets pull each other + star moves (N-body)")
    ax2.semilogy(times[1:], worst(numerical), color=MUTED, label="our numerical error (leapfrog, 40 000 steps/yr)")
    ax2.semilogy(times[1:], worst(gr_shift), color=BLUE, label="general relativity")
    ax2.axhline(1, color=INK, ls="--", lw=1.2)
    ax2.text(times[1], 1, " API limit: 1%", va="bottom", color=INK, fontsize=9)
    ax2.set_xlabel("time [yr]")
    ax2.set_ylabel("worst planet so far [%]")
    ax2.set_title("Which correction matters? GR is the smallest")
    ax2.legend(fontsize=8, loc="lower right")

    # What the N-body curve hides: two gas giants nearly collide
    dist, i, j, k = closest_encounter(helio)
    for p in (i, j):
        ax3.plot(newton[0, p], newton[1, p], color=INK, lw=0.8, ls="--")
        ax3.plot(helio[0, p], helio[1, p], color=PLANET_COLORS[p], lw=1.5,
                 label=f"planet {p} ({system.masses[p] / 9.55e-4:.2f} Jupiter masses)")
        ax3.plot(*helio[:, p, k], "o", color=PLANET_COLORS[p], ms=8, mec=INK, mew=0.5)
    ax3.plot([], [], color=INK, lw=0.8, ls="--", label="fixed-star Kepler orbits (what the game uses)")
    ax3.annotate(f"t = {times[k]:.1f} yr: {dist:.3f} AU apart", helio[:, i, k], xytext=(0, -2.6),
                 textcoords="data", ha="center", fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax3.plot(0, 0, "*", color="#eda100", ms=14, mec=INK, mew=0.5)
    ax3.set_aspect("equal")
    lim = np.abs(helio[:, [i, j]]).max() * 1.05
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim * 1.5)  # Room for the legend
    ax3.set_xlabel("x [AU]")
    ax3.set_ylabel("y [AU]")
    ax3.set_title(f"N-body: planet {i} meets planet {j} and is thrown onto a new orbit")
    ax3.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"R2  Corrections to the fixed-star Newtonian orbits over {N_HOME_ORBITS} home orbits",
                 fontsize=12)
    save(fig, "r2_size_of_corrections")


# ===================================================== R3: exaggerated picture
def plot_rosette(system):
    """The most eccentric planet with c reduced so that GR turns the orbit by 20 degrees per orbit."""
    p = int(np.argmax(system.eccentricities))
    GM, a, e = G * system.star_mass, system.semi_major_axes[p], system.eccentricities[p]
    c_slow = np.sqrt(6 * np.pi * GM / (ROSETTE_PRECESSION * a * (1 - e ** 2)))
    x0, v0 = system.initial_positions[:, p], system.initial_velocities[:, p]
    h = x0[0] * v0[1] - x0[1] * v0[0]
    P = 2 * np.pi * np.sqrt(a ** 3 / GM)
    n_orbits = 18
    pos, vel = leapfrog_central([GM, GM], [0, 3 * h ** 2 / c_slow ** 2], np.vstack([x0, x0]), np.vstack([v0, v0]),
                        [P / 4000] * 2, n_orbits * 4000, save_every=10)
    # This strong, the LRL vector wobbles within each orbit, so use the actual perihelion passages (minima of r)
    r = np.hypot(*pos[:, 1])
    peri = np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0] + 1
    peri_angle = np.unwrap(np.arctan2(pos[1, 1, peri], pos[0, 1, peri]))
    measured = np.polyfit(np.arange(len(peri)), peri_angle, 1)[0]  # unwrap already removed the 2 pi per orbit
    # Beyond first order, for a near-circular orbit with this h: the circular solution of the Binet
    # equation, 3GM u0^2/c^2 - u0 + GM/h^2 = 0, and the oscillation about it, d^2du/dphi^2 + (1 - 6GM u0/c^2) du = 0,
    # give 2 pi (1 / sqrt(1 - 6 GM u0 / c^2) - 1). The extra pull makes the orbit smaller than Newton's a.
    rs = GM / c_slow ** 2
    u0 = (1 - np.sqrt(1 - 12 * rs * GM / h ** 2)) / (6 * rs)
    exact = 2 * np.pi * (1 / np.sqrt(1 - 6 * rs * u0) - 1)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(pos[0, 1], pos[1, 1], color=PLANET_COLORS[p], lw=1, label=f"GR, {n_orbits} orbits")
    ax.plot(pos[0, 0], pos[1, 0], color=INK, lw=1.2, ls="--", label="Newton (closed ellipse)")
    ax.plot(pos[0, 1, peri], pos[1, 1, peri], "o", color=PLANET_COLORS[p], ms=5, mec=INK, mew=0.5,
            label="perihelion passages")
    ax.plot(0, 0, "*", color="#eda100", ms=16, mec=INK, mew=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"R3  Planet {p} if light were {C / c_slow:.0f}x slower (e = {e:.3f} drawn to scale)\n"
                 f"perihelion advance {np.degrees(measured):.2f} deg/orbit (first order: "
                 f"{np.degrees(ROSETTE_PRECESSION):.0f}, all orders, circular: {np.degrees(exact):.2f})", fontsize=11)
    save(fig, "r3_exaggerated_precession")
    return C / c_slow, measured, exact


def main():
    system = get_mission().system
    print(f"R1    perihelion precession: Newton and GR runs, {N_ORBITS} orbits x {STEPS_PER_ORBIT} steps per body ...")
    res = measure_precession(system)
    print("      body       e        GM/(c^2 a)   predicted       measured        ratio-1   leapfrog's own")
    print("                                       [\"/century]     [\"/century]              [\"/century]")
    for i in np.argsort(-res["predicted"] * res["century"]):
        scale = res["century"][i] * ARCSEC
        a = system.semi_major_axes[i] if i < system.number_of_planets else MERCURY["a"]
        M = system.star_mass if i < system.number_of_planets else MERCURY["M"]
        print(f"      {res['names'][i]:<9} {res['e'][i]:.4f}   {G * M / (C ** 2 * a):.2e}   "
              f"{res['predicted'][i] * scale:12.4f}   {res['measured'][i] * scale:12.4f}   "
              f"{res['measured'][i] / res['predicted'][i] - 1:+.1e}   {res['spurious'][i] * scale:+12.4f}")
    fast = int(np.argmin(system.semi_major_axes))
    plot_precession(res, fast)
    merc = len(res["names"]) - 1 if res["names"][-1].lower().startswith("merc") else None
    if merc is not None:
        record("gr-mercury", res["measured"][merc] * res["century"][merc] * ARCSEC, "{:.2f}")
    record("gr-fast-planet", fast, "{:d}")
    record("gr-fast-pred", res["predicted"][fast] * res["century"][fast] * ARCSEC, "{:.1f}")
    record("gr-fast-meas", res["measured"][fast] * res["century"][fast] * ARCSEC, "{:.1f}")

    print(f"R2    {N_HOME_ORBITS} home orbits: Newton, GR and full N-body (~20 s) ...")
    times, numerical, gr_shift, nbody_shift, helio, newton = compare_corrections(system)
    radius_km = np.linalg.norm(system.initial_positions, axis=0) * const.AU / 1e3
    print("      planet   GR shift          N-body shift      numerical error  (largest over the run)")
    for p in np.argsort(system.semi_major_axes):
        print(f"      {p:>6}   {gr_shift[p].max():.1e} ({gr_shift[p].max() * radius_km[p]:7.0f} km)   "
              f"{nbody_shift[p].max():.1e}           {numerical[p].max():.1e}")
    dist, i, j, k = closest_encounter(helio)
    print(f"      N-body: planets {i} and {j} pass {dist:.4f} AU apart at t = {times[k]:.2f} yr")
    record("gr-home-km", gr_shift[0].max() * radius_km[0], "{:.0f}")
    record("nbody-pair", f"{i} and {j}")
    record("nbody-dist", dist, "{:.3f}")
    record("nbody-time", times[k], "{:.1f}")
    plot_corrections(system, times, numerical, gr_shift, nbody_shift, helio, newton)

    factor, measured, exact = plot_rosette(system)
    print(f"R3    exaggerated: c / {factor:.0f} gives {np.degrees(measured):.2f} deg per orbit "
          f"(first order {np.degrees(ROSETTE_PRECESSION):.0f}, all orders for a circular orbit {np.degrees(exact):.2f})")


if __name__ == "__main__":
    main()
