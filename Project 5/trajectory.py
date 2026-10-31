"""
Part 5, challenge A: a general simulator for the spacecraft's trajectory.

Input: start time t0, position r0 [AU] and velocity v0 [AU/yr], end time t1.
Output: the trajectory, final position and final velocity. While coasting, the
only force is gravity from the star and all the planets:

    a = -G M_s r / |r|^3 - sum_i G M_i (r - r_i) / |r - r_i|^3

Planet positions (Ephemeris): exact positions and velocities from Kepler's
equation (Part 2) on a uniform time grid, interpolated in between. On a uniform
grid the two neighbouring points are found by arithmetic
(index = floor((t - t0) / dt)) instead of a search.
  * Linear interpolation (what the assignment asks for) is implemented and
    measured: the error falls as dt^2.
  * The simulator uses cubic Hermite interpolation, which also matches the
    planets' exact velocities at the grid points. The error falls as dt^4, so
    a 20x coarser table is 1000x more accurate. More importantly, the planets'
    velocities are continuous. With linear interpolation they jump at every
    grid point, and an adaptive high-order integrator near a planet keeps
    shrinking its step to resolve those artificial kicks (~6x more work for the same trip).

Integrator: an adaptive 8th-order Runge-Kutta (DOP853). The trip has two very
different time scales: minutes near a planet (the launch ends just 700 km
above the surface) and months in between. A fixed step small enough for the
first would waste millions of steps on the second.

Clever bits:
  * Batches: simulate() takes many spacecraft at once ((n, 2) arrays). The
    acceleration is computed for all of them with NumPy broadcasting over
    (spacecraft, planet). The planner uses this to get Jacobians and parameter
    scans for about the cost of a single trajectory.
  * The same physics as the API (exact Kepler planets, star at rest), so our
    predictions match the API's flight to within kilometres.

Run:  ../.venv/bin/python trajectory.py
Figures go to figures/a*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(2)

import time
from dataclasses import dataclass

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from common.config import get_mission
from common.plot_style import BLUE, INK, MUTED, ORANGE, PLANET_COLORS, save
from orbits import G, kepler_state

DT_TABLE = 1e-4  # Ephemeris grid step [yr] (~53 min), Hermite interpolation
RTOL = 1e-10  # Relative tolerance of the integrator
ATOL = 1e-13  # Absolute tolerance [AU, AU/yr] (1e-13 AU = 1.5 cm)
KM = const.AU / 1e3  # km per AU


# =================================================================== ephemeris
class Ephemeris:
    """Planet positions from a uniform grid of exact Kepler states, interpolated in time."""

    def __init__(self, system, t_start, t_end, dt=DT_TABLE, method="hermite"):
        self.t_start, self.dt, self.method = t_start, dt, method
        self.times = t_start + dt * np.arange(int(np.ceil((t_end - t_start) / dt)) + 2)
        positions, velocities = kepler_state(system, self.times)
        self.positions = np.ascontiguousarray(positions.transpose(2, 1, 0))  # (n_t, n_planets, 2)
        self.velocities = np.ascontiguousarray(velocities.transpose(2, 1, 0))

    def __call__(self, t):
        """Positions (n_planets, 2) at time t (or (..., n_planets, 2) for an array of times)."""
        s = (np.asarray(t, dtype=float) - self.t_start) / self.dt
        i = np.clip(np.floor(s).astype(int), 0, len(self.times) - 2)
        w = (s - i)[..., None, None]
        p0, p1 = self.positions[i], self.positions[i + 1]
        if self.method == "linear":
            return (1 - w) * p0 + w * p1
        # Cubic Hermite: matches position and velocity at both grid points
        m0, m1 = self.velocities[i] * self.dt, self.velocities[i + 1] * self.dt
        w2, w3 = w * w, w * w * w
        return ((2 * w3 - 3 * w2 + 1) * p0 + (w3 - 2 * w2 + w) * m0 + (-2 * w3 + 3 * w2) * p1 + (w3 - w2) * m1)

    def covers(self, t0, t1):
        return self.times[0] <= min(t0, t1) and max(t0, t1) <= self.times[-1]


# ================================================================== simulation
@dataclass
class Trajectory:
    """times (n_t,), positions and velocities (n_craft, 2, n_t); `dense` evaluates the solution at any time."""
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    dense: object = None
    n_evaluations: int = 0

    @property
    def final_position(self):
        return self.positions[..., -1]

    @property
    def final_velocity(self):
        return self.velocities[..., -1]

    def state(self, t):
        """Positions and velocities (n_craft, 2) at time(s) t, from the dense output."""
        y = self.dense(t).reshape(len(self.positions), 4, *np.shape(t))
        return y[:, :2], y[:, 2:]


class Simulator:
    """The spacecraft's equations of motion in a given solar system."""

    def __init__(self, system, ephemeris):
        self.system, self.ephemeris = system, ephemeris
        self.GM_star = G * system.star_mass
        self.GM_planets = G * system.masses  # (n_planets,)

    def acceleration(self, t, r):
        """Gravitational acceleration [AU/yr^2] at positions r (n_craft, 2) at time t."""
        d = r[:, None, :] - self.ephemeris(t)[None]  # (n_craft, n_planets, 2)
        planets = np.einsum("p,cpk->ck", self.GM_planets, d / np.sum(d ** 2, axis=2, keepdims=True) ** 1.5)
        star = self.GM_star * r / np.sum(r ** 2, axis=1, keepdims=True) ** 1.5
        return -(star + planets)

    def _rhs(self, t, y):
        s = y.reshape(-1, 4)
        return np.concatenate([s[:, 2:], self.acceleration(t, s[:, :2])], axis=1).ravel()

    def simulate(self, t0, r0, v0, t1, n_out=None, dense=False, rtol=RTOL):
        """
        Coast from (t0, r0, v0) to t1 (t1 < t0 integrates backwards).

        r0, v0: (2,) for one spacecraft or (n, 2) for a batch. n_out: number of
        evenly spaced output times (default: the integrator's own steps).
        """
        r0, v0 = np.atleast_2d(r0), np.atleast_2d(v0)
        if not self.ephemeris.covers(t0, t1):
            raise ValueError(f"the ephemeris does not cover {t0:.4f}-{t1:.4f} yr")
        y0 = np.concatenate([r0, v0], axis=1).ravel()
        t_eval = None if n_out is None else np.linspace(t0, t1, n_out)
        sol = integrate.solve_ivp(self._rhs, (t0, t1), y0, method="DOP853", t_eval=t_eval, rtol=rtol, atol=ATOL,
                                  dense_output=dense)
        if not sol.success:
            raise RuntimeError(f"trajectory integration failed: {sol.message}")
        y = sol.y.reshape(len(r0), 4, -1)
        return Trajectory(sol.t, y[:, :2], y[:, 2:], sol.sol, sol.nfev)

    def distance_to_planet(self, trajectory, planet):
        """|r - r_planet| along a trajectory, (n_craft, n_t) [AU]."""
        planets = self.ephemeris(trajectory.times)[:, planet].T  # (2, n_t)
        return np.linalg.norm(trajectory.positions - planets[None], axis=1)

    def closest_approach(self, trajectory, planet, n_fine=4000):
        """Time [yr] and distance [AU] of the closest approach to a planet, per spacecraft (needs dense output)."""
        coarse = self.distance_to_planet(trajectory, planet)
        times = trajectory.times
        result = []
        for c in range(coarse.shape[0]):
            k = int(np.argmin(coarse[c]))
            fine_t = np.linspace(times[max(k - 1, 0)], times[min(k + 1, len(times) - 1)], n_fine)
            pos, _ = trajectory.state(fine_t)
            d = np.linalg.norm(pos[c] - self.ephemeris(fine_t)[:, planet].T, axis=0)
            j = int(np.argmin(d))
            result.append((fine_t[j], d[j]))
        return np.array(result)


def make_simulator(system, t_start, t_end):
    return Simulator(system, Ephemeris(system, t_start, t_end))


# ===================================================================== tests
def ephemeris_error_study(system, steps=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6)):
    """Largest interpolation error per planet vs. grid step, linear and Hermite (halfway between grid points)."""
    errors = {"linear": [], "hermite": []}
    for dt in steps:
        for method in errors:
            eph = Ephemeris(system, 0.0, 0.1, dt, method)
            mid = eph.times[:-2] + dt / 2
            exact = kepler_state(system, mid)[0].transpose(2, 1, 0)
            errors[method].append(np.linalg.norm(eph(mid) - exact, axis=2).max(axis=0))
    return np.array(steps), {m: np.array(e) * KM for m, e in errors.items()}  # (n_steps, n_planets) [km]


def interpolation_cost_test(system, t0, r0, v0, duration):
    """Force evaluations and wall time for the same trajectory with linear vs. Hermite planet positions."""
    out = {}
    for method, dt in (("linear", 5e-6), ("hermite", DT_TABLE)):
        sim = Simulator(system, Ephemeris(system, t0 - 0.01, t0 + duration + 0.01, dt, method))
        start = time.perf_counter()
        traj = sim.simulate(t0, r0, v0, t0 + duration)
        out[method] = (traj.n_evaluations, time.perf_counter() - start, traj.final_position[0])
    return out


def kepler_test(system, n_orbits=20):
    """A massless test particle on the home planet's orbit, star only: compare with Kepler's equation."""
    class StarOnly(Simulator):
        def acceleration(self, t, r):
            return -self.GM_star * r / np.sum(r ** 2, axis=1, keepdims=True) ** 1.5

    sim = StarOnly(system, Ephemeris(system, 0.0, 0.01))
    P = 2 * np.pi * np.sqrt(system.semi_major_axes[0] ** 3 / (G * system.star_mass))
    sim.ephemeris.covers = lambda t0, t1: True
    results = []
    for rtol in (1e-6, 1e-8, 1e-10, 1e-12):
        t0 = time.perf_counter()
        traj = sim.simulate(0.0, system.initial_positions[:, 0], system.initial_velocities[:, 0], n_orbits * P,
                            n_out=2000, rtol=rtol)
        exact = kepler_state(system, traj.times)[0][:, 0]
        results.append((rtol, np.linalg.norm(traj.positions[0] - exact, axis=0).max() * KM, traj.n_evaluations,
                        time.perf_counter() - t0))
    return results


def batch_speed_test(sim, t0, r0, v0, duration, sizes=(1, 4, 16, 64)):
    """Wall time per spacecraft for batches of spacecraft with slightly different starting velocities."""
    rng = np.random.default_rng(9)
    out = []
    for n in sizes:
        v = v0 + rng.normal(0, 1e-3, (n, 2))
        t_start = time.perf_counter()
        sim.simulate(t0, np.tile(r0, (n, 1)), v, t0 + duration)
        out.append((n, (time.perf_counter() - t_start) / n))
    return out


# ===================================================================== figures
def plot_tests(system, steps, eph_errors, kepler_results, batch_results, cost):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ax = axes[0]
    for method, style in (("linear", "o--"), ("hermite", "s-")):
        for p in (7, 0, 1):
            ax.loglog(steps, np.maximum(eph_errors[method][:, p], 1e-9), style, color=PLANET_COLORS[p], ms=4, lw=1.3,
                      label=f"{method}, planet {p}")
    ax.axvline(DT_TABLE, color=INK, lw=1.2)
    ax.text(DT_TABLE, 1e-8, " used (Hermite)", fontsize=9)
    ax.set_xlabel("grid step of the planet table [yr]")
    ax.set_ylabel("largest interpolation error [km]")
    ax.set_title(f"Planet positions: linear (dt$^2$) vs. Hermite (dt$^4$)\nsame trip: linear needs "
                 f"{cost['linear'][0] / cost['hermite'][0]:.1f}x the force evaluations", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    rtols = [r[0] for r in kepler_results]
    ax.loglog(rtols, [r[1] for r in kepler_results], "o-", color=BLUE, label="position error after 20 orbits [km]")
    ax.loglog(rtols, [r[2] for r in kepler_results], "s-", color=ORANGE, label="force evaluations")
    ax.axvline(RTOL, color=INK, lw=1.2)
    ax.set_xlabel("integrator tolerance (rtol)")
    ax.set_title("Integrator check: test particle vs. Kepler's equation", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    sizes = [b[0] for b in batch_results]
    ax.loglog(sizes, [b[1] * 1e3 for b in batch_results], "o-", color=BLUE, label="measured")
    ax.loglog(sizes, [batch_results[0][1] * 1e3 / s for s in sizes], "--", color=MUTED, lw=1, label="ideal: 1 / batch size")
    ax.set_xlabel("spacecraft simulated together")
    ax.set_ylabel("wall time per spacecraft [ms]")
    ax.set_title("Batching: many trajectories for the price of one", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("A  The trajectory simulator", fontsize=12)
    save(fig, "a_simulator_tests")


def main():
    system = get_mission().system
    steps, eph_errors = ephemeris_error_study(system)
    i = int(np.argmin(np.abs(steps - DT_TABLE)))
    for method, errors in eph_errors.items():
        print(f"A     {method:>7} interpolation, table every {DT_TABLE:.0e} yr: error at most "
              f"{errors[i].max() * 1e3:10.3f} m")
    kepler_results = kepler_test(system)
    for rtol, err, nfev, wall in kepler_results:
        print(f"      test particle, 20 orbits, rtol {rtol:.0e}: error {err:9.4f} km, {nfev:6d} force evaluations, "
              f"{wall:.2f} s")
    # A typical cruise: leave the home planet 0.002 AU out, 1 AU/yr faster than the planet, for 0.3 yr
    r0 = system.initial_positions[:, 0] + np.array([0.002, 0])
    v0 = system.initial_velocities[:, 0] * 1.15
    cost = interpolation_cost_test(system, 0.0, r0, v0, 0.3)
    print(f"      same 0.3 yr trip: linear table {cost['linear'][0]} evaluations ({cost['linear'][1]:.2f} s), "
          f"Hermite {cost['hermite'][0]} ({cost['hermite'][1]:.2f} s); end points differ by "
          f"{np.linalg.norm(cost['linear'][2] - cost['hermite'][2]) * KM:.2f} km")
    sim = make_simulator(system, 0.0, 0.35)
    batch_results = batch_speed_test(sim, 0.0, r0, v0, 0.3)
    for n, per in batch_results:
        print(f"      batch of {n:>3} spacecraft: {per * 1e3:7.1f} ms per spacecraft")
    plot_tests(system, steps, eph_errors, kepler_results, batch_results, cost)


if __name__ == "__main__":
    main()
