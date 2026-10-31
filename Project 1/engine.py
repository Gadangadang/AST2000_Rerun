"""
Part 1, challenge C: Introducing a nozzle.

The box from challenge B gets a square hole of area 0.25 L^2 in the middle of
the x = L wall. Every particle that leaves through it carries momentum m*vx out
of the engine (that is the thrust) and mass m (that is the fuel consumption).

The number of particles must stay constant, so every escaped particle has to
be replaced. How we refill turns out to matter, because it changes the gas:

  "inlet"     (default) A fuel inlet the same size as the hole, in the middle
              of the opposite wall x = 0. It injects one particle per escaped
              particle, with the velocity distribution a wall at temperature T
              emits (vx ~ flux-weighted, vy and vz Gaussian). The gas keeps
              its temperature and settles into a steady flow towards the
              nozzle.
  "closed"    Count the particle as escaped but reflect it, as in challenge B.
              This is the gas in exact equilibrium: the analytical reference.
  "opposite"  Same particle, same velocity, moved to the opposite wall. The
              column in front of the hole turns into a one-way beam, and the
              rate keeps growing.
  "random"    Fresh Maxwell-Boltzmann particle at a random position. The
              fastest particles escape most often but are replaced by average
              ones, so the gas cools down and the thrust keeps falling.

compare_refill_methods() runs all four and plots this.

The engine numbers per box are written to engine_box.json for challenges D-F.

Run:  ../.venv/bin/python engine.py
Figures are written to figures/c*.png.
"""
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from ast2000tools.constants import k_B, m_H2
from gas_box import analytical, initial_state
from part_a_statistics import sample_velocities, sigma_v
from plot_style import AQUA, BLUE, INK, MUTED, ORANGE, VIOLET, save

# ------------------------------------------------------------------ parameters
L = 1e-6  # Box side length [m]
T = 3e3  # Temperature [K]
N = int(1e5)  # Number of particles in the box (kept constant)
HOLE_AREA_FRACTION = 0.25  # Hole area / L^2
TAU = 2e-9  # Total simulated time [s]
N_STEPS = 2000  # -> dt = 1e-12 s
BURN_IN = 0.5e-9  # Ignore the first part while the gas settles into its steady flow [s]
REFILL = "inlet"  # One of "inlet", "closed", "opposite", "random"
SEED = 2000

RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_box.json")


def ideal_engine(N=N, L=L, T=T, hole_area_fraction=HOLE_AREA_FRACTION, m=m_H2):
    """
    Effusion from a gas in equilibrium (all from challenge A):
      particles out per second   n A <vx, positive> = n A sqrt(kT / 2 pi m)
      thrust                     n A m <vx^2, positive> = P A / 2
      exhaust speed              thrust / (m * rate) = sqrt(pi kT / 2m)
    """
    ana = analytical(N, L, T, m)
    A = hole_area_fraction * L ** 2
    rate = ana["flux"] * A
    thrust = ana["P"] * A / 2
    return {"rate": rate, "thrust": thrust, "mass_loss_rate": m * rate, "exhaust_speed": thrust / (m * rate)}


def refill(positions, velocities, escaped, method, L, T, hole_half_side, dt, rng):
    """Put the escaped particles back into the box according to `method` (see module docstring)."""
    k = np.count_nonzero(escaped)
    if k == 0:
        return
    if method == "closed":
        positions[escaped, 0] = 2 * L - positions[escaped, 0]
        velocities[escaped, 0] *= -1
    elif method == "opposite":
        positions[escaped, 0] -= L
    elif method == "random":
        positions[escaped] = rng.uniform(0, L, size=(k, 3))
        velocities[escaped] = sample_velocities(k, T, rng)
    elif method == "inlet":
        s = sigma_v(T)
        # A wall at temperature T emits vx with probability ~ vx exp(-vx^2 / 2s^2) (a Rayleigh
        # distribution): fast particles cross the opening more often. Inverse-transform sample it.
        vx = s * np.sqrt(-2 * np.log(1 - rng.random(k)))
        velocities[escaped, 0] = vx
        velocities[escaped, 1:] = rng.normal(0, s, size=(k, 2))
        positions[escaped, 1:] = rng.uniform(L / 2 - hole_half_side, L / 2 + hole_half_side, size=(k, 2))
        # Spread the entry over the step, so injected particles don't all sit exactly at x = 0
        positions[escaped, 0] = vx * dt * rng.random(k)
    else:
        raise ValueError(f"unknown refill method {method!r}")


def simulate_engine(N=N, L=L, T=T, hole_area_fraction=HOLE_AREA_FRACTION, tau=TAU, n_steps=N_STEPS,
                    refill_method=REFILL, seed=SEED, keep_escaped=True):
    """
    Simulate one engine box with a hole in the x = L wall.

    Per time step we record the number of escaped particles, the momentum they
    carry out (sum of m vx) and the gas temperature (from <E> = 3/2 kT).
    """
    rng = np.random.default_rng(seed)
    dt = tau / n_steps
    positions, velocities = initial_state(N, L, T, rng)
    hole_half_side = np.sqrt(hole_area_fraction) * L / 2

    max_step_distance = 6 * sigma_v(T) * dt  # Practically nobody is faster than 6 sigma
    assert max_step_distance < L, f"dt too large: a fast particle moves {max_step_distance:.2e} m per step"

    escapes = np.zeros(n_steps, dtype=np.int64)
    momentum_out = np.zeros(n_steps)
    temperature = np.zeros(n_steps)
    escaped_vx = []

    for i in range(n_steps):
        positions += velocities * dt  # Free motion between collisions (exact)

        # y and z walls first, so y and z are inside the box when we check the hole
        for axis in (1, 2):
            below = positions[:, axis] < 0
            above = positions[:, axis] > L
            positions[below, axis] *= -1
            positions[above, axis] = 2 * L - positions[above, axis]
            velocities[below | above, axis] *= -1

        # x = 0 is a solid wall
        below = positions[:, 0] < 0
        positions[below, 0] *= -1
        velocities[below, 0] *= -1

        # x = L: through the hole, or bounce off the wall around it
        above = positions[:, 0] > L
        in_hole = above & (np.abs(positions[:, 1] - L / 2) <= hole_half_side) \
                        & (np.abs(positions[:, 2] - L / 2) <= hole_half_side)
        wall = above & ~in_hole
        positions[wall, 0] = 2 * L - positions[wall, 0]
        velocities[wall, 0] *= -1

        escapes[i] = np.count_nonzero(in_hole)
        momentum_out[i] = m_H2 * np.sum(velocities[in_hole, 0])
        if keep_escaped:
            escaped_vx.append(velocities[in_hole, 0].copy())
        refill(positions, velocities, in_hole, refill_method, L, T, hole_half_side, dt, rng)

        temperature[i] = m_H2 * np.mean(np.sum(velocities ** 2, axis=1)) / (3 * k_B)

    return {
        "N": N, "L": L, "T": T, "tau": tau, "n_steps": n_steps, "dt": dt, "refill": refill_method,
        "hole_area": (2 * hole_half_side) ** 2, "hole_area_fraction": hole_area_fraction,
        "time": np.arange(1, n_steps + 1) * dt,
        "escapes": escapes, "momentum_out": momentum_out, "temperature": temperature,
        "escaped_vx": np.concatenate(escaped_vx) if keep_escaped else None,
        "positions": positions, "velocities": velocities,
    }


def engine_performance(sim, burn_in=BURN_IN):
    """Thrust [N] and fuel consumption [kg/s] of ONE box, averaged after the burn-in."""
    steady = sim["time"] > burn_in
    duration = steady.sum() * sim["dt"]
    n_out = sim["escapes"][steady].sum()
    thrust = sim["momentum_out"][steady].sum() / duration
    mass_loss_rate = m_H2 * n_out / duration
    return {
        "thrust": thrust,
        "mass_loss_rate": mass_loss_rate,
        "rate": n_out / duration,
        "exhaust_speed": thrust / mass_loss_rate,
        "escaped_in_average": int(n_out),
        "relative_statistical_error": 1 / np.sqrt(n_out),  # Poisson noise in the count
    }


# ===================================================================== figures
def binned(values, n_bins):
    """Sum per-step values into n_bins equal time bins."""
    return values[: len(values) // n_bins * n_bins].reshape(n_bins, -1).sum(axis=1)


def plot_time_series(sim, perf, ideal):
    n_bins = 40
    t_bin = binned(sim["time"], n_bins) / (len(sim["time"]) // n_bins)  # Bin centres
    bin_time = sim["tau"] / n_bins
    rate = binned(sim["escapes"], n_bins) / bin_time
    thrust = binned(sim["momentum_out"], n_bins) / bin_time

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, values, ref, color, name, unit in (
            (axes[0], rate, ideal["rate"], BLUE, "particles out per second", "1/s"),
            (axes[1], thrust, ideal["thrust"], ORANGE, "thrust per box", "N")):
        ax.plot(t_bin * 1e9, values, "o-", color=color, ms=4, lw=1.5, label="simulation (per 0.05 ns)")
        ax.axhline(ref, color=INK, ls="--", lw=1.2, label="equilibrium formula")
        steady_value = perf["rate"] if unit == "1/s" else perf["thrust"]
        ax.axhline(steady_value, color=color, ls=":", lw=1.5, label=f"steady average = {steady_value:.3e} {unit}")
        ax.axvspan(0, BURN_IN * 1e9, color=MUTED, alpha=0.12, lw=0)
        ax.set_xlabel("time [ns]")
        ax.set_ylabel(f"{name} [{unit}]")
        ax.set_title(name)
        ax.legend(loc="center right", fontsize=8)
    axes[0].text(BURN_IN * 1e9 / 2, 0.95, "settling\n(ignored)", ha="center", va="top", color=MUTED, fontsize=9,
                 transform=axes[0].get_xaxis_transform())

    ax = axes[2]
    ax.plot(sim["time"] * 1e9, sim["temperature"], color=AQUA, lw=1.2, label="gas temperature from <E>")
    ax.axhline(sim["T"], color=INK, ls="--", lw=1.2, label=f"T = {sim['T']:.0f} K")
    ax.set_ylim(sim["T"] * 0.95, sim["T"] * 1.05)
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("T [K]")
    ax.set_title("temperature stays constant")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"C  One engine box: N = {sim['N']:.0e}, T = {sim['T']:.0f} K, hole = "
                 f"{sim['hole_area_fraction']} L$^2$, refill = '{sim['refill']}'", fontsize=12)
    save(fig, "c_engine_time_series")


def plot_exhaust(sim, perf):
    """The escaping particles are not a random sample of the gas: fast ones leave more often."""
    s = sigma_v(sim["T"])
    vx = np.linspace(0, 6 * s, 400)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.hist(sim["escaped_vx"] / 1e3, bins=80, range=(0, 6 * s / 1e3), density=True, color=ORANGE, alpha=0.5,
            label=f"escaped particles ({len(sim['escaped_vx']):,})")
    flux_weighted = vx / s ** 2 * np.exp(-vx ** 2 / (2 * s ** 2))
    gas = 2 * np.exp(-vx ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    ax.plot(vx / 1e3, flux_weighted * 1e3, color=INK, label=r"flux-weighted $\propto v_x e^{-mv_x^2/2kT}$")
    ax.plot(vx / 1e3, gas * 1e3, color=BLUE, ls="--", lw=1.5, label=r"gas inside, $v_x > 0$ (A2.1)")
    mean_out = np.mean(sim["escaped_vx"])
    ax.axvline(mean_out / 1e3, color=ORANGE, lw=1.5, ls=":")
    ax.text(mean_out / 1e3, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 0.2,
            f"  mean exhaust $v_x$ = {mean_out:.0f} m/s", color=INK, fontsize=9)
    ax.set_xlabel(r"$v_x$ [km/s]")
    ax.set_ylabel("probability density [s/km]")
    ax.set_title("C  Velocities of particles leaving through the nozzle")
    ax.legend(loc="center right")
    save(fig, "c_exhaust_velocities")


def plot_flow_map(sim):
    """Mean vx in the slab in front of the hole: the gas streams towards the nozzle."""
    L = sim["L"]
    pos, vel = sim["positions"], sim["velocities"]
    half = np.sqrt(sim["hole_area_fraction"]) * L / 2
    slab = np.abs(pos[:, 2] - L / 2) <= half  # Same z-range as the hole
    bins = 20
    sum_vx, xe, ye = np.histogram2d(pos[slab, 0] / L, pos[slab, 1] / L, bins=bins, range=[[0, 1], [0, 1]],
                                    weights=vel[slab, 0])
    count, _, _ = np.histogram2d(pos[slab, 0] / L, pos[slab, 1] / L, bins=bins, range=[[0, 1], [0, 1]])
    mean_vx = np.where(count > 0, sum_vx / np.maximum(count, 1), 0)
    lim = np.max(np.abs(mean_vx))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(xe, ye, mean_vx.T / 1e3, cmap="coolwarm", vmin=-lim / 1e3, vmax=lim / 1e3)
    fig.colorbar(im, ax=ax, label=r"mean $v_x$ [km/s]")
    for x0, name, align in ((1, "nozzle", "right"), (0, "inlet", "left")):
        ax.plot([x0, x0], [0.5 - half / L, 0.5 + half / L], color=INK, lw=5, solid_capstyle="butt")
        ax.text(x0, 0.5 + half / L + 0.02, name, ha=align, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.8))
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_aspect("equal")
    ax.set_title("C  Mean $v_x$ in the slab in front of the hole (end of run)")
    save(fig, "c_flow_map")


def compare_refill_methods(tau=TAU):
    """Run all refill methods with the same initial gas and plot how each one changes the gas."""
    methods = [("closed", INK, "closed box (equilibrium)"),
               ("inlet", BLUE, "inlet opposite the nozzle (used)"),
               ("opposite", ORANGE, "same velocity, opposite wall"),
               ("random", AQUA, "fresh MB particle, random spot")]
    ideal = ideal_engine()
    n_bins = 20
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    print("C     refill comparison (thrust / equilibrium formula, averaged after burn-in)")
    for method, color, label in methods:
        sim = simulate_engine(tau=tau, n_steps=int(round(tau / 1e-12)), refill_method=method, keep_escaped=False)
        bin_time = tau / n_bins
        t_bin = (np.arange(n_bins) + 0.5) * bin_time
        thrust = binned(sim["momentum_out"], n_bins) / bin_time
        ax1.plot(t_bin * 1e9, thrust / ideal["thrust"], "o-", ms=3, lw=1.5, color=color, label=label)
        ax2.plot(sim["time"] * 1e9, sim["temperature"], lw=1.5, color=color, label=label)
        perf = engine_performance(sim)
        print(f"      {method:<9} thrust ratio {perf['thrust'] / ideal['thrust']:.3f}, "
              f"final temperature {sim['temperature'][-1]:.0f} K")
    ax1.axhline(1, color=MUTED, lw=1, ls=":")
    ax1.set_xlabel("time [ns]")
    ax1.set_ylabel("thrust / (PA/2)")
    ax1.set_title("Thrust depends on how the box is refilled")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("time [ns]")
    ax2.set_ylabel("gas temperature [K]")
    ax2.set_title("Refilling with average particles cools the gas")
    ax2.legend(fontsize=8)
    save(fig, "c_refill_comparison")


def main():
    ideal = ideal_engine()
    print(f"C     N = {N:.0e}, L = {L} m, T = {T:.0f} K, hole = {HOLE_AREA_FRACTION} L^2, "
          f"tau = {TAU} s, dt = {TAU / N_STEPS} s, refill = '{REFILL}'")
    t0 = time.perf_counter()
    sim = simulate_engine()
    print(f"      simulation took {time.perf_counter() - t0:.1f} s")
    assert np.all((sim["positions"] >= 0) & (sim["positions"] <= L)), "a particle is outside the box"

    perf = engine_performance(sim)
    print(f"      averaged over t > {BURN_IN:.1e} s: {perf['escaped_in_average']} escaped particles "
          f"(statistical error ~{perf['relative_statistical_error']:.1%})")
    for key, unit in (("thrust", "N"), ("mass_loss_rate", "kg/s"), ("rate", "1/s"), ("exhaust_speed", "m/s")):
        print(f"      {key:<15} sim = {perf[key]:.5e} {unit:<5} equilibrium formula = {ideal[key]:.5e} {unit:<5} "
              f"ratio = {perf[key] / ideal[key]:.3f}")
    print(f"      mean gas temperature after burn-in = {np.mean(sim['temperature'][sim['time'] > BURN_IN]):.1f} K")

    with open(RESULT_FILE, "w") as f:
        json.dump({"parameters": {"N": N, "L": L, "T": T, "hole_area_fraction": HOLE_AREA_FRACTION, "tau": TAU,
                                  "n_steps": N_STEPS, "burn_in": BURN_IN, "refill": REFILL, "seed": SEED},
                   "thrust_per_box": perf["thrust"],
                   "mass_loss_rate_per_box": perf["mass_loss_rate"],
                   "exhaust_speed": perf["exhaust_speed"]}, f, indent=2)
    print(f"      -> {os.path.basename(RESULT_FILE)}")

    plot_time_series(sim, perf, ideal)
    plot_exhaust(sim, perf)
    plot_flow_map(sim)
    compare_refill_methods()


if __name__ == "__main__":
    main()
