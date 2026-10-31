"""
Part 1, challenge B: Simulating energetic gas particles in a closed cubic box.

N H2 molecules start uniformly distributed in a box of side L with velocities
drawn from the Maxwell-Boltzmann distribution. They move in straight lines
(no gravity, no particle-particle collisions) and bounce elastically off the
walls.

Key idea: between bounces a particle moves in an exact straight line, and a
bounce is an exact mirror image. So instead of "flip the velocity and hope",
we mirror the position back through the wall. The simulation is then exact
for ANY time step as long as no particle crosses the whole box in one step.
The only error left is statistical (finite N and finite tau), and that is
what we test and plot here.

Checks against challenge A:
  - vx and |v| distributions, and uniform positions
  - <v> = sqrt(8kT/(pi m)),  <E> = 3/2 kT
  - P = nkT, from the momentum transferred to the walls
  - wall hit rate = n A sqrt(kT/(2 pi m)) (the escape rate through a hole in C)

Run:  ../.venv/bin/python gas_box.py
Figures are written to figures/b*.png.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from ast2000tools.constants import k_B, m_H2
from part_a_statistics import mb_1d, mb_speed, sample_velocities, sigma_v
from plot_style import AQUA, BLUE, INK, MUTED, ORANGE, SERIES, VIOLET, save

# ------------------------------------------------------------------ parameters
L = 1e-6  # Box side length [m]
T = 3e3  # Temperature [K]
N = int(1e5)  # Number of particles
TAU = 1e-9  # Total simulated time [s]
N_STEPS = 1000  # Number of time steps -> dt = 1e-12 s
SEED = 2000  # RNG seed, so runs are reproducible
HOLE_AREA_FRACTION = 0.25  # Size of the future nozzle (challenge C), as a fraction of L^2
N_TRACKED = 4  # Particles whose full path we store for the 3D plot

WALL_NAMES = ["x=0", "x=L", "y=0", "y=L", "z=0", "z=L"]


def analytical(N=N, L=L, T=T, m=m_H2):
    """Everything challenge A predicts for this box."""
    n = N / L ** 3
    flux = n * sigma_v(T, m) / np.sqrt(2 * np.pi)  # Hits per wall area per second
    return {
        "n": n,
        "<v>": np.sqrt(8 * k_B * T / (np.pi * m)),
        "<E>": 1.5 * k_B * T,
        "P": n * k_B * T,
        "wall_hit_rate": flux * L ** 2,  # Hits per second on ONE wall
        "flux": flux,
    }


def initial_state(N, L, T, rng):
    """Uniform positions in [0, L]^3; each velocity component ~ N(0, sqrt(kT/m))."""
    positions = rng.uniform(0, L, size=(N, 3))
    velocities = sample_velocities(N, T, rng)
    return positions, velocities


def reflect_walls(positions, velocities, L, hole_half_side):
    """
    Elastic collisions with all six walls, done in place.

    A particle that has crossed a wall during the last step is mirrored back
    inside (x -> -x or x -> 2L - x), which puts it exactly where it would be
    had it bounced mid-step, and its velocity component normal to that wall
    flips sign.

    Returns, for each of the six walls (order of WALL_NAMES):
      momentum   momentum given to the wall this step, 2 m |v_normal| per hit
      hits       number of hits this step
    plus the number of hits and outward momentum (m |vx|, one way) on the
    central patch of the x = L wall where the nozzle will be in challenge C.
    """
    momentum = np.zeros(6)
    hits = np.zeros(6, dtype=np.int64)
    for axis in range(3):
        below = positions[:, axis] < 0
        above = positions[:, axis] > L

        if axis == 0:
            # Particles hitting the would-be hole: central square of the x = L wall
            in_patch = above & (np.abs(positions[:, 1] - L / 2) <= hole_half_side) \
                             & (np.abs(positions[:, 2] - L / 2) <= hole_half_side)
            patch_hits = np.count_nonzero(in_patch)
            patch_momentum = m_H2 * np.sum(velocities[in_patch, 0])

        positions[below, axis] *= -1
        positions[above, axis] = 2 * L - positions[above, axis]

        momentum[2 * axis] = 2 * m_H2 * np.sum(np.abs(velocities[below, axis]))
        momentum[2 * axis + 1] = 2 * m_H2 * np.sum(np.abs(velocities[above, axis]))
        hits[2 * axis] = np.count_nonzero(below)
        hits[2 * axis + 1] = np.count_nonzero(above)

        velocities[below | above, axis] *= -1
    return momentum, hits, patch_hits, patch_momentum


def simulate(N=N, L=L, T=T, tau=TAU, n_steps=N_STEPS, seed=SEED, n_tracked=N_TRACKED):
    """Run the closed-box simulation. Returns a dict with the final state and per-step wall data."""
    rng = np.random.default_rng(seed)
    dt = tau / n_steps
    positions, velocities = initial_state(N, L, T, rng)
    initial_positions, initial_velocities = positions.copy(), velocities.copy()

    # Mirroring is exact only if nobody can cross the whole box in one step
    max_step_distance = np.max(np.abs(velocities)) * dt
    assert max_step_distance < L, f"dt too large: fastest particle moves {max_step_distance:.2e} m per step"

    hole_half_side = np.sqrt(HOLE_AREA_FRACTION) * L / 2
    wall_momentum = np.zeros((n_steps, 6))
    wall_hits = np.zeros((n_steps, 6), dtype=np.int64)
    patch_hits = np.zeros(n_steps, dtype=np.int64)
    patch_momentum = np.zeros(n_steps)
    tracks = np.zeros((n_steps + 1, n_tracked, 3))
    tracks[0] = positions[:n_tracked]

    for i in range(n_steps):
        positions += velocities * dt  # Free motion between collisions (exact)
        wall_momentum[i], wall_hits[i], patch_hits[i], patch_momentum[i] = \
            reflect_walls(positions, velocities, L, hole_half_side)
        tracks[i + 1] = positions[:n_tracked]

    return {
        "N": N, "L": L, "T": T, "tau": tau, "n_steps": n_steps, "dt": dt,
        "initial_positions": initial_positions, "initial_velocities": initial_velocities,
        "positions": positions, "velocities": velocities,
        "wall_momentum": wall_momentum, "wall_hits": wall_hits,
        "patch_hits": patch_hits, "patch_momentum": patch_momentum,
        "hole_area": (2 * hole_half_side) ** 2,
        "tracks": tracks,
    }


def measured(sim):
    """Turn raw simulation output into the same quantities as analytical()."""
    L, tau = sim["L"], sim["tau"]
    speeds = np.linalg.norm(sim["velocities"], axis=1)
    total_momentum = sim["wall_momentum"].sum(axis=0)  # Per wall
    return {
        "<v>": np.mean(speeds),
        "<E>": np.mean(0.5 * m_H2 * speeds ** 2),
        "P": total_momentum.sum() / (6 * L ** 2 * tau),  # Average over all six walls
        "P_x=L": total_momentum[1] / (L ** 2 * tau),  # Only one wall, as in the assignment
        "wall_hit_rate": sim["wall_hits"].sum() / (6 * tau),
        "patch_rate": sim["patch_hits"].sum() / tau,
        "patch_force": sim["patch_momentum"].sum() / tau,
    }


# ===================================================================== figures
def plot_distributions(sim):
    v0, v1 = sim["initial_velocities"], sim["velocities"]
    T = sim["T"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    vx = np.linspace(-2.5e4, 2.5e4, 500)
    ax.hist(v1[:, 0] / 1e3, bins=100, density=True, color=BLUE, alpha=0.5, label="simulation, end")
    ax.plot(vx / 1e3, mb_1d(vx, T) * 1e3, color=INK, label="analytical (A2.1)")
    ax.set_xlabel(r"$v_x$ [km/s]")
    ax.set_ylabel("probability density [s/km]")
    ax.set_title(r"$v_x$ distribution")
    ax.legend(loc="upper left")

    ax = axes[1]
    v = np.linspace(0, 3e4, 500)
    ax.hist(np.linalg.norm(v1, axis=1) / 1e3, bins=100, density=True, color=ORANGE, alpha=0.5,
            label="simulation, end")
    ax.plot(v / 1e3, mb_speed(v, T) * 1e3, color=INK, label="analytical (A2.3)")
    ax.set_xlabel(r"$v$ [km/s]")
    ax.set_title("speed distribution")
    ax.legend()

    ax = axes[2]
    L = sim["L"]
    bins = np.linspace(0, L, 41)
    for axis, (name, color) in enumerate(zip("xyz", SERIES)):
        ax.hist(sim["positions"][:, axis] / L, bins=bins / L, density=True, histtype="step",
                color=color, lw=1.5, label=name)
    ax.axhline(1, color=INK, lw=1.5, ls="--", label="uniform")
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("position / L")
    ax.set_title("positions stay uniform")
    ax.legend(ncol=4, loc="lower center")

    fig.suptitle(f"B  Gas after {sim['n_steps']} steps (N = {sim['N']:.0e}, T = {T:.0f} K): "
                 "still Maxwell-Boltzmann", fontsize=12)
    save(fig, "b_distributions")


def plot_trajectories(sim):
    """3D paths of a few particles, showing straight lines and mirror bounces."""
    L = sim["L"]
    tracks = sim["tracks"] / L
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(projection="3d")
    # Box edges
    for a in (0, 1):
        for b in (0, 1):
            for line in (([0, 1], [a, a], [b, b]), ([a, a], [0, 1], [b, b]), ([a, a], [b, b], [0, 1])):
                ax.plot(*line, color=MUTED, lw=0.8)
    for k, color in zip(range(tracks.shape[1]), SERIES + [VIOLET]):
        ax.plot(tracks[:, k, 0], tracks[:, k, 1], tracks[:, k, 2], color=color, lw=1.4)
        ax.scatter(*tracks[0, k], color=color, s=25)
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_zlabel("z / L")
    ax.set_title(f"B  Paths of {tracks.shape[1]} particles over tau = {sim['tau']:.0e} s (dots = start)")
    save(fig, "b_trajectories_3d")


def plot_pressure_and_hits(sim, ana):
    """Running pressure and hit rate vs time: do they settle on the analytical values?"""
    L, dt = sim["L"], sim["dt"]
    t = np.arange(1, sim["n_steps"] + 1) * dt
    cumulative = np.cumsum(sim["wall_momentum"], axis=0)
    p_one = cumulative[:, 1] / (L ** 2 * t)
    p_all = cumulative.sum(axis=1) / (6 * L ** 2 * t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))
    ax1.plot(t * 1e9, p_one / ana["P"], color=BLUE, lw=1.5, label="wall x = L only")
    ax1.plot(t * 1e9, p_all / ana["P"], color=ORANGE, lw=1.5, label="mean of all 6 walls")
    ax1.axhline(1, color=INK, lw=1, ls="--", label="nkT (A3.2)")
    ax1.set_ylim(0.9, 1.1)
    ax1.set_xlabel("time [ns]")
    ax1.set_ylabel(r"$P_{\rm sim}(t)\,/\,nkT$")
    ax1.set_title("Pressure = momentum given to wall / (area x time)")
    ax1.legend()

    hits = sim["wall_hits"].sum(axis=1) / 6
    expected = ana["wall_hit_rate"] * dt
    ax2.plot(t * 1e9, hits, color=AQUA, lw=0.8, alpha=0.8, label="hits per wall per step (mean of 6)")
    ax2.axhline(expected, color=INK, lw=1.5, ls="--",
                label=rf"$nA\sqrt{{kT/2\pi m}}\,dt$ = {expected:.1f}")
    ax2.set_xlabel("time [ns]")
    ax2.set_ylabel("hits per step")
    ax2.set_title("Wall hit rate: the escape rate a nozzle will see in C")
    ax2.legend(loc="lower right")
    save(fig, "b_pressure_and_hits")


def plot_scorecard(sim, ana, meas):
    """Relative error of every check next to the statistical noise we expect from A3."""
    N = sim["N"]
    total_hits = sim["wall_hits"].sum()
    rows = [
        (r"$\langle v\rangle$", meas["<v>"], ana["<v>"], np.sqrt(3 * np.pi / 8 - 1) / np.sqrt(N)),
        (r"$\langle E\rangle$", meas["<E>"], ana["<E>"], np.sqrt(2 / 3) / np.sqrt(N)),
        ("P, all 6 walls", meas["P"], ana["P"], np.sqrt(2 / 3) / np.sqrt(N)),
        ("P, wall x = L", meas["P_x=L"], ana["P"], np.sqrt(2 / N)),
        ("wall hit rate", meas["wall_hit_rate"], ana["wall_hit_rate"], 1 / np.sqrt(total_hits)),
    ]
    labels = [r[0] for r in rows]
    errors = [abs(r[1] / r[2] - 1) * 100 for r in rows]
    noise = [r[3] * 100 for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(y, errors, height=0.5, color=BLUE, label="|simulation / analytical - 1|")
    ax.scatter(noise, y, marker="|", s=500, color=ORANGE, lw=2.5, zorder=3,
               label=r"expected 1$\sigma$ statistical noise (approx.)")
    for yi, e in zip(y, errors):
        ax.text(e, yi, f"  {e:.3f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("relative error [%]")
    ax.set_xlim(0, max(max(errors), max(noise)) * 1.4)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"B  Scorecard: N = {N:.0e}, tau = {sim['tau']:.0e} s, dt = {sim['dt']:.0e} s")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "b_scorecard")


def convergence_study():
    """
    How accurate is the simulation, and what controls it?

    1) Vary N: the pressure error should fall like 1/sqrt(N) (pure statistics).
    2) Vary the number of time steps at fixed tau: because free flight plus
       mirroring is exact, the result should NOT depend on dt at all.
    """
    print("B     convergence study (takes ~1 min)")
    Ns = [100, 300, 1000, 3000, 10000, 30000]
    runs = {100: 30, 300: 30, 1000: 20, 3000: 10, 10000: 6, 30000: 3}
    rms = []
    for n_i in Ns:
        errs = []
        for s in range(runs[n_i]):
            sim = simulate(N=n_i, seed=100 + s, n_tracked=1)
            errs.append(measured(sim)["P"] / analytical(N=n_i)["P"] - 1)
        rms.append(np.sqrt(np.mean(np.square(errs))))
        print(f"      N = {n_i:>6}: rms pressure error over {runs[n_i]} runs = {rms[-1]:.2%}")

    step_counts = [50, 100, 300, 1000, 3000, 10000]
    n_dt = 10000
    p_ratio, run_time = [], []
    for steps in step_counts:
        t0 = time.perf_counter()
        sim = simulate(N=n_dt, n_steps=steps, seed=7, n_tracked=1)
        run_time.append(time.perf_counter() - t0)
        p_ratio.append(measured(sim)["P"] / analytical(N=n_dt)["P"])
        print(f"      N = {n_dt}, steps = {steps:>5} (dt = {TAU / steps:.0e} s): "
              f"P/nkT = {p_ratio[-1]:.6f}, {run_time[-1]:.2f} s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))
    Ns = np.array(Ns)
    ax1.loglog(Ns, rms, "o", color=BLUE, ms=6, label="simulation (rms over seeds)")
    ax1.loglog(Ns, np.sqrt(2 / 3) / np.sqrt(Ns), "--", color=MUTED, lw=1.2, label=r"$0.816/\sqrt{N}$ (from A3)")
    ax1.set_xlabel("number of particles N")
    ax1.set_ylabel("rms relative error in P")
    ax1.set_title("Accuracy is set by N: error falls as $1/\\sqrt{N}$")
    ax1.legend()

    ax2.semilogx(step_counts, p_ratio, "o-", color=ORANGE, ms=6, label=f"P / nkT, N = {n_dt:.0e}, same seed")
    ax2.axhline(1, color=INK, lw=1, ls="--")
    ax2.set_ylim(0.97, 1.03)
    ax2.set_xlabel(f"number of time steps in tau = {TAU:.0e} s")
    ax2.set_ylabel("P / nkT")
    ax2.set_title("...not by dt: exact mirroring makes the result step-independent")
    ax2.legend(loc="upper left")
    save(fig, "b_convergence")


def main():
    ana = analytical()
    print(f"B     N = {N:.0e}, L = {L} m, T = {T:.0f} K, tau = {TAU} s, dt = {TAU / N_STEPS} s")
    t0 = time.perf_counter()
    sim = simulate()
    print(f"      simulation took {time.perf_counter() - t0:.1f} s")
    assert np.all((sim["positions"] >= 0) & (sim["positions"] <= L)), "a particle escaped the closed box"

    meas = measured(sim)
    units = {"<v>": "m/s", "<E>": "J", "P": "Pa", "P_x=L": "Pa", "wall_hit_rate": "1/s"}
    for key, unit in units.items():
        reference = ana["P"] if key == "P_x=L" else ana[key]
        print(f"      {key:<14} sim = {meas[key]:.5e} {unit:<4} analytical = {reference:.5e} {unit:<4} "
              f"rel. error = {abs(meas[key] / reference - 1):.3%}")

    # Preview of challenge C: what a hole of area 0.25 L^2 in the x = L wall would let out.
    # Rate out = n A <v_x, positive> = flux * A; momentum out per second = n A m <vx^2>_+ = P A / 2.
    A = sim["hole_area"]
    print(f"C-preview  hole area {A:.2e} m^2 (= {HOLE_AREA_FRACTION} L^2) on the x = L wall:")
    print(f"      particles out per second: sim = {meas['patch_rate']:.4e}, analytical = {ana['flux'] * A:.4e}")
    print(f"      thrust (momentum out/s):  sim = {meas['patch_force']:.4e} N, analytical PA/2 = {ana['P'] * A / 2:.4e} N")
    print(f"      exhaust speed F/(m rate): sim = {meas['patch_force'] / (m_H2 * meas['patch_rate']):.1f} m/s, "
          f"analytical sqrt(pi kT / 2m) = {np.sqrt(np.pi * k_B * T / (2 * m_H2)):.1f} m/s")

    plot_distributions(sim)
    plot_trajectories(sim)
    plot_pressure_and_hits(sim, ana)
    plot_scorecard(sim, ana, meas)
    convergence_study()


if __name__ == "__main__":
    main()
