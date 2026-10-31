"""
Part 5, challenges C and D: the real flight with ast2000tools.

1. Launch at the planned time from the planned site (Part 3 generalised
   launch, verified by the API), then the manual orientation with our Part 4
   software (verified by the API).
2. Departure boost: re-aimed from the actual state after launch. Our launch
   prediction is ~1 km and ~7 m/s off the API's, which would grow to a big
   miss after four months.
3. Course corrections: at a few points on the way the spacecraft orients
   itself (travel.orient), and Newton's method (planner.aim) finds the small
   boost that puts it back on target from where it really is. Each correction
   uses the same simulator as the plan.
4. Arrival: our simulator predicts the closest approach from the last
   orientation; we coast there, orient, and inject into a circular orbit.
5. D: orbital elements from the orientation right after injection, then from
   orientations over several orbits (they must stay constant), and a long
   coast to check that the orbit is stable.

Run:  ../.venv/bin/python journey.py   (plans first, then flies)
Figures go to figures/c*.png and figures/d*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(1, 2, 3, 4)

import io
import shutil
from contextlib import redirect_stdout

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission

import orientation
import planner
from common.config import JOURNEY_MISSION_FILE, MCAST_DATA_DIR, XML_DIR, get_mission
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, VIOLET, save
from generalised_launch import launch_state
from orbit_analysis import orbital_elements, polar_components
from orbits import kepler_state
from trajectory import KM, make_simulator

AU_PER_YR = const.AU / const.yr
# Course corrections, as fractions of the way to the target time. The first comes a few days after departure:
# tiny differences between our integrator and the API's near the home planet grow fast (b6_sensitivity), and
# early on they are cheapest to fix.
CORRECTIONS = (0.02, 0.25, 0.6, 0.85, 0.97)
MISS_TOLERANCE_KM = 100  # Correct the course when heading more than this far off the aim point [km]
N_SAMPLES = 120  # Orientations recorded along the way (for the figures)
N_ORBITS_CHECK = 10  # Orbits sampled right after injection (D)
LONG_COAST = 0.25  # Stability check: coast this long in orbit [yr]


def quiet(function, *args, **kwargs):
    """Call an API function with its printing silenced; return (result, printed text)."""
    log = io.StringIO()
    with redirect_stdout(log):
        result = function(*args, **kwargs)
    return result, log.getvalue()


# ===================================================================== launch
def launch_for_real(ctx, plan):
    """Launch at the planned time and site, verify it, and do the manual orientation."""
    mission = get_mission()
    mission.verbose = True
    engine, launch = ctx.engine, ctx.launch
    before, after, _, _ = launch_state(launch, ctx.home, ctx.system, plan.t_launch, plan.launch_angle)
    quiet(mission.set_launch_parameters, engine["thrust"], engine["mass_loss_rate"], engine["fuel_mass"],
          launch["duration"] * 1.05, before, plan.t_launch)
    quiet(mission.launch_rocket)
    _, text = quiet(mission.verify_launch_result, after)
    print("C     launch: " + ("VERIFIED" if mission.launch_result_verified else text))
    library = orientation.sky_pictures.load_or_build_library(orientation.sky_pictures.load_sky())
    angle, velocity, position = orientation.orient(mission, library)
    _, text = quiet(mission.verify_manual_orientation, position, velocity, angle)
    print("      manual orientation (Part 4 software): "
          + ("VERIFIED" if mission.manual_orientation_verified else text))
    return mission


# ===================================================================== the flight
class Flight:
    """The spacecraft in flight, with a log of everything it measures and does."""

    def __init__(self, ctx, plan, mission):
        self.ctx, self.plan = ctx, plan
        self.travel = mission.begin_interplanetary_travel()
        self.travel.verbose = False
        self.samples = []  # (t, r, v) from orientations
        self.boosts = []  # (t, dv, name)

    def orient(self):
        t, r, v = self.travel.orient()
        self.samples.append((t, r, v))
        return t, r, v

    def boost(self, dv, name):
        t, _, _ = self.travel.orient()
        self.travel.boost(dv)
        self.boosts.append((t, np.array(dv), name))

    def coast_to(self, t_end, n_samples=1):
        """Coast in steps, orienting along the way (for the figures)."""
        t_now = self.travel.orient()[0]
        for t in np.linspace(t_now, t_end, n_samples + 1)[1:]:
            self.travel.coast_until_time(t)
            self.orient()
        return self.samples[-1]

    @property
    def fuel(self):
        return self.travel.remaining_fuel_mass


def fly(ctx, plan, video=True):
    mission = launch_for_real(ctx, plan)
    flight = Flight(ctx, plan, mission)
    sim, dest = ctx.sim, ctx.destination

    # Departure boost, re-aimed from where the API says we are
    t, r, v = flight.orient()
    dv, miss = planner.aim(sim, t, r, v, plan.dv_boost, plan.t_target, plan.target)
    flight.boost(dv, "departure")
    print(f"C     departure boost {np.linalg.norm(dv) * AU_PER_YR:.2f} m/s (plan: "
          f"{np.linalg.norm(plan.dv_boost) * AU_PER_YR:.2f}); predicted miss {miss * KM:.2f} km")

    # Course corrections
    samples_per_leg = N_SAMPLES // (len(CORRECTIONS) + 1)
    for k, fraction in enumerate(CORRECTIONS):
        t_c = plan.t_boost + fraction * (plan.t_target - plan.t_boost)
        t, r, v = flight.coast_to(t_c, samples_per_leg)
        predicted = sim.simulate(t, r, v, plan.t_target).final_position[0]
        off = np.linalg.norm(predicted - plan.target) * KM
        correct = off > MISS_TOLERANCE_KM
        if correct:
            dv, miss = planner.aim(sim, t, r, v, np.zeros(2), plan.t_target, plan.target)
            flight.boost(dv, f"correction {k + 1}")
        print(f"C     t = {t:.5f} yr: heading {off:10.1f} km off the aim point -> "
              + (f"correction {np.linalg.norm(dv) * AU_PER_YR * 100:8.3f} cm/s" if correct else "on course"))

    # Arrival: find the closest approach from here, coast there, inject
    t, r, v = flight.orient()
    approach = sim.simulate(t, r, v, plan.t_target + 0.01, dense=True)
    t_close, d_close = sim.closest_approach(approach, dest)[0]
    if video:
        flight.travel.look_in_direction_of_planet(dest)
        flight.travel.start_video()
    t, r, v = flight.coast_to(t_close, 10)
    dv_inj = planner.injection_boost(ctx, t, r, v)
    flight.boost(dv_inj, "injection")
    print(f"C     closest approach at t = {t:.6f} yr, {d_close * KM:,.0f} km from planet {dest} "
          f"(planned {plan.r_orbit * KM:,.0f} km); injection {np.linalg.norm(dv_inj) * AU_PER_YR:,.1f} m/s")
    return mission, flight, video


# ===================================================================== in orbit (D)
def relative_state(ctx, t, r, v):
    pos, vel = kepler_state(ctx.system, np.atleast_1d(t))
    return (np.atleast_2d(r) - pos[:, ctx.destination].T), (np.atleast_2d(v) - vel[:, ctx.destination].T)


def check_orbit(ctx, flight, video):
    """Orbital elements right after injection, over N_ORBITS_CHECK orbits, and after a long coast."""
    GM = ctx.GM_dest
    t0, r0, v0 = flight.orient()
    first = orbital_elements(*relative_state(ctx, t0, r0, v0), GM)
    period = first["P"][0]
    start = len(flight.samples)
    flight.coast_to(t0 + N_ORBITS_CHECK * period, 40 * N_ORBITS_CHECK)
    if video:
        quiet(flight.travel.finish_video, filename="journey_arrival.xml", number_of_frames=1000)
        copy_video("journey_arrival.xml")
    orbit_samples = flight.samples[start:]
    t1 = flight.coast_to(t0 + LONG_COAST, 200)[0]
    long_samples = flight.samples[start + len(orbit_samples):]
    return first, orbit_samples, long_samples


def copy_video(name):
    path = os.path.join(XML_DIR, name)
    if os.path.exists(path) and os.path.isdir(MCAST_DATA_DIR):
        shutil.copy(path, MCAST_DATA_DIR)
        print(f"      MCAst video {name} copied into {MCAST_DATA_DIR}")


# ===================================================================== figures
def samples_array(samples):
    t = np.array([s[0] for s in samples])
    r = np.array([s[1] for s in samples])
    v = np.array([s[2] for s in samples])
    return t, r, v


def plot_flight(ctx, plan, flight):
    """Actual vs. planned: the trip, the deviation from the plan, and the boosts."""
    t, r, v = samples_array([s for s in flight.samples if s[0] <= plan.t_arrival])
    _, r0, v0, _ = launch_state(ctx.launch, ctx.home, ctx.system, plan.t_launch, plan.launch_angle)
    planned = ctx.sim.simulate(plan.t_boost, r0, v0 + plan.dv_boost, plan.t_arrival, dense=True)
    planned_pos, _ = planned.state(t)
    deviation = np.linalg.norm(r - planned_pos[0].T, axis=1) * KM

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 400)
    for p in (0, ctx.destination):
        a = ctx.system.semi_major_axes[p]
        ax.plot(a * np.cos(theta), a * np.sin(theta), color=PLANET_COLORS[p], lw=1, ls=":")
    ax.plot(planned.positions[0, 0], planned.positions[0, 1], color=MUTED, lw=3, alpha=0.5, label="plan")
    ax.plot(r[:, 0], r[:, 1], ".", color=INK, ms=3, label="actual (orientations)")
    for tb, dv, name in flight.boosts:
        i = np.argmin(np.abs(t - tb))
        ax.plot(r[i, 0], r[i, 1], "o", color=ORANGE if "correction" in name else BLUE, ms=8, mec=INK, mew=0.5)
    ax.plot(0, 0, "*", color="#eda100", ms=16, mec=INK, mew=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title("The flight (dots: boosts)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    days = (t - plan.t_launch) * 365.25
    ax.semilogy(days, np.maximum(deviation, 1e-3), color=INK, lw=1.5)
    for tb, dv, name in flight.boosts:
        if "correction" in name:
            ax.axvline((tb - plan.t_launch) * 365.25, color=ORANGE, lw=1, ls=":")
    ax.set_xlabel("days since launch")
    ax.set_ylabel("distance from the planned path [km]")
    ax.set_title("Actual vs. planned position (dotted: course correction);\n"
                 "the corrections steer back so we arrive where the plan arrives", fontsize=10)

    ax = axes[2]
    names = [name for _, _, name in flight.boosts]
    sizes = [np.linalg.norm(dv) * AU_PER_YR for _, dv, _ in flight.boosts]
    colors = [BLUE if n == "departure" else VIOLET if n == "injection" else ORANGE for n in names]
    ax.bar(range(len(sizes)), sizes, color=colors)
    ax.set_yscale("log")
    for i, s in enumerate(sizes):
        ax.annotate(f"{s:.3g} m/s", (i, s), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("delta-v [m/s]")
    ax.set_title(f"All boosts; fuel left {flight.fuel:,.0f} kg", fontsize=10)
    fig.suptitle("C  The real flight with ast2000tools: launch, departure, course corrections, injection", fontsize=12)
    save(fig, "c1_flight")


def predicted_orbit(ctx, t0, r0, v0, duration=LONG_COAST, n_out=400):
    """Our own simulation of the orbit from the state right after injection (for comparison with the API)."""
    sim = make_simulator(ctx.system, t0 - 0.01, t0 + duration + 0.01)
    traj = sim.simulate(t0, r0, v0, t0 + duration, n_out=n_out)
    r_rel, v_rel = relative_state(ctx, traj.times, traj.positions[0].T, traj.velocities[0].T)
    return traj.times, orbital_elements(r_rel, v_rel, ctx.GM_dest)


def plot_orbit(ctx, first, orbit_samples, long_samples):
    GM, R = ctx.GM_dest, ctx.system.radii[ctx.destination]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    t, r, v = samples_array(orbit_samples)
    r_rel, v_rel = relative_state(ctx, t, r, v)
    elements = orbital_elements(r_rel, v_rel, GM)
    ax = axes[0]
    ax.plot(r_rel[:, 0] * KM / R, r_rel[:, 1] * KM / R, ".-", color=VIOLET, ms=2, lw=0.8,
            label=f"{N_ORBITS_CHECK} orbits after injection")
    tl, rl, vl = samples_array(long_samples)
    rl_rel, vl_rel = relative_state(ctx, tl, rl, vl)
    ax.plot(rl_rel[:, 0] * KM / R, rl_rel[:, 1] * KM / R, ".", color=ORANGE, ms=3,
            label=f"orientations over {LONG_COAST} yr")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(np.cos(theta), np.sin(theta), color=PLANET_COLORS[ctx.destination], alpha=0.6, lw=0)
    ax.set_aspect("equal")
    ax.set_xlabel("x relative to the planet [planet radii]")
    ax.set_ylabel("y [planet radii]")
    ax.set_title(f"In orbit around planet {ctx.destination}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    hours = (t - t[0]) * const.yr / 3600
    for key, color, label in (("a", BLUE, "semi-major axis a"), ("periapsis", AQUA, "periapsis"),
                              ("apoapsis", ORANGE, "apoapsis"), ("r", INK, "distance r")):
        ax.plot(hours, elements[key] * KM, color=color, lw=1.4 if key != "r" else 0.8, label=label)
    ax.set_xlabel("hours after injection")
    ax.set_ylabel("[km]")
    ax.set_title(f"Elements from each orientation: e = {elements['e'].mean():.2e}, "
                 f"P = {elements['P'].mean() * const.yr / 3600:.2f} h", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    long_elements = orbital_elements(rl_rel, vl_rel, GM)
    days = (tl - t[0]) * 365.25
    t_ours, ours = predicted_orbit(ctx, t[0], r[0], v[0])
    days_ours = (t_ours - t[0]) * 365.25
    ax.plot(days, long_elements["a"] * KM, color=BLUE, lw=1.6, label="a, API (RK45)")
    ax.plot(days_ours, ours["a"] * KM, color=BLUE, lw=1.6, ls="--", label="a, our simulator (DOP853)")
    ax.fill_between(days, long_elements["periapsis"] * KM, long_elements["apoapsis"] * KM, color=ORANGE, alpha=0.3,
                    lw=0, label="periapsis-apoapsis, API")
    ax.fill_between(days_ours, ours["periapsis"] * KM, ours["apoapsis"] * KM, color=AQUA, alpha=0.3, lw=0,
                    label="periapsis-apoapsis, ours")
    ax.set_xlabel("days after injection")
    ax.set_ylabel("[km]")
    ax.set_title(f"{LONG_COAST} yr later: still in orbit. Our simulator keeps a constant;\n"
                 f"the API's orbit shrinks {1 - long_elements['a'][-1] / ours['a'][0]:.1%}: energy lost by its "
                 "non-symplectic RK45", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("D  The orbit around the destination", fontsize=12)
    save(fig, "d_orbit")
    return elements, long_elements


def main(ctx=None, plan=None, video=True):
    if plan is None:
        ctx, plan = planner.main(ctx)
    mission, flight, video = fly(ctx, plan, video)
    first, orbit_samples, long_samples = check_orbit(ctx, flight, video)
    print("D     right after injection (one orientation, two-body formulas):")
    for key, unit, scale in (("r", "km", KM), ("v_r", "m/s", AU_PER_YR), ("v_theta", "m/s", AU_PER_YR),
                             ("a", "km", KM), ("b", "km", KM), ("e", "", 1), ("P", "h", const.yr / 3600),
                             ("periapsis", "km", KM), ("apoapsis", "km", KM)):
        print(f"      {key:>9} = {first[key][0] * scale:14.6g} {unit}")
    plot_flight(ctx, plan, flight)
    elements, long_elements = plot_orbit(ctx, first, orbit_samples, long_samples)
    print(f"D     over {N_ORBITS_CHECK} orbits: a varies {np.ptp(elements['a']) / elements['a'].mean():.1e} "
          f"(relative), e between {elements['e'].min():.1e} and {elements['e'].max():.1e}")
    print(f"      after {LONG_COAST} yr: periapsis {long_elements['periapsis'][-1] * KM:,.0f} km, apoapsis "
          f"{long_elements['apoapsis'][-1] * KM:,.0f} km: still in orbit")
    flight.travel.record_destination(ctx.destination)
    quiet(SpaceMission.save, JOURNEY_MISSION_FILE, mission)
    print(f"      fuel left {flight.fuel:,.0f} kg; destination recorded; mission saved to "
          f"{os.path.basename(JOURNEY_MISSION_FILE)} for Part 6")
    return mission, flight


if __name__ == "__main__":
    main()
