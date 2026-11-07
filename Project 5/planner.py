"""
Part 5, challenge B: planning the journey to the destination planet.

1. Porkchop plot. For every (launch date, travel time) on a grid, Lambert's
   problem (lambert.py) gives the transfer orbit around the star. Its velocity
   relative to each planet at the ends, v_inf, sets the burns:
     * Departure: right after launch the rocket has exactly escape speed
       (that is how Part 1 ends the launch). Adding speed deep in the gravity
       well is cheap (the Oberth effect): to leave with v_inf we only need
           dv_dep = sqrt(v_inf^2 + v_esc^2) - v_esc.
     * Arrival: falling in from v_inf to a circular orbit of radius r costs
           dv_arr = sqrt(v_inf^2 + 2 G M / r) - sqrt(G M / r),
       which is smallest for r = 2 G M / v_inf^2 (then dv_arr = v_inf / sqrt 2).
       So we also choose the orbit radius, not just the route.
   The best point on the grid is our launch date and travel time.

2. Launch site. The launch goes straight up from the launch site (Part 3).
   Choosing the site whose "up" points along the departure direction means the
   rocket leaves the planet on a straight line: the planet's gravity slows it
   down but cannot bend it. The departure boost right after launch then only
   needs to add speed.

3. Shooting. The patched-conic numbers are a first guess. The boost is refined
   in the full N-body simulation with Newton's method: the miss vector at the
   arrival time is linear in the boost for small changes, and the 2 x 2
   Jacobian comes from one batched run of three spacecraft.

4. Arrival: at the closest approach the injection boost
       dv_inj = v_planet + v_circ e_theta - v   (planet frame: circular orbit)
   puts the spacecraft in a circular orbit.

Run:  ../.venv/bin/python planner.py
Figures go to figures/b*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(1, 2, 3)

import io
from contextlib import redirect_stdout
from dataclasses import dataclass, field

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np

from common.config import get_mission
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, VIOLET, save
from common.report import record, record_many
from destination import capture_distance, get_destination, hohmann
from generalised_launch import design, launch_state
from lambert import lambert
from launch import HomePlanet
from orbits import G, kepler_state
from trajectory import KM, make_simulator

AU_PER_YR = const.AU / const.yr  # [m/s] per [AU/yr]
LAUNCH_WINDOW = (0.0, 2.5)  # Launch dates searched [yr]
TRAVEL_TIMES = (0.1, 0.9)  # Travel times searched [yr]
GRID = 300  # Porkchop resolution per axis
AIM_TOLERANCE = 1.0 / KM  # Newton stops when the miss is below 1 km [AU]
MIN_ORBIT_RADII = 2.0  # Don't orbit lower than 2 planet radii (atmosphere, Part 6)


# ===================================================================== context
@dataclass
class Context:
    """Everything about the mission that the planner and the flight need."""
    mission: object
    system: object
    home: object
    engine: dict
    launch: dict
    destination: int
    exhaust_speed: float  # [m/s]
    fuel_after_launch: float  # [kg]
    sim: object = None

    @property
    def GM_dest(self):
        return G * self.system.masses[self.destination]

    @property
    def radius_dest(self):
        return self.system.radii[self.destination] / KM  # [AU]

    def escape_speed_after_launch(self):
        """|v| relative to the home planet when the launch ends [AU/yr] (exactly the escape speed there)."""
        return np.sqrt(2 * G * self.system.masses[0] / (self.launch["r"] / const.AU))

    def fuel_for(self, dv_list):
        """Fuel [kg] used by each boost in turn (rocket equation with the engine's exhaust speed)."""
        mass = self.mission.spacecraft_mass + self.fuel_after_launch
        used = []
        for dv in dv_list:
            burn = mass * (1 - np.exp(-np.linalg.norm(dv) * AU_PER_YR / self.exhaust_speed))
            used.append(burn)
            mass -= burn
        return np.array(used)


def make_context(mission=None):
    mission = mission or get_mission()
    home = HomePlanet(mission)
    with redirect_stdout(io.StringIO()):
        engine, launch = design(mission, home)
    return Context(mission, mission.system, home, engine, launch, get_destination(mission.system),
                   engine["thrust"] / engine["mass_loss_rate"], launch["fuel_left"])


# ===================================================================== porkchop
def porkchop(ctx, launch_window=LAUNCH_WINDOW, travel_times=TRAVEL_TIMES, n=GRID):
    """Departure, arrival and total delta-v [AU/yr] on a (launch date, travel time) grid."""
    system, dest = ctx.system, ctx.destination
    t_launch = np.linspace(*launch_window, n)
    t_travel = np.linspace(*travel_times, n)
    TL, TT = np.meshgrid(t_launch, t_travel, indexing="ij")
    pos_home, vel_home = kepler_state(system, t_launch)
    pos_dest, vel_dest = kepler_state(system, (TL + TT).ravel())
    r1 = np.broadcast_to(pos_home[:, 0].T[:, None, :], TL.shape + (2,))
    r2 = pos_dest[:, dest].T.reshape(TL.shape + (2,))
    v1, v2 = lambert(r1, r2, TT, G * system.star_mass)
    v_inf_dep = np.linalg.norm(v1 - vel_home[:, 0].T[:, None, :], axis=-1)
    v_inf_arr = np.linalg.norm(v2 - vel_dest[:, dest].T.reshape(TL.shape + (2,)), axis=-1)
    v_esc = ctx.escape_speed_after_launch()
    dv_dep = np.sqrt(v_inf_dep ** 2 + v_esc ** 2) - v_esc
    r_orbit = injection_radius(ctx, v_inf_arr)
    dv_arr = np.sqrt(v_inf_arr ** 2 + 2 * ctx.GM_dest / r_orbit) - np.sqrt(ctx.GM_dest / r_orbit)
    return {"t_launch": t_launch, "t_travel": t_travel, "dv_dep": dv_dep, "dv_arr": dv_arr, "dv": dv_dep + dv_arr,
            "v_inf_dep": v_inf_dep, "v_inf_arr": v_inf_arr, "v1": v1, "v2": v2}


def injection_radius(ctx, v_inf):
    """Cheapest circular-orbit radius 2 G M / v_inf^2, kept between 2 planet radii and half the capture distance."""
    l = capture_distance(ctx.system, ctx.destination, ctx.system.semi_major_axes[ctx.destination])
    return np.clip(2 * ctx.GM_dest / np.asarray(v_inf) ** 2, MIN_ORBIT_RADII * ctx.radius_dest, 0.5 * l)


# ===================================================================== shooting
def aim(sim, t0, r0, v0, dv_guess, t_target, target, max_iterations=12, h=1e-6):
    """
    Boost dv at (t0, r0, v0) such that the spacecraft is at `target` at time t_target (Newton's method).

    Each iteration simulates three spacecraft at once: dv, dv + h x, dv + h y,
    giving the miss and its 2 x 2 Jacobian. Returns dv and the final miss [AU].
    """
    dv = np.array(dv_guess, dtype=float)
    for _ in range(max_iterations):
        batch_dv = dv + np.array([[0, 0], [h, 0], [0, h]])
        traj = sim.simulate(t0, np.tile(r0, (3, 1)), v0 + batch_dv, t_target)
        ends = traj.final_position
        miss = ends[0] - target
        if np.linalg.norm(miss) < AIM_TOLERANCE:
            return dv, np.linalg.norm(miss)
        jacobian = np.column_stack([(ends[1] - ends[0]) / h, (ends[2] - ends[0]) / h])
        step = np.linalg.solve(jacobian, -miss)
        dv = dv + step * min(1.0, 0.2 / max(np.linalg.norm(step), 1e-12))  # Cap steps at 0.2 AU/yr (~1 km/s)
    return dv, np.linalg.norm(miss)


def aim_offset(ctx, v_rel_arrival, impact):
    """Offset from the planet's centre to aim at: `impact` AU to the side that gives a counterclockwise orbit."""
    v_hat = v_rel_arrival / np.linalg.norm(v_rel_arrival)
    return impact * np.array([v_hat[1], -v_hat[0]])


def plan_arrival(ctx, t0, r0, v0, dv_guess, t_arrival, v_rel_arrival, r_orbit, n_outer=6):
    """
    The boost at t0 that brings the spacecraft to a closest approach of r_orbit from the destination.

    Inner loop: aim (Newton) at a point beside the planet at t_arrival. Outer loop:
    adjust how far beside (the planet's gravity pulls the path in) until the
    closest approach equals r_orbit (secant method).
    """
    sim, dest = ctx.sim, ctx.destination
    v_inf = np.linalg.norm(v_rel_arrival)
    impact = r_orbit * np.sqrt(1 + 2 * ctx.GM_dest / (r_orbit * v_inf ** 2))  # Gravitational focusing
    history = []
    dv = dv_guess
    for _ in range(n_outer):
        target = kepler_state(ctx.system, [t_arrival])[0][:, dest, 0] + aim_offset(ctx, v_rel_arrival, impact)
        dv, miss = aim(sim, t0, r0, v0, dv, t_arrival, target)
        traj = sim.simulate(t0, r0, v0 + dv, t_arrival + 0.02, dense=True)
        t_close, d_close = sim.closest_approach(traj, dest)[0]
        history.append((impact, d_close))
        if abs(d_close - r_orbit) < 0.01 * r_orbit:
            break
        if len(history) >= 2:  # Secant step on impact -> closest distance
            (b0, d0), (b1, d1) = history[-2], history[-1]
            impact = b1 + (r_orbit - d1) * (b1 - b0) / (d1 - d0) if d1 != d0 else b1 * r_orbit / d1
        else:
            impact *= r_orbit / d_close
    return dv, t_close, d_close, miss, target


def injection_boost(ctx, t, r, v):
    """Boost that turns the current velocity into a circular orbit around the destination (planet frame)."""
    pos, vel = kepler_state(ctx.system, [t])
    r_rel, v_rel = r - pos[:, ctx.destination, 0], v - vel[:, ctx.destination, 0]
    distance = np.linalg.norm(r_rel)
    e_r = r_rel / distance
    e_theta = np.array([-e_r[1], e_r[0]])
    sign = np.sign(r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0]) or 1.0  # Keep the current sense of rotation
    v_circular = sign * np.sqrt(ctx.GM_dest / distance) * e_theta
    return v_circular - v_rel


# ===================================================================== the plan
@dataclass
class Plan:
    t_launch: float
    launch_angle: float  # Launch-site angle [rad]
    t_boost: float  # Right after launch
    dv_boost: np.ndarray
    t_arrival: float  # Closest approach
    r_orbit: float  # Planned orbit radius [AU]
    t_target: float = None  # Shooting target: be at `target` at time t_target
    target: np.ndarray = None
    dv_injection: np.ndarray = None
    porkchop_choice: dict = field(default_factory=dict)

    def describe(self):
        return (f"launch at t = {self.t_launch:.5f} yr from the site at {np.degrees(self.launch_angle):.2f} deg; "
                f"boost {np.linalg.norm(self.dv_boost) * AU_PER_YR:.1f} m/s right after; arrive at "
                f"t = {self.t_arrival:.5f} yr, {self.r_orbit * KM:,.0f} km from the planet")


def make_plan(ctx, chart):
    """Best porkchop point -> launch site -> shooting in the full simulation."""
    i, j = np.unravel_index(np.nanargmin(chart["dv"]), chart["dv"].shape)
    t_launch, t_travel = chart["t_launch"][i], chart["t_travel"][j]
    _, vel = kepler_state(ctx.system, [t_launch, t_launch + t_travel])
    v_inf_dep_vec = chart["v1"][i, j] - vel[:, 0, 0]
    v_rel_arrival = chart["v2"][i, j] - vel[:, ctx.destination, 1]
    launch_angle = np.arctan2(v_inf_dep_vec[1], v_inf_dep_vec[0]) % (2 * np.pi)  # "Up" along the departure

    _, r0, v0, t_boost = launch_state(ctx.launch, ctx.home, ctx.system, t_launch, launch_angle)
    ctx.sim = make_simulator(ctx.system, t_launch - 0.01, t_launch + t_travel + 0.1)

    # First guess for the boost: relative velocity along the departure direction, at sqrt(v_inf^2 + v_esc^2)
    _, home_vel = kepler_state(ctx.system, [t_boost])
    v_rel = v0 - home_vel[:, 0, 0]
    v_inf = np.linalg.norm(v_inf_dep_vec)
    speed = np.sqrt(v_inf ** 2 + np.linalg.norm(v_rel) ** 2)
    dv_guess = speed * v_inf_dep_vec / v_inf - v_rel

    r_orbit = float(injection_radius(ctx, np.linalg.norm(v_rel_arrival)))
    dv, t_close, d_close, miss, target = plan_arrival(ctx, t_boost, r0, v0, dv_guess, t_launch + t_travel,
                                                      v_rel_arrival, r_orbit)
    plan = Plan(t_launch, launch_angle, t_boost, dv, t_close, d_close, t_launch + t_travel, target,
                porkchop_choice={"t_travel": t_travel, "dv_dep": chart["dv_dep"][i, j], "dv_arr": chart["dv_arr"][i, j],
                                 "v_inf_dep": v_inf, "v_inf_arr": np.linalg.norm(v_rel_arrival), "dv_guess": dv_guess})
    traj = ctx.sim.simulate(t_boost, r0, v0 + dv, t_close, dense=True)
    plan.dv_injection = injection_boost(ctx, t_close, traj.final_position[0], traj.final_velocity[0])
    return plan


def simulate_plan(ctx, plan, orbit_time=0.01, n_out=6000):
    """The whole planned journey: launch end -> boost -> arrival -> injection -> a while in orbit."""
    _, r0, v0, _ = launch_state(ctx.launch, ctx.home, ctx.system, plan.t_launch, plan.launch_angle)
    cruise = ctx.sim.simulate(plan.t_boost, r0, v0 + plan.dv_boost, plan.t_arrival, n_out=n_out, dense=True)
    r_arr, v_arr = cruise.final_position[0], cruise.final_velocity[0]
    orbit = ctx.sim.simulate(plan.t_arrival, r_arr, v_arr + plan.dv_injection, plan.t_arrival + orbit_time, n_out=2000)
    coast = ctx.sim.simulate(plan.t_boost, r0, v0, plan.t_boost + 0.02, n_out=2000)  # Without the boost
    return cruise, orbit, coast, (r0, v0)


# ===================================================================== figures
def plot_porkchop(ctx, chart, plan):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.4, 1]})
    dv = chart["dv"] * AU_PER_YR / 1e3
    levels = np.linspace(np.nanmin(dv), np.nanmin(dv) + 15, 31)
    cs = ax1.contourf(chart["t_launch"], chart["t_travel"] * 365.25, dv.T, levels=levels, cmap="viridis_r",
                      extend="max")
    ax1.contour(chart["t_launch"], chart["t_travel"] * 365.25, dv.T, levels=levels[::5], colors="white",
                linewidths=0.5)
    fig.colorbar(cs, ax=ax1, label="total delta-v (departure + capture) [km/s]")
    ax1.plot(plan.t_launch, plan.porkchop_choice["t_travel"] * 365.25, "*", color="white", ms=18, mec=INK)
    ax1.annotate(f"chosen: launch t = {plan.t_launch:.3f} yr\n{plan.porkchop_choice['t_travel'] * 365.25:.0f} days, "
                 f"{np.nanmin(dv):.2f} km/s", (plan.t_launch, plan.porkchop_choice["t_travel"] * 365.25),
                 xytext=(12, -30), textcoords="offset points", color="white", fontsize=9)
    ax1.set_xlabel("launch date [yr]")
    ax1.set_ylabel("travel time [days]")
    ax1.set_title(f"Porkchop plot: cheapest trips to planet {ctx.destination} (one window per synodic period)",
                  fontsize=10)
    ax1.grid(False)

    # Along the best travel time for each launch date: departure vs. capture cost
    best = np.nanargmin(chart["dv"], axis=1)
    rows = np.arange(len(best))
    for key, color, label in (("dv_dep", BLUE, "departure burn"), ("dv_arr", ORANGE, "capture burn"),
                              ("dv", INK, "total")):
        ax2.plot(chart["t_launch"], chart[key][rows, best] * AU_PER_YR / 1e3, color=color, lw=1.8, label=label)
    ax2.axvline(plan.t_launch, color=MUTED, ls=":", lw=1)
    ax2.set_xlabel("launch date [yr]")
    ax2.set_ylabel("delta-v with the best travel time [km/s]")
    ax2.set_title("Launch windows: the cost dips every synodic period", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle("B  When to go: Lambert's problem on a 300 x 300 grid", fontsize=12)
    save(fig, "b1_porkchop")


def plot_journey(ctx, plan, cruise, orbit):
    system = ctx.system
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
    t = cruise.times
    planets = kepler_state(system, t)[0]
    theta = np.linspace(0, 2 * np.pi, 400)
    for p in (0, ctx.destination):
        a = system.semi_major_axes[p]
        ax1.plot(a * np.cos(theta), a * np.sin(theta), color=PLANET_COLORS[p], lw=0.8, ls=":")
        ax1.plot(planets[0, p], planets[1, p], color=PLANET_COLORS[p], lw=2, alpha=0.6, label=f"planet {p} during the trip")
        ax1.plot(planets[0, p, 0], planets[1, p, 0], "o", color=PLANET_COLORS[p], ms=9, mec=INK, mew=0.5)
        ax1.plot(planets[0, p, -1], planets[1, p, -1], "s", color=PLANET_COLORS[p], ms=9, mec=INK, mew=0.5)
    ax1.plot(cruise.positions[0, 0], cruise.positions[0, 1], color=INK, lw=1.6, label="spacecraft")
    ax1.plot(0, 0, "*", color="#eda100", ms=16, mec=INK, mew=0.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [AU]")
    ax1.set_ylabel("y [AU]")
    ax1.set_title(f"The planned trip: {plan.t_arrival - plan.t_launch:.3f} yr "
                  f"({(plan.t_arrival - plan.t_launch) * 365.25:.0f} days); circles: launch, squares: arrival", fontsize=10)
    ax1.legend(fontsize=8, loc="lower left")

    d = ctx.sim.distance_to_planet(cruise, ctx.destination)[0] * KM
    l = capture_distance(system, ctx.destination, np.linalg.norm(cruise.positions[0], axis=0)) * KM
    ax2.semilogy((t - plan.t_launch) * 365.25, d, color=INK, lw=1.6, label=f"distance to planet {ctx.destination}")
    ax2.semilogy((t - plan.t_launch) * 365.25, l, color=AQUA, lw=1.4, ls="--",
                 label=r"capture distance $l = |r|\sqrt{M_p/10M_\star}$")
    ax2.axhline(system.radii[ctx.destination], color=PLANET_COLORS[ctx.destination], lw=1.2, ls=":",
                label="planet radius")
    ax2.axhline(plan.r_orbit * KM, color=VIOLET, lw=1.2, label=f"planned orbit: {plan.r_orbit * KM:,.0f} km")
    ax2.set_xlabel("days since launch")
    ax2.set_ylabel("distance [km]")
    ax2.set_title("Closing in: inside the capture distance only at the very end", fontsize=10)
    ax2.legend(fontsize=8, loc="lower left")
    fig.suptitle("B  The planned journey (full N-body simulation)", fontsize=12)
    save(fig, "b2_planned_journey")


def plot_departure(ctx, plan, cruise, coast, start):
    """Zoom on the first hours: why a boost right after launch is needed."""
    e_along = np.array([np.cos(plan.launch_angle), np.sin(plan.launch_angle)])
    e_side = np.array([-e_along[1], e_along[0]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    for traj, color, label in ((coast, ORANGE, "no boost (launch only)"), (cruise, BLUE, "with the departure boost")):
        t = traj.times
        home = kepler_state(ctx.system, t)
        rel = traj.positions[0] - home[0][:, 0]
        vrel = traj.velocities[0] - home[1][:, 0]
        hours = (t - plan.t_boost) * const.yr / 3600
        window = hours <= 48
        ax1.plot(rel[:, window].T @ e_along * KM / 1e3, rel[:, window].T @ e_side * KM, color=color, lw=1.8,
                 label=label)
        ax2.plot(hours[window], np.linalg.norm(vrel[:, window], axis=0) * AU_PER_YR / 1e3, color=color, lw=1.8,
                 label=label)
    ax1.set_xlabel("distance along the launch direction [1000 km]")
    ax1.set_ylabel("sideways drift [km]")
    ax1.set_title("First 48 hours, home planet frame: ~700 000 km out, only a few thousand km sideways\n"
                  "(launching 'up' along the departure: gravity slows the rocket but hardly bends it)", fontsize=10)
    ax1.legend(fontsize=8)
    v_inf = plan.porkchop_choice["v_inf_dep"] * AU_PER_YR / 1e3
    ax2.axhline(v_inf, color=INK, ls="--", lw=1, label=f"planned v_inf = {v_inf:.2f} km/s")
    ax2.set_xlabel("hours after launch")
    ax2.set_ylabel("speed relative to the home planet [km/s]")
    ax2.set_title("Escape speed alone leaves us with ~0 speed far away: the boost is needed at once", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle(f"B  Departure: a {np.linalg.norm(plan.dv_boost) * AU_PER_YR:.0f} m/s boost right after launch "
                 f"(Oberth: cheaper than the {v_inf * 1e3:.0f} m/s needed far away)", fontsize=12)
    save(fig, "b3_departure")


def plot_arrival(ctx, plan, cruise, orbit):
    """Destination frame: the approach and the circular orbit after injection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    dest = ctx.destination
    R = ctx.system.radii[dest]
    for ax, span in ((ax1, 40), (ax2, 4)):
        for traj, color, label in ((cruise, INK, "approach"), (orbit, VIOLET, "after injection")):
            planet = kepler_state(ctx.system, traj.times)[0][:, dest]
            rel = (traj.positions[0] - planet) * KM / R
            keep = np.linalg.norm(rel, axis=0) < span * 1.5
            ax.plot(rel[0, keep], rel[1, keep], color=color, lw=1.6, label=label)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.fill(np.cos(theta), np.sin(theta), color=PLANET_COLORS[dest], alpha=0.6, lw=0)
        l = capture_distance(ctx.system, dest, ctx.system.semi_major_axes[dest]) * KM / R
        if span > 10:
            ax.plot(l * np.cos(theta), l * np.sin(theta), color=AQUA, ls="--", lw=1.2, label="capture distance l")
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_xlabel("x relative to the planet [planet radii]")
        ax.set_ylabel("y [planet radii]")
        ax.legend(fontsize=8, loc="upper left")
    ax1.set_title(f"Approach to planet {dest}", fontsize=10)
    ax2.set_title(f"Zoom: circular orbit at {plan.r_orbit * KM:,.0f} km ({plan.r_orbit * KM / R:.1f} radii)", fontsize=10)
    fig.suptitle(f"B  Arrival: injection boost {np.linalg.norm(plan.dv_injection) * AU_PER_YR:.0f} m/s at the closest "
                 "approach", fontsize=12)
    save(fig, "b4_arrival")


def plot_budget(ctx, plan):
    boosts = [("departure", plan.dv_boost), ("capture", plan.dv_injection)]
    fuel = ctx.fuel_for([dv for _, dv in boosts])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    names = [name for name, _ in boosts]
    dvs = [np.linalg.norm(dv) * AU_PER_YR for _, dv in boosts]
    ax1.bar(names, dvs, color=[BLUE, ORANGE], width=0.5)
    for i, v in enumerate(dvs):
        ax1.annotate(f"{v:,.0f} m/s", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax1.set_ylabel("delta-v [m/s]")
    ax1.set_title("Planned boosts", fontsize=10)
    left = ctx.fuel_after_launch - np.cumsum(fuel)
    ax2.bar(["after launch"] + [f"after {n}" for n in names], np.r_[ctx.fuel_after_launch, left],
            color=[MUTED, BLUE, ORANGE], width=0.5)
    for i, v in enumerate(np.r_[ctx.fuel_after_launch, left]):
        ax2.annotate(f"{v:,.0f} kg", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax2.set_ylabel("fuel in the tank [kg]")
    ax2.set_title(f"Fuel: {left[-1]:,.0f} kg left for course corrections and landing", fontsize=10)
    fig.suptitle(f"B  Budget (exhaust speed {ctx.exhaust_speed:,.0f} m/s, spacecraft {ctx.mission.spacecraft_mass:.0f} kg)",
                 fontsize=12)
    save(fig, "b5_budget")
    return fuel


def plot_trades(ctx, plan):
    """
    Two choices the planner makes, and why. (1) Capture: the burn to circularise
    at radius r after falling in from v_inf is sqrt(v_inf^2 + 2GM/r) - sqrt(GM/r),
    smallest at r = 2GM/v_inf^2. (2) Departure: burning right after launch, at
    escape speed deep in the home planet's well, against burning far away.
    """
    choice = plan.porkchop_choice
    v_inf_arr = choice["v_inf_arr"] * AU_PER_YR
    GM = const.G * ctx.system.masses[ctx.destination] * const.m_sun
    R = ctx.system.radii[ctx.destination] * 1e3
    r = np.logspace(np.log10(R), np.log10(capture_distance(ctx.system, ctx.destination,
                                                           ctx.system.semi_major_axes[ctx.destination]) * const.AU), 400)
    dv_capture = np.sqrt(v_inf_arr ** 2 + 2 * GM / r) - np.sqrt(GM / r)
    r_best = 2 * GM / v_inf_arr ** 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    ax1.semilogx(r / R, dv_capture, color=INK, label=f"falling in at $v_\\infty$ = {v_inf_arr / 1e3:.2f} km/s")
    ax1.axvline(r_best / R, color=MUTED, ls=":", lw=1.2, label=f"optimum $2GM/v_\\infty^2$ = {r_best / R:.1f} radii")
    ax1.axvline(MIN_ORBIT_RADII, color=ORANGE, ls="--", lw=1.2, label=f"lowest allowed ({MIN_ORBIT_RADII:.0f} radii)")
    ax1.plot(plan.r_orbit * KM * 1e3 / R, np.linalg.norm(plan.dv_injection) * AU_PER_YR, "*", color=VIOLET, ms=16,
             mec=INK, label=f"our orbit: {plan.r_orbit * KM:,.0f} km")
    ax1.set_xlabel("circular orbit radius [planet radii]")
    ax1.set_ylabel("capture burn [m/s]")
    ax1.set_title("Capture: a low orbit costs more, and so does a high one", fontsize=10)
    ax1.legend(fontsize=8)

    v_inf = np.linspace(0, 8e3, 300)
    v_esc = ctx.escape_speed_after_launch() * AU_PER_YR
    v_esc_surface = ctx.home.v_esc
    ax2.plot(v_inf / 1e3, v_inf / 1e3, color=MUTED, ls="--", label="burn far from the planet: $\\Delta v = v_\\infty$")
    ax2.plot(v_inf / 1e3, (np.sqrt(v_inf ** 2 + v_esc ** 2) - v_esc) / 1e3, color=BLUE,
             label=f"burn right after launch (at {v_esc / 1e3:.2f} km/s)")
    ax2.plot(v_inf / 1e3, (np.sqrt(v_inf ** 2 + v_esc_surface ** 2) - v_esc_surface) / 1e3, color=AQUA, lw=1.2,
             label="ideal: at the surface")
    v_dep = choice["v_inf_dep"] * AU_PER_YR
    ax2.plot(v_dep / 1e3, np.linalg.norm(plan.dv_boost) * AU_PER_YR / 1e3, "*", color=VIOLET, ms=16, mec=INK,
             label=f"our departure: {np.linalg.norm(plan.dv_boost) * AU_PER_YR:,.0f} m/s for {v_dep / 1e3:.2f} km/s")
    ax2.set_xlabel("$v_\\infty$ needed far from the home planet [km/s]")
    ax2.set_ylabel("departure burn [km/s]")
    ax2.set_title(f"Oberth effect: the burn after launch saves {(v_dep - np.linalg.norm(plan.dv_boost) * AU_PER_YR) / 1e3:.1f} km/s",
                  fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle("B  Two trades: where to orbit, and where to burn", fontsize=12)
    save(fig, "b7_trades")
    return r_best


def sensitivity(ctx, plan, n_points=25, h=1e-6):
    """How far the arrival point moves per m/s of velocity error, for errors at different times on the way."""
    _, r0, v0, _ = launch_state(ctx.launch, ctx.home, ctx.system, plan.t_launch, plan.launch_angle)
    cruise = ctx.sim.simulate(plan.t_boost, r0, v0 + plan.dv_boost, plan.t_target, dense=True)
    times = plan.t_boost + (plan.t_target - plan.t_boost) * (1 - np.geomspace(1, 1e-3, n_points))
    km_per_ms = []
    for t in times:
        r, v = cruise.state(t)
        batch_v = v[0] + np.array([[0, 0], [h, 0], [0, h]])
        ends = ctx.sim.simulate(t, np.tile(r[0], (3, 1)), batch_v, plan.t_target).final_position
        jacobian = np.column_stack([ends[1] - ends[0], ends[2] - ends[0]]) / h  # [AU per AU/yr]
        km_per_ms.append(np.linalg.norm(jacobian, 2) * KM / AU_PER_YR)  # Largest miss per unit error
    return times, np.array(km_per_ms)


def plot_sensitivity(ctx, plan, times, km_per_ms):
    days = (times - plan.t_launch) * 365.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(days, km_per_ms, "o-", color=BLUE, ms=5)
    l = capture_distance(ctx.system, ctx.destination, ctx.system.semi_major_axes[ctx.destination]) * KM
    ax.axhline(l, color=AQUA, ls="--", lw=1.2, label=f"capture distance l = {l:,.0f} km")
    ax.axhline(plan.r_orbit * KM, color=VIOLET, lw=1.2, label=f"planned orbit radius {plan.r_orbit * KM:,.0f} km")
    ax.set_xlabel("day of the trip when the velocity error happens")
    ax.set_ylabel("miss at arrival per 1 m/s error [km]")
    ax.set_title(f"A 1 m/s error right after launch misses by {km_per_ms[0]:,.0f} km, so the departure boost must be "
                 f"exact. The flip side: fixing a 1000 km miss\ncosts {1000 / km_per_ms[0] * 100:.2f} cm/s at the start but "
                 f"{1000 / km_per_ms[-1]:.0f} m/s a day before arrival, so correct early", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("B  Sensitivity of the trip to velocity errors (Jacobian from batched runs)", fontsize=12)
    save(fig, "b6_sensitivity")


def main(ctx=None):
    ctx = ctx or make_context()
    print(f"B     destination: planet {ctx.destination}; fuel after launch {ctx.fuel_after_launch:,.0f} kg, "
          f"exhaust speed {ctx.exhaust_speed:,.0f} m/s")
    chart = porkchop(ctx)
    print(f"      porkchop: {GRID}x{GRID} Lambert transfers; cheapest total delta-v "
          f"{np.nanmin(chart['dv']) * AU_PER_YR / 1e3:.2f} km/s")
    plan = make_plan(ctx, chart)
    choice = plan.porkchop_choice
    print(f"      patched conics: v_inf {choice['v_inf_dep'] * AU_PER_YR / 1e3:.2f} km/s out, "
          f"{choice['v_inf_arr'] * AU_PER_YR / 1e3:.2f} km/s in; burns {choice['dv_dep'] * AU_PER_YR:.0f} + "
          f"{choice['dv_arr'] * AU_PER_YR:.0f} m/s")
    print(f"B     plan: {plan.describe()}")
    print(f"      first guess for the boost was off by "
          f"{np.linalg.norm(plan.dv_boost - choice['dv_guess']) * AU_PER_YR:.1f} m/s (fixed by shooting)")
    cruise, orbit, coast, start = simulate_plan(ctx, plan)
    print(f"      injection {np.linalg.norm(plan.dv_injection) * AU_PER_YR:.0f} m/s at t = {plan.t_arrival:.5f} yr")
    plot_porkchop(ctx, chart, plan)
    plot_journey(ctx, plan, cruise, orbit)
    plot_departure(ctx, plan, cruise, coast, start)
    plot_arrival(ctx, plan, cruise, orbit)
    fuel = plot_budget(ctx, plan)
    times, km_per_ms = sensitivity(ctx, plan)
    plot_sensitivity(ctx, plan, times, km_per_ms)
    print(f"      sensitivity: 1 m/s error right after launch -> {km_per_ms[0]:,.0f} km miss; "
          f"halfway -> {km_per_ms[len(km_per_ms) // 2]:,.0f} km; a day before arrival -> {km_per_ms[-1]:,.0f} km")
    print(f"      fuel: {fuel[0]:,.0f} kg + {fuel[1]:,.0f} kg -> {ctx.fuel_after_launch - fuel.sum():,.0f} kg left")
    r_best = plot_trades(ctx, plan)
    v = lambda x: np.linalg.norm(x) * AU_PER_YR
    record_many({
        "dest": (ctx.destination, "{:d}"), "u": (ctx.exhaust_speed, "{:,.0f}"),
        "fuel-after-launch": (ctx.fuel_after_launch, "{:,.0f}"), "grid": (GRID, "{:d}"),
        "window-lo": (LAUNCH_WINDOW[0], "{:g}"), "window-hi": (LAUNCH_WINDOW[1], "{:g}"),
        "porkchop-min": (np.nanmin(chart["dv"]) * AU_PER_YR / 1e3, "{:.2f}"),
        "t-launch": (plan.t_launch, "{:.4f}"), "travel-days": (choice["t_travel"] * 365.25, "{:.0f}"),
        "launch-angle": (np.degrees(plan.launch_angle), "{:.1f}"),
        "vinf-dep": (choice["v_inf_dep"] * AU_PER_YR / 1e3, "{:.2f}"),
        "vinf-arr": (choice["v_inf_arr"] * AU_PER_YR / 1e3, "{:.2f}"),
        "pc-dv-dep": (choice["dv_dep"] * AU_PER_YR, "{:,.0f}"), "pc-dv-arr": (choice["dv_arr"] * AU_PER_YR, "{:,.0f}"),
        "plan-boost": (v(plan.dv_boost), "{:,.1f}"), "plan-inj": (v(plan.dv_injection), "{:,.0f}"),
        "plan-total": ((v(plan.dv_boost) + v(plan.dv_injection)) / 1e3, "{:.2f}"),
        "guess-off": (v(plan.dv_boost - choice["dv_guess"]), "{:.1f}"),
        "r-orbit": (plan.r_orbit * KM, "{:,.0f}"), "r-orbit-radii": (plan.r_orbit * KM / ctx.system.radii[ctx.destination], "{:.1f}"),
        "r-best-radii": (r_best / (ctx.system.radii[ctx.destination] * 1e3), "{:.1f}"),
        "plan-fuel-dep": (fuel[0], "{:,.0f}"), "plan-fuel-inj": (fuel[1], "{:,.0f}"),
        "plan-fuel-left": (ctx.fuel_after_launch - fuel.sum(), "{:,.0f}"),
        "sens-start": (km_per_ms[0], "{:,.0f}"), "sens-mid": (km_per_ms[len(km_per_ms) // 2], "{:,.0f}"),
        "sens-end": (km_per_ms[-1], "{:,.0f}"),
        "arrival-days": ((plan.t_arrival - plan.t_launch) * 365.25, "{:.1f}"),
        "hohmann-dv": (sum(hohmann(G * ctx.system.star_mass, ctx.system.semi_major_axes[0],
                                   ctx.system.semi_major_axes[ctx.destination])[:2]) * AU_PER_YR / 1e3, "{:.2f}"),
    })
    return ctx, plan


if __name__ == "__main__":
    main()
