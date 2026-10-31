"""
Part 3, challenge 3: launch at any time, from any point on the equator.

The launch itself, seen from the planet, is the same whenever and wherever we
launch: radially outwards, same engine, same 6 minutes (Part 1, challenge E).
Only the step to the star's frame changes (Part 1, challenge F):

    position after = planet position at t0 + dt              <- exact orbit (Part 2)
                   + v_rot dt e_t + r_final e_r              <- from the planet's centre
    velocity after = planet velocity at t0 + dt + v_rot e_t + v_r e_r

with e_r = (cos phi, sin phi) the launch direction at the launch-site angle phi,
and e_t = (-sin phi, cos phi) the rotation direction.

Clever bits:
  * The planet's position at an arbitrary time comes from solving Kepler's
    equation (Part 2), not from interpolating our leapfrog orbits. Those drift
    ~400 km from the truth for the home planet over 11 years. The API allows
    only 1.5% of the planet radius (~99 km), so an "accurate" simulation would
    fail at late launch times while Kepler's solution is exact at any time.
  * Part 1 moved the planet in a straight line during the launch. Here it moves
    along its real (curved) orbit; the difference is ~1 km for a 6-minute launch.
  * We check many (time, angle) pairs with the API, not just one, and read the
    API's own answer back to report how many km we miss by.

Run:  ../.venv/bin/python generalised_launch.py
Figures go to figures/3*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(1, 2)

import io
from contextlib import redirect_stdout

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np

from common.config import get_mission
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, save
from launch import HomePlanet, design_engine, simulate_launch, to_solar_system_frame
from orbits import kepler_state
from rocket import load_engine_box

AU_PER_YR = const.AU / const.yr  # [m/s] per [AU/yr]
# Launches we test with the API: (time [yr], launch-site angle [rad]). Spread over 20 home orbits and all directions.
TEST_LAUNCHES = [(0.0, 0.0), (0.0, np.pi / 2), (0.25, np.pi), (0.97, 1.0), (2.345, 4.0), (5.5, 5.5),
                 (8.123, 2.2), (10.4, 3.6)]


def launch_state(launch, planet, system, time_of_launch, launch_angle):
    """Star-frame position before, position and velocity after [AU, AU/yr], and time after [yr]."""
    dt = launch["duration"]
    pos, vel = kepler_state(system, [time_of_launch, time_of_launch + dt / const.yr])
    e_r = np.array([np.cos(launch_angle), np.sin(launch_angle)])
    e_t = np.array([-np.sin(launch_angle), np.cos(launch_angle)])
    position_before = pos[:, 0, 0] + planet.radius * e_r / const.AU
    position_after = pos[:, 0, 1] + (planet.v_rot * dt * e_t + launch["r"] * e_r) / const.AU
    velocity_after = vel[:, 0, 1] + (planet.v_rot * e_t + launch["v"] * e_r) / AU_PER_YR
    return position_before, position_after, velocity_after, time_of_launch + dt / const.yr


def api_launch(mission, engine, launch, state, time_of_launch):
    """Official launch and check. Returns (verified, API's position, API's velocity, API's message)."""
    position_before, position_after, _, _ = state
    log = io.StringIO()
    mission.verbose = True
    with redirect_stdout(log):
        mission.set_launch_parameters(engine["thrust"], engine["mass_loss_rate"], engine["fuel_mass"],
                                      launch["duration"] * 1.05, position_before, time_of_launch)
        mission.launch_rocket()
        try:
            mission.verify_launch_result(position_after)
        except RuntimeError:
            pass
    # Read the API's own answer back, only to report how far off we are
    return (mission.launch_result_verified, np.array(mission._position_after_launch),
            np.array(mission._velocity_after_launch), log.getvalue())


def design(mission, planet):
    """The Part 1 engine and launch (same design: 6 min, launch.FUEL_RESERVE kg left)."""
    thrust_box, mdot_box = load_engine_box()
    n_boxes, fuel = design_engine(thrust_box, mdot_box, mission.spacecraft_mass, planet)
    engine = {"thrust": thrust_box * n_boxes, "mass_loss_rate": mdot_box * n_boxes, "fuel_mass": fuel}
    launch = simulate_launch(engine["thrust"], engine["mass_loss_rate"], fuel, mission.spacecraft_mass, planet,
                             dense=True)
    return engine, launch


# ===================================================================== figures
def plot_launch_sites(launch, planet, system):
    """Planet frame: where the rocket ends up for launch sites all around the equator."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    R = planet.radius / 1e3
    theta = np.linspace(0, 2 * np.pi, 300)
    angles = np.radians(np.arange(0, 360, 45))
    dt = launch["duration"]
    sol = launch["solution"]
    t = np.linspace(0, dt, 200)
    r = sol.sol(t)[0] / 1e3  # Radial distance during the launch [km]
    for ax in (ax1, ax2):
        ax.fill(R * np.cos(theta), R * np.sin(theta), color=PLANET_COLORS[0], alpha=0.25, lw=0)
        ax.plot(R * np.cos(theta), R * np.sin(theta), color=PLANET_COLORS[0], lw=1.5)
    for phi in angles:
        e_r = np.array([np.cos(phi), np.sin(phi)])
        e_t = np.array([-np.sin(phi), np.cos(phi)])
        path = r[None, :] * e_r[:, None] + planet.v_rot * t[None, :] / 1e3 * e_t[:, None]
        for ax in (ax1, ax2):
            ax.plot(*path, color=ORANGE, lw=1.8)
            ax.plot(*(R * e_r), "o", color=INK, ms=5)
            ax.plot(*path[:, -1], "o", color=ORANGE, ms=7, mec=INK, mew=0.5)
        ax1.annotate(f"{np.degrees(phi):.0f}$^\\circ$", R * e_r * 0.82, ha="center", va="center", fontsize=8)
    ax1.set_title(f"Launch from 8 sites: the rocket ends {launch['r'] / 1e3 - R:,.0f} km up after {dt:.0f} s", fontsize=10)
    lim = launch["r"] / 1e3 * 1.15
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    # Zoom on one site: the sideways drift from the planet's rotation
    phi = 0.0
    drift = planet.v_rot * dt / 1e3
    ax2.set_xlim(R * 0.97, launch["r"] / 1e3 * 1.03)
    span = (launch["r"] / 1e3 * 1.03 - R * 0.97) / 2
    ax2.set_ylim(-span * 0.4, span * 1.6)
    ax2.annotate("", xy=(launch["r"] / 1e3, drift), xytext=(launch["r"] / 1e3, 0),
                 arrowprops={"arrowstyle": "<->", "color": BLUE})
    ax2.text(launch["r"] / 1e3, drift / 2, f"  {drift:.0f} km sideways\n  (rotation, {planet.v_rot:.0f} m/s)",
             color=BLUE, fontsize=9, va="center")
    ax2.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax2.set_title("Zoom on the 0$^\\circ$ site: straight up, but carried along by the spin", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xlabel("x from the planet's centre [km]")
        ax.set_ylabel("y [km]")
    fig.suptitle("3  Launching from anywhere on the equator (planet frame; same launch, rotated)", fontsize=12)
    save(fig, "3a_launch_sites")


def plot_launch_times(system, results):
    """Star frame: the test launches along the home orbit, and how close we are to the API."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.2]})
    t = np.linspace(0, 1.05 * 2 * np.pi * np.sqrt(system.semi_major_axes[0] ** 3 / (const.G_sol * system.star_mass)), 400)
    pos, _ = kepler_state(system, t)
    ax1.plot(pos[0, 0], pos[1, 0], color=PLANET_COLORS[0], lw=1.5)
    ax1.plot(0, 0, "*", color="#eda100", ms=16, mec=INK, mew=0.5)
    for i, res in enumerate(results):
        x, y = res["state"][1]
        vx, vy = res["state"][2]
        ax1.plot(x, y, "o", color=BLUE, ms=7, mec=INK, mew=0.5)
        ax1.annotate(f"{i + 1}", (x, y), xytext=(6, 4 - 12 * (i % 2)), textcoords="offset points", fontsize=9)
        ax1.arrow(x, y, vx * 0.02, vy * 0.02, width=0.004, color=MUTED, length_includes_head=True)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [AU]")
    ax1.set_ylabel("y [AU]")
    ax1.set_title("Test launches along the home orbit (arrows: velocity)", fontsize=10)

    miss = np.array([res["miss_km"] for res in results])
    tolerance = 0.015 * system.radii[0]
    x = np.arange(len(results))
    ax2.bar(x, miss, color=[AQUA if res["verified"] else ORANGE for res in results], width=0.6)
    ax2.axhline(tolerance, color=INK, ls="--", lw=1.2)
    ax2.text(-0.4, tolerance, f" API tolerance: 1.5% of R = {tolerance:.0f} km", va="bottom", fontsize=9)
    for xi, res in zip(x, results):
        ax2.annotate(f"{res['miss_km']:.1f} km", (xi, res["miss_km"]), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8, color=MUTED)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{i + 1}\nt={res['t0']:.2f} yr\n{np.degrees(res['phi']):.0f}$^\\circ$"
                         for i, res in enumerate(results)], fontsize=8)
    ax2.set_ylabel("distance from the API's position [km]")
    ax2.set_title("How far our position after launch is from the API's (all verified)" if all(
        res["verified"] for res in results) else "How far our position is from the API's (orange: failed)", fontsize=10)
    fig.suptitle("3  Generalised launch: any time, any site, checked by the API", fontsize=12)
    save(fig, "3b_launch_times")


def main():
    mission = get_mission()
    system = mission.system
    planet = HomePlanet(mission)
    engine, launch = design(mission, planet)
    print(f"3     Part 1 launch: thrust {engine['thrust']:.4e} N, {engine['mass_loss_rate']:.3f} kg/s, "
          f"fuel {engine['fuel_mass']:,.1f} kg -> escape after {launch['duration']:.2f} s")

    # The straight-line (Part 1) and curved (here) planet motion during the launch differ by:
    state = launch_state(launch, planet, system, 0.0, 0.0)
    pos0, vel0 = (q[:, 0, 0] for q in kepler_state(system, [0.0]))
    linear = to_solar_system_frame(launch, planet, 0.0, 0.0, pos0, vel0)
    print(f"      straight-line vs. curved planet motion during launch: "
          f"{np.linalg.norm(linear[1] - state[1]) * const.AU / 1e3:.2f} km")

    print(f"API   {len(TEST_LAUNCHES)} test launches (time, launch-site angle):")
    results = []
    for t0, phi in TEST_LAUNCHES:
        state = launch_state(launch, planet, system, t0, phi)
        verified, api_pos, api_vel, log = api_launch(mission, engine, launch, state, t0)
        miss = np.linalg.norm(state[1] - api_pos) * const.AU / 1e3
        miss_v = np.linalg.norm(state[2] - api_vel) * AU_PER_YR
        results.append({"t0": t0, "phi": phi, "state": state, "verified": verified, "miss_km": miss, "miss_v": miss_v})
        print(f"      t0 = {t0:6.3f} yr, phi = {np.degrees(phi):5.1f} deg: {'VERIFIED' if verified else 'FAILED  '}  "
              f"off by {miss:6.2f} km in position, {miss_v:5.2f} m/s in velocity")
        if not verified:
            print("      " + "\n      ".join(line for line in log.splitlines() if line.strip()))
    if all(r["verified"] for r in results):
        last = log.splitlines()
        print("      API says: " + " / ".join(line.strip() for line in last if "Well done" in line or "Achievement" in line))

    plot_launch_sites(launch, planet, system)
    plot_launch_times(system, results)
    return engine, launch, results


if __name__ == "__main__":
    main()
