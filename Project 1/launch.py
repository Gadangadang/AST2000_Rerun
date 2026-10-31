"""
Part 1, challenges E and F: Simulating a rocket launch, and entering the
solar system frame. Ends with the official check through the API.

E: The rocket launches radially outwards from the surface of the home planet.
   Forces: engine thrust and the planet's gravity (no air resistance). The mass
   drops as fuel burns. The planet's rotation gives the rocket a sideways speed
   v_rot = 2 pi R / P_rot, which stays unchanged during the launch (it is
   perpendicular to every force). The launch is done when the rocket's kinetic
   energy, including v_rot, equals the gravitational binding energy:

       1/2 (v_r^2 + v_rot^2) = G M / r.

   We integrate (r, v_r, m) with an adaptive Runge-Kutta solver, and let it
   find the exact moment of escape (or running out of fuel) as an "event",
   instead of checking by hand every time step.

   Designing the engine: the number of boxes sets the thrust, and the launch
   must take 5-20 min while leaving FUEL_RESERVE kg for later parts. Both
   unknowns (fuel to load, number of boxes) are found by root finding, so the
   design adjusts itself if the engine or the reserve changes.

F: Positions and velocities in the star's frame, in AU, AU/yr and yr.
   During a launch of a few minutes the planet moves in an essentially straight
   line (its orbit bends by only a few km), so
       position = planet position + (v_planet + v_rot) t + r_final * e_launch
       velocity = v_planet + v_rot + v_r e_launch
   The launch site is given by an angle phi on the equator (phi = 0 is the +x
   side of the planet, as in figure 2), and the launch time is t0 = 0 for now.

Run:  ../.venv/bin/python launch.py
Figures are written to figures/e*.png and figures/f*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import io
from contextlib import redirect_stdout

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize

from common.config import MISSION_FILE, get_mission
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, VIOLET, save
from rocket import fuel_consumed, load_engine_box

# ------------------------------------------------------------------ design choices
# Aim for a 6 minute launch. The allowed window is 5-20 min, but with T = 3000 K (u = 4.4 km/s) the
# longest launch physically possible is ~6.25 min (thrust/weight -> 1 at liftoff); see e_design_space.png.
# A higher engine temperature allows longer launches and less fuel.
TARGET_DURATION = 6 * 60  # [s]
FUEL_RESERVE = 3000.0  # Fuel left in the tank after launch, for later parts (educated guess) [kg]
LAUNCH_ANGLE = 0.0  # Launch site angle on the equator [rad]; 0 = the +x side of the planet (figure 2)
TIME_OF_LAUNCH = 0.0  # [yr]

MIN_DURATION, MAX_DURATION = 5 * 60, 20 * 60  # Allowed by the assignment and the API [s]


class HomePlanet:
    """The numbers we need about the home planet, in SI units."""

    def __init__(self, mission):
        system = mission.system
        self.mass = system.masses[0] * const.m_sun  # [kg]
        self.radius = system.radii[0] * 1e3  # [m]
        self.GM = const.G * self.mass
        self.rotation_period = system.rotational_periods[0] * const.day  # [s]
        self.v_rot = 2 * np.pi * self.radius / self.rotation_period  # Surface speed at the equator [m/s]
        self.v_esc = np.sqrt(2 * self.GM / self.radius)
        self.position_AU = system.initial_positions[:, 0]  # At t = 0
        self.velocity_AU = system.initial_velocities[:, 0]


# ======================================================================= E
def simulate_launch(thrust, mass_loss_rate, fuel_mass, spacecraft_mass, planet, t_max=2 * MAX_DURATION,
                    dense=False):
    """
    Integrate the radial launch until escape, empty tank or t_max.

    State y = (r, v_r, m). While the thrust cannot yet lift the rocket, the ground
    holds it (no motion), just like a real rocket sitting on the launch pad.
    """
    GM, R, v_rot = planet.GM, planet.radius, planet.v_rot

    def rhs(t, y):
        r, v, m = y
        a = thrust / m - GM / r ** 2
        if r <= R and v <= 0 and a < 0:
            a = 0.0  # Resting on the ground
        return [v, a, -mass_loss_rate]

    def escaped(t, y):
        r, v, m = y
        return 0.5 * (v ** 2 + v_rot ** 2) - GM / r

    def out_of_fuel(t, y):
        return y[2] - spacecraft_mass

    escaped.terminal = out_of_fuel.terminal = True
    escaped.direction = 1

    m0 = spacecraft_mass + fuel_mass
    sol = integrate.solve_ivp(rhs, (0, t_max), [R, 0.0, m0], method="RK45", events=(escaped, out_of_fuel),
                              rtol=1e-10, atol=[1e-3, 1e-6, 1e-6], dense_output=dense, max_step=1.0)
    success = len(sol.t_events[0]) > 0
    if success:
        t_end, (r, v, m) = sol.t_events[0][0], sol.y_events[0][0]
    else:
        t_end, (r, v, m) = sol.t[-1], sol.y[:, -1]
    return {"success": success, "out_of_fuel": len(sol.t_events[1]) > 0, "duration": t_end,
            "r": r, "v": v, "mass": m, "fuel_left": m - spacecraft_mass, "fuel_used": m0 - m, "solution": sol,
            "thrust": thrust, "mass_loss_rate": mass_loss_rate, "fuel_mass": fuel_mass}


def normalised_launch(a0, exhaust_speed, planet, dense=False):
    """
    The launch for a rocket of initial mass 1 kg and thrust a0 N.

    Dividing the equations of motion by the thrust F shows that, for a given
    exhaust speed u = F/mdot, the whole launch depends on ONE number: the initial
    thrust acceleration a0 = F/m0. Any real rocket with the same a0 flies the
    same path; only its masses are scaled. So we design in this normalised form
    and scale up afterwards.
    """
    return simulate_launch(a0, a0 / exhaust_speed, 1.0 - 1e-12, 1e-12, planet, dense=dense)


def scale_to_rocket(a0, exhaust_speed, planet, final_mass):
    """Scale the normalised launch so the rocket has exactly final_mass kg left at escape."""
    unit = normalised_launch(a0, exhaust_speed, planet)
    if not unit["success"]:
        return None
    k = final_mass / unit["mass"]  # [kg] of real rocket per kg of normalised rocket
    thrust = a0 * k
    return {"thrust": thrust, "mass_loss_rate": thrust / exhaust_speed, "initial_mass": k,
            "duration": unit["duration"]}


def design_engine(thrust_box, mdot_box, spacecraft_mass, planet, target=TARGET_DURATION, reserve=FUEL_RESERVE):
    """
    Number of boxes and fuel load for a launch that lasts `target` seconds and
    leaves `reserve` kg of fuel. Only a 1D root search over a0 is needed.
    """
    u = thrust_box / mdot_box
    g = planet.GM / planet.radius ** 2

    def duration_error(log_tw):
        unit = normalised_launch(g * 10 ** log_tw, u, planet)
        return (unit["duration"] if unit["success"] else 1e9) - target

    # Thrust/weight at liftoff must exceed 1; longer launches need it closer to 1
    log_tw = np.linspace(1e-4, 1.5, 300)
    errors = np.array([duration_error(x) for x in log_tw])
    crossing = np.nonzero((errors[:-1] > 0) & (errors[1:] <= 0) & (errors[:-1] < 1e8))[0]
    if len(crossing) == 0:
        longest = np.max(errors[errors < 1e8]) + target
        raise RuntimeError(f"a {target:.0f} s launch is impossible with u = {u:.0f} m/s "
                           f"(the longest possible is {longest:.0f} s); raise the engine temperature")
    i = crossing[0]
    a0 = g * 10 ** optimize.brentq(duration_error, log_tw[i], log_tw[i + 1], xtol=1e-10)
    rocket = scale_to_rocket(a0, u, planet, spacecraft_mass + reserve)
    n_boxes = rocket["thrust"] / thrust_box
    return n_boxes, rocket["initial_mass"] - spacecraft_mass


# ======================================================================= F
def to_solar_system_frame(launch, planet, launch_angle=LAUNCH_ANGLE, time_of_launch=TIME_OF_LAUNCH,
                          planet_position_AU=None, planet_velocity_AU=None):
    """
    Convert the launch result (planet frame, SI) to the star frame (AU, AU/yr, yr).

    planet_position_AU / planet_velocity_AU are the planet's state at the launch
    time; by default the t = 0 values from the SolarSystem (in Part 3 they will
    come from our simulated orbits).
    """
    if planet_position_AU is None:
        planet_position_AU, planet_velocity_AU = planet.position_AU, planet.velocity_AU
    e_r = np.array([np.cos(launch_angle), np.sin(launch_angle)])  # Launch direction (radially out)
    e_t = np.array([-np.sin(launch_angle), np.cos(launch_angle)])  # Rotation direction (counterclockwise)

    planet_pos = planet_position_AU * const.AU  # [m]
    planet_vel = planet_velocity_AU * const.AU / const.yr  # [m/s]
    v_surface = planet.v_rot * e_t
    t = launch["duration"]

    position_before = (planet_pos + planet.radius * e_r) / const.AU
    position_after = (planet_pos + (planet_vel + v_surface) * t + launch["r"] * e_r) / const.AU
    velocity_after = (planet_vel + v_surface + launch["v"] * e_r) * const.yr / const.AU
    time_after = time_of_launch + t / const.yr
    return position_before, position_after, velocity_after, time_after


# ===================================================================== figures
def plot_launch(launch, planet, spacecraft_mass):
    sol = launch["solution"]
    t = np.linspace(0, launch["duration"], 1500)
    r, v, m = sol.sol(t)
    v_esc_local = np.sqrt(np.maximum(2 * planet.GM / r - planet.v_rot ** 2, 0))  # Radial speed needed to escape
    a = launch["thrust"] / m - planet.GM / r ** 2
    g0 = planet.GM / planet.radius ** 2

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(t / 60, (r - planet.radius) / 1e3, color=BLUE)
    ax.set_ylabel("height above surface [km]")
    ax.set_title(f"Height: {(launch['r'] - planet.radius) / 1e3:,.0f} km at escape")

    ax = axes[0, 1]
    ax.plot(t / 60, v / 1e3, color=ORANGE, label="radial speed $v_r$")
    ax.plot(t / 60, v_esc_local / 1e3, color=INK, ls="--", lw=1.5,
            label=r"needed: $\sqrt{2GM/r - v_{rot}^2}$")
    ax.plot(launch["duration"] / 60, launch["v"] / 1e3, "o", color=ORANGE, ms=8)
    ax.annotate(f"escape after {launch['duration'] / 60:.2f} min", (launch["duration"] / 60, launch["v"] / 1e3),
                textcoords="offset points", xytext=(-150, -5), fontsize=9)
    ax.set_ylabel("speed [km/s]")
    ax.set_title("Escape: the speed curves meet")
    ax.legend(loc="center left", fontsize=8)

    ax = axes[1, 0]
    ax.plot(t / 60, m - spacecraft_mass, color=AQUA, label="fuel on board")
    ax.axhline(launch["fuel_left"], color=MUTED, ls=":", lw=1)
    ax.axhline(0, color=INK, lw=1)
    ax.text(0, launch["fuel_left"], f"  {launch['fuel_left']:,.0f} kg left for later parts", va="bottom",
            color=INK, fontsize=9)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("fuel [kg]")
    ax.set_title(f"Fuel: {launch['fuel_used']:,.0f} kg burned")

    ax = axes[1, 1]
    ax.plot(t / 60, a / g0, color=VIOLET)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("net acceleration [surface g]")
    ax.set_title("Acceleration grows as the rocket gets lighter")
    for ax in axes.flat:
        ax.set_xlim(0, launch["duration"] / 60 * 1.02)
    fig.suptitle(f"E  Launch: thrust {launch['thrust'] / 1e3:,.0f} kN, fuel use {launch['mass_loss_rate']:.1f} kg/s, "
                 f"{launch['fuel_mass']:,.0f} kg fuel loaded", fontsize=12)
    save(fig, "e_launch")


def plot_design_space(thrust_box, mdot_box, spacecraft_mass, planet, chosen_tw):
    """Launch duration and fuel vs. thrust/weight at liftoff, for several engine temperatures."""
    g = planet.GM / planet.radius ** 2
    tw = np.logspace(np.log10(1.001), np.log10(3), 60)
    final_mass = spacecraft_mass + FUEL_RESERVE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    # Every speed in the gas scales as sqrt(T): thrust per box ~ T, fuel use per box ~ sqrt(T)
    for T_i, color in zip((3000, 6000, 10000), (BLUE, ORANGE, AQUA)):
        u = thrust_box / mdot_box * np.sqrt(T_i / 3000)
        rockets = [scale_to_rocket(g * x, u, planet, final_mass) for x in tw]
        duration = np.array([r["duration"] for r in rockets])
        fuel = np.array([r["initial_mass"] - final_mass for r in rockets])
        label = f"T = {T_i} K (u = {u / 1e3:.1f} km/s)" + ("  <- used" if T_i == 3000 else "")
        ax1.plot(tw, duration / 60, color=color, label=label)
        ax2.semilogy(tw, fuel, color=color, label=label)
    ax1.axhspan(MIN_DURATION / 60, MAX_DURATION / 60, color=AQUA, alpha=0.12, lw=0)
    ax1.text(1.25, MAX_DURATION / 60, " allowed: 5-20 min", va="top", color=INK, fontsize=9)
    ax1.set_ylim(0, 21)
    for ax in (ax1, ax2):
        ax.axvline(chosen_tw, color=INK, ls="--", lw=1.2)
        ax.set_xlabel("thrust / weight at liftoff")
        ax.legend(fontsize=8, loc="upper right")
    ax1.text(chosen_tw, 1, f" chosen: {chosen_tw:.3f}", color=INK, fontsize=9)
    ax1.set_ylabel("launch duration [min]")
    ax1.set_title("A gentler liftoff gives a longer launch, but only up to a limit")
    ax2.set_ylabel("fuel burned during launch [kg]")
    ax2.set_title("...and costs more fuel (gravity losses)")
    fig.suptitle(f"E  Engine design space (always {FUEL_RESERVE:,.0f} kg fuel left after launch)", fontsize=12)
    save(fig, "e_design_space")


def plot_solar_system_frame(launch, planet, frame):
    """The launch seen from the star's frame, relative to where the planet centre was at t = 0."""
    position_before, position_after, _, _ = frame
    sol = launch["solution"]
    t = np.linspace(0, launch["duration"], 400)
    r = sol.sol(t)[0]
    e_r, e_t = np.array([1.0, 0.0]), np.array([0.0, 1.0])  # LAUNCH_ANGLE = 0
    planet_vel = planet.velocity_AU * const.AU / const.yr
    centre = np.outer(t, planet_vel)  # Planet centre relative to its t = 0 position [m]
    rocket = centre + np.outer(t, planet.v_rot * e_t) + np.outer(r, e_r)

    fig, ax = plt.subplots(figsize=(8, 7))
    for k, (tc, label, ls) in enumerate(((0, "planet at launch", "-"), (launch["duration"], "planet at escape", "--"))):
        c = tc * planet_vel / 1e3
        ax.add_patch(plt.Circle(c, planet.radius / 1e3, fill=False, color=MUTED, lw=1.5, ls=ls))
        ax.text(c[0] - planet.radius / 1e3 * 0.5, c[1], label, ha="center", va="center", color=MUTED, fontsize=9)
    ax.plot(centre[:, 0] / 1e3, centre[:, 1] / 1e3, color=MUTED, lw=1, ls=":", label="planet centre")
    ax.plot(rocket[:, 0] / 1e3, rocket[:, 1] / 1e3, color=ORANGE, label="rocket")
    start = (position_before - planet.position_AU) * const.AU / 1e3
    end = (position_after - planet.position_AU) * const.AU / 1e3
    ax.plot(*start, "o", color=BLUE, ms=8, label="launch site (F: position before)")
    ax.plot(*end, "*", color=ORANGE, ms=14, label="escape (F: position after)")
    ax.set_aspect("equal")
    ax.set_xlabel("x relative to planet at t = 0 [km]")
    ax.set_ylabel("y relative to planet at t = 0 [km]")
    ax.set_title("F  Launch seen from the star's frame: the planet moves on while we climb")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    save(fig, "f_solar_system_frame")


# ============================================================ verification
def verify_with_api(mission, thrust, mass_loss_rate, fuel_mass, launch, frame):
    """Hand our numbers to ast2000tools and let it check our final position."""
    position_before, position_after, velocity_after, time_after = frame
    # The API simulates up to 1.01 x our estimate, so a small safety margin on the estimate is harmless
    estimated_duration = launch["duration"] * 1.05
    mission.verbose = True  # Let the API tell us what it thinks
    log = io.StringIO()
    with redirect_stdout(log):
        mission.set_launch_parameters(thrust, mass_loss_rate, fuel_mass, estimated_duration, position_before,
                                      TIME_OF_LAUNCH)
        mission.launch_rocket()
        mission.verify_launch_result(position_after)
    print("      " + "\n      ".join(line for line in log.getvalue().splitlines() if line.strip()))
    api_duration = (mission.time_after_launch - TIME_OF_LAUNCH) * const.yr
    print(f"      launch duration: ours {launch['duration']:.2f} s, API {api_duration:.2f} s "
          f"(diff {abs(api_duration - launch['duration']):.2f} s; the API uses 0.01 s Euler steps)")
    return mission.launch_result_verified


def main():
    mission = get_mission()
    planet = HomePlanet(mission)
    craft = mission.spacecraft_mass
    thrust_box, mdot_box = load_engine_box()
    print(f"E     home planet: R = {planet.radius / 1e3:.1f} km, M = {planet.mass:.4e} kg, "
          f"g = {planet.GM / planet.radius ** 2:.2f} m/s^2")
    print(f"      v_esc = {planet.v_esc:.1f} m/s, rotation speed at equator = {planet.v_rot:.1f} m/s, "
          f"spacecraft mass = {craft:.0f} kg")

    n_boxes, fuel = design_engine(thrust_box, mdot_box, craft, planet)
    thrust, mdot = thrust_box * n_boxes, mdot_box * n_boxes
    launch = simulate_launch(thrust, mdot, fuel, craft, planet, dense=True)
    assert launch["success"] and MIN_DURATION <= launch["duration"] <= MAX_DURATION
    print(f"E     design: {n_boxes:.4e} boxes -> thrust {thrust:.4e} N, fuel use {mdot:.3f} kg/s, "
          f"fuel loaded {fuel:,.1f} kg")
    liftoff_tw = thrust / ((craft + fuel) * planet.GM / planet.radius ** 2)
    print(f"      thrust / weight at liftoff = {liftoff_tw:.3f}")
    print(f"      escape after {launch['duration']:.2f} s ({launch['duration'] / 60:.2f} min) at height "
          f"{(launch['r'] - planet.radius) / 1e3:,.1f} km, v_r = {launch['v']:.1f} m/s")
    print(f"      fuel burned {launch['fuel_used']:,.1f} kg, left {launch['fuel_left']:,.1f} kg, "
          f"final mass {launch['mass']:,.1f} kg")
    no_gravity = fuel_consumed(thrust, mdot, craft + fuel, launch["v"])
    print(f"      (D without gravity would need only {no_gravity:,.0f} kg for this speed: "
          f"{launch['fuel_used'] - no_gravity:,.0f} kg goes to fighting gravity)")

    frame = to_solar_system_frame(launch, planet)
    position_before, position_after, velocity_after, time_after = frame
    print(f"F     position before launch = {position_before} AU")
    print(f"      position after launch  = {position_after} AU")
    print(f"      velocity after launch  = {velocity_after} AU/yr")
    print(f"      time after launch      = {time_after:.6e} yr")

    plot_launch(launch, planet, craft)
    plot_solar_system_frame(launch, planet, frame)
    plot_design_space(thrust_box, mdot_box, craft, planet, liftoff_tw)

    print("API   set_launch_parameters -> launch_rocket -> verify_launch_result")
    verified = verify_with_api(mission, thrust, mdot, fuel, launch, frame)
    if verified:
        mission.save(MISSION_FILE, mission, verbose=False)
        print(f"      VERIFIED. Mission saved to {MISSION_FILE.split('/')[-1]} for the next parts.")
    return verified


if __name__ == "__main__":
    main()
