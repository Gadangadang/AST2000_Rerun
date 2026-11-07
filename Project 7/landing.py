"""
Part 7, challenge C: landing on the chosen site.

The plan, all in landing-sequence time (t = 0 when Part 6's sequence begins):
  1. Enter the Part 6 low orbit (same commands, so the same orbit).
  2. Release the lander with DEORBIT_DV against its motion (more if the orbit needs it to
     reach the ground). Its new orbit dips into the atmosphere; drag brakes it from orbital
     speed to its terminal velocity, and from then on it drifts with the co-rotating air:
     its planet-fixed position hardly changes any more.
  3. Open the parachute once the lander is CHUTE_HEIGHT above the ground and float down
     at V_TARGET. The parachute (and thrusters, in thin air) come from drag.landing_gear.
Where it lands (in planet-fixed longitude) depends on when it is released. We scan
the release time over SEARCH_ORBITS orbits with our simulator (B), find every release
that lands on the site (root finding on the wrapped longitude error), and keep the
one with the star highest at touchdown.

Flying it with ast2000tools, we do not trust the plan blindly: at the release time
the spacecraft orients itself, the release boost is computed from the measured
velocity, and the parachute time is predicted again from the lander's actual state.

Run:  ../.venv/bin/python landing.py
Figures go to figures/c*.png; MCAst video landing.xml and picture landed.xml.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(6)

import io
import json
import shutil
from contextlib import redirect_stdout

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

import landing_sites
import low_orbit
from common.config import JOURNEY_STATE_FILE, MCAST_DATA_DIR, XML_DIR
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, VIOLET, save
from common.report import record, record_many
from descent import Commands, DescentSimulator, check, low_orbit_altitude
from drag import (BURN_LIMIT, PARACHUTE_LIMIT, V_SAFE, V_TARGET, Environment, drag_velocity, landing_gear,
                  terminal_velocity)

DEORBIT_DV = 200.0  # Smallest release boost against the motion [m/s]
CHUTE_HEIGHT = 3000.0  # Open the parachute at this height [m]
SEARCH_ORBITS = 2  # Release times searched (orbits after reaching the low orbit)
N_SCAN = 80  # Release times in the coarse scan
CAMERA_OFFSET = 300.0  # MCAst camera height above the surface for the landing video [m]


# ===================================================================== planning
def fixed_longitude(r, t, env):
    """Planet-fixed longitude [rad] of position r at time t (the site's convention: phi at t = 0)."""
    return np.mod(landing_sites.to_spherical(r)[..., 2] - env.omega * t, 2 * np.pi)


def release_boost(env, r):
    """Release boost [m/s]: DEORBIT_DV, or twice what lowers the periapsis of a circular orbit at r to the ground."""
    r = np.linalg.norm(r)
    graze = np.sqrt(env.GM / r) - np.sqrt(2 * env.GM * env.R / (r * (r + env.R)))
    return max(DEORBIT_DV, 2 * graze)


class Planner:
    def __init__(self, env, sim, t0, r0, v0):
        self.env, self.sim = env, sim
        self.t0, self.r0, self.v0 = t0, r0, v0
        self.dv = release_boost(env, r0)
        self.chute_area, self.thrust, self.thrust_height = landing_gear(env)

    def state_at(self, t):
        """The spacecraft (not yet released) at time t, from our simulator."""
        if t <= self.t0:
            return self.r0, self.v0
        traj = self.sim.simulate(self.t0, self.r0, self.v0, Commands(np.inf, np.zeros(3)), t_max=t)
        return traj.r[-1], traj.v[-1]

    def descend(self, t_launch, r, v, chute_height=CHUTE_HEIGHT):
        """
        Release at t_launch from (r, v): first without a parachute to find when the lander passes
        chute_height, then again with the parachute opening at that moment. Returns (trajectory, commands).
        """
        boost = -self.dv * v / np.linalg.norm(v)
        free = self.sim.simulate(t_launch, r, v, Commands(t_launch, boost), t_max=t_launch + 4 * 3600)
        t_chute = free.time_through(self.env.R + chute_height)
        commands = Commands(t_launch, boost, t_chute=t_chute, chute_area=self.chute_area, thrust=self.thrust,
                            thrust_height=self.thrust_height)
        return self.sim.simulate(t_launch, r, v, commands, t_max=t_launch + 6 * 3600), commands

    def landing_point(self, t_launch):
        r, v = self.state_at(t_launch)
        traj, commands = self.descend(t_launch, r, v)
        return fixed_longitude(traj.r[-1], traj.t[-1], self.env), traj, commands


def scan_releases(planner, phi_site, period):
    """Wrapped longitude miss against release time, and every release that hits the site (refined by Brent)."""
    t = planner.t0 + np.linspace(0, SEARCH_ORBITS * period, N_SCAN)
    miss = np.array([np.angle(np.exp(1j * (planner.landing_point(x)[0] - phi_site))) for x in t])
    roots = []
    for i in np.flatnonzero((miss[:-1] < 0) & (miss[1:] >= 0)):  # Upward crossings (not the +-pi wraps)
        f = lambda x: np.angle(np.exp(1j * (planner.landing_point(x)[0] - phi_site)))
        roots.append(optimize.brentq(f, t[i], t[i + 1], xtol=0.01))
    return t, miss, roots


def sun_elevation(r, t, env, t0_years):
    star = landing_sites.star_direction(env.mission.system, env.planet.index, t0_years, t)[0]
    coords = landing_sites.to_spherical(r)
    return float(landing_sites.solar_elevation(coords, star))


# ===================================================================== flying it
def fly(orbiter, env, sim, t_launch, video=True, dt=10.0):
    """
    The real landing in ast2000tools. Orients at the release time, computes the boost from the measured velocity,
    predicts the parachute time from the lander's actual state, and falls to the ground. The parachute opens at
    that time, or earlier if the measured height is already below CHUTE_HEIGHT. Returns the log.
    """
    seq = orbiter.sequence
    log = io.StringIO()
    samples, events = [], {}
    planner = Planner(env, sim, *low_orbit.quiet(seq.orient))
    with redirect_stdout(log):
        seq.verbose = True
        seq.fall_until_time(t_launch)
        t, r, v = seq.orient()
        samples.append((t, r, v))
        traj, commands = planner.descend(t, r, v)  # Plan from the measured state
        if video:
            seq.look_in_direction_of_planet()
            seq.start_video()
        seq.launch_lander(commands.dv_launch)
        if commands.thrust:  # Switch on by themselves below their height
            seq.adjust_landing_thruster(force=commands.thrust, min_height=commands.thrust_height)
        events["launch"] = t
        # Open at the planned time, or earlier if the altimeter says we are already down to CHUTE_HEIGHT
        while t < commands.t_chute - 1e-6 and np.linalg.norm(r) - env.R > CHUTE_HEIGHT:
            seq.fall_until_time(min(t + dt, commands.t_chute))
            t, r, v = seq.orient()
            samples.append((t, r, v))
        events["chute_early"] = commands.t_chute - t
        seq.adjust_parachute_area(commands.chute_area)
        seq.deploy_parachute()
        events["chute"] = t
        while not seq.reached_surface:
            seq.fall(dt)
            t, r, v = seq.orient()
            samples.append((t, r, v))
        events["ground"] = t
        if video:
            seq.finish_video(filename="landing.xml", number_of_frames=1000, radial_camera_offset=CAMERA_OFFSET)
        seq.look_in_direction_of_motion()
        seq.take_picture(filename="landed.xml")
    for name in ("landing.xml", "landed.xml"):
        path = os.path.join(XML_DIR, name)
        if os.path.exists(path) and os.path.isdir(MCAST_DATA_DIR):
            shutil.copy(path, MCAST_DATA_DIR)
    t = np.array([s[0] for s in samples])
    return {"t": t, "r": np.array([s[1] for s in samples]), "v": np.array([s[2] for s in samples]),
            "events": events, "plan": traj, "commands": commands, "messages": log.getvalue()}


# ===================================================================== figures
def plot_plan(scan_t, miss, roots, choice, period, t0, elevations):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot((scan_t - t0) / period, np.degrees(miss), "o-", color=BLUE, ms=3, lw=1.2,
            label="landing longitude minus the site's (our simulator)")
    for root, elev in zip(roots, elevations):
        ax.plot((root - t0) / period, 0, "v" if root != choice else "*", color=VIOLET if root == choice else MUTED,
                ms=10 if root != choice else 18, mec=INK)
        ax.annotate(f"sun {np.degrees(elev):+.0f}$^\\circ$", ((root - t0) / period, 0), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("release time [orbits after reaching the low orbit]")
    ax.set_ylabel("longitude miss [deg]")
    ax.set_title("Every orbit there is one release time that lands on the site; we pick the one in sunlight "
                 "(star: chosen)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("C  When to release the lander", fontsize=12)
    save(fig, "c1_release_time")


def plot_descent(env, sim, flight):
    plan, t, r, v = flight["plan"], flight["t"], flight["r"], flight["v"]
    t0 = flight["events"]["launch"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    ax = axes[0, 0]
    ax.plot((plan.t - t0) / 60, (plan.height - env.R) / 1e3, color=BLUE, lw=2, label="plan (our simulator)")
    ax.plot((t - t0) / 60, (np.linalg.norm(r, axis=1) - env.R) / 1e3, ".", color=ORANGE, ms=3, label="ast2000tools")
    ax.set_ylabel("height [km]")
    ax.set_title("Height", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    u_plan = np.linalg.norm(drag_velocity(plan.r, plan.v, env.omega), axis=1)
    ax.semilogy((plan.t - t0) / 60, u_plan, color=BLUE, lw=2, label="plan")
    ax.semilogy((t - t0) / 60, np.linalg.norm(drag_velocity(r, v, env.omega), axis=1), ".", color=ORANGE, ms=3,
                label="ast2000tools")
    ax.axhline(V_SAFE, color=INK, ls="--", lw=1, label=f"{V_SAFE:g} m/s")
    ax.set_ylabel("speed relative to the air [m/s]")
    ax.set_title(f"Braking: {u_plan[0] / 1e3:.1f} km/s to terminal velocity, then the parachute", fontsize=10)
    ax.legend(fontsize=8)
    F, q, _ = sim.loads(plan)
    ax = axes[1, 0]
    ax.semilogy((plan.t - t0) / 60, np.maximum(q, 1e-6), color=VIOLET, label="drag per area")
    ax.axhline(BURN_LIMIT, color=ORANGE, ls="--", lw=1.2, label="burn limit 10$^7$ Pa")
    ax.set_ylabel("drag per area [Pa]")
    ax.set_title(f"Heat shield: peak {q.max():.2e} Pa, {BURN_LIMIT / q.max():.0f}x below the limit", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1, 1]
    ax.semilogy((plan.t - t0) / 60, np.maximum(F, 1e-3), color=AQUA, label="drag force")
    ax.axhline(PARACHUTE_LIMIT, color=ORANGE, ls="--", lw=1.2, label="parachute limit 250 kN")
    ax.axhline(env.lander_mass * env.g0, color=MUTED, ls=":", lw=1, label="the lander's weight")
    chute = plan.t >= flight["commands"].t_chute
    ax.set_ylabel("drag force [N]")
    ax.set_title(f"Parachute: peak {F[chute].max() / 1e3:.1f} kN when it opens", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    for ax in axes.flat:
        ax.set_xlabel("minutes after the release")
        ax.axvline((flight["commands"].t_chute - t0) / 60, color=AQUA, ls=":", lw=1.2)
    fig.suptitle("C  The landing: plan against the real flight (dotted: parachute opens)", fontsize=12)
    save(fig, "c2_descent")


def plot_final(env, flight):
    t, r, v = flight["t"], flight["r"], flight["v"]
    h = np.linalg.norm(r, axis=1) - env.R
    v_r = -np.sum(v * r, axis=1) / np.linalg.norm(r, axis=1)
    near = h < 1.5 * CHUTE_HEIGHT
    plan = flight["plan"]
    hp = plan.height - env.R
    vp = -np.sum(plan.v * plan.r, axis=1) / plan.height
    fig, ax = plt.subplots(figsize=(10, 5.5))
    keep = hp < 1.5 * CHUTE_HEIGHT
    ax.plot(vp[keep], hp[keep] / 1e3, color=BLUE, lw=2, label="plan")
    ax.plot(v_r[near], h[near] / 1e3, "o", color=ORANGE, ms=4, label="ast2000tools")
    hh = np.linspace(0, 1.5 * CHUTE_HEIGHT, 200)
    chute = flight["commands"].chute_area
    ax.plot(terminal_velocity(hh, env.lander_area + chute, env.lander_mass, env), hh / 1e3, color=INK, ls="--",
            lw=1.2, label=f"terminal velocity with the {chute:g} m$^2$ parachute")
    ax.plot(terminal_velocity(hh, env.lander_area, env.lander_mass, env), hh / 1e3, color=MUTED, ls=":", lw=1.2,
            label="terminal velocity of the lander alone")
    ax.axvline(V_SAFE, color=ORANGE, ls="--", lw=1, label=f"limit {V_SAFE:g} m/s")
    ax.set_xscale("log")
    ax.set_xlabel("falling speed [m/s]")
    ax.set_ylabel("height [km]")
    ax.set_title(f"Touchdown at {v_r[-1]:.2f} m/s", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("C  The last kilometres", fontsize=12)
    save(fig, "c3_final_approach")


def plot_ground_track(env, flight, site_phi, site_index, t0_years):
    """Planet-fixed view from above the north pole: the lander's path from release to the ground, and the site."""
    t, r = flight["t"], flight["r"]
    phi = fixed_longitude(r, t, env)
    rad = np.linalg.norm(r, axis=1) / env.R
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    star = landing_sites.star_direction(env.mission.system, env.planet.index, t0_years, t[-1])[0]
    angle = np.arctan2(star[1], star[0]) - env.omega * t[-1]
    theta = np.linspace(0, 2 * np.pi, 400)
    ax1.fill(np.cos(theta), np.sin(theta), color="#3b3a37", lw=0)
    day = np.linspace(angle - np.pi / 2, angle + np.pi / 2, 200)
    ax1.fill(np.cos(day), np.sin(day), color="#f3d37a", lw=0)
    ax1.plot(rad * np.cos(phi), rad * np.sin(phi), color=ORANGE, lw=2, label="lander (planet-fixed)")
    ax1.plot(np.cos(site_phi), np.sin(site_phi), "*", color=VIOLET, ms=18, mec="white", label=f"site {site_index}")
    ax1.plot(rad[-1] * np.cos(phi[-1]), rad[-1] * np.sin(phi[-1]), "o", color=INK, ms=7, label="touchdown")
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Planet-fixed frame, from above the north pole (day and night at touchdown)", fontsize=10)
    surface_km = env.R / 1e3 * np.degrees(np.angle(np.exp(1j * (phi - site_phi)))) * np.pi / 180
    ax2.plot(surface_km, (np.linalg.norm(r, axis=1) - env.R) / 1e3, color=ORANGE, lw=2)
    ax2.axvline(0, color=VIOLET, lw=1.5, label=f"site {site_index}")
    ax2.set_xlabel("distance along the equator from the site [km]")
    ax2.set_ylabel("height [km]")
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_ylim(0, (np.linalg.norm(r, axis=1).max() - env.R) / 1e3 * 1.3)
    ax2.set_title(f"Down {surface_km[-1]:+.1f} km from the site: once braked, the lander drifts with the air", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle("C  Where it came down", fontsize=12)
    save(fig, "c4_ground_track")
    return surface_km[-1]


# ===================================================================== main
def main():
    env = Environment()
    sim = DescentSimulator(env)
    with open(os.path.join(os.path.dirname(landing_sites.SITE_FILE), "landing_site.json")) as f:
        site = json.load(f)
    with open(JOURNEY_STATE_FILE) as f:
        t0_years = json.load(f)["time_in_orbit"]
    site_phi, site_index = site["planet_fixed_coords_at_t0"][2], site["index"]
    altitude = low_orbit_altitude()

    orbiter, planet, _, _ = low_orbit.enter_low_orbit(altitude=altitude)
    t0, r0, v0 = orbiter.orient()
    period = planet.period(np.linalg.norm(r0))
    planner = Planner(env, sim, t0, r0, v0)
    scan_t, miss, roots = scan_releases(planner, site_phi, period)
    options = []
    for root in roots:
        _, traj, _ = planner.landing_point(root)
        options.append((root, traj, sun_elevation(traj.r[-1], traj.t[-1], env, t0_years)))
    choice, plan, elevation = max(options, key=lambda o: o[2])
    v_r, v_t = sim.touchdown(plan)
    limits = check(plan, sim)
    print(f"C     low orbit reached at t = {t0 / 3600:.2f} h; site {site_index} at planet-fixed phi = "
          f"{np.degrees(site_phi):.1f} deg")
    for root, traj, elev in options:
        print(f"      release at t = {root / 3600:.3f} h -> lands at t = {traj.t[-1] / 3600:.3f} h, sun {np.degrees(elev):+.0f} deg"
              + ("   <- chosen" if root == choice else ""))
    thrusters = (f", thrusters {planner.thrust:.0f} N below {planner.thrust_height:.0f} m" if planner.thrust else
                 ", no thrusters")
    print(f"      plan: release {planner.dv:.0f} m/s retrograde, parachute {planner.chute_area:g} m^2 at {CHUTE_HEIGHT:g} m"
          f"{thrusters}; touchdown {v_r:.2f} m/s; peak drag/area {limits['max_q']:.2e} Pa, peak chute force "
          f"{limits['max_chute_force'] / 1e3:.1f} kN")
    if not (limits["burn_ok"] and limits["chute_ok"] and v_r <= V_SAFE):
        print(f"      WARNING: the plan breaks a limit (burn ok {limits['burn_ok']}, parachute ok {limits['chute_ok']}, "
              f"touchdown {v_r:.2f} m/s); flying it anyway to see what the API says")

    orbiter.restart()  # Fly it for real, from the same start
    orbiter, planet, _, _ = low_orbit.enter_low_orbit(altitude=altitude)
    flight = fly(orbiter, env, sim, choice)
    t, r, v = flight["t"], flight["r"], flight["v"]
    landed = orbiter.sequence.reached_surface
    phi_land = fixed_longitude(r[-1], t[-1], env)
    miss_km = env.R / 1e3 * np.angle(np.exp(1j * (phi_land - site_phi)))
    elev_land = sun_elevation(r[-1], t[-1], env, t0_years)
    messages = [line.strip() for line in flight["messages"].splitlines()
                if any(k in line.lower() for k in ("launched", "success", "crash", "fail", "burn", "deployed", "reached"))]
    print(f"      ast2000tools: landed = {landed}; at t = {t[-1] / 3600:.3f} h, {miss_km:+.1f} km from the site, "
          f"sun {np.degrees(elev_land):+.0f} deg")
    if flight["events"]["chute_early"] > 1.0:
        print(f"      the altimeter opened the parachute {flight['events']['chute_early']:.0f} s before the plan said")
    print("      API says: " + " / ".join(dict.fromkeys(messages)))

    plot_plan(scan_t, miss, roots, choice, period, t0, [o[2] for o in options])
    plot_descent(env, sim, flight)
    plot_final(env, flight)
    along = plot_ground_track(env, flight, site_phi, site_index, t0_years)
    plan_r = flight["plan"]
    plan_land = fixed_longitude(plan_r.r[-1], plan_r.t[-1], env)
    F, q, _ = sim.loads(flight["plan"])
    chute = flight["plan"].t >= flight["commands"].t_chute
    import re
    found = re.search(r"with velocity ([\d.]+) m/s", flight["messages"])  # The API's own touchdown speed
    v_final = float(found.group(1)) if found else np.nan
    record_many({
        "deorbit-dv": (planner.dv, "{:.0f}"), "chute-area": (planner.chute_area, "{:g}"),
        "chute-height": (CHUTE_HEIGHT / 1e3, "{:g}"), "v-target": (V_TARGET, "{:g}"), "site-index": (site_index, "{:d}"),
        "v-rel-orbit": (np.linalg.norm(drag_velocity(r0, v0, env.omega)) / 1e3, "{:.1f}"),
        "thruster-text": (f"and landing thrusters of {planner.thrust:.0f}\\,N below {planner.thrust_height:.0f}\\,m, "
                          "because a parachute big enough on its own would tear on opening" if planner.thrust
                          else "and no thrusters"),
        "n-releases": (len(roots), "{:d}"), "t-release": (choice / 3600, "{:.3f}"), "t-low": (t0 / 3600, "{:.2f}"),
        "t-land": (t[-1] / 3600, "{:.3f}"), "descent-min": ((t[-1] - choice) / 60, "{:.1f}"),
        "t-chute": ((flight["commands"].t_chute - choice) / 60, "{:.1f}"),
        "landed": "yes" if landed else "NO", "miss-km": (abs(miss_km), "{:.1f}"),
        "miss-dir": "west" if miss_km < 0 else "east",  # phi grows eastwards (the planet spins counterclockwise)
        "plan-miss-km": (abs(env.R / 1e3 * np.angle(np.exp(1j * (plan_land - phi_land)))), "{:.2f}"),
        "sun-land": (np.degrees(elev_land), "{:.0f}"), "v-touch": (v_final, "{:.2f}"), "v-touch-plan": (v_r, "{:.2f}"),
        "max-q": (q.max(), "{:.2e}"), "burn-margin": (BURN_LIMIT / q.max(), "{:.0f}"),
        "chute-peak": (F[chute].max() / 1e3, "{:.1f}"), "site-phi": (np.degrees(site_phi), "{:.1f}"),
        "land-phi": (np.degrees(phi_land), "{:.2f}"), "camera-offset": (CAMERA_OFFSET, "{:.0f}"),
    })
    return flight


if __name__ == "__main__":
    main()
