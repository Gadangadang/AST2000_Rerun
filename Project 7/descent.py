"""
Part 7, challenge B: our own simulator of the descent.

Input:  start time, position and velocity (planet frame, SI), and the commands
        (when and how hard to launch the lander, when to open the parachute and
        how big it is, the landing thrusters' force and switch-on height).
Output: the trajectory, the final time, position and velocity, and the loads
        (drag force and drag per area) to compare with the limits.

Forces (Part 7: all of them must be present):
    gravity     -G M r / |r|^3                        (point mass)
    drag        -1/2 rho(h) C_d A |v - w| (v - w)     (air co-rotating, drag.py; rho from Part 6)
    thrusters   +F_L r_hat once below their height    (radial, as in equation 3)
Phases: spacecraft (1100 kg, 16 m^2) until the lander launch; then the lander
(90 kg, 0.3 m^2); after the parachute opens, the area is lander + parachute.

The integrator is adaptive (DOP853), so it takes tiny steps through the
violent moments (entry, the parachute opening, deceleration ~ 100 g) and long
ones while coasting: the tips' warning about "too large time steps" throwing
the lander back to space is handled automatically. The ground is an event
(h = 0), found exactly.

Run:  ../.venv/bin/python descent.py   (validation against ast2000tools)
Figures go to figures/b*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(6)

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

import low_orbit
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, VIOLET, save
from common.report import record_many
from drag import BURN_LIMIT, PARACHUTE_LIMIT, C_D, Environment, drag_force, drag_velocity, landing_gear

RTOL, ATOL = 1e-10, 1e-6
VALID_DV = 150.0  # Test descent: release boost [m/s] ...
VALID_CHUTE_HEIGHT = 10e3  # ... and parachute opening height [m] (the real landing uses 200+ m/s and 3 km)


@dataclass
class Commands:
    """Everything the landing sequence is told, in landing-sequence time [s]."""
    t_launch: float
    dv_launch: np.ndarray  # Boost given to the lander at launch [m/s] (3,)
    t_chute: float = np.inf  # When the parachute opens
    chute_area: float = 0.0  # [m^2]
    thrust: float = 0.0  # Landing thruster force [N]
    thrust_height: float = 0.0  # Thrusters switch on below this height [m]


@dataclass
class Trajectory:
    t: np.ndarray
    r: np.ndarray  # (n, 3)
    v: np.ndarray
    area: np.ndarray  # Cross-section at each sample
    mass: np.ndarray
    landed: bool = False
    events: dict = field(default_factory=dict)

    @property
    def height(self):
        """Distance from the planet's centre [m] (subtract R for the height above the ground)."""
        return np.linalg.norm(self.r, axis=1)

    def time_through(self, radius):
        """First time [s] the trajectory comes down through this distance from the centre (linear between samples)."""
        d = self.height
        below = np.flatnonzero(d <= radius)
        if not below.size or below[0] == 0:
            raise RuntimeError(f"the trajectory does not come down through r = {radius / 1e3:.0f} km "
                               f"(lowest {d.min() / 1e3:.0f} km)")
        i = below[0]
        return self.t[i - 1] + (d[i - 1] - radius) * (self.t[i] - self.t[i - 1]) / (d[i - 1] - d[i])


class DescentSimulator:
    def __init__(self, env):
        self.env = env

    def _rhs(self, t, y, mass, area, thrust):
        r, v = y[:3], y[3:]
        d = np.linalg.norm(r)
        a = -self.env.GM * r / d ** 3 + drag_force(r, v, area, self.env) / mass
        if thrust:
            a = a + thrust / mass * r / d
        return np.concatenate([v, a])

    def simulate(self, t0, r0, v0, commands, t_max, n_per_segment=400):
        """Integrate from t0 until the ground or t_max, applying the commands. Returns a Trajectory."""
        env = self.env
        y = np.concatenate([r0, v0]).astype(float)
        mass, area, thrust = env.craft_mass, env.craft_area, 0.0
        launched, chute, thrusting = False, False, False
        stops = sorted(t for t in (commands.t_launch, commands.t_chute, t_max) if np.isfinite(t) and t >= t0)
        out_t, out_y, out_A, out_m, events = [], [], [], [], {}
        t = t0
        ground = lambda t, y, *args: np.linalg.norm(y[:3]) - env.R
        ground.terminal, ground.direction = True, -1
        thruster_on = lambda t, y, *args: np.linalg.norm(y[:3]) - env.R - commands.thrust_height
        thruster_on.terminal, thruster_on.direction = True, -1
        while t < t_max:
            if not launched and t >= commands.t_launch:
                y[3:] += commands.dv_launch
                mass, area, launched = env.lander_mass, env.lander_area, True
                events["launch"] = t
            if launched and not chute and t >= commands.t_chute:
                area, chute = env.lander_area + commands.chute_area, True
                events["chute"] = t
            t_next = min([s for s in stops if s > t] + [t_max])
            watch = [ground] + ([thruster_on] if launched and commands.thrust and not thrusting else [])
            sol = integrate.solve_ivp(self._rhs, (t, t_next), y, args=(mass, area, thrust), method="DOP853",
                                      rtol=RTOL, atol=ATOL, events=watch, dense_output=True)
            t_end = sol.t[-1]
            ts = np.linspace(t, t_end, n_per_segment)
            out_t.append(ts)
            out_y.append(sol.sol(ts).T)
            out_A.append(np.full(len(ts), area))
            out_m.append(np.full(len(ts), mass))
            y, t = sol.y[:, -1], t_end
            if len(sol.t_events[0]):  # Ground
                events["ground"] = t
                break
            if len(watch) > 1 and len(sol.t_events[1]):  # Thrusters switch on
                thrust, thrusting = commands.thrust, True
                events["thrusters"] = t
        Y = np.concatenate(out_y)
        return Trajectory(np.concatenate(out_t), Y[:, :3], Y[:, 3:], np.concatenate(out_A), np.concatenate(out_m),
                          landed="ground" in events, events=events)

    # ------------------------------------------------------------- derived quantities
    def loads(self, traj):
        """Drag force [N], drag per area [Pa] and speed relative to the air [m/s] along a trajectory."""
        F = np.linalg.norm(drag_force(traj.r, traj.v, traj.area, self.env), axis=1)
        u = np.linalg.norm(drag_velocity(traj.r, traj.v, self.env.omega), axis=1)
        return F, F / traj.area, u

    def touchdown(self, traj):
        """Radial and tangential (relative to the ground) speed at the end [m/s]."""
        r, v = traj.r[-1], traj.v[-1]
        r_hat = r / np.linalg.norm(r)
        u = drag_velocity(r, v, self.env.omega)  # The ground moves with the air
        return -np.dot(u, r_hat), np.linalg.norm(u - np.dot(u, r_hat) * r_hat)


def check(traj, sim):
    """Peak loads against the limits (parachute load only after it opens)."""
    F, q, u = sim.loads(traj)
    chute = traj.t >= traj.events.get("chute", np.inf)
    return {"max_q": q.max(), "burn_ok": q.max() < BURN_LIMIT,
            "max_chute_force": F[chute].max() if chute.any() else 0.0,
            "chute_ok": (F[chute].max() if chute.any() else 0.0) < PARACHUTE_LIMIT}


# ===================================================================== validation against the API
def fly_api(orbiter, commands, t_end, dt=5.0):
    """The same commands in ast2000tools; orientations every dt seconds. Returns (t, r, v), landed, messages."""
    import io
    from contextlib import redirect_stdout
    seq = orbiter.sequence
    log = io.StringIO()
    samples = []
    with redirect_stdout(log):
        t, r, v = seq.orient()
        samples.append((t, r, v))
        plan = [(commands.t_launch, "launch"), (commands.t_chute, "chute")]
        for t_cmd, name in plan:
            if not np.isfinite(t_cmd):
                continue
            while t < t_cmd - 1e-9 and not seq.reached_surface:
                seq.fall_until_time(min(t + dt, t_cmd))
                t, r, v = seq.orient()
                samples.append((t, r, v))
            if name == "launch":
                seq.launch_lander(np.asarray(commands.dv_launch, dtype=float))
                if commands.thrust:
                    seq.adjust_landing_thruster(force=commands.thrust, min_height=commands.thrust_height)
            else:
                seq.adjust_parachute_area(commands.chute_area)
                seq.deploy_parachute()
        while t < t_end and not seq.reached_surface:
            try:
                seq.fall_until_time(t + dt)
            except RuntimeError:
                break
            t, r, v = seq.orient()
            samples.append((t, r, v))
    t = np.array([s[0] for s in samples])
    return t, np.array([s[1] for s in samples]), np.array([s[2] for s in samples]), seq.reached_surface, log.getvalue()


def plot_validation(env, ours, api, title, name):
    t_a, r_a, v_a = api
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    t0 = ours.events["launch"]
    h_o = (ours.height - env.R) / 1e3
    h_a = (np.linalg.norm(r_a, axis=1) - env.R) / 1e3
    axes[0].plot((ours.t - t0) / 60, h_o, color=BLUE, lw=2, label="our simulator")
    axes[0].plot((t_a - t0) / 60, h_a, ".", color=ORANGE, ms=3, label="ast2000tools")
    axes[0].set_ylabel("height [km]")
    u_o = np.linalg.norm(drag_velocity(ours.r, ours.v, env.omega), axis=1)
    u_a = np.linalg.norm(drag_velocity(r_a, v_a, env.omega), axis=1)
    axes[1].semilogy((ours.t - t0) / 60, u_o, color=BLUE, lw=2, label="our simulator")
    axes[1].semilogy((t_a - t0) / 60, u_a, ".", color=ORANGE, ms=3, label="ast2000tools")
    axes[1].set_ylabel("speed relative to the air [m/s]")
    common = (t_a >= ours.t[0]) & (t_a <= ours.t[-1])
    r_interp = np.column_stack([np.interp(t_a[common], ours.t, ours.r[:, k]) for k in range(3)])
    diff = np.linalg.norm(r_interp - r_a[common], axis=1)
    axes[2].semilogy((t_a[common] - t0) / 60, np.maximum(diff, 1e-3), color=INK)
    axes[2].set_ylabel("distance between the two [m]")
    for ax in axes:
        ax.set_xlabel("minutes after the lander launch")
        for key, color in (("chute", AQUA), ("thrusters", VIOLET)):
            if key in ours.events:
                ax.axvline((ours.events[key] - t0) / 60, color=color, ls=":", lw=1.2)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[2].set_title(f"largest difference {diff.max():.3g} m", fontsize=10)
    fig.suptitle(title, fontsize=12)
    save(fig, name)
    return diff.max()


def main():
    env = Environment()
    sim = DescentSimulator(env)
    altitude = low_orbit_altitude()
    orbiter, planet, _, _ = low_orbit.enter_low_orbit(altitude=altitude)
    t0, r0, v0 = orbiter.orient()
    # A test descent, different from the real one: release 5 min in, parachute (A's size) at VALID_CHUTE_HEIGHT
    chute_area = landing_gear(env)[0]
    test = Commands(t_launch=t0 + 300.0, dv_launch=-VALID_DV * v0 / np.linalg.norm(v0))
    free = sim.simulate(t0, r0, v0, test, t_max=t0 + 4 * 3600)
    commands = Commands(test.t_launch, test.dv_launch, t_chute=free.time_through(env.R + VALID_CHUTE_HEIGHT),
                        chute_area=chute_area)
    ours = sim.simulate(t0, r0, v0, commands, t_max=t0 + 4 * 3600)
    api = fly_api(orbiter, commands, t_end=t0 + 4 * 3600)
    biggest = plot_validation(env, ours, api[:3], f"B  Our simulator against ast2000tools (test descent: lander released "
                              f"with -{VALID_DV:.0f} m/s, {chute_area:g} m$^2$ parachute at "
                              f"{VALID_CHUTE_HEIGHT / 1e3:g} km)", "b1_validation")
    v_r, v_t = sim.touchdown(ours)
    print(f"B     test descent: ours lands after {(ours.events.get('ground', np.nan) - commands.t_launch) / 60:.1f} min at "
          f"{v_r:.2f} m/s; API landed: {api[3]}; largest difference {biggest:.3g} m")
    print("      API says: " + " / ".join(line.strip() for line in api[4].splitlines()
                                         if "land" in line.lower() or "fail" in line.lower() or "crash" in line.lower())[:600])
    record_many({"valid-diff": (biggest, "{:.3g}"), "valid-dv": (VALID_DV, "{:.0f}"), "valid-chute": (chute_area, "{:g}"),
                 "valid-chute-height": (VALID_CHUTE_HEIGHT / 1e3, "{:g}")})
    return sim


def low_orbit_altitude():
    import json
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Project 6", "paper",
                           "results.json")) as f:
        return float(json.load(f)["low-alt"]) * 1e3


if __name__ == "__main__":
    main()
