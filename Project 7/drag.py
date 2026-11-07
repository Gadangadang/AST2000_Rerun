"""
Part 7, challenge A: air resistance, terminal velocity, parachute and thrusters.

1. The atmosphere co-rotates with the planet (angular velocity omega about z), so the
   air at position r moves with
       w(r) = omega z_hat x r = omega (-y, x, 0),
   and the drag acts on the velocity relative to the air,
       v_drag = v - w,      F_d = -1/2 rho(h) C_d A |v_drag| v_drag   (C_d = 1).
2. Why v_theta -> 0 relative to the air: the drag always points against v_drag, so it
   only ever removes relative velocity, and gravity is radial: nothing drives the
   tangential relative velocity, which therefore decays (on the time scale m / (rho A |v_drag|)).
   Why v_r becomes constant: falling faster increases drag as v^2 until it balances gravity.
3. Terminal velocity (v_theta = 0, drag = gravity):
       1/2 rho A v_t^2 = G M m / r^2   ->   v_t = sqrt(2 G M m / (rho A r^2)),
   and the area needed for a given landing speed v at the surface:
       A = 2 G M m / (rho0 R^2 v^2).
4. Landing thrusters: with radial forces only, F_d + F_L = F_G at v_safe gives
       F_L = 1/2 rho0 A (v_t^2 - v_safe^2).

Also the two limits that shape the descent: the parachute tears above 250 kN, and the lander
burns up above a drag of 1e7 Pa per area.

Run:  ../.venv/bin/python drag.py
Figures go to figures/a*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(6)

import json

import matplotlib.pyplot as plt
import numpy as np

import atmosphere as atmosphere_model
import low_orbit
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, SERIES, VIOLET, save
from common.report import record_many

C_D = 1.0  # Drag coefficient (Part 7)
V_SAFE = 3.0  # Fastest allowed touchdown [m/s]
PARACHUTE_LIMIT = 250e3  # The parachute tears above this drag force [N]
BURN_LIMIT = 1e7  # The lander burns above this drag per area [Pa]
V_TARGET = 2.0  # Touchdown speed we design the landing for [m/s]
CHUTE_MARGIN = 0.8  # Largest parachute we use, as a fraction of the one that tears on opening
THRUST_HEIGHT = 400.0  # Landing thrusters (if needed) switch on below this height [m] (the API allows 500 m)
PART6_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Project 6")


class Environment:
    """Planet, atmosphere and spacecraft numbers for the landing (SI units)."""

    def __init__(self):
        self.mission, self.planet, self.fuel, self.exhaust_speed = low_orbit.load()
        with open(os.path.join(PART6_DIR, "atmosphere_model.json")) as f:
            model = json.load(f)
        self.atmosphere = atmosphere_model.Atmosphere(self.planet, model["T0"], model["mu"])
        self.omega = 2 * np.pi / self.planet.rotation_period
        self.lander_mass = self.mission.lander_mass
        self.lander_area = self.mission.lander_area
        self.craft_mass = self.mission.spacecraft_mass
        self.craft_area = self.mission.spacecraft_area
        self.GM, self.R = self.planet.GM, self.planet.R
        self.g0 = self.GM / self.R ** 2
        self.rho0 = self.planet.rho0

    def density(self, h):
        return self.atmosphere.density(np.maximum(h, 0.0))


# ===================================================================== A1: drag in rotating air
def air_velocity(r, omega):
    """Velocity of the co-rotating air at positions r (..., 3): omega z_hat x r."""
    r = np.asarray(r, dtype=float)
    return omega * np.stack([-r[..., 1], r[..., 0], np.zeros(r.shape[:-1])], axis=-1)


def drag_velocity(r, v, omega):
    """Velocity relative to the air."""
    return np.asarray(v, dtype=float) - air_velocity(r, omega)


def drag_force(r, v, area, env, c_d=C_D):
    """Drag force [N] (..., 3) on a body of cross-section `area` at position r and velocity v (planet frame)."""
    r = np.asarray(r, dtype=float)
    u = drag_velocity(r, v, env.omega)
    rho = env.density(np.linalg.norm(r, axis=-1) - env.R)
    area = np.asarray(area, dtype=float)
    area = area[..., None] if area.ndim else area  # One area per sample, or one for all
    return -0.5 * rho[..., None] * c_d * area * np.linalg.norm(u, axis=-1, keepdims=True) * u


# ===================================================================== A2, A3: terminal velocity
def terminal_velocity(h, area, mass, env, c_d=C_D):
    """Speed [m/s] at which drag balances gravity at height h."""
    g = env.GM / (env.R + np.asarray(h, dtype=float)) ** 2
    return np.sqrt(2 * mass * g / (env.density(h) * c_d * area))


def area_for_speed(v, mass, env, c_d=C_D):
    """Total area [m^2] that gives terminal velocity v at the surface."""
    return 2 * mass * env.g0 / (env.rho0 * c_d * np.asarray(v, dtype=float) ** 2)


def thruster_force(area, mass, env, v_safe=V_SAFE):
    """Thruster force [N] that, with drag, holds the radial speed at v_safe near the surface (equation 3)."""
    v_t = terminal_velocity(0.0, area, mass, env)
    return 0.5 * env.rho0 * area * (v_t ** 2 - v_safe ** 2)


def deploy_force(total_area, lander_area, mass, g, speed_factor=1.0):
    """
    Parachute drag right after opening, when the lander falls at speed_factor x its own terminal
    velocity: F = 1/2 rho A_tot v^2 = m g (A_tot / A_lander) speed_factor^2, independent of height.
    """
    return mass * g * total_area / lander_area * speed_factor ** 2


def landing_gear(env):
    """
    (parachute area [m^2], thruster force [N], thruster height [m]) for a V_TARGET touchdown.
    The parachute alone needs A = 2GMm/(rho0 R^2 v^2) in total. Opened at the lander's terminal speed its
    load is m g A / A_lander, so in thin air that area would tear it: then we take CHUTE_MARGIN of the
    largest that survives, and thrusters F_L = 1/2 rho0 A (v_t^2 - v^2) (equation 3) do the rest.
    """
    need = area_for_speed(V_TARGET, env.lander_mass, env)
    cap = CHUTE_MARGIN * PARACHUTE_LIMIT / (env.lander_mass * env.g0) * env.lander_area
    chute = float(np.ceil(need - env.lander_area) if need <= cap else np.floor(cap - env.lander_area))
    total = chute + env.lander_area
    if total >= need:
        return chute, 0.0, 0.0
    return chute, float(np.ceil(thruster_force(total, env.lander_mass, env, v_safe=V_TARGET))), THRUST_HEIGHT


# ===================================================================== figures
def plot_terminal(env, areas):
    h = np.linspace(0, 60e3, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    labels = [f"lander alone ({env.lander_area} m$^2$)"] + [f"lander + parachute: {a:.1f} m$^2$" for a in areas[1:]]
    for area, label, color in zip(areas, labels, [INK] + SERIES + [VIOLET]):
        ax1.semilogx(terminal_velocity(h, area, env.lander_mass, env), h / 1e3, color=color, label=label)
    ax1.axvline(V_SAFE, color=ORANGE, ls="--", lw=1.2, label=f"safe landing: {V_SAFE:g} m/s")
    ax1.set_xlabel("terminal velocity [m/s]")
    ax1.set_ylabel("height [km]")
    ax1.set_title("Terminal velocity $v_t = \\sqrt{2GMm/\\rho A r^2}$: thin air high up, slow near the ground",
                  fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")

    v = np.linspace(0.5, 10, 300)
    A = area_for_speed(v, env.lander_mass, env)
    ax2.semilogy(v, A, color=BLUE, label="area needed at the surface, $A = 2GMm/\\rho_0R^2v^2$")
    A_max = PARACHUTE_LIMIT / (env.lander_mass * env.g0) * env.lander_area
    ax2.axhline(A_max, color=ORANGE, ls="--", lw=1.2,
                label=f"largest parachute that survives opening at terminal speed: {A_max:.0f} m$^2$")
    ax2.axvline(V_SAFE, color=INK, ls=":", lw=1)
    ax2.plot(V_SAFE, area_for_speed(V_SAFE, env.lander_mass, env), "o", color=INK, ms=8)
    ax2.annotate(f" {area_for_speed(V_SAFE, env.lander_mass, env):.1f} m$^2$ for {V_SAFE:g} m/s",
                 (V_SAFE, area_for_speed(V_SAFE, env.lander_mass, env)), fontsize=9)
    ax2.set_xlabel("landing speed [m/s]")
    ax2.set_ylabel("lander + parachute area [m$^2$]")
    ax2.set_title("How big a parachute? (lander 90 kg)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    fig.suptitle("A  Terminal velocity and the parachute", fontsize=12)
    save(fig, "a1_terminal_velocity")


def plot_thrusters(env):
    areas = np.linspace(env.lander_area, area_for_speed(V_SAFE, env.lander_mass, env) * 1.3, 300)
    F = thruster_force(areas, env.lander_mass, env)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(areas, np.maximum(F, 0), color=VIOLET)
    ax.axhline(env.lander_mass * env.g0, color=MUTED, ls=":", lw=1, label=f"the lander's weight, {env.lander_mass * env.g0:.0f} N")
    ax.axvline(area_for_speed(V_SAFE, env.lander_mass, env), color=INK, ls="--", lw=1,
               label="no thrust needed beyond this area")
    ax.set_xlabel("lander + parachute area [m$^2$]")
    ax.set_ylabel("thruster force $F_L$ [N]")
    ax.set_title("Landing thrusters: $F_L = \\frac{1}{2}\\rho_0A(v_t^2 - v_{safe}^2)$ makes up for a small parachute",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("A  Thrust needed for a 3 m/s touchdown", fontsize=12)
    save(fig, "a2_thrusters")


def main():
    env = Environment()
    A_safe = area_for_speed(V_SAFE, env.lander_mass, env)
    v_lander = terminal_velocity(0.0, env.lander_area, env.lander_mass, env)
    F_lander = thruster_force(env.lander_area, env.lander_mass, env)
    A_max = PARACHUTE_LIMIT / (env.lander_mass * env.g0) * env.lander_area
    print(f"A     co-rotating air: omega = {env.omega:.4e} rad/s, {env.omega * env.R:.0f} m/s at the equator")
    print(f"      terminal velocity at the surface: lander alone {v_lander:.1f} m/s; area for {V_SAFE} m/s: {A_safe:.2f} m^2")
    print(f"      thrusters with no parachute: F_L = {F_lander:.0f} N (weight {env.lander_mass * env.g0:.0f} N)")
    print(f"      largest parachute that survives opening at terminal speed: {A_max:.0f} m^2")
    chute, thrust, height = landing_gear(env)
    print(f"      for {V_TARGET:g} m/s we take a {chute:g} m^2 parachute"
          + (f" and {thrust:.0f} N of thrusters below {height:.0f} m (thin air)" if thrust else " and no thrusters"))
    plot_terminal(env, [env.lander_area, 5.0, A_safe, 25.0, 50.0])
    plot_thrusters(env)
    record_many({
        "omega": (env.omega, "{:.4e}"), "w-equator": (env.omega * env.R, "{:.0f}"),
        "lander-mass": (env.lander_mass, "{:.0f}"), "lander-area": (env.lander_area, "{:g}"),
        "craft-mass": (env.craft_mass, "{:.0f}"), "craft-area": (env.craft_area, "{:g}"),
        "g0": (env.g0, "{:.2f}"), "rho0": (env.rho0, "{:.2f}"), "v-safe": (V_SAFE, "{:g}"),
        "vt-lander": (v_lander, "{:.1f}"), "A-safe": (A_safe, "{:.2f}"), "FL-lander": (F_lander, "{:.0f}"),
        "weight": (env.lander_mass * env.g0, "{:.0f}"), "A-max": (A_max, "{:.0f}"),
        "chute-limit": (PARACHUTE_LIMIT / 1e3, "{:.0f}"), "burn-limit": (BURN_LIMIT, "{:.0e}"),
    })
    return env


if __name__ == "__main__":
    main()
