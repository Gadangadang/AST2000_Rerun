"""
Part 1, challenge D: The rocket engine's performance.

Input:  rocket thrust [N], fuel consumption [kg/s], initial rocket mass [kg], speed boost [m/s]
Output: fuel consumed [kg]

Gravity is ignored, so momentum conservation gives Tsiolkovsky's rocket
equation. With exhaust speed u = thrust / fuel consumption:

    m dv = -u dm   ->   m_final = m_initial exp(-dv / u)
    fuel consumed = m_initial (1 - exp(-dv / u))

Key insight: the fuel needed depends only on u, not on how many engine boxes
we use. The number of boxes only sets HOW FAST the boost happens. We check
the formula against a brute-force step-by-step simulation of a boost.

Run:  ../.venv/bin/python rocket.py
Figures are written to figures/d*.png.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from config import get_mission
from engine import RESULT_FILE
from plot_style import AQUA, BLUE, INK, MUTED, ORANGE, SERIES, save


def load_engine_box():
    """Thrust and fuel consumption of ONE engine box, from challenge C (run engine.py first)."""
    with open(RESULT_FILE) as f:
        box = json.load(f)
    return box["thrust_per_box"], box["mass_loss_rate_per_box"]


def fuel_consumed(thrust, mass_loss_rate, initial_mass, delta_v):
    """Fuel [kg] burned to change the speed by delta_v [m/s], starting from initial_mass [kg] (rocket + fuel)."""
    exhaust_speed = thrust / mass_loss_rate
    return initial_mass * (1 - np.exp(-np.asarray(delta_v) / exhaust_speed))


def boost_duration(thrust, mass_loss_rate, initial_mass, delta_v):
    """How long the engine must burn [s] for the boost."""
    return fuel_consumed(thrust, mass_loss_rate, initial_mass, delta_v) / mass_loss_rate


def simulate_boost(thrust, mass_loss_rate, initial_mass, delta_v, dt=1e-3):
    """Brute force: step dv = F/m dt and dm = -mdot dt until the boost is done. Returns fuel used [kg]."""
    v, m = 0.0, initial_mass
    while v < delta_v:
        v += thrust / m * dt
        m -= mass_loss_rate * dt
    # Remove the overshoot of the last step by linear interpolation
    overshoot = (v - delta_v) / (thrust / m)
    return initial_mass - m - mass_loss_rate * overshoot


def main():
    mission = get_mission()
    craft = mission.spacecraft_mass
    thrust_box, mdot_box = load_engine_box()
    u = thrust_box / mdot_box
    print(f"D     exhaust speed u = thrust / fuel consumption = {u:.1f} m/s (spacecraft mass {craft:.0f} kg)")

    # Check the formula against the brute-force simulation for a few cases
    n_boxes = 1e15  # Any engine size; only the burn time depends on it
    F, mdot = thrust_box * n_boxes, mdot_box * n_boxes
    print("      delta_v [m/s]  m0 [kg]   formula [kg]   step sim [kg]   rel. diff   burn time [s]")
    for dv, m0 in ((1000, craft + 500), (3000, craft + 3000), (10000, craft + 30000)):
        exact = fuel_consumed(F, mdot, m0, dv)
        sim = simulate_boost(F, mdot, m0, dv)
        print(f"      {dv:>12}  {m0:>7.0f}   {exact:>12.3f}   {sim:>13.3f}   {abs(sim / exact - 1):.1e}"
              f"   {boost_duration(F, mdot, m0, dv):>12.1f}")

    # Figure 1: fuel vs delta_v for different starting fuel loads, with the brute force on top
    dv = np.linspace(0, 15000, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for extra_fuel, color in zip((1000, 5000, 20000), SERIES):
        m0 = craft + extra_fuel
        fuel = fuel_consumed(F, mdot, m0, dv)
        possible = fuel <= extra_fuel
        ax1.plot(dv[possible] / 1e3, fuel[possible], color=color, label=f"{extra_fuel:,} kg fuel on board")
        ax1.plot(dv[~possible] / 1e3, fuel[~possible], color=color, lw=1, ls=":")
        check = np.linspace(500, dv[possible][-1], 5)
        ax1.plot(check / 1e3, [simulate_boost(F, mdot, m0, d, dt=1e-2) for d in check], "o", color=color, ms=6,
                 mfc="none", mew=1.5)
    ax1.plot([], [], "o", color=MUTED, mfc="none", mew=1.5, label="step-by-step simulation")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"speed boost $\Delta v$ [km/s]")
    ax1.set_ylabel("fuel consumed [kg]")
    ax1.set_title(f"D  Rocket equation (spacecraft {craft:.0f} kg, u = {u:.0f} m/s)\ndotted = more fuel than on board")
    ax1.legend(fontsize=8)

    # Figure 1b: the ratio delta_v / u sets everything; mass ratio grows exponentially
    ratio = np.linspace(0, 5, 300)
    ax2.plot(ratio, np.exp(ratio), color=INK)
    for k, color in zip((1, 2, 3), SERIES):
        ax2.plot(k, np.exp(k), "o", color=color, ms=8)
        ax2.annotate(f"$\\Delta v$ = {k}u = {k * u / 1e3:.1f} km/s\n{100 * (1 - np.exp(-k)):.0f}% of the mass is fuel",
                     (k, np.exp(k)), textcoords="offset points", xytext=(-150 if k == 3 else 10, 0), fontsize=8,
                     va="center")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\Delta v\,/\,u$")
    ax2.set_ylabel(r"$m_{\rm initial}\,/\,m_{\rm final}$")
    ax2.set_title("Required mass ratio grows exponentially with $\\Delta v$")
    save(fig, "d_rocket_equation")

    # Figure 2: engine size changes the burn time, not the fuel
    m0 = craft + 5000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    for boxes, color in zip((1e14, 1e15, 1e16), SERIES):
        Fb, mb = thrust_box * boxes, mdot_box * boxes
        ax1.plot(dv / 1e3, fuel_consumed(Fb, mb, m0, dv), color=color, lw=2, label=f"{boxes:.0e} boxes")
        ax2.plot(dv / 1e3, boost_duration(Fb, mb, m0, dv) / 60, color=color, label=f"{boxes:.0e} boxes")
    ax1.set_xlabel(r"$\Delta v$ [km/s]")
    ax1.set_ylabel("fuel consumed [kg]")
    ax1.set_title(f"Fuel: identical for every engine size (m$_0$ = {m0:,.0f} kg)")
    ax1.legend(fontsize=8)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\Delta v$ [km/s]")
    ax2.set_ylabel("burn time [min]")
    ax2.set_title("Burn time: inversely proportional to the number of boxes")
    ax2.legend(fontsize=8)
    save(fig, "d_engine_size")


if __name__ == "__main__":
    main()
