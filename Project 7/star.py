"""
Part 7, challenge D: exploring the star (luminosity two ways).

1. From the surface: L = 4 pi R^2 sigma T^4, and the star's place in an HR diagram
   (ast2000tools' StarPopulation as the background population).
2. Core temperature, for a star of uniform density rho0 = M / (4/3 pi R^3):
       M(r) = M r^3 / R^3                                  (mass inside r)
       dP/dr = -G M(r) rho0 / r^2,   P = rho0 k T / (mu m_H)  (hydrostatic + ideal gas)
       -> dT/dr = -(4 pi / 3) G rho0 (mu m_H / k) r
       -> T_c = T(R) + (2 pi / 3) G rho0 (mu m_H / k) R^2
   with mu for a fully ionised gas of 75% H and 25% He (by mass): 1/mu = 2X + 3Y/4.
3. Energy production in the core (r < 0.2 R, temperature T_c everywhere). For T_c < 90 MK:
   the pp chain and the CNO cycle, with X = 0.745, Y = 0.253, X_CNO = 0.002, and the
   AST2000 lecture notes' approximations (T6 = T / 10^6 K):
       eps_pp  = 1.08e-12 W m^3/kg^2 * X^2 rho T6^4
       eps_CNO = 8.24e-31 W m^3/kg^2 * X X_CNO rho T6^20
   L = (eps_pp + eps_CNO) * M_core, and the surface temperature that L would imply.
   (For T_c > 90 MK it would be the triple-alpha process; the code reports which branch applies.)

Run:  ../.venv/bin/python star.py
Figures go to figures/d*.png; the answers are in paper/part7_star.tex.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.star_population import StarPopulation
from scipy import constants as sc
from scipy import optimize

from common.config import get_mission
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, VIOLET, save
from common.report import record_many

X_SURFACE, Y_SURFACE = 0.75, 0.25  # Whole star, for mu (question 2)
X_CORE, Y_CORE, X_CNO = 0.745, 0.253, 0.002  # Core composition for the pp + CNO branch
X_CORE_3A, Y_CORE_3A = 0.20, 0.80  # Core composition if the triple-alpha branch applies
EPS0_PP, EPS0_CNO, EPS0_3A = 1.08e-12, 8.24e-31, 3.86e-18  # [W m^3 kg^-2] (3a: [W m^6 kg^-3])
CORE_FRACTION = 0.2  # Nuclear reactions happen inside 0.2 R
T_SPLIT = 90e6  # [K]
M_H = sc.m_p  # Hydrogen mass ~ proton mass [kg]
L_SUN = const.L_sun
R_SUN = 6.957e8


def mean_molecular_weight(X, Y, ionised=True):
    """mu in units of m_H: fully ionised 1/mu = 2X + 3Y/4, neutral 1/mu = X + Y/4."""
    return 1 / (2 * X + 0.75 * Y) if ionised else 1 / (X + 0.25 * Y)


def core_temperature(M, R, T_surface, mu):
    rho0 = M / (4 / 3 * np.pi * R ** 3)
    return T_surface + 2 * np.pi / 3 * sc.G * rho0 * mu * M_H / sc.k * R ** 2, rho0


def temperature_profile(r, M, R, T_surface, mu):
    rho0 = M / (4 / 3 * np.pi * R ** 3)
    return T_surface + 2 * np.pi / 3 * sc.G * rho0 * mu * M_H / sc.k * (R ** 2 - r ** 2)


def energy_rates(T, rho, X=X_CORE, X_cno=X_CNO):
    """pp-chain and CNO energy release per kg [W/kg] at temperature T [K] and density rho."""
    T6 = np.asarray(T) / 1e6
    return EPS0_PP * X ** 2 * rho * T6 ** 4, EPS0_CNO * X * X_cno * rho * T6 ** 20


def triple_alpha(T, rho, Y=Y_CORE_3A):
    return EPS0_3A * rho ** 2 * Y ** 3 * (np.asarray(T) / 1e8) ** 41


# ===================================================================== figures
def plot_hr(T, L):
    pop = StarPopulation(seed=1)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(pop.temperatures, pop.luminosities, c=pop.colors, s=1, alpha=0.6, lw=0)
    ax.plot(T, L, "*", color=ORANGE, ms=22, mec=INK, mew=1, label=f"our star: {T:.0f} K, {L:.2f} L$_\\odot$")
    ax.plot(5772, 1.0, "o", color="#eda100", ms=10, mec=INK, label="the Sun")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2000)
    ax.set_ylim(3e-5, 3e6)  # The population puts its faintest dwarfs on a floor near 1e-5
    ax.set_xlabel("surface temperature [K]")
    ax.set_ylabel("luminosity [L$_\\odot$]")
    ax.set_facecolor("#1b1b1b")
    ax.grid(False)
    for name, (x, y) in {"main sequence": (12000, 40), "giants": (4500, 300), "supergiants": (8000, 3e5),
                         "white dwarfs": (12000, 3e-3)}.items():
        ax.text(x, y, name, color="white", fontsize=10, ha="center")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("HR diagram (50 000 stars from ast2000tools' StarPopulation)", fontsize=10)
    fig.suptitle("D1  Is our star a main-sequence star?", fontsize=12)
    save(fig, "d1_hr_diagram")


def plot_core(M, R, T_surface, mu, Tc, rho0, L_surface, T_needed):
    r = np.linspace(0, R, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    ax1.plot(r / R, temperature_profile(r, M, R, T_surface, mu) / 1e6, color=ORANGE)
    ax1.axvspan(0, CORE_FRACTION, color=AQUA, alpha=0.2, lw=0, label=f"core (r < {CORE_FRACTION} R): reactions")
    ax1.set_xlabel("r / R")
    ax1.set_ylabel("temperature [MK]")
    ax1.set_title(f"Uniform density: $T(r) = T_c - \\frac{{2\\pi}}{{3}}G\\rho_0\\frac{{\\mu m_H}}{{k}}r^2$, "
                  f"$T_c$ = {Tc / 1e6:.2f} MK", fontsize=10)
    ax1.legend(fontsize=8)
    T = np.logspace(6, 8, 300)
    pp, cno = energy_rates(T, rho0)
    M_core = rho0 * 4 / 3 * np.pi * (CORE_FRACTION * R) ** 3
    ax2.loglog(T / 1e6, pp * M_core / L_SUN, color=BLUE, label="pp chain")
    ax2.loglog(T / 1e6, cno * M_core / L_SUN, color=VIOLET, label="CNO cycle")
    ax2.loglog(T / 1e6, (pp + cno) * M_core / L_SUN, color=INK, lw=1.2, ls="--", label="total")
    ax2.axhline(L_surface / L_SUN, color=ORANGE, lw=1.5, label=f"from the surface: {L_surface / L_SUN:.2f} L$_\\odot$")
    ax2.axvline(Tc / 1e6, color=MUTED, ls=":", lw=1.2, label=f"our T$_c$ = {Tc / 1e6:.1f} MK")
    ax2.axvline(T_needed / 1e6, color=AQUA, ls="--", lw=1.2, label=f"T$_c$ that would match: {T_needed / 1e6:.1f} MK")
    ax2.set_ylim(1e-8, 1e4)
    ax2.set_xlabel("core temperature [MK]")
    ax2.set_ylabel("luminosity of the core [L$_\\odot$]")
    ax2.set_title(f"Fusion in the core at the uniform density {rho0:.0f} kg/m$^3$: steep in T (T$^4$, T$^{{20}}$)",
                  fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    fig.suptitle("D2, D3  Core temperature and energy production", fontsize=12)
    save(fig, "d2_core")


def main():
    system = get_mission().system
    M = system.star_mass * const.m_sun
    R = system.star_radius * 1e3
    T_surface = system.star_temperature
    L_surface = 4 * np.pi * R ** 2 * sc.Stefan_Boltzmann * T_surface ** 4

    mu = mean_molecular_weight(X_SURFACE, Y_SURFACE)
    mu_neutral = mean_molecular_weight(X_SURFACE, Y_SURFACE, ionised=False)
    Tc, rho0 = core_temperature(M, R, T_surface, mu)
    Tc_neutral = core_temperature(M, R, T_surface, mu_neutral)[0]
    branch = "pp + CNO" if Tc < T_SPLIT else "triple alpha"
    M_core = rho0 * 4 / 3 * np.pi * (CORE_FRACTION * R) ** 3
    pp, cno = energy_rates(Tc, rho0)
    L_core = (pp + cno) * M_core if Tc < T_SPLIT else triple_alpha(Tc, rho0) * M_core
    T_from_core = (L_core / (4 * np.pi * R ** 2 * sc.Stefan_Boltzmann)) ** 0.25
    T_needed = optimize.brentq(lambda T: np.sum(energy_rates(T, rho0)) * M_core - L_surface, 1e6, 1e8)
    print(f"D1    star: M = {system.star_mass:.3f} Msun, R = {R / R_SUN:.3f} Rsun, T = {T_surface:.0f} K -> "
          f"L = {L_surface:.3e} W = {L_surface / L_SUN:.3f} Lsun")
    print(f"D2    uniform density {rho0:.0f} kg/m^3, mu = {mu:.3f} (ionised) -> T_c = {Tc / 1e6:.2f} MK "
          f"(neutral gas, mu = {mu_neutral:.2f}: {Tc_neutral / 1e6:.1f} MK); branch: {branch}")
    print(f"D3    core mass {M_core / M:.3f} M; eps_pp = {pp:.3e} W/kg, eps_CNO = {cno:.3e} W/kg -> L = {L_core:.3e} W "
          f"= {L_core / L_SUN:.2e} Lsun ({L_surface / L_core:.0f}x too low)")
    print(f"      surface temperature from that L: {T_from_core:.0f} K (true {T_surface:.0f} K); "
          f"T_c needed for the observed L: {T_needed / 1e6:.1f} MK")
    plot_hr(T_surface, L_surface / L_SUN)
    plot_core(M, R, T_surface, mu, Tc, rho0, L_surface, T_needed)
    record_many({
        "star-M": (system.star_mass, "{:.3f}"), "star-R": (R / R_SUN, "{:.3f}"), "star-R-km": (R / 1e3, "{:,.0f}"),
        "star-T": (T_surface, "{:.0f}"), "star-L": (L_surface, "{:.3e}"), "star-L-sun": (L_surface / L_SUN, "{:.2f}"),
        "star-mu": (mu, "{:.3f}"), "star-mu-neutral": (mu_neutral, "{:.2f}"), "star-rho0": (rho0, "{:.0f}"),
        "star-Tc": (Tc / 1e6, "{:.2f}"), "star-Tc-neutral": (Tc_neutral / 1e6, "{:.1f}"), "star-branch": branch,
        "star-core-mass": (M_core / M, "{:.3f}"), "star-eps-pp": (pp, "{:.2e}"), "star-eps-cno": (cno, "{:.2e}"),
        "star-L-core": (L_core, "{:.2e}"), "star-L-core-sun": (L_core / L_SUN, "{:.1e}"),
        "star-L-ratio": (L_surface / L_core, "{:.0f}"), "star-T-core-model": (T_from_core, "{:.0f}"),
        "star-T-needed": (T_needed / 1e6, "{:.1f}"), "star-X": (X_CORE, "{:g}"), "star-Y": (Y_CORE, "{:g}"),
        "star-Xcno": (X_CNO, "{:g}"), "star-eps0-pp": (EPS0_PP, "{:.2e}"), "star-eps0-cno": (EPS0_CNO, "{:.2e}"),
        "star-cno-share": (cno / (pp + cno) * 100, "{:.1e}"),
    })


if __name__ == "__main__":
    main()
