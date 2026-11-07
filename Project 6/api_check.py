"""
Part 6, after the fact: our atmosphere against ast2000tools' own.

Everything in challenges A-D was decided from measurements (spectrum, drag)
without looking at what the API uses internally. Here, afterwards, we read
the API's hidden atmosphere (private attributes of the LandingSequence: its
molecule mass, surface temperature, drag coefficient, gravity model and
density profile) and compare. Nothing here feeds back into the model.

Run:  ../.venv/bin/python api_check.py   (after atmosphere.py)
Figure: figures/c4_api_check.png
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as sc

import atmosphere
import low_orbit
from common.plot_style import BLUE, INK, MUTED, ORANGE, save
from common.report import record_many

HERE = os.path.dirname(os.path.abspath(__file__))


def api_atmosphere():
    """The API's own atmosphere and its parameters (private attributes; read only for this check)."""
    mission, planet, fuel, u = low_orbit.load()
    sequence = low_orbit.Orbiter(mission, fuel, u).sequence
    params = {"molecule_mass_u": sequence._mean_molecular_mass * sc.m_p / sc.atomic_mass,
              "molecule_mass_mp": sequence._mean_molecular_mass, "T0": sequence._surface_temperature,
              "C_d": sequence._drag_coefficient, "uniform_gravity": sequence._assume_uniform_gravity_atmosphere}
    return sequence._atmosphere, params, planet


def main():
    api, params, planet = api_atmosphere()
    with open(os.path.join(HERE, "atmosphere_model.json")) as f:
        ours = json.load(f)
    with open(atmosphere.RESULT_FILE) as f:
        spectral = json.load(f)
    model = atmosphere.Atmosphere(planet, ours["T0"], ours["mu"])
    detected = (atmosphere.Atmosphere(planet, ours["T0"], atmosphere.mean_molecular_weight(spectral["detected"]))
                if spectral["detected"] else None)  # None when there was no spectrum for this seed
    varying = atmosphere.Atmosphere(planet, ours["T0"], ours["mu"], uniform_gravity=False)

    with open(os.path.join(HERE, "paper", "results.json")) as f:
        probe_mu = float(json.load(f)["mu-probe"])  # The drag measurement from atmosphere.py
    h = np.linspace(0, 400e3, 801)
    api_rho = np.array([api._density(x) for x in h])
    api_T = np.array([api._temperature(x) for x in h])
    ratio = model.density(h) / api_rho
    print(f"check API: molecule mass {params['molecule_mass_mp']} m_p = {params['molecule_mass_u']:.3f} u "
          f"(ours {ours['mu']:.3f} u, {ours['mu'] / params['molecule_mass_u'] - 1:+.2%}); T0 {params['T0']:.3f} K "
          f"(ours {ours['T0']:.3f} K); C_d {params['C_d']}; uniform gravity {params['uniform_gravity']}")
    print(f"      density ours/API: {ratio[0]:.4f} at the surface, {np.interp(100e3, h, ratio):.3f} at 100 km, "
          f"{np.interp(300e3, h, ratio):.3f} at 300 km; temperature differs by at most {np.max(np.abs(model.temperature(h) - api_T)):.2f} K")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.2, 1, 0.9]})
    ax = axes[0]
    curves = [(model, BLUE, f"adopted ({ours['mu']:.2f} u, uniform g)"),
              (varying, MUTED, "adopted $\\mu$, g falling with height")]
    if detected is not None:
        curves.append((detected, ORANGE, f"spectrum's detections only ({detected.mu:.2f} u)"))
    for m, color, name in curves:
        ax.semilogy(h / 1e3, m.density(h) / api_rho, color=color, lw=2 if m is model else 1.3, label=name)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xlabel("height [km]")
    ax.set_ylabel(r"$\rho_{\rm ours} / \rho_{\rm API}$")
    ax.set_title("Density: our model divided by the API's", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    ax.plot(h / 1e3, model.temperature(h) - api_T, color=BLUE)
    ax.set_xlabel("height [km]")
    ax.set_ylabel(r"$T_{\rm ours} - T_{\rm API}$ [K]")
    ax.set_title("Temperature difference", fontsize=10)
    ax = axes[2]
    ax.axis("off")
    rows = [("molecule mass", f"{ours['mu']:.2f} u", f"{params['molecule_mass_mp']} m$_p$ = {params['molecule_mass_u']:.2f} u"),
            ("surface temperature", f"{ours['T0']:.2f} K", f"{params['T0']:.2f} K"),
            ("drag coefficient", "1 (Part 7 text)", f"{params['C_d']:g}"),
            ("gravity", "uniform (drag data)", "uniform" if params["uniform_gravity"] else "falling"),
            ("composition", ", ".join(ours["composition"][:2]) + ",\n" + ", ".join(ours["composition"][2:]),
             "mean A = 23.5 =\n(32+18+16+28)/4")]
    table = ax.table(cellText=[[a, b, c] for a, b, c in rows], colLabels=["", "ours", "API (hidden)"], loc="center",
                     cellLoc="left", colWidths=[0.34, 0.33, 0.38])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    ax.set_title("Parameters", fontsize=10)
    fig.suptitle("C  After the fact: our atmosphere against ast2000tools' hidden one", fontsize=12)
    save(fig, "c4_api_check")

    record_many({
        "api-mu-mp": (params["molecule_mass_mp"], "{:g}"), "api-mu-u": (params["molecule_mass_u"], "{:.2f}"),
        "api-mu-err": (abs(ours["mu"] / params["molecule_mass_u"] - 1) * 100, "{:.2f}"),
        "api-probe-err": (abs(probe_mu / params["molecule_mass_u"] - 1) * 100, "{:.2f}"),
        "api-T0": (params["T0"], "{:.2f}"), "api-Cd": (params["C_d"], "{:g}"),
        "api-rho-ratio-0": (ratio[0], "{:.4f}"), "api-rho-ratio-100": (np.interp(100e3, h, ratio), "{:.3f}"),
        "api-rho-ratio-300": (np.interp(300e3, h, ratio), "{:.3f}"),
        "api-T-diff": (np.max(np.abs(model.temperature(h) - api_T)), "{:.2f}"),
        "api-detected-ratio-300": (np.interp(300e3, h, detected.density(h) / api_rho) if detected else np.nan, "{:.0f}"),
    })


if __name__ == "__main__":
    main()
