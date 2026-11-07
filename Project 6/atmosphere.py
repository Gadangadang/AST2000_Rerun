"""
Part 6, challenge C: a model of the atmosphere, density and temperature against height.

Assumptions (from the assignment): uniform composition, spherically symmetric,
hydrostatic equilibrium, ideal gas, surface temperature T0 from Part 3 and
surface density rho0 from ast2000tools; adiabatic (gamma = 1.4) up to the
height where T = T0/2, isothermal above.

Equations, with h the height above the surface and g(h) = GM / (R + h)^2:
    hydrostatic        dP/dh = -rho g
    ideal gas          P = rho k T / (mu m_u)
    adiabatic          P^(1-gamma) T^gamma = const  ->  dT/dh = -(gamma - 1)/gamma * mu m_u g / k
    so                 d ln(rho)/dh = -(mu m_u g / (k T)) - (1/T) dT/dh
We integrate (ln rho, T) upwards with an adaptive Runge-Kutta solver, which
stops exactly where T = T0/2 (an event) and continues isothermally. Working
with ln(rho) keeps the solution accurate over the 10^-13 range it spans.
density(h) and temperature(h) interpolate the solution (cubic in ln rho).

Checks:
  * With uniform gravity g0 the model has a closed form: T falls linearly
    (lapse rate Gamma = (gamma-1)/gamma mu m_u g0 / k), rho = rho0 (T/T0)^(1/(gamma-1)),
    then rho decays exponentially with scale height H = k (T0/2) / (mu m_u g0).
    The numerical solution must reproduce it when run with uniform gravity, and
    the difference with real gravity answers "how good is uniform g?".
  * The drag probe from challenge A measured the density between ~250 and
    ~340 km: with F = 1/2 rho C_d A v_rel^2 (C_d = 1, Part 7) and air that
    co-rotates with the planet, a circular orbit loses
        delta a per orbit = 2 pi rho (A/m) a^2 (v_rel / v)^2.
    The same model with mu free, fitted to those densities, gives mu
    independently of the spectrum.

Composition: the spectrum (challenge B) gives the detected gases and any
marginal ones. Each allowed composition gives a mu (equal amounts of each gas);
we adopt the one the drag measurement agrees with.

Gravity: the assignment allows uniform g. The drag data decide: with uniform g
the fitted model matches both the level and the slope of the measured
densities; with g falling with height the scale height near 300 km comes out
~15% too large. So the adopted model uses uniform g (also ast2000tools'
default, begin_landing_sequence(assume_uniform_gravity_atmosphere=True)), and
c3_gravity shows what falling gravity would change.

Run:  ../.venv/bin/python atmosphere.py   (after spectrum.py)
Figures go to figures/c*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(3)

import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as sc
from scipy import integrate, interpolate, optimize

import low_orbit
from common.config import SEED
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, SERIES, VIOLET, save
from common.report import record, record_many
from habitable_zone import Star, planet_temperature
from spectrum import LINES, RESULT_FILE, mean_molecular_weight

GAMMA = 1.4
MU_TOLERANCE = 0.03  # The drag probe's mu and the spectrum's composition must agree this well
H_MAX = 600e3  # Model top [m]; above this the isothermal formula continues it
PROBE_HEIGHTS = np.arange(200e3, 400e3, 5e3)  # Drag-probe altitudes used for the density check [m]
U = sc.atomic_mass


class Atmosphere:
    """Temperature, density and pressure against height, from the numerical model."""

    def __init__(self, planet, T0, mu, gamma=GAMMA, uniform_gravity=True, h_max=H_MAX):
        self.planet, self.T0, self.mu, self.gamma = planet, T0, mu, gamma
        self.uniform_gravity = uniform_gravity
        self.m = mu * U  # Mean molecule mass [kg]
        h, ln_rho, T, self.h_iso = self._integrate(h_max)
        self._ln_rho = interpolate.CubicSpline(h, ln_rho)
        self._T = interpolate.interp1d(h, T)
        self.h_max = h_max
        self.H_top = sc.k * T[-1] / (self.m * self.gravity(h_max))  # Scale height used above h_max

    def gravity(self, h):
        g0 = self.planet.GM / self.planet.R ** 2
        return np.full_like(np.asarray(h, dtype=float), g0) if self.uniform_gravity else \
            self.planet.GM / (self.planet.R + np.asarray(h, dtype=float)) ** 2

    def _rhs(self, h, y, adiabatic):
        ln_rho, T = y
        g = self.gravity(h)
        dT = -(self.gamma - 1) / self.gamma * self.m * g / sc.k if adiabatic else 0.0
        return [-self.m * g / (sc.k * T) - dT / T, dT]

    def _integrate(self, h_max, step=200.0):
        """Adiabatic layer up to the event T = T0/2, then the isothermal layer; sampled every `step` metres."""
        half = lambda h, y, adiabatic: y[1] - self.T0 / 2
        half.terminal = True
        y0 = [np.log(self.planet.rho0), self.T0]
        low = integrate.solve_ivp(self._rhs, (0, h_max), y0, args=(True,), events=half, dense_output=True,
                                  rtol=1e-11, atol=1e-11, method="DOP853")
        h_iso = low.t_events[0][0]
        y_iso = low.y_events[0][0]
        high = integrate.solve_ivp(self._rhs, (h_iso, h_max), y_iso, args=(False,), dense_output=True,
                                   rtol=1e-11, atol=1e-11, method="DOP853")
        h = np.arange(0, h_max + step, step)
        y = np.where(h <= h_iso, low.sol(np.minimum(h, h_iso)), high.sol(np.maximum(h, h_iso)))
        return h, y[0], y[1], h_iso

    # ----------------------------------------------------------- the profiles
    def temperature(self, h):
        """Temperature [K] at height h [m] (scalar or array)."""
        h = np.asarray(h, dtype=float)
        return np.where(h <= self.h_max, self._T(np.clip(h, 0, self.h_max)), self.T0 / 2)

    def density(self, h):
        """Density [kg/m^3] at height h [m] (scalar or array)."""
        h = np.asarray(h, dtype=float)
        inside = self._ln_rho(np.clip(h, 0, self.h_max))
        return np.exp(np.where(h <= self.h_max, inside, inside - (h - self.h_max) / self.H_top))

    def pressure(self, h):
        return self.density(h) * sc.k * self.temperature(h) / self.m


def analytic(planet, T0, mu, h, gamma=GAMMA):
    """Closed form with uniform gravity g0: linear temperature, polytropic then exponential density."""
    g0 = planet.GM / planet.R ** 2
    lapse = (gamma - 1) / gamma * mu * U * g0 / sc.k
    h_iso = T0 / 2 / lapse
    H = sc.k * T0 / 2 / (mu * U * g0)
    T = np.maximum(T0 - lapse * h, T0 / 2)
    rho_iso = planet.rho0 * 0.5 ** (1 / (gamma - 1))
    rho = np.where(h <= h_iso, planet.rho0 * (T / T0) ** (1 / (gamma - 1)), rho_iso * np.exp(-(h - h_iso) / H))
    return T, rho, {"lapse": lapse, "h_iso": h_iso, "H": H, "rho_iso": rho_iso}


# ===================================================================== drag check
def probe_densities(heights=PROBE_HEIGHTS, floor_height=600e3):
    """
    Density [kg/m^3] at each height from the drag probe (orbital decay per orbit); NaN where unusable.
    The API's integrator adds a tiny steady drift (~ -0.012 m per orbit), measured at floor_height where
    there is no drag, and subtracted.
    """
    mission, planet, fuel, u = low_orbit.load()
    orbiter = low_orbit.Orbiter(mission, fuel, u)
    decay, _ = low_orbit.decay_probe(orbiter, planet, np.append(heights, floor_height))
    decay, drift = decay[:-1] - decay[-1], decay[-1]
    floor = 0.05  # [m per orbit] use only decays well above the drift
    a = planet.R + heights
    v = np.sqrt(planet.GM / a)
    v_rel = v - 2 * np.pi / planet.rotation_period * a  # Prograde equatorial orbit, co-rotating air
    rho = decay * mission.spacecraft_mass / (2 * np.pi * mission.spacecraft_area * a ** 2 * (v_rel / v) ** 2)
    return np.where(np.isfinite(decay) & (decay > floor), rho, np.nan), drift


def fit_mu(planet, T0, heights, rho, uniform_gravity=True):
    """mu that makes the model match the probe densities (least squares in ln rho), with its 1-sigma error."""
    ok = np.isfinite(rho)
    residual = lambda p: np.log(Atmosphere(planet, T0, p[0], uniform_gravity=uniform_gravity,
                                           h_max=heights[ok].max() + 10e3).density(heights[ok])) - np.log(rho[ok])
    fit = optimize.least_squares(residual, [25.0], bounds=([10.0], [60.0]))
    scatter = np.sqrt(np.sum(fit.fun ** 2) / max(ok.sum() - 1, 1))
    error = scatter / np.sqrt(fit.jac.T @ fit.jac)[0, 0]
    return fit.x[0], error, scatter


def compositions(detected, marginal):
    """Every composition the spectrum allows: the detected gases plus any subset of the marginal ones."""
    options = []
    for k in range(len(marginal) + 1):
        for extra in itertools.combinations(marginal, k):
            gases = sorted(set(detected) | set(extra), key=list(LINES).index)
            options.append((gases, mean_molecular_weight(gases)))
    return options


def missed_gases(detected, mu_probe):
    """
    When no composition the spectrum allows matches the drag probe, the spectrum has missed gases whose
    lines are too weak to see. The detected gases plus the fewest undetected ones that come closest to
    mu_probe; returns (gases, mu, the other undetected gases that would give the same mu).
    """
    undetected = [g for g in LINES if g not in detected]
    candidates = [(sorted(set(detected) | set(extra), key=list(LINES).index), set(extra))
                  for k in range(1, len(undetected) + 1) for extra in itertools.combinations(undetected, k)]
    best, extra = min(candidates, key=lambda c: (round(abs(mean_molecular_weight(c[0]) - mu_probe), 2), len(c[1])))
    mu = mean_molecular_weight(best)
    twins = {g for gases, e in candidates if len(e) == len(extra) and e != extra
             and abs(mean_molecular_weight(gases) - mu) < 1e-9 for g in e - extra}
    return best, mu, sorted(twins)


# ===================================================================== figures
def label(gases):
    return " + ".join(g.replace("2", "$_2$").replace("4", "$_4$") for g in gases)


def plot_profiles(model, varying, ana_T, ana_rho, h, low_orbit_h):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    km = h / 1e3
    ax = axes[0]
    ax.plot(model.temperature(h), km, color=ORANGE, label="numerical model (uniform g)")
    ax.plot(varying.temperature(h), km, color=MUTED, lw=1, label="numerical, g falling with height")
    ax.plot(ana_T, km, color=INK, ls="--", lw=1, label="closed form (uniform g)")
    ax.axhline(model.h_iso / 1e3, color=MUTED, ls=":", lw=1)
    ax.text(model.T0 * 0.75, model.h_iso / 1e3, f" adiabatic below, isothermal above ({model.h_iso / 1e3:.1f} km)",
            va="bottom", fontsize=8, color=MUTED)
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel("height above the surface [km]")
    ax.set_title(f"T falls {((model.gamma - 1) / model.gamma * model.m * model.gravity(0) / sc.k) * 1e3:.2f} K/km, "
                 f"then stays at T$_0$/2 = {model.T0 / 2:.0f} K", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[1]
    ax.semilogx(model.density(h), km, color=BLUE, label="numerical model (uniform g, adopted)")
    ax.semilogx(varying.density(h), km, color=MUTED, lw=1, label="numerical, g falling with height")
    ax.semilogx(ana_rho, km, color=INK, ls="--", lw=1, label="closed form (uniform g)")
    ax.axhline(low_orbit_h / 1e3, color=VIOLET, lw=1.2, label=f"our low orbit ({low_orbit_h / 1e3:.0f} km)")
    ax.set_xlabel("density [kg/m$^3$]")
    ax.set_title(f"Density: {model.planet.rho0:.2f} kg/m$^3$ at the surface", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[2]
    ax.semilogx(model.pressure(h) / 1e5, km, color=AQUA)
    ax.set_xlabel("pressure [bar]")
    ax.set_title(f"Pressure: {model.pressure(0) / 1e5:.2f} bar at the surface", fontsize=10)
    axes[0].set_ylim(0, h[-1] / 1e3)
    fig.suptitle(f"C  The atmosphere of planet {model.planet.index} ($\\mu$ = {model.mu:.2f}, T$_0$ = {model.T0:.0f} K)",
                 fontsize=12)
    save(fig, "c1_profiles")


def plot_drag_check(planet, T0, heights, rho, options, mu_fit, mu_err, adopted, mu_fit_g, mu_err_g):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ok = np.isfinite(rho)
    h = np.linspace(heights[ok].min() - 30e3, heights[ok].max() + 30e3, 200)
    ax1.semilogy(heights[ok] / 1e3, rho[ok], "o", color=INK, ms=6, label="measured by orbital decay (challenge A)",
                 zorder=3)
    for (gases, mu), color in zip(options, SERIES):
        model = Atmosphere(planet, T0, mu, h_max=h.max())
        ax1.semilogy(h / 1e3, model.density(h), color=color, lw=2 if gases == adopted else 1.3,
                     label=f"model, {label(gases)} ($\\mu$ = {mu:.2f})")
    model = Atmosphere(planet, T0, mu_fit_g, uniform_gravity=False, h_max=h.max())
    ax1.semilogy(h / 1e3, model.density(h), color=MUTED, ls="--", lw=1.2,
                 label=f"model with g falling with height, best fit $\\mu$ = {mu_fit_g:.2f}")
    ax1.set_xlabel("height [km]")
    ax1.set_ylabel("density [kg/m$^3$]")
    ax1.set_title("Model curves use uniform g. The spacecraft weighs the upper atmosphere: "
                  "$\\Delta a = 2\\pi\\rho (A/m) a^2 (v_{rel}/v)^2$", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    names = [label(g) for g, _ in options] + ["drag, uniform g", "drag, g falling with height"]
    values = [mu for _, mu in options] + [mu_fit, mu_fit_g]
    y = np.arange(len(names))[::-1]
    ax2.errorbar(values[-2], y[-2], xerr=mu_err, fmt="o", color=INK, ms=8, capsize=5)
    ax2.errorbar(values[-1], y[-1], xerr=mu_err_g, fmt="o", color=MUTED, ms=8, capsize=5)
    if options:
        ax2.scatter(values[:-2], y[:-2], color=SERIES[:len(options)], s=90, edgecolor=INK, zorder=3)
    ax2.axvspan(mu_fit - mu_err, mu_fit + mu_err, color=MUTED, alpha=0.15, lw=0)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("mean molecular weight $\\mu$")
    ax2.set_title("Which composition agrees with the drag?", fontsize=10)
    ax2.grid(axis="y", visible=False)
    fig.suptitle("C  Checking the model against the drag probe", fontsize=12)
    save(fig, "c2_drag_check")


def plot_gravity(planet, T0, mu, h):
    """How good is uniform gravity? The atmosphere is thin, but density errors grow with height."""
    real = Atmosphere(planet, T0, mu, uniform_gravity=False, h_max=h.max())
    flat = Atmosphere(planet, T0, mu, uniform_gravity=True, h_max=h.max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    ax1.plot(h / 1e3, real.gravity(h) / real.gravity(0), color=BLUE)
    ax1.set_xlabel("height [km]")
    ax1.set_ylabel("g(h) / g(0)")
    ax1.set_title(f"Gravity at height h: {(1 - real.gravity(100e3) / real.gravity(0)):.1%} weaker at 100 km", fontsize=10)
    ax2.plot(h / 1e3, real.density(h) / flat.density(h), color=ORANGE)
    ax2.set_xlabel("height [km]")
    ax2.set_ylabel(r"$\rho$(real g) / $\rho$(uniform g)")
    ax2.set_yscale("log")
    ax2.set_title("...but over many scale heights the density difference compounds\n"
                  "(the drag probe prefers uniform g, figure c2)", fontsize=10)
    fig.suptitle("C  Is uniform gravity a good approximation?", fontsize=12)
    save(fig, "c3_gravity")
    return real, flat


# ===================================================================== main
def destination_temperature(system, planet):
    """T0: Part 3's black-body surface temperature of the destination, at its mean distance."""
    star = Star(system)
    return float(planet_temperature(star.T, star.R, system.semi_major_axes[planet.index]))


def main():
    mission, planet, _, _ = low_orbit.load()
    T0 = destination_temperature(mission.system, planet)
    with open(RESULT_FILE) as f:
        spectral = json.load(f)
    # Without this seed's spectrum (B skipped, or a file from another seed) the drag probe alone gives mu
    has_spectrum = spectral.get("seed") == SEED and bool(spectral["detected"])
    options = compositions(spectral["detected"], spectral["marginal"]) if has_spectrum else []
    heights = PROBE_HEIGHTS
    rho, drift = probe_densities(heights)
    mu_fit, mu_err, scatter = fit_mu(planet, T0, heights, rho)  # Uniform gravity
    mu_fit_g, mu_err_g, scatter_g = fit_mu(planet, T0, heights, rho, uniform_gravity=False)
    adopted, mu = min(options, key=lambda o: abs(o[1] - mu_fit)) if options else ([], mu_fit)
    missed, twins = [], []
    if not has_spectrum:
        print(f"C     no spectral analysis for seed {SEED}: adopting the drag probe's mu = {mu_fit:.2f} directly")
    elif abs(mu / mu_fit - 1) > MU_TOLERANCE:
        # The drag weighs the real air: trust it over the spectrum's non-detections
        best, mu_best, twins = missed_gases(spectral["detected"], mu_fit)
        missed = [g for g in best if g not in spectral["detected"]]
        print(f"C     WARNING: the closest composition the spectrum allows (mu = {mu:.2f}) is {mu / mu_fit - 1:+.0%} "
              f"from the drag probe's {mu_fit:.2f}: the spectrum has missed gas. Adding undetected "
              f"{'+'.join(missed)}" + (f" (or {'/'.join(twins)}: same mass)" if twins else "") + f" -> mu = {mu_best:.2f}")
        adopted, mu = best, mu_best
        options = options + [(best, mu_best)]
    print(f"C     T0 = {T0:.1f} K (Part 3), rho0 = {planet.rho0:.3f} kg/m^3, gamma = {GAMMA}")
    for gases, m in options:
        print(f"      spectrum allows {'+'.join(gases):<20} mu = {m:.2f}" + ("   <- adopted" if gases == adopted else ""))
    print(f"      drag probe ({np.isfinite(rho).sum()} heights, integrator drift {drift:+.4f} m/orbit removed): "
          f"mu = {mu_fit:.2f} +- {mu_err:.2f} with uniform g (scatter {scatter:.2f} in ln rho), "
          f"{mu_fit_g:.2f} +- {mu_err_g:.2f} with g falling with height (scatter {scatter_g:.2f})")

    model = Atmosphere(planet, T0, mu)
    h = np.linspace(0, 450e3, 901)
    ana_T, ana_rho, ana = analytic(planet, T0, mu, h)
    flat = model  # The adopted model already uses uniform gravity
    check = np.max(np.abs(flat.density(h) / ana_rho - 1))
    print(f"      numerical (uniform g) vs closed form: largest relative density difference {check:.1e}")
    print(f"      adiabatic up to {model.h_iso / 1e3:.2f} km (lapse {ana['lapse'] * 1e3:.2f} K/km), then isothermal at "
          f"{T0 / 2:.1f} K with H = {ana['H'] / 1e3:.2f} km (surface g)")
    for hh in (0, 10e3, model.h_iso, 100e3, 300e3, 360e3):
        print(f"      h = {hh / 1e3:6.1f} km: T = {model.temperature(hh):6.1f} K, rho = {model.density(hh):.3e} kg/m^3")
    varying = Atmosphere(planet, T0, mu, uniform_gravity=False)
    plot_profiles(model, varying, ana_T, ana_rho, h, low_orbit_altitude())
    plot_drag_check(planet, T0, heights, rho, options, mu_fit, mu_err, adopted, mu_fit_g, mu_err_g)
    real, uniform = plot_gravity(planet, T0, mu, h)

    record_many({
        "T0": (T0, "{:.0f}"), "gamma": (GAMMA, "{:g}"), "mu-adopted": (mu, "{:.2f}"),
        "composition": (label(adopted) if adopted else "unknown (no spectrum; drag probe only)")
                       + (f" ({label(missed)} added: too weak for the spectrum, required by the drag)" if missed else ""),
        "n-options": (len(options), "{:d}"),
        "mu-detected": (options[0][1] if options else np.nan, "{:.2f}"), "mu-probe": (mu_fit, "{:.2f}"), "mu-probe-err": (mu_err, "{:.2f}"),
        "probe-scatter": (scatter, "{:.2f}"), "mu-probe-g": (mu_fit_g, "{:.2f}"), "mu-probe-g-err": (mu_err_g, "{:.2f}"),
        "probe-scatter-g": (scatter_g, "{:.2f}"), "probe-drift": (drift, "{:+.4f}"),
        "H-varying-300": (sc.k * T0 / 2 / (mu * U * planet.GM / (planet.R + 300e3) ** 2) / 1e3, "{:.2f}"), "n-probe-heights": (int(np.isfinite(rho).sum()), "{:d}"),
        "probe-h-lo": (heights[np.isfinite(rho)].min() / 1e3, "{:.0f}"),
        "probe-h-hi": (heights[np.isfinite(rho)].max() / 1e3, "{:.0f}"),
        "h-iso": (model.h_iso / 1e3, "{:.1f}"), "lapse": (ana["lapse"] * 1e3, "{:.2f}"), "H-iso": (ana["H"] / 1e3, "{:.2f}"),
        "T-iso": (T0 / 2, "{:.0f}"), "rho-iso": (model.density(model.h_iso), "{:.3f}"),
        "P0": (model.pressure(0) / 1e5, "{:.2f}"),
        "rho-100": (model.density(100e3), "{:.2e}"), "rho-300": (model.density(300e3), "{:.2e}"),
        "rho-low": (model.density(low_orbit_altitude()), "{:.2e}"),
        "closed-form-check": (check, "{:.0e}"),
        "g-drop-100": ((1 - real.gravity(100e3) / real.gravity(0)) * 100, "{:.1f}"),
        "uniform-ratio-100": (real.density(100e3) / uniform.density(100e3), "{:.2f}"),
        "uniform-ratio-300": (real.density(300e3) / uniform.density(300e3), "{:.1f}"),
        "probe-ratio-detected": (np.nanmedian(Atmosphere(planet, T0, options[0][1]).density(heights) / rho)
                                 if options else np.nan, "{:.0f}"),
        "probe-ratio-adopted": (np.nanmedian(model.density(heights) / rho), "{:.2f}"),
    })
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "atmosphere_model.json"), "w") as f:
        json.dump({"T0": T0, "mu": mu, "gamma": GAMMA, "composition": adopted}, f, indent=2)
    return model


def low_orbit_altitude():
    """The low orbit chosen in challenge A (read back from its recorded results)."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "results.json")) as f:
        return float(json.load(f)["low-alt"]) * 1e3


if __name__ == "__main__":
    main()
