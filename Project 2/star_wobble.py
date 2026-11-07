"""
Part 2, challenge C: Can extraterrestrials discover the planets in our system?

Now the star moves too. These orbits are ONLY for this challenge.

C1  Star + one planet. We pick the planet that makes the star wobble the most
    (star speed ~ m / sqrt(a): the heaviest planet close in). Both bodies are
    integrated with leapfrog in the centre-of-mass (CM) frame:
        R_cm = (M r_star + m r_planet) / (M + m)
    Starting with the star at rest at the origin makes the CM drift with
    V_cm = m v_planet / (M + m); subtracting it gives the CM frame.
    Energy check: E = 1/2 mu v^2 - G M mu / r (reduced mass mu, relative v, r).

C2  Radial velocity curve: v_r = v_pec + v_star,x sin(i), observer along +x,
    plus Gaussian noise with sigma = max|signal| / 5. We then analyse the
    curve as an outside observer would (subtract v_pec, read amplitude and
    period) and get the planet's minimum mass
        m sin i = M^(2/3) v_star P^(1/3) / (2 pi G)^(1/3).
    The curve is exported to rv_curve.txt for the partner group, and
    `--analyse-rv FILE --star-mass M` analyses one you receive.

C3  Light curve (i = 90 deg): flux drops by (R_p / R_star)^2 during transit,
    linear during ingress/egress (as the assignment suggests). Noise 1e-4.
    From depth and ingress time we estimate the planet's radius and density.

C4  Radial velocity with several planets (star + the three gas giants, full
    N-body). A periodogram finds the periods hidden in the noisy curve.

Run:  ../.venv/bin/python star_wobble.py
Figures go to figures/c*.png, data for the partner group to rv_curve.txt and light_curve.txt.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import argparse

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal

from common.config import get_mission
from common.integrators import leapfrog
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, PLANET_COLORS, VIOLET, save
from common.report import record, record_many
from orbits import G

HERE = os.path.dirname(os.path.abspath(__file__))
AU_PER_YR = const.AU / const.yr  # [m/s] per [AU/yr]
STEPS_PER_YEAR = 4000
SEED = 2000

# Our free choices for the observer (C2)
INCLINATION = np.radians(70)  # Angle between the line of sight and the orbital plane's normal (90 = edge-on)
PECULIAR_VELOCITY = 2500.0  # [m/s] CM velocity along the line of sight
N_OBSERVATIONS = 1500  # Number of radial-velocity measurements over the observation period
NOISE_FRACTION = 1 / 5  # Noise sigma relative to the maximum of the signal
LIGHT_NOISE = 1e-4  # Light curve noise (relative flux)


# ================================================================== N-body
def simulate_nbody(masses, x0, v0, duration, steps_per_year=STEPS_PER_YEAR, save_every=10):
    """
    Leapfrog for n bodies that all attract each other, in the CM frame.

    masses (n,) [Msun], x0, v0 (n, 2) [AU, AU/yr] in any frame. The CM
    position and velocity are removed first, so the CM stays at rest at the
    origin. Returns times (n_out,), positions and velocities (n_out, n, 2),
    and the CM's original position and velocity (to undo the shift if wanted).
    """
    masses = np.asarray(masses, dtype=float)
    total = masses.sum()
    r_cm = masses @ x0 / total
    v_cm = masses @ v0 / total
    x = x0 - r_cm
    v = v0 - v_cm

    def acceleration(x):
        d = x[None, :, :] - x[:, None, :]  # d[i, j] = x_j - x_i
        dist3 = np.sum(d ** 2, axis=2) ** 1.5
        np.fill_diagonal(dist3, np.inf)
        return G * np.sum(masses[None, :, None] * d / dist3[:, :, None], axis=1)

    n_out = int(np.ceil(duration * steps_per_year / save_every))
    dt = duration / (n_out * save_every)
    positions, velocities = leapfrog(acceleration, x, v, dt, n_out, save_every)
    return np.linspace(0, duration, n_out + 1), positions, velocities, r_cm, v_cm


def star_and_planets(system, planets):
    """Masses and initial state with the star at rest at the origin, as the assignment says."""
    masses = np.concatenate([[system.star_mass], system.masses[planets]])
    x0 = np.vstack([[0.0, 0.0], system.initial_positions[:, planets].T])
    v0 = np.vstack([[0.0, 0.0], system.initial_velocities[:, planets].T])
    return masses, x0, v0


# ======================================================================= C1
def run_c1(system):
    M = system.star_mass
    wobble = system.masses / np.sqrt(system.semi_major_axes)  # Star speed ~ m sqrt(G/a)
    planet = int(np.argmax(wobble))
    print(f"C1    star speed ~ m/sqrt(a): best planet is {planet} (m = {system.masses[planet]:.3e} Msun, "
          f"a = {system.semi_major_axes[planet]:.2f} AU)")
    m = system.masses[planet]
    P = 2 * np.pi * np.sqrt(system.semi_major_axes[planet] ** 3 / (G * (M + m)))
    duration = 3 * P
    masses, x0, v0 = star_and_planets(system, [planet])
    times, pos, vel, r_cm, v_cm = simulate_nbody(masses, x0, v0, duration)
    print(f"      simulated {duration:.1f} yr (3 orbits); CM drift in the start frame: {np.linalg.norm(v_cm) * AU_PER_YR:.2f} m/s")

    # Energy of the two-body system from the relative motion
    mu = M * m / (M + m)
    r_rel = pos[:, 1] - pos[:, 0]
    v_rel = vel[:, 1] - vel[:, 0]
    energy = 0.5 * mu * np.sum(v_rel ** 2, axis=1) - G * (M + m) * mu / np.linalg.norm(r_rel, axis=1)
    print(f"      energy E = 1/2 mu v^2 - G M mu / r: max |E/E0 - 1| = {np.max(np.abs(energy / energy[0] - 1)):.2e}")

    # Period of the relative orbit vs Newton's 3rd law. The relative orbit's semi-major axis follows
    # from the energy, E = -G M mu / (2a), with M = M_star + m (not the fixed-star value).
    angle = np.unwrap(np.arctan2(r_rel[:, 1], r_rel[:, 0]))
    turns = (angle - angle[0]) / (2 * np.pi)
    crossings = np.interp(np.arange(1, int(turns[-1]) + 1), turns, times)
    P_sim = crossings[-1] / len(crossings)
    a_rel = -G * (M + m) * mu / (2 * energy[0])
    P_newton = 2 * np.pi * np.sqrt(a_rel ** 3 / (G * (M + m)))
    P_kepler = 2 * np.pi * np.sqrt(a_rel ** 3 / (G * M))
    print(f"      relative orbit: a = {a_rel:.5f} AU; period simulated {P_sim:.5f} yr, Newton P^2 = a^3/(M+m) "
          f"{P_newton:.5f} yr, Kepler P^2 = a^3/M {P_kepler:.5f} yr")
    print(f"      -> Newton matches to {abs(P_sim / P_newton - 1):.1e}; Kepler is off by {abs(P_sim / P_kepler - 1):.1e} "
          f"({(P_kepler - P_sim) * 365.25:.1f} days): Newton's correction is measurable here")

    star_speed = np.linalg.norm(vel[:, 0], axis=1) * AU_PER_YR
    record_many({"c1-planet": (planet, "{:d}"), "c1-mass": (m, "{:.3e}"), "c1-a": (system.semi_major_axes[planet], "{:.2f}"),
                 "c1-star-speed": (star_speed.mean(), "{:.1f}"),
                 "c1-energy-err": (np.max(np.abs(energy / energy[0] - 1)), "{:.1e}"),
                 "c1-P-sim": (P_sim, "{:.4f}"), "c1-P-newton": (P_newton, "{:.4f}"), "c1-P-kepler": (P_kepler, "{:.4f}"),
                 "c1-kepler-days": ((P_kepler - P_sim) * 365.25, "{:.1f}"),
                 "c1-wobble-km": (np.ptp(pos[:, 0, 0]) * const.AU / 1e3 / 2, "{:,.0f}")})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
    ax = axes[0]
    start_frame = pos + r_cm + v_cm * times[:, None, None]  # Undo the CM shift: the frame the star started at rest in
    ax.plot(start_frame[:, 0, 0] * 1e3, start_frame[:, 0, 1] * 1e3, color="#eda100", lw=1.5, label="star (start frame)")
    ax.plot(pos[:, 0, 0] * 1e3, pos[:, 0, 1] * 1e3, color=INK, lw=1.5, label="star (CM frame)")
    ax.plot(0, 0, "+", color=INK, ms=12, mew=2, label="centre of mass")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mAU]")
    ax.set_ylabel("y [mAU]")
    ax.set_title("The star's wobble: drifting in the start frame,\nan ellipse in the CM frame")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(pos[:, 1, 0], pos[:, 1, 1], color=PLANET_COLORS[planet], lw=1.5, label=f"planet {planet}")
    ax.plot(pos[:, 0, 0], pos[:, 0, 1], color=INK, lw=3,
            label=f"star (orbit radius only {np.ptp(pos[:, 0, 0]) * const.AU / 1e9 / 2:.1f} million km)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title(f"Both orbit the CM (star speed {star_speed.mean():.1f} m/s)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    ax.plot(times, energy / energy[0] - 1, color=VIOLET, lw=1.2)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("time [yr]")
    ax.set_ylabel(r"$E/E_0 - 1$")
    ax.set_title(r"Energy $\frac{1}{2}\hat\mu v^2 - GM\hat\mu/r$ is conserved")
    fig.suptitle(f"C1  Star and planet {planet} in the centre-of-mass frame", fontsize=12)
    save(fig, "c1_star_orbit")
    return planet, times, pos, vel


# ======================================================================= C2
def radial_velocity_curve(times, star_velocity, rng, inclination=INCLINATION, v_pec=PECULIAR_VELOCITY,
                          n_obs=N_OBSERVATIONS):
    """Observed v_r [m/s] at n_obs times, with noise. star_velocity (n_times, 2) in AU/yr, CM frame."""
    idx = np.linspace(0, len(times) - 1, n_obs).astype(int)
    signal_true = star_velocity[idx, 0] * AU_PER_YR * np.sin(inclination)
    noise = rng.normal(0, NOISE_FRACTION * np.max(np.abs(signal_true)), size=n_obs)
    return times[idx], v_pec + signal_true, v_pec + signal_true + noise


def smooth(y, n):
    """Running mean over n points: what the eye does when reading a noisy curve."""
    return np.convolve(y, np.ones(n) / n, mode="same")


def analyse_rv(t, v, star_mass):
    """
    Read a noisy radial-velocity curve like an observer. t [yr], v [m/s], star_mass [Msun].

    1) v_pec = mean of the data (removes the drift)
    2) smooth, keep only clear maxima/minima (prominent, like the eye ignores
       small noise bumps); period = spacing of the maxima, amplitude = half the
       average peak-to-trough height
    3) a sine fit as a numerical cross-check
    4) m sin i from the amplitude and period (assuming m << M and a circular orbit)
    """
    v_pec = np.mean(v)
    dv = v - v_pec
    window = max(3, len(t) // 30)
    s = smooth(dv, window)
    inner = slice(window, len(t) - window)  # Avoid the edges, where the running mean is biased
    prominence = 0.5 * np.ptp(s[inner])
    peaks = signal.find_peaks(s[inner], prominence=prominence)[0] + window
    troughs = signal.find_peaks(-s[inner], prominence=prominence)[0] + window
    extrema = np.sort(np.concatenate([peaks, troughs]))
    # Period: maxima are one period apart, and so are minima; a max and the next min half a period
    P_eye = 2 * np.mean(np.diff(t[extrema])) if len(extrema) > 1 else np.ptp(t)
    amp_eye = 0.5 * (np.mean(s[peaks]) - np.mean(s[troughs]))

    model = lambda t, A, P, phi, c: A * np.sin(2 * np.pi * t / P + phi) + c
    (A, P_fit, phi, c), _ = optimize.curve_fit(model, t, v, p0=[amp_eye, P_eye, 0, v_pec])
    A = abs(A)

    def min_mass(amp, P):
        return (star_mass * const.m_sun) ** (2 / 3) * amp * (P * const.yr) ** (1 / 3) / (2 * np.pi * const.G) ** (1 / 3)

    return {"v_pec": v_pec, "P_eye": P_eye, "amp_eye": amp_eye, "peaks": extrema, "smooth": s + v_pec,
            "A_fit": A, "P_fit": P_fit,
            "m_sin_i_eye": min_mass(amp_eye, P_eye) / const.m_sun, "m_sin_i_fit": min_mass(A, P_fit) / const.m_sun}


def plot_rv(t, v_true, v_obs, result, planet, true_m_sin_i, name="c2_radial_velocity", title=None):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, v_obs, ".", color=MUTED, ms=2.5, alpha=0.6, label="observations (with noise)")
    ax.plot(t, result["smooth"], color=BLUE, lw=2, label="smoothed (what the eye sees)")
    ax.plot(t, v_true, color=INK, lw=1, ls="--", label="true signal")
    ax.axhline(result["v_pec"], color=ORANGE, lw=1.5, label=f"peculiar velocity = mean = {result['v_pec']:.0f} m/s")
    for k in (1, -1):
        ax.axhline(result["v_pec"] + k * result["amp_eye"], color=AQUA, lw=1.2, ls=":")
    ax.plot(t[result["peaks"]], result["smooth"][result["peaks"]], "v", color=AQUA, ms=10,
            label=f"read off: P = {result['P_eye']:.2f} yr, amplitude = {result['amp_eye']:.1f} m/s")
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("radial velocity [m/s]")
    ax.set_title(title or (f"C2  Radial velocity of our star (i = {np.degrees(INCLINATION):.0f}°): "
                           f"m sin i = {result['m_sin_i_eye']:.2e} Msun read by eye, "
                           f"true {true_m_sin_i:.2e} Msun"))
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    save(fig, name)


def run_c2(system, planet, times, vel, rng):
    t, v_true, v_obs = radial_velocity_curve(times, vel[:, 0], rng)
    path = os.path.join(HERE, "rv_curve.txt")
    np.savetxt(path, np.column_stack([t, v_obs]), fmt="%.6f",
               header=f"Radial velocity of our star. Star mass = {system.star_mass:.6f} Msun\ntime [yr]  v_r [m/s]")
    print(f"C2    radial velocity curve: i = {np.degrees(INCLINATION):.0f} deg, v_pec = {PECULIAR_VELOCITY:.0f} m/s, "
          f"noise sigma = {NOISE_FRACTION * np.max(np.abs(v_true - PECULIAR_VELOCITY)):.1f} m/s -> {os.path.basename(path)}")

    result = analyse_rv(t, v_obs, system.star_mass)
    result["noise"] = NOISE_FRACTION * np.max(np.abs(v_true - PECULIAR_VELOCITY))
    true_m = system.masses[planet]
    print(f"      by eye:  v_pec {result['v_pec']:.1f} m/s, P {result['P_eye']:.3f} yr, amplitude {result['amp_eye']:.2f} m/s"
          f" -> m sin i = {result['m_sin_i_eye']:.3e} Msun")
    print(f"      sine fit:                       P {result['P_fit']:.3f} yr, amplitude {result['A_fit']:.2f} m/s"
          f" -> m sin i = {result['m_sin_i_fit']:.3e} Msun")
    print(f"      true: m = {true_m:.3e} Msun, m sin i = {true_m * np.sin(INCLINATION):.3e} Msun "
          f"(eye off by {result['m_sin_i_eye'] / (true_m * np.sin(INCLINATION)) - 1:+.1%})")
    plot_rv(t, v_true, v_obs, result, planet, true_m * np.sin(INCLINATION))
    record_many({"c2-incl": (np.degrees(INCLINATION), "{:.0f}"), "c2-vpec": (PECULIAR_VELOCITY, "{:.0f}"),
                 "c2-noise": (NOISE_FRACTION * np.max(np.abs(v_true - PECULIAR_VELOCITY)), "{:.1f}"),
                 "c2-nobs": (N_OBSERVATIONS, "{:d}"),
                 "c2-P-eye": (result["P_eye"], "{:.2f}"), "c2-amp-eye": (result["amp_eye"], "{:.1f}"),
                 "c2-msini-eye": (result["m_sin_i_eye"], "{:.3e}"), "c2-msini-fit": (result["m_sin_i_fit"], "{:.3e}"),
                 "c2-msini-true": (true_m * np.sin(INCLINATION), "{:.3e}"),
                 "c2-eye-err": ((result["m_sin_i_eye"] / (true_m * np.sin(INCLINATION)) - 1) * 100, "{:+.1f}"),
                 "c2-P-fit": (result["P_fit"], "{:.3f}"), "c2-amp-fit": (result["A_fit"], "{:.2f}")})
    return result


# ======================================================================= C3
def transit_flux(d, R_star, R_planet):
    """Relative flux for projected star-planet distance d: linear ingress/egress, flat bottom."""
    depth = (R_planet / R_star) ** 2
    covered = np.clip((R_star + R_planet - d) / (2 * R_planet), 0, 1)  # 0 outside, 1 fully in front
    return 1 - depth * covered


def run_c3(system, planet, times, pos, vel, rng, mass_estimate):
    R_star = system.star_radius * 1e3
    R_p = system.radii[planet] * 1e3
    r_rel = (pos[:, 1] - pos[:, 0]) * const.AU
    v_rel = (vel[:, 1] - vel[:, 0]) * AU_PER_YR
    # Observer far away along +x (i = 90 deg): transit when the planet crosses y = 0 on the +x side
    crossing = np.nonzero((r_rel[:-1, 1] < 0) & (r_rel[1:, 1] >= 0) & (r_rel[:-1, 0] > 0))[0][0]
    frac = -r_rel[crossing, 1] / (r_rel[crossing + 1, 1] - r_rel[crossing, 1])
    t_c = (times[crossing] + frac * (times[crossing + 1] - times[crossing])) * const.yr  # [s]
    v_transverse = v_rel[crossing, 1]  # Straight-line motion is exact enough over two days
    # Observe +-1.5 days around the transit, one point per 2 minutes
    t = np.arange(t_c - 1.5 * const.day, t_c + 1.5 * const.day, 120.0)
    flux_true = transit_flux(np.abs(v_transverse * (t - t_c)), R_star, R_p)
    flux = flux_true + rng.normal(0, LIGHT_NOISE, size=len(t))
    hours = (t - t_c) / 3600
    np.savetxt(os.path.join(HERE, "light_curve.txt"), np.column_stack([hours, flux]), fmt="%.8f",
               header=f"Light curve of our star (i = 90 deg). Star radius = {system.star_radius:.1f} km\n"
                      "time from mid-transit [h]  relative flux")

    # Analysis: fit the trapezoid (start of ingress t1, full at t2, depth) to the noisy data
    def trapezoid(h, t1, t2, depth):
        rise = np.clip((np.abs(h) - t2) / (t1 - t2), 0, 1)
        return 1 - depth * (1 - rise)

    (t1, t2, depth), _ = optimize.curve_fit(trapezoid, hours, flux, p0=[15, 11, 0.02])
    ingress = (t1 - t2) * 3600  # [s]
    # Relative speed from the RV analysis: v_rel = v_star (1 + M/m)
    v_rel_est = mass_estimate["v_star"] * (1 + system.star_mass / mass_estimate["m"])
    R_from_ingress = v_rel_est * ingress / 2
    R_from_depth = R_star * np.sqrt(depth)
    m_kg = mass_estimate["m"] * const.m_sun
    rho = m_kg / (4 / 3 * np.pi * R_from_depth ** 3)
    rho_true = system.masses[planet] * const.m_sun / (4 / 3 * np.pi * R_p ** 3)
    print(f"C3    transit of planet {planet}: depth {depth:.5f} (true {(R_p / R_star) ** 2:.5f}), noise {LIGHT_NOISE:.0e} "
          f"-> easily detectable")
    print(f"      radius from depth: {R_from_depth / 1e3:,.0f} km, from ingress ({ingress / 3600:.2f} h at "
          f"{v_rel_est / 1e3:.2f} km/s): {R_from_ingress / 1e3:,.0f} km, true {R_p / 1e3:,.0f} km")
    print(f"      density: {rho:.0f} kg/m^3 (true {rho_true:.0f} kg/m^3)")
    home_depth = (system.radii[0] / system.star_radius) ** 2
    record_many({"c3-depth": (depth, "{:.5f}"), "c3-depth-true": ((R_p / R_star) ** 2, "{:.5f}"),
                 "c3-noise": (LIGHT_NOISE, "{:.0e}"), "c3-ingress-h": (ingress / 3600, "{:.2f}"),
                 "c3-R-depth": (R_from_depth / 1e3, "{:,.0f}"), "c3-R-ingress": (R_from_ingress / 1e3, "{:,.0f}"),
                 "c3-R-true": (R_p / 1e3, "{:,.0f}"), "c3-rho": (rho, "{:.0f}"), "c3-rho-true": (rho_true, "{:.0f}"),
                 "c3-home-depth": (home_depth, "{:.1e}")})
    print(f"      (our home planet would dip only {home_depth:.1e}, below the {LIGHT_NOISE:.0e} noise per point)")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, flux, ".", color=MUTED, ms=2.5, alpha=0.6, label=f"observations (noise {LIGHT_NOISE:.0e})")
    ax.plot(hours, flux_true, color=INK, lw=1, ls="--", label="true light curve")
    ax.plot(hours, trapezoid(hours, t1, t2, depth), color=ORANGE, lw=2, label="fitted trapezoid")
    for x in (-t1, -t2, t2, t1):
        ax.axvline(x, color=AQUA, lw=1, ls=":")
    ax.annotate("", xy=(-t2, 1 - depth * 1.15), xytext=(-t1, 1 - depth * 1.15),
                arrowprops=dict(arrowstyle="<->", color=AQUA, lw=1.5))
    ax.text(-(t1 + t2) / 2, 1 - depth * 1.2, f"ingress\n{ingress / 3600:.2f} h", ha="center", va="top", fontsize=9)
    ax.annotate("", xy=(0, 1 - depth), xytext=(0, 1), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.5))
    ax.text(0.5, 1 - depth / 2, f"depth = {depth:.4f}\n$R_p = R_\\star\\sqrt{{depth}}$ = {R_from_depth / 1e3:,.0f} km",
            fontsize=9, va="center")
    ax.set_ylim(1 - depth * 1.45, 1 + depth * 0.25)
    ax.set_xlabel("time from mid-transit [h]")
    ax.set_ylabel("relative flux")
    ax.set_title(f"C3  Light curve: planet {planet} crossing the star "
                 f"(estimated density {rho:.0f} kg/m$^3$, true {rho_true:.0f})")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "c3_light_curve")


# ======================================================================= C4
def run_c4(system, rng):
    planets = [p for p in np.argsort(system.semi_major_axes) if system.types[p] == "gas"]
    P = 2 * np.pi * np.sqrt(system.semi_major_axes[planets] ** 3 / (G * system.star_mass))
    duration = 2.3 * P.max()
    masses, x0, v0 = star_and_planets(system, planets)
    print(f"C4    star + gas giants {planets} (full N-body), {duration:.0f} yr ...")
    times, pos, vel, _, _ = simulate_nbody(masses, x0, v0, duration, steps_per_year=2000, save_every=10)
    t, v_true, v_obs = radial_velocity_curve(times, vel[:, 0], rng, n_obs=3000)

    # Each planet's contribution, from a star + that planet only (to show what is summed)
    singles = {}
    for p in planets:
        m1, x1, w1 = star_and_planets(system, [p])
        tt, _, vv, _, _ = simulate_nbody(m1, x1, w1, duration, steps_per_year=2000, save_every=10)
        singles[p] = np.interp(t, tt, vv[:, 0, 0] * AU_PER_YR * np.sin(INCLINATION))

    # Prewhitening: find the strongest period with a Lomb-Scargle periodogram, fit and subtract
    # that sine, and search the residual again. Each round can reveal one more planet.
    periods = np.logspace(np.log10(1), np.log10(duration), 3000)
    residual = v_obs - v_obs.mean()
    sine = lambda t, A, P, phi, c: A * np.sin(2 * np.pi * t / P + phi) + c
    rounds = []
    for _ in range(len(planets)):
        power = signal.lombscargle(t, residual, 2 * np.pi / periods, normalize=True)
        P_peak = periods[np.argmax(power)]
        (A, P_fit, phi, c), _ = optimize.curve_fit(sine, t, residual, p0=[np.std(residual), P_peak, 0, 0])
        rounds.append({"power": power, "P": abs(P_fit), "A": abs(A)})
        residual = residual - sine(t, A, P_fit, phi, c)
    # Each round was fitted with the other planets still in the data, which biases it.
    # Refine all sines together in one joint fit, starting from the rounds.
    def joint(t, *q):
        return q[-1] + sum(q[3 * k] * np.sin(2 * np.pi * t / q[3 * k + 1] + q[3 * k + 2]) for k in range(len(rounds)))

    guess = [x for r in rounds for x in (r["A"], r["P"], 0.0)] + [0.0]
    best, best_cost = None, np.inf
    for phases in np.random.default_rng(0).uniform(0, 2 * np.pi, size=(20, len(rounds))):  # Try several start phases
        guess[2::3] = phases
        try:
            q, _ = optimize.curve_fit(joint, t, v_obs - v_obs.mean(), p0=guess, maxfev=20000)
        except RuntimeError:
            continue
        cost = np.sum((v_obs - v_obs.mean() - joint(t, *q)) ** 2)
        if cost < best_cost:
            best, best_cost = q, cost
    joint_fit = sorted((abs(best[3 * k + 1]), abs(best[3 * k])) for k in range(len(rounds)))
    noise_sigma = NOISE_FRACTION * np.max(np.abs(v_true - PECULIAR_VELOCITY))
    print(f"      true periods {np.round(P, 2)} yr; prewhitening rounds find "
          + ", ".join(f"{r['P']:.2f} yr ({r['A']:.1f} m/s)" for r in rounds))
    print("      joint fit: " + ", ".join(f"P = {Pk:.2f} yr ({Ak:.1f} m/s)" for Pk, Ak in joint_fit))
    print(f"      residual scatter {np.sqrt(best_cost / len(t)):.1f} m/s vs. noise {noise_sigma:.1f} m/s")

    record("c4-planets", ", ".join(str(p) for p in planets))
    record("c4-true-P", ", ".join(f"{x:.2f}" for x in sorted(P)))
    record("c4-fit-P", ", ".join(f"{Pk:.2f}" for Pk, _ in joint_fit))
    record("c4-duration", duration, "{:.0f}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    ax1.plot(t, v_obs, ".", color=MUTED, ms=2, alpha=0.5, label="observations")
    ax1.plot(t, v_true, color=INK, lw=1.2, label="true total signal")
    for p in planets:
        ax1.plot(t, PECULIAR_VELOCITY + singles[p], color=PLANET_COLORS[p], lw=1.5,
                 label=f"planet {p} alone (P = {2 * np.pi * np.sqrt(system.semi_major_axes[p] ** 3 / (G * system.star_mass)):.1f} yr)")
    ax1.set_xlabel("time [yr]")
    ax1.set_ylabel("radial velocity [m/s]")
    ax1.set_title("Several planets: the curve is the sum of their wobbles")
    ax1.legend(fontsize=8, ncol=3, loc="upper right")
    for k, (r, color) in enumerate(zip(rounds, (BLUE, ORANGE, AQUA)), 1):
        ax2.semilogx(periods, r["power"], color=color, lw=1.5,
                     label=f"round {k}: strongest P = {r['P']:.2f} yr ({r['A']:.1f} m/s)")
    for p, Pp in zip(planets, P):
        ax2.axvline(Pp, color=PLANET_COLORS[p], ls="--", lw=1, label=f"true period planet {p}")
    ax2.set_xlabel("trial period [yr]")
    ax2.set_ylabel("periodogram power")
    ax2.set_title("Prewhitening: find the strongest period, subtract that sine, search again\n"
                  "joint fit of all three: " + ", ".join(f"{Pk:.2f} yr" for Pk, _ in joint_fit))
    ax2.legend(fontsize=8)
    fig.suptitle("C4  Radial velocity with more planets", fontsize=12)
    save(fig, "c4_multi_planet_rv")


# ======================================================================= C5
def plot_detectability(system, rv_noise, light_noise=LIGHT_NOISE, n_points=None):
    """
    Could an observer far away find each planet on its own? For every planet:
      RV semi-amplitude  K = (2 pi G / P)^(1/3) m sin i / M^(2/3)   (m << M, circular)
      transit depth      (R_p / R_star)^2, and the chance the orbit is lined up, R_star / a.
    A signal is detectable once its size beats the noise averaged over the
    observations: noise / sqrt(number of points in the signal).
    """
    GM = G * system.star_mass
    a = system.semi_major_axes
    P = 2 * np.pi * np.sqrt(a ** 3 / GM)
    m = system.masses * const.m_sun
    K = (2 * np.pi * const.G / (P * const.yr)) ** (1 / 3) * m * np.sin(INCLINATION) / (system.star_mass * const.m_sun) ** (2 / 3)
    depth = (system.radii / system.star_radius) ** 2
    probability = system.star_radius * 1e3 / (a * const.AU)
    n_points = n_points or N_OBSERVATIONS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    for p in range(len(a)):
        size = 11 if system.types[p] == "gas" else 8
        ax1.loglog(P[p], K[p], "o", color=PLANET_COLORS[p], ms=size, mec=INK, mew=0.5)
        ax1.annotate(f" {p}", (P[p], K[p]), fontsize=9)
        ax2.loglog(P[p], depth[p], "o", color=PLANET_COLORS[p], ms=size, mec=INK, mew=0.5)
        ax2.annotate(f" {p} ({probability[p]:.1%})", (P[p], depth[p]), fontsize=8)
    ax1.axhline(rv_noise, color=ORANGE, ls="--", lw=1.2, label=f"noise per point in C2: {rv_noise:.1f} m/s")
    ax1.axhline(rv_noise / np.sqrt(n_points), color=ORANGE, ls=":", lw=1.2,
                label=f"... averaged over {n_points} points")
    ax1.axhline(1.0, color=MUTED, ls="-.", lw=1, label="1 m/s (good real spectrograph)")
    ax1.set_xlabel("orbital period [yr]")
    ax1.set_ylabel("star's radial-velocity amplitude K [m/s]")
    ax1.set_title(f"Radial velocity (i = {np.degrees(INCLINATION):.0f}$^\\circ$): only the gas giants stand out")
    ax1.legend(fontsize=8, loc="lower left")
    ax2.axhline(light_noise, color=ORANGE, ls="--", lw=1.2, label=f"noise per point in C3: {light_noise:.0e}")
    ax2.axhline(light_noise / np.sqrt(100), color=ORANGE, ls=":", lw=1.2, label="... averaged over 100 points in transit")
    ax2.set_xlabel("orbital period [yr]")
    ax2.set_ylabel("transit depth $(R_p/R_\\star)^2$")
    ax2.set_title("Transits (label: chance the orbit is lined up, $R_\\star/a$)")
    ax2.legend(fontsize=8, loc="lower left")
    fig.suptitle("C5  Which of our planets could extraterrestrials discover?", fontsize=12)
    save(fig, "c5_detectability")
    detectable_rv = [p for p in range(len(a)) if K[p] > rv_noise / np.sqrt(n_points) * 5]
    detectable_tr = [p for p in range(len(a)) if depth[p] > light_noise / np.sqrt(100) * 5]
    return K, depth, probability, detectable_rv, detectable_tr


def main():
    rng = np.random.default_rng(SEED)
    system = get_mission().system
    planet, times, pos, vel = run_c1(system)
    rv = run_c2(system, planet, times, vel, rng)
    # C3 needs the planet's mass and the star's full speed; we know the inclination we chose in C2
    run_c3(system, planet, times, pos, vel, rng,
           {"m": rv["m_sin_i_eye"] / np.sin(INCLINATION), "v_star": rv["amp_eye"] / np.sin(INCLINATION)})
    run_c4(system, rng)
    K, depth, probability, by_rv, by_transit = plot_detectability(system, rv["noise"])
    record("c5-home-K", K[0] * 100, "{:.1f}")  # [cm/s]
    giants = [p for p in range(system.number_of_planets) if system.types[p] == "gas"]
    weakest = min(giants, key=lambda p: K[p])
    record("c5-weakest-giant", weakest, "{:d}")
    record("c5-weakest-giant-K", K[weakest], "{:.1f}")
    record("c5-max-K", K.max(), "{:.1f}")
    record("c5-by-rv", ", ".join(map(str, by_rv)) or "none")
    record("c5-by-transit", ", ".join(map(str, by_transit)) or "none")
    record("c5-home-prob", probability[0] * 100, "{:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2 challenge C")
    parser.add_argument("--analyse-rv", metavar="FILE", help="analyse a partner's radial velocity file (t [yr], v [m/s])")
    parser.add_argument("--star-mass", type=float, help="the partner's star mass [Msun]")
    args = parser.parse_args()
    if args.analyse_rv:
        t, v = np.loadtxt(args.analyse_rv, unpack=True)
        res = analyse_rv(t, v, args.star_mass)
        print(f"v_pec {res['v_pec']:.1f} m/s, P {res['P_eye']:.3f} yr, amplitude {res['amp_eye']:.2f} m/s "
              f"-> m sin i = {res['m_sin_i_eye']:.3e} Msun (sine fit: {res['m_sin_i_fit']:.3e} Msun)")
        plot_rv(t, np.full_like(t, np.nan), v, res, 0, np.nan, name="c2_partner_rv", title="C2  Partner's radial velocity curve")
    else:
        main()
