"""
Part 4, challenge C: the spacecraft's velocity from the Doppler shift of two stars.

1. Doppler: a source moving away at radial speed v_r shifts a line by
       delta_lambda / lambda_0 = v_r / c   ->   v_r = c delta_lambda / lambda_0
   (lambda_0 = 656.3 nm, H-alpha; speeds << c, so the classical formula is enough).
2. Seen from the Sun, star i recedes at v_i,sun = c dl_i,sun / lambda_0.
   Seen from the spacecraft it recedes at v_i,ship = c dl_i,ship / lambda_0.
   The difference is the spacecraft's own speed towards star i, relative to the Sun:
       v_i = v_i,sun - v_i,ship
3. Turn the two numbers (v_1, v_2), given along the star directions
   u_i = (cos phi_i, sin phi_i), into (v_x, v_y).

Step 3 has two possible readings, and they differ unless the stars are 90 deg apart:
  * Textbook (projections): v_i = v . u_i, so (v_1, v_2) = B v with rows u_1, u_2,
    and v = B^-1 (v_1, v_2) (the assignment's equation 12).
  * ast2000tools: (v_1, v_2) are the coefficients of v along the two directions,
    v = v_1 u_1 + v_2 u_2. This is how the API generates the measured shifts
    (we read its source), so this is the reading that gives the right answer
    with the API's data. Our stars are 248 deg apart, and the textbook reading
    is 27% off.

Run:  ../.venv/bin/python doppler.py
Figures go to figures/c*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np

from common.config import get_mission
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, save

LAMBDA_0 = 656.3  # H-alpha rest wavelength [nm]
AU_PER_YR = const.AU / const.yr  # [m/s] per [AU/yr]


def radial_velocity(delta_lambda, lambda_0=LAMBDA_0):
    """Speed away from us [m/s] from a Doppler shift [nm]."""
    return const.c * np.asarray(delta_lambda, dtype=float) / lambda_0


def star_directions(star_angles_deg):
    """Unit vectors towards the two reference stars, as the columns of a 2 x 2 matrix."""
    phi = np.radians(star_angles_deg)
    return np.array([np.cos(phi), np.sin(phi)])


def components_towards_stars(shifts_ship, shifts_sun):
    """The spacecraft's speed towards each star relative to the Sun [m/s]."""
    return radial_velocity(shifts_sun) - radial_velocity(shifts_ship)


def spacecraft_velocity(shifts_ship, shifts_sun, star_angles_deg, convention="api"):
    """
    (v_x, v_y) of the spacecraft relative to the Sun [AU/yr] from the two Doppler shifts [nm].

    convention "api": v = v_1 u_1 + v_2 u_2 (how ast2000tools defines the measurement)
    convention "projection": v_i = v . u_i (textbook; equals "api" only for perpendicular stars)
    """
    v_i = components_towards_stars(shifts_ship, shifts_sun) / AU_PER_YR
    U = star_directions(star_angles_deg)
    if convention == "api":
        return U @ v_i
    return np.linalg.solve(U.T, v_i)


def shifts_for_velocity(velocity, shifts_sun, star_angles_deg, convention="api"):
    """Inverse of spacecraft_velocity: the shifts the spacecraft would measure (for testing)."""
    U = star_directions(star_angles_deg)
    v_i = np.linalg.solve(U, velocity) if convention == "api" else U.T @ velocity
    return np.asarray(shifts_sun) - v_i * AU_PER_YR * LAMBDA_0 / const.c


# ===================================================================== tests
def self_tests(mission):
    """Answers we know in advance."""
    angles, sun = mission.star_direction_angles, np.array(mission.star_doppler_shifts_at_sun)
    rng = np.random.default_rng(6)
    results = {}
    # 1. The spacecraft sees the same shifts as the Sun -> it moves with the Sun
    results["same shifts as the Sun -> v = 0"] = np.abs(spacecraft_velocity(sun, sun, angles)).max()
    # 2. Round trip: any velocity -> shifts -> back
    worst = max(np.abs(spacecraft_velocity(shifts_for_velocity(v, sun, angles), sun, angles) - v).max()
                for v in rng.normal(0, 10, (100, 2)))
    results["velocity -> shifts -> velocity (100 random)"] = worst
    # 3. Perpendicular stars: both conventions must agree
    perpendicular = (30.0, 120.0)
    v = np.array([3.0, -4.0])
    shifts = shifts_for_velocity(v, sun, perpendicular)
    results["stars 90 deg apart: api - projection"] = np.abs(
        spacecraft_velocity(shifts, sun, perpendicular) - spacecraft_velocity(shifts, sun, perpendicular, "projection")).max()
    return results


# ===================================================================== figures
def plot_doppler(mission, velocity, true_velocity=None):
    angles = mission.star_direction_angles
    U = star_directions(angles)
    v_i = components_towards_stars(mission.measure_star_doppler_shifts(), mission.star_doppler_shifts_at_sun) / AU_PER_YR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    scale = 1.4 * np.abs(velocity).max()
    for i, color in enumerate((BLUE, ORANGE)):
        ax1.plot([-scale * U[0, i], scale * U[0, i]], [-scale * U[1, i], scale * U[1, i]], color=color, lw=0.8, ls=":")
        ax1.annotate(f"towards star {i + 1}\n($\\phi$ = {angles[i]:.1f}$^\\circ$)", scale * U[:, i], color=color,
                     fontsize=9, ha="center", va="center", xytext=(0, 14 if U[1, i] > 0 else -14),
                     textcoords="offset points")
        part = v_i[i] * U[:, i]
        ax1.annotate("", xy=part, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
    # Parallelogram: v = v_1 u_1 + v_2 u_2
    p1, p2 = v_i[0] * U[:, 0], v_i[1] * U[:, 1]
    ax1.plot([p1[0], velocity[0], p2[0]], [p1[1], velocity[1], p2[1]], color=MUTED, lw=1, ls="--")
    ax1.annotate("", xy=velocity, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2.5})
    ax1.annotate(f"v = ({velocity[0]:.3f}, {velocity[1]:.3f}) AU/yr", velocity, xytext=(-10, 10),
                 textcoords="offset points", ha="right", fontsize=9)
    ax1.plot(0, 0, "o", color=INK)
    ax1.set_aspect("equal")
    ax1.set_xlim(-scale * 1.15, scale * 1.15)
    ax1.set_ylim(-scale * 1.15, scale * 1.15)
    ax1.set_xlabel("$v_x$ [AU/yr]")
    ax1.set_ylabel("$v_y$ [AU/yr]")
    ax1.set_title("Velocity built from the two star directions: v = v$_1$u$_1$ + v$_2$u$_2$", fontsize=10)

    # How much the two conventions differ, as a function of the angle between the stars
    separation = np.linspace(10, 170, 200)
    rng = np.random.default_rng(7)
    v_test = rng.normal(0, 8, (200, 2))
    sun = np.array(mission.star_doppler_shifts_at_sun)
    error = []
    for sep in separation:
        stars = (angles[0], angles[0] + sep)
        e = [np.linalg.norm(spacecraft_velocity(shifts_for_velocity(v, sun, stars), sun, stars, "projection") - v)
             / np.linalg.norm(v) for v in v_test]
        error.append(np.median(e))
    ax2.plot(separation, np.array(error) * 100, color=BLUE)
    our_sep = abs((angles[1] - angles[0] + 180) % 360 - 180)
    ax2.axvline(our_sep, color=ORANGE, lw=1.2)
    ax2.text(our_sep, ax2.get_ylim()[1] * 0.9, f"  our stars: {our_sep:.1f}$^\\circ$ apart", color=ORANGE, fontsize=9)
    ax2.axvline(90, color=MUTED, lw=0.8, ls=":")
    ax2.set_yscale("log")
    ax2.set_xlabel("angle between the reference stars [deg]")
    ax2.set_ylabel("textbook-projection error [%] (median)")
    ax2.set_title("Reading the data as projections v$\\cdot$u$_i$ only works for perpendicular stars", fontsize=10)
    fig.suptitle("C  Spacecraft velocity from Doppler shifts", fontsize=12)
    save(fig, "c_doppler_velocity")


def main(mission=None):
    """mission: a launched, verified SpaceMission (Part 1/3). Without one, only the self-tests run."""
    m = mission if mission is not None else get_mission()
    print(f"C     reference stars at phi = {m.star_direction_angles[0]:.2f} and {m.star_direction_angles[1]:.2f} deg; "
          f"shifts seen from the Sun {m.star_doppler_shifts_at_sun[0]:+.6f}, {m.star_doppler_shifts_at_sun[1]:+.6f} nm")
    sun_v = radial_velocity(m.star_doppler_shifts_at_sun)
    print(f"      the stars recede from the Sun at {sun_v[0]:+.1f} and {sun_v[1]:+.1f} m/s")
    for name, value in self_tests(m).items():
        print(f"      test: {name}: {value:.1e} AU/yr")
    if mission is None:
        return None
    shifts = mission.measure_star_doppler_shifts()
    velocity = spacecraft_velocity(shifts, mission.star_doppler_shifts_at_sun, mission.star_direction_angles)
    textbook = spacecraft_velocity(shifts, mission.star_doppler_shifts_at_sun, mission.star_direction_angles, "projection")
    print(f"C     spacecraft measures {shifts[0]:+.6f}, {shifts[1]:+.6f} nm -> v = ({velocity[0]:.4f}, {velocity[1]:.4f}) AU/yr"
          f" = {np.linalg.norm(velocity) * AU_PER_YR / 1e3:.2f} km/s")
    print(f"      (textbook projection reading would give ({textbook[0]:.4f}, {textbook[1]:.4f}) AU/yr)")
    plot_doppler(mission, velocity)
    return velocity


if __name__ == "__main__":
    main()
