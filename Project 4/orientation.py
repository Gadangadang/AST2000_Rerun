"""
Part 4, manual orientation: the spacecraft finds its angle, velocity and position
right after launch with our software, and the API checks the result.

  angle     take_picture                -> compare with the 360 references (B)
  velocity  measure_star_doppler_shifts -> two star directions (C)
  position  measure_distances           -> trilateration (D)

We also compare with what we predicted for this launch in Parts 1 and 3
(generalised_launch.launch_state): two independent routes to the same numbers.

`orient(mission)` is the reusable "onboard software" for later parts.

Run:  ../.venv/bin/python orientation.py   (needs the launched mission from Part 1)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/
from common.paths import use_parts

use_parts(1, 2, 3)

import io
from contextlib import redirect_stdout

import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission
from PIL import Image

import doppler
import sky_pictures
import trilateration
from common.config import MISSION_FILE, ORIENTED_MISSION_FILE
from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, save
from common.report import record

PICTURE_FILE = os.path.join(HERE, "sky_picture.png")
AU_PER_YR = const.AU / const.yr


def orient(mission, library=None):
    """Angle [deg], velocity [AU/yr] and position [AU] of the spacecraft from its own instruments."""
    if library is None:
        library = sky_pictures.load_or_build_library(sky_pictures.load_sky())
    with redirect_stdout(io.StringIO()):
        mission.take_picture(filename=PICTURE_FILE, full_sky_image_path=sky_pictures.SKY_FILE)
    angle, _ = sky_pictures.find_pointing(np.array(Image.open(PICTURE_FILE).convert("RGB")), library)
    velocity = doppler.spacecraft_velocity(mission.measure_star_doppler_shifts(), mission.star_doppler_shifts_at_sun,
                                           mission.star_direction_angles)
    position, _ = trilateration.trilaterate(mission.system, mission.time_after_launch, mission.measure_distances())
    return angle, velocity, position


def predicted_state(mission):
    """Our own prediction for the launch in MISSION_FILE (Part 1: t0 = 0, launch-site angle 0)."""
    import generalised_launch
    from launch import HomePlanet, LAUNCH_ANGLE, TIME_OF_LAUNCH
    planet = HomePlanet(mission)
    with redirect_stdout(io.StringIO()):
        _, launch = generalised_launch.design(mission, planet)
    _, position, velocity, _ = generalised_launch.launch_state(launch, planet, mission.system, TIME_OF_LAUNCH,
                                                               LAUNCH_ANGLE)
    return position, velocity


def plot_summary(mission, angle, velocity, position, pred_position, pred_velocity, library):
    """Mission-control view of the fix: the picture, our best reference, and the three measurements vs. prediction."""
    picture = np.array(Image.open(PICTURE_FILE).convert("RGB"))
    sky = sky_pictures.load_sky()
    reference = sky_pictures.make_picture(sky, np.radians(angle))
    fig = plt.figure(figsize=(17, 8.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    for i, (img, title) in enumerate(((picture, "What the spacecraft's camera sees"),
                                      (reference, f"Our projection at the angle found, {angle:.2f}$^\\circ$"),
                                      (None, "difference (std {:.2f} per colour)".format(
                                          np.std(picture.astype(int) - reference.astype(int)))))):
        ax = fig.add_subplot(grid[0, i])
        if img is None:
            ax.imshow((picture.astype(int) - reference.astype(int)).mean(axis=2), cmap="gray", vmin=-30, vmax=30)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    # verify_manual_orientation's rules (ast2000tools source): position within 0.01 AU, velocity within 1% of |v|
    tolerances = {"position": 0.01 * const.AU / 1e3, "velocity": 0.01 * np.linalg.norm(velocity) * AU_PER_YR}
    rows = [("position [km]", np.linalg.norm(position - pred_position) * const.AU / 1e3, tolerances["position"]),
            ("velocity [m/s]", np.linalg.norm(velocity - pred_velocity) * AU_PER_YR, tolerances["velocity"])]
    ax = fig.add_subplot(grid[1, :2])
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[BLUE, ORANGE], height=0.45, label="onboard fix minus launch prediction")
    ax.scatter([r[2] for r in rows], y, marker="|", s=900, color=INK, lw=2.5, label="API tolerance")
    for yi, r in zip(y, rows):
        ax.text(r[1], yi, f"  {r[1]:.2f}", va="center", fontsize=10)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_title("Two independent routes agree: instruments (Part 4) vs. launch simulation (Parts 1 and 3)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    text = (f"t = {mission.time_after_launch:.6e} yr\n\n"
            f"pointing   {angle:.2f} deg\n"
            f"velocity   ({velocity[0]:.4f}, {velocity[1]:.4f}) AU/yr\n"
            f"position   ({position[0]:.7f}, {position[1]:.7f}) AU\n\n"
            f"API: {'VERIFIED' if mission.manual_orientation_verified else 'not verified'}")
    ax.text(0, 0.5, text, family="monospace", fontsize=11, va="center")
    fig.suptitle("MO  Manual orientation right after launch", fontsize=12)
    save(fig, "e_orientation_summary")


def main(library=None):
    if not os.path.exists(MISSION_FILE):
        raise FileNotFoundError(f"{MISSION_FILE} is missing: run Part 1 (pro_1.py) first")
    with redirect_stdout(io.StringIO()):
        mission = SpaceMission.load(MISSION_FILE)
    angle, velocity, position = orient(mission, library)
    print(f"MO    our onboard software, right after launch (t = {mission.time_after_launch:.6e} yr):")
    print(f"      pointing angle  {angle:.2f} deg      (picture saved as {os.path.basename(PICTURE_FILE)})")
    print(f"      velocity        ({velocity[0]:.5f}, {velocity[1]:.5f}) AU/yr")
    print(f"      position        ({position[0]:.8f}, {position[1]:.8f}) AU")

    pred_position, pred_velocity = predicted_state(mission)
    print(f"      vs. our launch prediction (Part 3): position differs by "
          f"{np.linalg.norm(position - pred_position) * const.AU / 1e3:.1f} km, velocity by "
          f"{np.linalg.norm(velocity - pred_velocity) * AU_PER_YR:.1f} m/s")

    mission.verbose = True
    log = io.StringIO()
    with redirect_stdout(log):
        try:
            mission.verify_manual_orientation(position, velocity, angle)
        except RuntimeError:
            pass
    print("API   verify_manual_orientation")
    print("      " + "\n      ".join(line for line in log.getvalue().splitlines() if line.strip()))
    if mission.manual_orientation_verified:
        with redirect_stdout(io.StringIO()):
            SpaceMission.save(ORIENTED_MISSION_FILE, mission)
        print(f"      VERIFIED. Mission saved to {os.path.basename(ORIENTED_MISSION_FILE)} for Part 5.")
    plot_summary(mission, angle, velocity, position, pred_position, pred_velocity, library)
    record("mo-verified", "passed" if mission.manual_orientation_verified else "FAILED")
    record("mo-angle", angle, "{:.2f}")
    record("mo-pos-diff", np.linalg.norm(position - pred_position) * const.AU / 1e3, "{:.1f}")
    record("mo-vel-diff", np.linalg.norm(velocity - pred_velocity) * AU_PER_YR, "{:.1f}")
    record("mo-time", mission.time_after_launch, "{:.4e}")
    return mission, (angle, velocity, position)


if __name__ == "__main__":
    main()
