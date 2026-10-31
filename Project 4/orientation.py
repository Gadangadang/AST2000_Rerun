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
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO)  # Repo root, for common/
for part in ("Project 1", "Project 2", "Project 3"):
    sys.path.insert(0, os.path.join(REPO, part))

import io
from contextlib import redirect_stdout

import ast2000tools.constants as const
import numpy as np
from ast2000tools.space_mission import SpaceMission
from PIL import Image

import doppler
import sky_pictures
import trilateration
from common.config import MISSION_FILE, ORIENTED_MISSION_FILE

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
    return mission, (angle, velocity, position)


if __name__ == "__main__":
    main()
