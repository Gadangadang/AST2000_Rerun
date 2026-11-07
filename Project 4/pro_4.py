"""
Part 4, everything in one go: the onboard orientation software.

    ../.venv/bin/python pro_4.py     # ~30 s (first run also builds the 360 reference pictures)

  A, B  sky_pictures.py   Stereographic projection, 360 reference pictures, pointing angle
  C     doppler.py        Velocity from the Doppler shifts of two stars
  D     trilateration.py  Position from distances to the planets and the star
  MO    orientation.py    Manual orientation right after launch, checked with the API

Needs himmelkule.npy and sample0000.png (course files) in this folder, and the
launched mission from Part 1. Saves mission_after_orientation.bin for Part 5.
Figures go to figures/.
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import time

from ast2000tools.space_mission import SpaceMission

import doppler
import orientation
import sky_pictures
import trilateration
from common.config import MISSION_FILE
from common.report import record_system
from common.runner import banner



def main():
    start = time.perf_counter()
    record_system()
    library = sky_pictures.load_or_build_library(sky_pictures.load_sky())
    banner("Manual orientation (uses A-D below)")
    orientation.main(library)
    banner("A, B  Sky pictures and pointing angle")
    sky_pictures.main(extra_pictures=[("spacecraft's picture", orientation.PICTURE_FILE)])
    with redirect_stdout(io.StringIO()):
        mission = SpaceMission.load(MISSION_FILE)
    banner("C  Velocity from Doppler shifts")
    doppler.main(mission)
    banner("D  Position from distances")
    trilateration.main(mission)
    banner(f"Part 4 done ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
