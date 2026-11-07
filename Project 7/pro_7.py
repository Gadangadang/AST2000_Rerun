"""
Part 7, everything in one go: the landing, and the star.

    ../.venv/bin/python pro_7.py     # ~1.5 min

  A  drag.py      Air resistance in the co-rotating atmosphere, terminal velocity, parachute, thrusters
  B  descent.py   Our descent simulator, validated against ast2000tools
  C  landing.py   Release time for the chosen site, the real landing, video (MCAst: landing.xml)
  D  star.py      Luminosity from the surface and from the core (separate document: paper/part7_star.tex)

Needs Part 6 (low orbit, atmosphere_model.json, landing_site.json).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import time

import descent
import drag
import landing
import star
from common.report import record_system
from common.runner import banner


def main():
    start = time.perf_counter()
    record_system()
    banner("A  Air resistance")
    drag.main()
    banner("B  Simulating the landing")
    descent.main()
    banner("C  Landing on the chosen site")
    landing.main()
    banner("D  Exploring the star")
    star.main()
    banner(f"Part 7 done ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
