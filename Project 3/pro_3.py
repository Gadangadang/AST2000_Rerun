"""
Part 3, everything in one go: preparing for the journey.

    ../.venv/bin/python pro_3.py     # ~10 s

  1      habitable_zone.py      Star's flux, planet temperatures, habitable zone, solar panels
  2,4,5  destination.py         Destination comparison, Hohmann routes and launch windows,
                                capture distance (k = 10) and camera distance
  3      generalised_launch.py  Launch at any time and site, checked with the API

Figures go to figures/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import time

import destination
import generalised_launch
import habitable_zone


def banner(text):
    print("\n" + "=" * 78 + f"\n  {text}\n" + "=" * 78)


def main():
    start = time.perf_counter()
    banner("1  The habitable zone")
    habitable_zone.main()
    banner("2, 4, 5  Choosing the destination; how close is close")
    destination.main()
    banner("3  Generalised launch")
    generalised_launch.main()
    banner(f"Part 3 done ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
