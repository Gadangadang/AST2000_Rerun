"""
Part 5, everything in one go: plan the journey and fly it.

    ../.venv/bin/python pro_5.py     # ~45 s

  A  trajectory.py      The simulator: planets from an exact Kepler table, adaptive DOP853, batches
  B  planner.py         Porkchop plot (Lambert), launch site, shooting in the full N-body simulation
  C  journey.py         The real flight with ast2000tools: launch, orientation, corrections, injection
  D  orbit_analysis.py  Orbital elements in orbit around the destination, and a stability check

Needs Part 3 (launch) and Part 4 (orientation software and himmelkule.npy).
Saves mission_in_orbit.bin for Part 6. Figures go to figures/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import time

import journey
import planner
import trajectory
from common.runner import banner


def main():
    start = time.perf_counter()
    banner("A  The trajectory simulator")
    trajectory.main()
    banner("B  Planning the journey")
    ctx, plan = planner.main()
    banner("C + D  The flight, and the orbit at the destination")
    journey.main(ctx, plan)
    banner(f"Part 5 done ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
