"""
Part 2, everything in one go: challenges A-C and the official API verification.

    ../.venv/bin/python pro_2.py           # everything, ~1 min
    ../.venv/bin/python pro_2.py --quick   # skip the time-step study and the GR extra, ~35 s

  A  orbits.py       Analytical and simulated orbits, verify_planet_positions, SSView video
  B  kepler.py       Kepler's 2nd and 3rd laws
  C  star_wobble.py  Star + planet in the CM frame, radial velocity and light curves
  +  relativity.py   Extra: general-relativistic corrections (does not change the verified orbits)

Outputs: figures/, simulated_orbits.npz, planet_trajectories.npz (exact orbits from the
API, used in later parts), rv_curve.txt and light_curve.txt (for the partner group),
and XMLs/orbit_video.xml in the repo root (also copied into MCAst/ssv_data/ for SSView).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import argparse
import time

import kepler
import orbits
import relativity
import star_wobble
from common.report import record_system
from common.runner import banner



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the time-step study in A and the GR extra")
    args = parser.parse_args()

    start = time.perf_counter()
    record_system()
    banner("A  Planetary orbits")
    orbits.main(study=not args.quick)
    banner("B  Kepler's laws")
    kepler.main()
    banner("C  Can extraterrestrials discover our planets?")
    star_wobble.main()
    if not args.quick:
        banner("Extra  General relativity")
        relativity.main()
    banner(f"Part 2 done ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
