"""
Part 1, everything in one go: challenges A-F and the official API verification.

    ../.venv/bin/python pro_1.py           # everything, ~2 min
    ../.venv/bin/python pro_1.py --quick   # skip the slow accuracy studies, ~40 s

Each challenge also runs on its own (part_a_statistics.py, gas_box.py,
engine.py, rocket.py, launch.py). Figures go to figures/, the engine numbers
to engine_box.json, and the verified mission to mission_after_launch.bin.

  A  part_a_statistics.py  Gaussian and Maxwell-Boltzmann statistics
  B  gas_box.py            Gas particles in a closed box
  C  engine.py             The nozzle: thrust and fuel use per box
  D  rocket.py             Fuel consumed for a speed boost
  E  launch.py             Launch simulation and engine design
  F  launch.py             Result in the solar system frame + API check
"""
import argparse
import sys
import time

import engine
import gas_box
import launch
import part_a_statistics
import rocket


def banner(text):
    print("\n" + "=" * 78 + f"\n  {text}\n" + "=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the slow accuracy studies in B and C")
    args = parser.parse_args()

    start = time.perf_counter()
    banner("A  Investigating Boltzmann statistics")
    part_a_statistics.main()
    banner("B  Simulating energetic gas particles")
    gas_box.main(convergence=not args.quick)
    banner("C  Introducing a nozzle")
    engine.main(compare=not args.quick)
    banner("D  The rocket engine's performance")
    rocket.main()
    banner("E + F  Rocket launch, solar system frame and API verification")
    verified = launch.main()

    banner(f"Part 1 {'VERIFIED by ast2000tools' if verified else 'NOT verified'} "
           f"({time.perf_counter() - start:.0f} s)")
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
