"""
The whole AST2000 project in one go: every part that is done so far, in order.

    .venv/bin/python run_all.py                    # our system (username in common/config.py)
    .venv/bin/python run_all.py --quick            # skip the slow side studies
    .venv/bin/python run_all.py --username abc123  # someone else's solar system
    .venv/bin/python run_all.py --seed 12345       # ... or pick the seed directly
    .venv/bin/python run_all.py --parts 3 4        # only some parts (later parts need earlier outputs)

Each part runs in its own folder (pro_N.py), so its figures land in "Project N/figures".
A new seed overwrites the generated files (figures, missions, orbits), which are all
regenerated here anyway. Each part feeds the next: the launch (1) is reused by the
orbits and launch windows (2, 3), and the launched mission by the orientation (4).
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
PARTS = {  # part -> (folder, script, extra arguments in --quick mode)
    1: ("Project 1", "pro_1.py", ["--quick"]),
    2: ("Project 2", "pro_2.py", ["--quick"]),
    3: ("Project 3", "pro_3.py", []),
    4: ("Project 4", "pro_4.py", []),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the slow side studies in each part")
    parser.add_argument("--username", help="use the solar system of this username")
    parser.add_argument("--seed", type=int, help="use this seed directly")
    parser.add_argument("--parts", type=int, nargs="+", default=sorted(PARTS), help="which parts to run")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.username:
        env["AST2000_USERNAME"] = args.username
    if args.seed is not None:
        env["AST2000_SEED"] = str(args.seed)
    seed = subprocess.run([sys.executable, "-c", "from common.config import SEED; print(SEED)"], cwd=REPO, env=env,
                          capture_output=True, text=True, check=True).stdout.strip()
    print(f"AST2000 project, seed {seed}, parts {args.parts}")

    start = time.perf_counter()
    for part in args.parts:
        folder, script, quick_args = PARTS[part]
        print("\n" + "#" * 78 + f"\n#  PART {part}\n" + "#" * 78, flush=True)
        t0 = time.perf_counter()
        result = subprocess.run([sys.executable, script, *(quick_args if args.quick else [])],
                                cwd=os.path.join(REPO, folder), env=env)
        if result.returncode != 0:
            sys.exit(f"\nPart {part} failed (exit code {result.returncode}); stopping here.")
        print(f"\n#  Part {part} finished in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"\nAll done: parts {args.parts} for seed {seed} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
