"""
Import code from other parts of the project.

The part folders ("Project 1", ...) have spaces in their names, so they can't be
Python packages. use_parts(1, 2) puts those folders on the import path, so that
`from launch import ...` (Part 1) and `from orbits import ...` (Part 2) work.
"""
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def part_dir(part):
    return os.path.join(REPO_DIR, f"Project {part}")


def use_parts(*parts):
    """Make the modules of the given parts importable (earlier entries win on name clashes)."""
    for part in reversed(parts):
        path = part_dir(part)
        if path not in sys.path:
            sys.path.insert(1, path)  # After the running script's own folder
