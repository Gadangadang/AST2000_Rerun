"""
Mission setup shared by all parts of the project.

The username decides the seed, and the seed decides the whole solar system,
so this is the one place to change it. run_all.py can also pick another system
without editing this file, through the environment variables AST2000_USERNAME
or AST2000_SEED.
"""
import os

import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission

USERNAME = os.environ.get("AST2000_USERNAME", "09012023")
SEED = int(os.environ["AST2000_SEED"]) if "AST2000_SEED" in os.environ else utils.get_seed(USERNAME)

# Destination planet for each seed we have chosen by hand (Part 3). For any other seed,
# Part 3 picks one automatically: a rocky planet, preferably habitable, with the cheapest trip.
DESTINATION_CHOICES = {42716: 1}
DESTINATION = DESTINATION_CHOICES.get(SEED)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(REPO_DIR, "XMLs")  # Where ast2000tools writes its XML files for MCAst/SSView
MISSION_FILE = os.path.join(REPO_DIR, "Project 1", "mission_after_launch.bin")  # Verified launch (Part 1)
ORIENTED_MISSION_FILE = os.path.join(REPO_DIR, "Project 4", "mission_after_orientation.bin")  # Oriented (Part 4)
TRAJECTORY_FILE = os.path.join(REPO_DIR, "Project 2", "planet_trajectories.npz")  # Exact orbits (Part 2)
MCAST_DATA_DIR = os.path.join(REPO_DIR, "MCAst", "data")  # MCAst reads its XML files from here
SSVIEW_DATA_DIR = os.path.join(REPO_DIR, "MCAst", "ssv_data")  # SSView reads from its own folder, not data/


def get_mission(verbose=False):
    """A fresh SpaceMission for our seed."""
    return SpaceMission(SEED, data_path=XML_DIR, verbose=verbose)
