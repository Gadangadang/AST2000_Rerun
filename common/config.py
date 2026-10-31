"""
Mission setup shared by all parts of the project.

The username decides the seed, and the seed decides the whole solar system,
so this is the one place to change it.
"""
import os

import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission

USERNAME = "09012023"
SEED = utils.get_seed(USERNAME)
DESTINATION = 1  # Planet we travel to (chosen in Part 3: habitable, and by far the cheapest trip)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(REPO_DIR, "XMLs")  # Where ast2000tools writes its XML files for MCAst/SSView
MISSION_FILE = os.path.join(REPO_DIR, "Project 1", "mission_after_launch.bin")  # Verified launch (Part 1)
TRAJECTORY_FILE = os.path.join(REPO_DIR, "Project 2", "planet_trajectories.npz")  # Exact orbits (Part 2)
MCAST_DATA_DIR = os.path.join(REPO_DIR, "MCAst", "data")  # MCAst reads its XML files from here
SSVIEW_DATA_DIR = os.path.join(REPO_DIR, "MCAst", "ssv_data")  # SSView reads from its own folder, not data/


def get_mission(verbose=False):
    """A fresh SpaceMission for our seed."""
    return SpaceMission(SEED, data_path=XML_DIR, verbose=verbose)
