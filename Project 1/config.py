"""
Mission setup shared by all challenges (and later parts of the project).

The username decides the seed, and the seed decides the whole solar system,
so this is the one place to change it.
"""
import os

import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission

USERNAME = "09012023"
SEED = utils.get_seed(USERNAME)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_DIR = os.path.join(PROJECT_DIR, "XMLs")  # Where ast2000tools writes its XML files for MCAst
MISSION_FILE = os.path.join(PROJECT_DIR, "mission_after_launch.bin")  # Verified mission, for later parts


def get_mission(verbose=False):
    """A fresh SpaceMission for our seed."""
    return SpaceMission(SEED, data_path=XML_DIR, verbose=verbose)
