"""
Part 6, everything in one go: preparing for the landing.

    ../.venv/bin/python pro_6.py     # ~1.5 min (the first run also caches the 330 MB spectrum as binary)

  A  low_orbit.py      Descent to a low orbit; drag probe of the upper atmosphere
  B  spectrum.py       chi^2 search for the gases' spectral lines (blind, then at the common Doppler shift)
  C  atmosphere.py     Density and temperature profiles, checked against the drag probe
  D  landing_sites.py  Coordinates on the spinning planet, scouting pictures, landing site
  +  api_check.py      Afterwards: our atmosphere against the one ast2000tools uses internally

Needs Part 5 (mission_in_orbit.bin and journey_state.json) and the course spectra in spectra/
(spectrum_seedXX.parquet, XX = last two digits of the seed, and sigma_noise.parquet; see spectrum.py).
Without this seed's spectrum, B is skipped and C uses the drag probe alone.
Writes landing_site.json and atmosphere_model.json for Part 7, MCAst pictures (site_k.xml) and
videos (descent.xml) to MCAst/data, and figures to figures/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import time

import api_check
import atmosphere
import landing_sites
import low_orbit
import spectrum
from common.report import record_system
from common.runner import banner


def main():
    start = time.perf_counter()
    record_system()
    banner("A  Enter a low orbit")
    low_orbit.main()
    banner("B  Spectral analysis of the atmosphere")
    spectrum.main()
    banner("C  Model the atmosphere")
    atmosphere.main()
    banner("D  Scouting for landing sites")
    landing_sites.main()
    banner("After the fact: our atmosphere against the API's hidden one")
    api_check.main()
    banner(f"Part 6 done ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
