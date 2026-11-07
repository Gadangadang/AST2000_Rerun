"""
Course data that is too large for git, kept on this repository's GitHub release instead.

    Project 4/himmelkule.npy               the sky sphere (126 MB)
    Project 6/spectra/*.parquet            the atmosphere spectra of all 100 seeds, and the noise (2.8 GB)
    XMLs.zip                               the MCAst/SSView files from our run

fetch() downloads one file when a part needs it: over plain HTTPS, or with the gh CLI if that fails (a private
repository needs gh's login).
"""
import os
import shutil
import subprocess
import urllib.request

REPOSITORY = "Gadangadang/AST2000_Rerun"
RELEASE_TAG = "spectra"  # The name is historical: it holds all the course data
RELEASE_URL = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{{}}"


def fetch(target, name=None):
    """
    Download the release file `name` (default: target's file name) to the path `target`. It is written under a
    temporary name first, so an interrupted download never leaves a half file. Returns True if it arrived.
    """
    name = name or os.path.basename(target)
    part = target + ".part"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"      downloading {name} from the GitHub release '{RELEASE_TAG}'", flush=True)
    try:
        with urllib.request.urlopen(RELEASE_URL.format(name), timeout=60) as response, open(part, "wb") as f:
            shutil.copyfileobj(response, f, 1 << 20)
    except OSError as error:  # HTTP and network errors alike
        done = (subprocess.run(["gh", "release", "download", RELEASE_TAG, "--repo", REPOSITORY, "--pattern", name,
                                "--output", part, "--clobber"], capture_output=True, text=True)
                if shutil.which("gh") else None)
        if done is None or done.returncode:
            print(f"      download failed: {error}" + (f"; gh: {done.stderr.strip()}" if done else ""))
            if os.path.exists(part):
                os.remove(part)
            return False
    os.replace(part, target)
    return True
