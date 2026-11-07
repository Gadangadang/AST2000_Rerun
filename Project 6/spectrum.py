"""
Part 6, challenge B: which gases are in the atmosphere? A chi^2 analysis of the spectrum.

Data: the flux meter's spectrum, 600-3000 nm in 1e7 points (0.00024 nm apart),
normalised so F = 1 without lines, and the noise sigma_i of every point.

Two numbers set the search (preparation 1 and 2):
  * Doppler: the spacecraft moves at most |v| = 10 km/s relative to the atmosphere,
        delta_lambda_max = lambda_0 v_max / c   (0.02 nm at 632 nm, 0.10 nm at 2870 nm)
  * Line width: every molecule's line-of-sight velocity is Gaussian with sigma_v = sqrt(kT/m)
    (Part 1), and delta_lambda / lambda_0 = v / c, so the line profile is a Gaussian with
        sigma = (lambda_0 / c) sqrt(k T / m)   (a few grid points for 150-450 K)

Model (equation a of the hints): F(lambda) = 1 + A exp(-(lambda - lambda_c)^2 / 2 sigma^2), A = F_min - 1.

Clever bits:
  * The model is LINEAR in the depth A. For fixed (lambda_c, sigma) the best A and its chi^2 follow
    exactly from two sums,
        S_fg = sum w (f - 1) g,   S_gg = sum w g^2,   w = 1/sigma_i^2
        A* = S_fg / S_gg,         chi^2 = chi^2_null - 2 A S_fg + A^2 S_gg,
    and as functions of lambda_c both sums are convolutions with a Gaussian kernel. So the
    whole (lambda_c, T) grid costs one convolution per temperature instead of a triple loop.
    A least-squares fit started at the grid minimum then refines all three parameters and
    gives their uncertainties.
  * Real line or fluke? Instead of judging by eye we measure it: the same search is run on
    hundreds of stretches of the spectrum with no listed line nearby. The distribution of
    delta chi^2 = chi^2_null - chi^2_min there is what noise alone can do, and a line counts as
    detected only if it beats that (false-alarm probability below FALSE_ALARM). Then the
    physics checks: the real lines share one Doppler shift, have a temperature in 150-450 K
    and F_min > 0.7.
  * Completeness: synthetic lines of known depth are injected into the noise and searched
    for, so a non-detection tells us how deep a missing gas's lines could at most be.

The detected gases give the mean molecular weight mu (equal amounts of each), for challenge C.

Run:  ../.venv/bin/python spectrum.py            (the analysis for our seed)
      ../.venv/bin/python spectrum.py --import   (course .txt/.zip files -> spectra/*.parquet, originals deleted)
      ../.venv/bin/python spectrum.py --download (all 100 spectra from the repository's GitHub release)
Figures go to figures/b*.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # Repo root, for common/

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as sc
from scipy import optimize, signal

from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, SERIES, VIOLET, save
from common.report import record, record_many

HERE = os.path.dirname(os.path.abspath(__file__))
# The course's spectra (one per seed, XX = last two digits) and the noise sigma (the same for all), stored as
# float32 in Parquet, zstd-compressed after splitting each float into byte planes (byte_stream_split). Lossless
# (the text files print float32 values), 28 MB per seed, ~1/12 of the text; noise has ~22 bits of information
# per point, so lossless storage cannot get much smaller. Loads in ~0.05 s.
# Course files (.txt, or the .zip as downloaded) dropped in this folder or in spectra/ are converted on first use;
# `python spectrum.py --import` converts every one it finds and deletes the originals.
# spectra/ is not in git (2.8 GB): the files live on the repository's GitHub release RELEASE_TAG, and a missing
# one is downloaded when needed (gh CLI, logged in; the release is private like the repository).
# `python spectrum.py --download` fetches all of them.
RELEASE_TAG = "spectra"
SPECTRA_DIR = os.path.join(HERE, "spectra")
SPECTRUM_FILE = os.path.join(SPECTRA_DIR, "spectrum_seed{:02d}.parquet")
NOISE_FILE = os.path.join(SPECTRA_DIR, "sigma_noise.parquet")
COURSE_NAMES = {  # course file name (without .zip) -> our file name
    r"spectrum_seed(\d\d)_600nm_3000nm\.txt": r"spectrum_seed\1.parquet",
    r"sigma_noise\.txt": "sigma_noise.parquet",
}
PARQUET = {"compression": "zstd", "compression_level": 19, "use_byte_stream_split": True, "use_dictionary": False}
N_POINTS = 10_000_000
LAMBDA_START, LAMBDA_STEP = 600.0, 0.00024  # [nm]; the text file rounds this, so we rebuild the exact grid

# Table II: spectral lines [nm] of the gases that may be present
LINES = {"O2": (632, 690, 760), "H2O": (720, 820, 940), "CO2": (1400, 1600), "CH4": (1660, 2200),
         "CO": (2340,), "N2O": (2870,)}
# Table I: (protons, neutrons) of each atom, and the atoms of each molecule
NUCLEI = {"H": (1, 0), "C": (6, 6), "N": (7, 7), "O": (8, 8)}
MOLECULES = {"O2": {"O": 2}, "H2O": {"H": 2, "O": 1}, "CO2": {"C": 1, "O": 2}, "CH4": {"C": 1, "H": 4},
             "CO": {"C": 1, "O": 1}, "N2O": {"N": 2, "O": 1}}

V_MAX = 10e3  # Largest speed of the spacecraft relative to the atmosphere [m/s]
T_RANGE = (150.0, 450.0)  # Expected atmospheric temperature [K]
T_GRID = np.linspace(100.0, 600.0, 51)  # Searched a little wider than expected, to see where fits want to go
DEPTH_MAX = 0.5  # Deepest line searched (F_min >= 0.5); real lines should have F_min > 0.7
WINDOW_SIGMAS = 6  # Window half-width beyond the Doppler range, in line widths
FALSE_ALARM = 1e-3  # A detection must beat noise with this false-alarm probability
MARGINAL = 1e-2  # Lines that beat only this (but pass the physics checks) are reported as marginal
RESULT_FILE = os.path.join(HERE, "spectral_result.json")  # Detected and marginal gases, for challenge C
N_NOISE_WINDOWS = 400  # Line-free windows per line for the noise distribution
U = sc.atomic_mass  # [kg]


# ===================================================================== data
def molecule_mass(name):
    """
    Mass [kg] from the protons and neutrons of its atoms (Table I), as the assignment asks. This is ~0.8%
    heavier than the real molecule: nuclear binding energy makes a nucleus lighter than its parts.
    """
    return sum(n * (Z * sc.m_p + N * sc.m_n) for atom, n in MOLECULES[name].items() for Z, N in [NUCLEI[atom]])


def save_array(path, values):
    """One float32 column to Parquet (written, then renamed: an interrupted run never leaves a half file)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"value": np.asarray(values, dtype=np.float32)}), path + ".part", **PARQUET)
    os.replace(path + ".part", path)


def load_array(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).column("value").to_numpy()


def stored_name(course_path):
    """Our file name for a course file ('... (1).zip' browser duplicates included), or None if it is not one."""
    import re
    name = re.sub(r"( \(\d+\))?\.zip$", "", os.path.basename(course_path))
    return next((re.sub(f"^{pattern}$", target, name) for pattern, target in COURSE_NAMES.items()
                 if re.fullmatch(pattern, name)), None)


def course_files(folders=(HERE, SPECTRA_DIR)):
    """Every course data file (text or zip) in these folders."""
    return sorted(os.path.join(d, f) for d in folders if os.path.isdir(d) for f in os.listdir(d) if stored_name(f))


def convert(course_path):
    """
    Read a course file (pandas reads the zip directly), check it is the full 600-3000 nm grid, and save
    column 2 as float32 Parquet in spectra/. If that file already exists (a duplicate download), check that the
    data are identical instead. Returns its path.
    """
    import pandas as pd
    table = pd.read_csv(course_path, sep=r"\s+", header=None, engine="c", dtype=np.float64).to_numpy()
    lam, values = table[:, 0], table[:, 1].astype(np.float32)
    grid = LAMBDA_START + LAMBDA_STEP * np.arange(N_POINTS)
    if len(lam) != N_POINTS or np.abs(lam - grid).max() > 1e-4 or not np.isfinite(values).all():
        raise ValueError(f"{course_path}: not a full {N_POINTS:,}-point spectrum on the 600-3000 nm grid")
    if not np.array_equal(values.astype(np.float64), table[:, 1].astype(np.float32).astype(np.float64)):
        raise ValueError(f"{course_path}: values do not fit in float32")
    target = os.path.join(SPECTRA_DIR, stored_name(course_path))
    if os.path.exists(target):
        if not np.array_equal(load_array(target), values):
            raise ValueError(f"{course_path} differs from the {os.path.basename(target)} we already have")
        return target
    os.makedirs(SPECTRA_DIR, exist_ok=True)
    save_array(target, values)
    return target


def import_course_files(delete=True, workers=4):
    """Convert every course file found (in parallel), and delete each original once its converted file is checked."""
    from concurrent.futures import ProcessPoolExecutor
    paths = course_files()
    # One file per target in parallel; the other copies (.txt and .zip, browser duplicates) afterwards,
    # so they are checked against the converted one instead of racing it
    first = list({stored_name(p): p for p in reversed(paths)}.values())
    rest = [p for p in paths if p not in first]
    print(f"converting {len(paths)} course files into {SPECTRA_DIR} ({workers} at a time)")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        done = list(pool.map(convert, first)) + [convert(p) for p in rest]
    for path, target in zip(first + rest, done):
        print(f"  {os.path.basename(path):<45} -> {os.path.basename(target)}" + ("   (deleted)" if delete else ""))
        if delete:
            os.remove(path)
    print(f"done in {time.perf_counter() - t0:.0f} s")


def download(names):
    """
    Fetch spectra files (names or glob patterns) from the repository's GitHub release into spectra/.
    Goes through a temporary folder, so an interrupted download never leaves a half file. Returns False
    if the gh CLI is missing or a file is not on the release.
    """
    import shutil
    import subprocess
    if not shutil.which("gh"):
        return False
    staging = os.path.join(SPECTRA_DIR, ".download")
    os.makedirs(staging, exist_ok=True)
    args = [arg for name in names for arg in ("--pattern", name)]
    print(f"      downloading {' '.join(names)} from the GitHub release '{RELEASE_TAG}'", flush=True)
    done = subprocess.run(["gh", "release", "download", RELEASE_TAG, *args, "--dir", staging, "--clobber"],
                          cwd=HERE, capture_output=True, text=True)
    for name in os.listdir(staging):
        os.replace(os.path.join(staging, name), os.path.join(SPECTRA_DIR, name))
    shutil.rmtree(staging, ignore_errors=True)
    if done.returncode:
        print(f"      download failed: {done.stderr.strip()}")
    return done.returncode == 0


def have_spectrum(seed_digits):
    """Is this seed's spectrum here (converted, or a course file still to convert), or downloadable from GitHub?"""
    def local(path):
        return os.path.exists(path) or any(stored_name(p) == os.path.basename(path) for p in course_files())
    missing = [os.path.basename(p) for p in (SPECTRUM_FILE.format(seed_digits), NOISE_FILE) if not local(p)]
    return not missing or download(missing)


def load_spectrum(seed_digits):
    """Wavelength grid [nm], flux and noise sigma for this seed (converting course files first if needed)."""
    for path in (SPECTRUM_FILE.format(seed_digits), NOISE_FILE):
        if not os.path.exists(path):
            convert(next(p for p in course_files() if stored_name(p) == os.path.basename(path)))
    flux, sigma = load_array(SPECTRUM_FILE.format(seed_digits)), load_array(NOISE_FILE)
    return LAMBDA_START + LAMBDA_STEP * np.arange(len(flux)), flux, sigma


def doppler_max(lambda0):
    """Largest Doppler shift [nm] the spacecraft can see (preparation 1)."""
    return lambda0 * V_MAX / sc.c


def line_sigma(lambda0, T, mass):
    """Gaussian line width [nm] for gas temperature T and molecule mass (preparation 2)."""
    return lambda0 / sc.c * np.sqrt(sc.k * T / mass)


def temperature_from_sigma(lambda0, sigma, mass):
    return mass * (sigma * sc.c / lambda0) ** 2 / sc.k


# ===================================================================== search
def window(lambda0, mass, n_points):
    """Index range of the analysis window around lambda0: Doppler range plus a few line widths."""
    half = doppler_max(lambda0) + WINDOW_SIGMAS * line_sigma(lambda0, T_GRID[-1], mass)
    i0 = int((lambda0 - half - LAMBDA_START) / LAMBDA_STEP)
    i1 = int(np.ceil((lambda0 + half - LAMBDA_START) / LAMBDA_STEP))
    return max(i0, 0), min(i1, n_points)


def full_band(lambda0):
    """Line centres allowed in the blind search: the whole Doppler range [nm, relative to lambda0]."""
    return -doppler_max(lambda0), doppler_max(lambda0)


def grid_search(lam, f, sigma, lambda0, mass, band=None):
    """
    chi^2 on the grid (lambda_c at every data point in the band of allowed centres) x (T in T_GRID),
    with the depth A solved exactly at every grid point. Returns a dict with the surfaces.
    """
    lo, hi = band or full_band(lambda0)
    w = 1 / sigma ** 2
    d = f - 1
    chi2_null = np.sum(w * d ** 2)
    centres = np.flatnonzero((lam - lambda0 >= lo) & (lam - lambda0 <= hi))
    if len(centres) == 0:  # Band narrower than a grid step: use the nearest point
        centres = np.array([np.argmin(np.abs(lam - lambda0 - (lo + hi) / 2))])
    chi2 = np.empty((len(T_GRID), len(centres)))
    depth = np.empty_like(chi2)
    for k, T in enumerate(T_GRID):
        s = line_sigma(lambda0, T, mass) / LAMBDA_STEP  # Width in grid points
        half = int(np.ceil(WINDOW_SIGMAS * s))
        x = np.arange(-half, half + 1)
        g = np.exp(-0.5 * (x / s) ** 2)
        S_fg = signal.fftconvolve(w * d, g, mode="same")[centres]  # Kernel is symmetric: convolution = correlation
        S_gg = signal.fftconvolve(w, g ** 2, mode="same")[centres]
        A = np.clip(S_fg / S_gg, -DEPTH_MAX, 0.0)  # Absorption only, not deeper than DEPTH_MAX
        chi2[k] = chi2_null - 2 * A * S_fg + A ** 2 * S_gg
        depth[k] = A
    return {"chi2": chi2, "depth": depth, "centres": centres, "chi2_null": chi2_null}


def refine(lam, f, sigma, lambda0, mass, grid, band=None):
    """Least-squares fit of (F_min, lambda_c, sigma) from the grid minimum; returns values and 1-sigma errors."""
    k, j = np.unravel_index(np.argmin(grid["chi2"]), grid["chi2"].shape)
    p0 = [1 + grid["depth"][k, j], lam[grid["centres"][j]], line_sigma(lambda0, T_GRID[k], mass)]
    model = lambda p: 1 + (p[0] - 1) * np.exp(-0.5 * ((lam - p[1]) / p[2]) ** 2)
    lo, hi = band or full_band(lambda0)
    lower = [1 - DEPTH_MAX, lambda0 + lo - LAMBDA_STEP / 2, line_sigma(lambda0, T_GRID[0], mass)]
    upper = [1.0, lambda0 + hi + LAMBDA_STEP / 2, line_sigma(lambda0, T_GRID[-1], mass)]
    p0 = np.clip(p0, np.nextafter(lower, upper), np.nextafter(upper, lower))
    fit = optimize.least_squares(lambda p: (f - model(p)) / sigma, p0, bounds=(lower, upper), x_scale=[0.1, 1e-3, 1e-3])
    J = fit.jac
    try:
        errors = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))  # chi^2 already uses the true sigma_i: no rescaling
    except np.linalg.LinAlgError:
        errors = np.full(3, np.nan)
    if 1 - fit.x[0] < 1e-3:  # No line: its position and width are meaningless, and so are their errors
        errors[1:] = np.nan
    chi2 = np.sum(fit.fun ** 2)
    return fit.x, errors, chi2, model


def analyse_line(lam_all, f_all, s_all, gas, lambda0, band=None):
    """Search one line (centres limited to `band`, default the whole Doppler range): grid, refined fit, derived values."""
    mass = molecule_mass(gas)
    i0, i1 = window(lambda0, mass, len(f_all))
    lam, f, sigma = lam_all[i0:i1], f_all[i0:i1].astype(float), s_all[i0:i1].astype(float)
    grid = grid_search(lam, f, sigma, lambda0, mass, band)
    p, err, chi2, model = refine(lam, f, sigma, lambda0, mass, grid, band)
    F_min, lambda_c, width = p
    T = temperature_from_sigma(lambda0, width, mass)
    return {"gas": gas, "lambda0": lambda0, "mass": mass, "lam": lam, "f": f, "sigma": sigma, "grid": grid,
            "F_min": F_min, "F_min_err": err[0], "lambda_c": lambda_c, "v": (lambda_c - lambda0) / lambda0 * sc.c,
            "v_err": err[1] / lambda0 * sc.c, "width": width, "T": T, "T_err": 2 * T * err[2] / width,
            "chi2": chi2, "delta_chi2": grid["chi2_null"] - chi2, "dof": len(f) - 3, "model": model(p),
            "noise": np.median(sigma),
            # The noise sigma_i oscillates along the spectrum: is the line sitting on a noise maximum?
            "noise_at_line": sigma[np.argmin(np.abs(lam - lambda_c))] / np.median(sigma)}


def noise_distribution(lam_all, f_all, s_all, gas, lambda0, band=None, n=N_NOISE_WINDOWS, seed=11):
    """
    delta chi^2 of the best "line" the search finds where no line is listed: the same search (same
    template widths and Doppler range) on windows shifted 1-25 nm away, avoiding every listed line.
    """
    rng = np.random.default_rng(seed + int(lambda0))
    mass = molecule_mass(gas)
    i0, i1 = window(lambda0, mass, len(f_all))
    width = i1 - i0
    listed = np.array([l for lines in LINES.values() for l in lines], dtype=float)
    out = []
    while len(out) < n:
        offset = rng.uniform(1, 25) * rng.choice([-1, 1])
        if np.min(np.abs(listed - (lambda0 + offset))) < 1.0:  # Keep 1 nm from every listed line
            continue
        j0 = i0 + int(round(offset / LAMBDA_STEP))
        if j0 < 0 or j0 + width > len(f_all):
            continue
        lam = lam_all[j0:j0 + width] - offset  # Pretend the window is centred on lambda0, so the templates match
        grid = grid_search(lam, f_all[j0:j0 + width].astype(float), s_all[j0:j0 + width].astype(float), lambda0, mass,
                           band)
        out.append(grid["chi2_null"] - grid["chi2"].min())
    return np.array(out)


def injection_test(lam_all, f_all, s_all, gas, lambda0, threshold, depths, band, n=30, T=300.0, seed=12):
    """Fraction of lines injected inside `band` (temperature T) and recovered above the threshold, per depth."""
    rng = np.random.default_rng(seed + int(lambda0))
    mass = molecule_mass(gas)
    i0, i1 = window(lambda0, mass, len(f_all))
    lam, sigma = lam_all[i0:i1], s_all[i0:i1].astype(float)
    width = line_sigma(lambda0, T, mass)
    rate = []
    for depth in depths:
        found = 0
        for _ in range(n):
            centre = lambda0 + rng.uniform(*band)
            fake = 1 - depth * np.exp(-0.5 * ((lam - centre) / width) ** 2) + rng.normal(0, sigma)
            grid = grid_search(lam, fake, sigma, lambda0, mass, band)
            found += grid["chi2_null"] - grid["chi2"].min() > threshold
        rate.append(found / n)
    return np.array(rate)


# ===================================================================== decisions
def threshold(noise, false_alarm=FALSE_ALARM):
    """
    The delta chi^2 that noise alone exceeds with probability false_alarm. With a few hundred noise windows we
    extrapolate the tail with an exponential fitted to the top 10% (the tail of a chi^2 maximum is close to it).
    """
    top = np.sort(noise)[-len(noise) // 10:]
    return top[0] + np.mean(top - top[0]) * np.log(0.1 / false_alarm)


def common_shift(results):
    """Weighted mean Doppler velocity of the given lines, and its error [m/s]."""
    w = np.array([1 / r["v_err"] ** 2 for r in results])
    return np.sum(w * np.array([r["v"] for r in results])) / w.sum(), 1 / np.sqrt(w.sum())


def physics_checks(r):
    """Temperature in the expected range and F_min > 0.7, both within 2 sigma."""
    return {"temperature": T_RANGE[0] - 2 * r["T_err"] <= r["T"] <= T_RANGE[1] + 2 * r["T_err"],
            "depth": r["F_min"] + 2 * r["F_min_err"] > 0.7}


def targeted_band(lambda0, v_common, v_width):
    """Allowed centres at the common Doppler shift, +-v_width [nm, relative to lambda0]."""
    return lambda0 * (v_common - v_width) / sc.c, lambda0 * (v_common + v_width) / sc.c


def mean_molecular_weight(gases):
    """mu = mean molecule mass in units of u, all detected gases in equal amounts."""
    return np.mean([molecule_mass(g) for g in gases]) / U


# ===================================================================== figures
def plot_overview(lam, f, sigma, results, bins=4000):
    """The whole spectrum (binned) and the noise level, with the listed lines and the verdicts."""
    n = len(f) // bins * bins
    lam_b = lam[:n].reshape(bins, -1).mean(axis=1)
    f_b = f[:n].reshape(bins, -1).mean(axis=1)
    f_lo = f[:n].reshape(bins, -1).min(axis=1)
    s_b = sigma[:n].reshape(bins, -1).mean(axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(lam_b, f_lo, color=MUTED, lw=0.5, label=f"lowest point per bin of {len(f) // bins:,}")
    ax1.plot(lam_b, f_b, color=INK, lw=0.8, label="mean per bin")
    for r in results:
        color = AQUA if r["real"] else (ORANGE if r["significant"] else MUTED)
        for ax in (ax1, ax2):
            ax.axvline(r["lambda0"], color=color, lw=1.2, ls="-" if r["real"] else ":")
        ax1.annotate(r["gas"].replace("2", "$_2$").replace("4", "$_4$"), (r["lambda0"], 1.0), xytext=(2, 70),
                     textcoords="offset points", rotation=90, fontsize=8, color=color)
    ax1.set_ylabel("normalised flux")
    ax1.set_ylim(f_lo.min() * 0.95, 1.2)
    ax1.set_title("10 million points: lines a few points wide are invisible at this scale, so each line is "
                  "searched in its own window (solid green: detected)", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax2.plot(lam_b, s_b, color=ORANGE)
    ax2.set_ylabel("noise $\\sigma_i$")
    ax2.set_xlabel("wavelength [nm]")
    ax2.set_title("The noise changes along the spectrum: chi$^2$ weights every point by $1/\\sigma_i^2$", fontsize=10)
    fig.suptitle("B  The flux meter's spectrum", fontsize=12)
    save(fig, "b1_spectrum")


def plot_fits(results):
    """Every listed line: the data in its window, the best fit and the verdict."""
    n = len(results)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.6 * rows))
    for ax, r in zip(axes.flat, results):
        v = (r["lam"] - r["lambda0"]) / r["lambda0"] * sc.c / 1e3  # Doppler velocity axis [km/s]
        ax.plot(v, r["f"], ".", color=MUTED, ms=2, alpha=0.6)
        ax.fill_between(v, 1 - r["sigma"], 1 + r["sigma"], color=MUTED, alpha=0.12, lw=0)
        color = AQUA if r["real"] else (ORANGE if r["significant"] else BLUE)
        ax.plot(v, r["model"], color=color, lw=1.8)
        ax.axvspan(-V_MAX / 1e3, V_MAX / 1e3, color=AQUA, alpha=0.04, lw=0)
        verdict = "REAL" if r["real"] else ("significant, fails: " + ", ".join(k for k, ok in r["checks"].items() if not ok)
                                            if r["significant"] else "noise")
        ax.set_title(f"{r['gas']} {r['lambda0']} nm: {verdict}\nF$_{{min}}$ = {r['F_min']:.2f}, v = {r['v'] / 1e3:+.2f} km/s, "
                     f"T = {r['T']:.0f} K, $\\Delta\\chi^2$ = {r['delta_chi2']:.0f}", fontsize=9, color=INK)
        ax.set_ylim(0.4, 1.6)
        ax.set_xlim(-1.6 * V_MAX / 1e3, 1.6 * V_MAX / 1e3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Doppler velocity [km/s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("flux")
    fig.suptitle("B  Every listed line, fitted by chi$^2$ (grey band: $\\pm\\sigma_i$; shaded: the $\\pm$10 km/s the "
                 "spacecraft can cause)", fontsize=12)
    save(fig, "b2_line_fits", rect=(0, 0, 1, 0.97))


def plot_chi2_maps(results, n_max=4):
    """chi^2 over (Doppler velocity, temperature) of the blind search for the strongest lines: one clear valley."""
    best = [r["blind"] for r in sorted(results, key=lambda r: -r["blind"]["delta_chi2"])[:n_max]]
    fig, axes = plt.subplots(1, len(best), figsize=(5 * len(best), 4.5))
    for ax, r in zip(np.atleast_1d(axes), best):
        g = r["grid"]
        v = (r["lam"][g["centres"]] - r["lambda0"]) / r["lambda0"] * sc.c / 1e3
        delta = g["chi2"] - g["chi2"].min()
        im = ax.pcolormesh(v, T_GRID, np.minimum(delta, 50), cmap="viridis", shading="auto")
        ax.contour(v, T_GRID, delta, levels=[2.3, 6.18, 11.8], colors="white", linewidths=0.8)
        ax.plot(r["v"] / 1e3, r["T"], "+", color="white", ms=12, mew=2)
        ax.axhspan(*T_RANGE, color="white", alpha=0.0)
        for T in T_RANGE:
            ax.axhline(T, color="white", ls=":", lw=0.8)
        ax.set_xlabel("Doppler velocity [km/s]")
        ax.set_ylabel("temperature [K]")
        ax.set_title(f"{r['gas']} {r['lambda0']} nm", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=np.atleast_1d(axes)[-1], label=r"$\chi^2 - \chi^2_{min}$ (capped at 50)")
    fig.suptitle("B  chi$^2$ surfaces of the strongest lines (contours: 1, 2, 3$\\sigma$ for two parameters)", fontsize=12)
    save(fig, "b3_chi2_maps")


def plot_evidence(results, noise, thresholds, v_common, v_common_err):
    """The three tests side by side for every line: significance, Doppler shift, temperature."""
    labels = [f"{r['gas']} {r['lambda0']}" for r in results]
    y = np.arange(len(results))[::-1]
    colors = [AQUA if r["real"] else (ORANGE if r["significant"] else MUTED) for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    ax = axes[0]
    for yi, r in zip(y, results):
        n = noise[(r["gas"], r["lambda0"])]
        ax.plot(np.percentile(n, [50, 99]), [yi - 0.15] * 2, color=MUTED, lw=5, alpha=0.4, solid_capstyle="butt")
        ax.plot(thresholds[(r["gas"], r["lambda0"])], yi - 0.15, "|", color=INK, ms=12, mew=2)
        ax.plot(r["blind_threshold"], yi + 0.15, "|", color=MUTED, ms=12, mew=2)
    ax.scatter([max(r["blind"]["delta_chi2"], 0.5) for r in results], y + 0.15, color="white", s=45, zorder=3,
               edgecolor=MUTED, label="blind search (+-10 km/s) and its threshold")
    ax.scatter([max(r["delta_chi2"], 0.5) for r in results], y - 0.15, color=colors, s=70, zorder=3, edgecolor=INK,
               label="targeted search (common shift) and its threshold")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta\chi^2 = \chi^2_{\rm no\,line} - \chi^2_{\rm best}$")
    ax.set_title(f"Significance (grey bars: noise-only windows, median to 99%;\n| : threshold for {FALSE_ALARM:g} "
                 "false-alarm probability)", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax = axes[1]
    ax.errorbar([r["blind"]["v"] / 1e3 for r in results], y, xerr=[r["blind"]["v_err"] / 1e3 for r in results],
                fmt="none", ecolor=MUTED)
    ax.scatter([r["blind"]["v"] / 1e3 for r in results], y, color=colors, s=60, zorder=3, edgecolor=INK)
    if np.isfinite(v_common):
        ax.axvspan((v_common - 3 * v_common_err) / 1e3, (v_common + 3 * v_common_err) / 1e3, color=AQUA, alpha=0.2, lw=0)
        ax.axvline(v_common / 1e3, color=AQUA, lw=1.5, label=f"common shift {v_common / 1e3:+.3f} km/s (targeted band)")
        ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(-V_MAX / 1e3 * 1.05, V_MAX / 1e3 * 1.05)
    ax.set_xlabel("Doppler velocity [km/s]")
    ax.set_title("Blind fits: the strong lines share one Doppler shift;\nnoise fits land anywhere in +-10 km/s", fontsize=10)
    ax = axes[2]
    ax.errorbar([r["T"] for r in results], y, xerr=[r["T_err"] for r in results], fmt="none", ecolor=MUTED)
    ax.scatter([r["T"] for r in results], y, color=colors, s=60, zorder=3, edgecolor=INK)
    ax.axvspan(*T_RANGE, color=AQUA, alpha=0.12, lw=0, label="expected 150-450 K")
    ax.set_xlim(T_GRID[0], T_GRID[-1])
    ax.set_xlabel("temperature from the line width [K]")
    ax.set_title("...and a plausible temperature", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle("B  Real lines or flukes? (green: real, orange: significant but fails a physics check, grey: noise)",
                 fontsize=12)
    save(fig, "b4_evidence")


def plot_completeness(depths, rates, noise_levels):
    fig, ax = plt.subplots(figsize=(10, 5))
    order = np.argsort(list(noise_levels.values()))
    keys = list(rates)
    for i, color in zip(order, SERIES * 4):
        gas, lambda0 = keys[i]
        ax.plot(depths, rates[keys[i]], "o-", ms=4, lw=1.4, color=color if i < 3 else None,
                label=f"{gas} {lambda0} nm (noise {noise_levels[keys[i]]:.2f})")
    ax.axhline(0.9, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("depth of the injected line, 1 - F$_{min}$")
    ax.set_ylabel("fraction recovered")
    ax.set_title("Injected lines (T = 300 K, random Doppler shift) recovered above the noise threshold:\n"
                 "a gas we did not detect cannot have lines deeper than about this", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("B  Completeness of the search", fontsize=12)
    save(fig, "b5_completeness")


# ===================================================================== main
def main(seed_digits=None):
    from common.config import SEED
    seed_digits = SEED % 100 if seed_digits is None else seed_digits
    if not have_spectrum(seed_digits):
        print(f"B     SKIPPED: no spectrum for seed {SEED} (spectra/spectrum_seed{seed_digits:02d}.parquet, or the course's\n"
              f"      spectrum_seed{seed_digits:02d}_600nm_3000nm.txt[.zip] in {HERE}) or no sigma_noise.\n"
              f"      Until then challenge C uses the drag probe alone.")
        with open(RESULT_FILE, "w") as fh:
            json.dump({"seed": SEED, "spectrum": False, "detected": [], "marginal": []}, fh, indent=2)
        return
    t0 = time.perf_counter()
    lam, f, sigma = load_spectrum(seed_digits)
    print(f"B     spectrum seed{seed_digits:02d}: {len(f):,} points, {lam[0]:.0f}-{lam[-1]:.0f} nm, noise sigma "
          f"{sigma.min():.2f}-{sigma.max():.2f} (loaded in {time.perf_counter() - t0:.1f} s)")
    print(f"      Doppler limit (10 km/s): {doppler_max(632):.4f} nm at 632 nm ... {doppler_max(2870):.4f} nm at 2870 nm; "
          f"line width O2 632 nm at 150-450 K: {line_sigma(632, 150, molecule_mass('O2')):.5f}-"
          f"{line_sigma(632, 450, molecule_mass('O2')):.5f} nm ({LAMBDA_STEP} nm per point)")

    lines = [(gas, l0) for gas, ls in LINES.items() for l0 in ls]

    # Stage 1, blind: every line anywhere in +-10 km/s. Strong lines fix the common Doppler shift.
    blind = {key: analyse_line(lam, f, sigma, *key) for key in lines}
    blind_noise = {key: noise_distribution(lam, f, sigma, *key) for key in lines}
    blind_thr = {key: threshold(n) for key, n in blind_noise.items()}
    strong = [r for key, r in blind.items() if r["delta_chi2"] > blind_thr[key] and all(physics_checks(r).values())]
    v_blind, v_blind_err = common_shift(strong)
    # A line agrees with that shift if within 3 of its own sigmas; the targeted band allows exactly that.
    v_width = 3 * np.median([r["v_err"] for r in strong])

    # Stage 2, targeted: every line again, but only at the common shift. Far fewer places for noise to hide,
    # so the threshold (again measured on line-free windows, with the same narrow band) is much lower.
    results, thresholds, noise = [], {}, {}
    for key in lines:
        band = targeted_band(key[1], v_blind, v_width)
        r = analyse_line(lam, f, sigma, *key, band=band)
        noise[key] = noise_distribution(lam, f, sigma, *key, band=band)
        thresholds[key] = threshold(noise[key])
        r["band"] = band
        r["blind"] = blind[key]
        r["blind_threshold"] = blind_thr[key]
        r["blind_significant"] = blind[key]["delta_chi2"] > blind_thr[key]
        r["significant"] = r["delta_chi2"] > thresholds[key]
        r["checks"] = physics_checks(r)
        r["real"] = r["significant"] and all(r["checks"].values())
        r["marginal"] = (not r["real"] and r["delta_chi2"] > threshold(noise[key], MARGINAL)
                         and all(r["checks"].values()))
        results.append(r)
    v_common, v_common_err = common_shift([r for r in results if r["real"]])

    print(f"      stage 1 (blind, +-{V_MAX / 1e3:.0f} km/s): {len(strong)} strong lines share v = {v_blind / 1e3:+.3f} "
          f"+- {v_blind_err / 1e3:.3f} km/s; stage 2 searches +-{v_width / 1e3:.2f} km/s around it")
    print("      line          blind dchi2/thr   targeted dchi2/thr   F_min          v [km/s]          T [K]        verdict")
    for r in results:
        key = (r["gas"], r["lambda0"])
        verdict = "REAL" if r["real"] else ("significant but fails " + ", ".join(k for k, ok in r["checks"].items() if not ok)
                                            if r["significant"] else "noise")
        print(f"      {r['gas']:>4} {r['lambda0']:>4} nm  {r['blind']['delta_chi2']:7.1f} / {r['blind_threshold']:5.1f}"
              f"     {r['delta_chi2']:7.1f} / {thresholds[key]:5.1f}     "
              f"{r['F_min']:.3f}+-{r['F_min_err']:.3f}  {r['v'] / 1e3:+7.3f}+-{r['v_err'] / 1e3:.3f}  "
              f"{r['T']:5.0f}+-{r['T_err']:4.0f}   {verdict}")
    gases = sorted({r["gas"] for r in results if r["real"]}, key=list(LINES).index)
    marginal = sorted({r["gas"] for r in results if r["marginal"]} - set(gases), key=list(LINES).index)
    mu = mean_molecular_weight(gases) if gases else np.nan
    with open(RESULT_FILE, "w") as fh:
        json.dump({"seed": SEED, "spectrum": True, "detected": gases, "marginal": marginal, "v_common": v_common,
                   "marginal_lines": [f"{r['gas']} {r['lambda0']}" for r in results if r["marginal"]]}, fh, indent=2)
    print(f"      common Doppler shift {v_common / 1e3:+.3f} +- {v_common_err / 1e3:.3f} km/s; gases: {', '.join(gases)}; "
          f"mu = {mu:.2f}; marginal (false alarm < {MARGINAL:g}): {', '.join(marginal) or 'none'}")

    depths = np.linspace(0.0, 0.3, 13)
    rates = {(r["gas"], r["lambda0"]): injection_test(lam, f, sigma, r["gas"], r["lambda0"],
                                                      thresholds[(r["gas"], r["lambda0"])], depths, r["band"])
             for r in results}
    noise_levels = {(r["gas"], r["lambda0"]): r["noise"] for r in results}
    depth90 = {k: depths[np.argmax(v >= 0.9)] if (v >= 0.9).any() else np.nan for k, v in rates.items()}

    plot_overview(lam, f, sigma, results)
    plot_fits(results)
    plot_chi2_maps(results)
    plot_evidence(results, noise, thresholds, v_common, v_common_err)
    plot_completeness(depths, rates, noise_levels)

    real = [r for r in results if r["real"]]
    record_many({
        "v-max": (V_MAX / 1e3, "{:.0f}"), "dl-632": (doppler_max(632), "{:.4f}"), "dl-2870": (doppler_max(2870), "{:.4f}"),
        "w-o2-150": (line_sigma(632, 150, molecule_mass("O2")), "{:.5f}"),
        "w-o2-450": (line_sigma(632, 450, molecule_mass("O2")), "{:.5f}"),
        "n-points": (len(f), "{:,}"), "step": (LAMBDA_STEP, "{:g}"),
        "noise-min": (sigma.min(), "{:.2f}"), "noise-max": (sigma.max(), "{:.2f}"),
        "n-lines": (len(results), "{:d}"), "n-real": (len(real), "{:d}"),
        "n-significant": (sum(r["significant"] for r in results), "{:d}"),
        "n-blind": (len(strong), "{:d}"), "v-blind": (v_blind / 1e3, "{:+.3f}"), "v-blind-err": (v_blind_err / 1e3, "{:.3f}"),
        "v-width": (v_width / 1e3, "{:.2f}"),
        "false-alarm": (FALSE_ALARM, "{:g}"), "n-noise-windows": (N_NOISE_WINDOWS, "{:d}"),
        "v-common": (v_common / 1e3, "{:+.3f}"), "v-common-err": (v_common_err / 1e3, "{:.3f}"),
        "gases": ", ".join(g.replace("2", "$_2$").replace("4", "$_4$") for g in gases) or "none",
        "marginal-gases": ", ".join(g.replace("2", "$_2$") for g in marginal) or "none", "marginal-fa": (MARGINAL, "{:g}"),
        "n-gases": (len(gases), "{:d}"), "mu": (mu, "{:.2f}"),
        "T-lines-mean": (np.mean([r["T"] for r in real]) if real else np.nan, "{:.0f}"),
    })
    for r in results:
        key = f"{r['gas']}-{r['lambda0']}"
        record_many({f"{key}-dchi2": (r["delta_chi2"], "{:.0f}"), f"{key}-Fmin": (r["F_min"], "{:.2f}"),
                     f"{key}-v": (r["v"] / 1e3, "{:+.2f}"), f"{key}-T": (r["T"], "{:.0f}"),
                     f"{key}-thr": (thresholds[(r["gas"], r["lambda0"])], "{:.0f}"),
                     f"{key}-blind": (r["blind"]["delta_chi2"], "{:.0f}"), f"{key}-blind-thr": (r["blind_threshold"], "{:.0f}"),
                     f"{key}-verdict": "real" if r["real"] else ("rejected" if r["significant"] else "noise"),
                     f"{key}-depth90": (depth90[(r["gas"], r["lambda0"])], "{:.3f}"),
                     f"{key}-noise-at": (r["noise_at_line"], "{:.2f}"), f"{key}-Fmin-err": (r["F_min_err"], "{:.2f}")})
    for gas in LINES:
        record(f"mass-{gas}", molecule_mass(gas) / U, "{:.2f}")
    print(f"      done in {time.perf_counter() - t0:.0f} s")
    return gases, mu, v_common, results


if __name__ == "__main__":
    if "--import" in sys.argv:  # Convert the course files to Parquet (add --keep to keep the originals)
        import_course_files(delete="--keep" not in sys.argv)
    elif "--download" in sys.argv:  # Every spectrum from the GitHub release
        download(["*.parquet"])
    else:
        main()
