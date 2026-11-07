"""
Part 4, challenges A and B: pictures of the sky, and which way we are pointing.

A. himmelkule.npy is the whole sky: one RGB colour per direction (theta, phi),
   stored as a HEALPix map (12 x 512^2 pixels of equal area). A camera picture
   is a stereographic projection of a 70 x 70 degree patch of that sphere:

     1. Make a grid of plane coordinates (X, Y) with one point per pixel,
        spanning +-X_max = +-2 sin(alpha/2) / (1 + cos(alpha/2)).
     2. Turn every (X, Y) into a direction (theta, phi) with the inverse
        projection (equations 3-4 in the assignment, theta0 = 90 deg).
     3. Look up the colour of each direction in the sky map.

   Check: our picture for phi0 = 0 must equal sample0000.png pixel by pixel.

B. To find where a new picture points, compare it with reference pictures for
   phi0 = 0, 1, ..., 359 degrees (least squares of the RGB difference) and take
   the best; then refine to 0.1 degree around it.

Clever bits:
  * The API's lookup get_sky_image_pixel(theta, phi) works one pixel at a
    time: 360 full pictures would be 110 million Python calls. We port its
    HEALPix formula to NumPy and check that it gives identical indices.
  * Every picture uses the same (X, Y) grid, so the inverse projection is done
    once and only phi0 changes.
  * Pictures are compared after averaging 8 x 8 pixel blocks. Single pixels are
    dominated by sharp stars and texture: a picture shifted by only 0.4 deg
    differs pixel by pixel as much as one shifted by 3 deg, so a raw comparison
    has no valley to slide into. The blurred pictures keep the large structure
    (the bright galactic band), which changes smoothly with angle, so the
    match score gets one wide, smooth minimum. It also makes the 360-picture
    library small (~20 MB instead of 330 MB).
  * The best whole degree is refined to 0.01 deg by fitting a parabola to the
    scores around the minimum.
  * Colours are converted to float before subtracting: uint8 arithmetic wraps
    (100 - 200 = 156).

Run:  ../.venv/bin/python sky_pictures.py
Figures go to figures/a*.png and figures/b*.png.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # Repo root, for common/

import time

import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission
from PIL import Image

from common.plot_style import AQUA, BLUE, INK, MUTED, ORANGE, VIOLET, save
from common.release import fetch
from common.report import record

SKY_FILE = os.path.join(HERE, "himmelkule.npy")  # Course file, on the GitHub release (common/release.py)
SAMPLE_FILES = {0.0: os.path.join(HERE, "sample0000.png")}  # Pictures with a known pointing angle
EXTRA_SAMPLE = os.path.join(HERE, "sample0200.png")  # Angle not stated: our software finds it
LIBRARY_FILE = os.path.join(HERE, "reference_library.npz")
FOV = np.radians(70)  # Field of view, both directions
WIDTH, HEIGHT = 640, 480  # Picture size [pixels]
BLOCK = 8  # Pictures are compared as averages of 8 x 8 pixel blocks
NSIDE = 512  # HEALPix resolution of himmelkule.npy
GRIDLINE = "#d8d7d2"


# ===================================================================== sky map
def have_sky():
    """Is the sky sphere here, or downloadable from the GitHub release?"""
    return os.path.exists(SKY_FILE) or fetch(SKY_FILE)


def load_sky():
    """RGB colour of every HEALPix pixel, shape (n_pixels, 3), uint8. Columns 2-4 of the file are R, G, B."""
    return np.load(SKY_FILE)[:, 2:5].astype(np.uint8)


def sky_pixel_index(theta, phi):
    """
    Vectorised SpaceMission.get_sky_image_pixel: HEALPix ring-scheme index of each direction.

    Line by line the same formulas as the API, but on whole arrays at once.
    """
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi, 2 * np.pi) / (np.pi / 2)  # In [0, 4)

    # Equatorial region, |z| <= 2/3
    temp1 = NSIDE * (0.5 + tt)
    temp2 = NSIDE * 0.75 * z
    jp = np.floor(temp1 - temp2).astype(np.int64)  # Ascending edge line
    jm = np.floor(temp1 + temp2).astype(np.int64)  # Descending edge line
    ir = NSIDE + 1 + jp - jm  # Ring number counted from z = 2/3
    kshift = 1 - ir % 2
    ip = (jp + jm - NSIDE + kshift + 1) // 2
    ip = np.where(ip >= 4 * NSIDE, ip - 4 * NSIDE, ip)
    equatorial = 2 * NSIDE * (NSIDE - 1) + 4 * NSIDE * (ir - 1) + ip

    # Polar caps
    tp = tt - np.floor(tt)
    tmp = NSIDE * np.sqrt(3.0 * (1.0 - za))
    jp = np.floor(tp * tmp).astype(np.int64)
    jm = np.floor((1.0 - tp) * tmp).astype(np.int64)
    ir = jp + jm + 1  # Ring number counted from the closest pole
    ip = np.floor(tt * ir).astype(np.int64)
    ip = np.where(ip >= 4 * ir, ip - 4 * ir, ip)
    north = 2 * ir * (ir - 1) + ip
    south = 12 * NSIDE ** 2 - 2 * ir * (ir + 1) + ip

    return np.where(za <= 2 / 3, equatorial, np.where(z > 0, north, south))


# ======================================================== stereographic projection
def plane_limits(fov=FOV):
    """X_max (= Y_max for a square field of view), equation 7."""
    return 2 * np.sin(fov / 2) / (1 + np.cos(fov / 2))


def picture_directions(width=WIDTH, height=HEIGHT, fov=FOV, stride=1):
    """
    (theta, phi - phi0) of every pixel, shape (height, width), for a picture centred on theta0 = 90 deg.

    X grows to the right, Y grows upwards (row 0 is the top of the picture).
    Equations 3-4 with theta0 = pi/2 (cos theta0 = 0, sin theta0 = 1).
    """
    X_max = Y_max = plane_limits(fov)
    X, Y = np.meshgrid(np.linspace(-X_max, X_max, width)[::stride], np.linspace(Y_max, -Y_max, height)[::stride])
    rho = np.hypot(X, Y)
    beta = 2 * np.arctan(rho / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.pi / 2 - np.arcsin(np.where(rho > 0, Y * np.sin(beta) / rho, 0.0))
        dphi = np.arctan2(X * np.sin(beta), rho * np.cos(beta))
    return theta, np.where(rho > 0, dphi, 0.0)


def make_picture(sky, phi0, directions=None):
    """Picture (height, width, 3) uint8 pointing at azimuth phi0 [rad] in the solar system plane."""
    theta, dphi = picture_directions() if directions is None else directions
    return sky[sky_pixel_index(theta, phi0 + dphi)]


# ============================================================ reference library
def blur(picture, block=BLOCK):
    """Average over block x block pixels -> (h/block, w/block, 3) float. Floats: uint8 arithmetic wraps."""
    h, w = picture.shape[0] // block * block, picture.shape[1] // block * block
    p = picture[:h, :w].astype(np.float64)
    return p.reshape(h // block, block, w // block, block, 3).mean(axis=(1, 3))


def build_library(sky):
    """Blurred reference pictures for phi0 = 0..359 deg, shape (360, h/BLOCK, w/BLOCK, 3) float32."""
    directions = picture_directions()
    return np.array([blur(make_picture(sky, np.radians(deg), directions)) for deg in range(360)], dtype=np.float32)


def load_or_build_library(sky):
    if os.path.exists(LIBRARY_FILE):
        return np.load(LIBRARY_FILE)["pictures"]
    t0 = time.perf_counter()
    library = build_library(sky)
    np.savez_compressed(LIBRARY_FILE, pictures=library)
    print(f"A3    built 360 reference pictures in {time.perf_counter() - t0:.1f} s -> {os.path.basename(LIBRARY_FILE)}")
    return library


def match_scores(picture, library):
    """Mean squared difference between the blurred picture and each blurred reference."""
    return np.mean((library.astype(np.float64) - blur(picture)[None]) ** 2, axis=(1, 2, 3))


def find_pointing(picture, library):
    """
    Azimuth [deg] the picture points at, and the score for every whole degree.
    Best whole degree, refined with a parabola through it and its two neighbours.
    """
    scores = match_scores(picture, library)
    best = int(np.argmin(scores))
    left, mid, right = scores[(best - 1) % 360], scores[best], scores[(best + 1) % 360]
    curvature = left - 2 * mid + right
    offset = 0.5 * (left - right) / curvature if curvature > 0 else 0.0
    return float(np.round(best + offset, 2) % 360), scores


# ===================================================================== figures
def plot_projection(sky, ours, sample):
    """The whole sky as a map, the 70-degree patch the camera sees, and ours vs. the sample picture."""
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    ax = fig.add_subplot(grid[0, :])
    theta_map, phi_map = np.meshgrid(np.linspace(0, np.pi, 300), np.linspace(0, 2 * np.pi, 600), indexing="ij")
    ax.imshow(sky[sky_pixel_index(theta_map, phi_map)], extent=(0, 360, 180, 0), aspect="auto")
    theta, dphi = picture_directions(stride=8)
    edge = np.concatenate([np.c_[theta[0], dphi[0]], np.c_[theta[:, -1], dphi[:, -1]],
                           np.c_[theta[-1, ::-1], dphi[-1, ::-1]], np.c_[theta[::-1, 0], dphi[::-1, 0]]])
    ax.plot(np.degrees(np.mod(edge[:, 1], 2 * np.pi)) % 360, np.degrees(edge[:, 0]), ".", color="white", ms=2)
    ax.axhline(90, color="white", lw=0.6, ls=":")
    ax.text(5, 87, "solar system plane (theta = 90 deg)", color="white", fontsize=8)
    ax.set_xlabel(r"azimuth $\phi$ [deg]")
    ax.set_ylabel(r"polar angle $\theta$ [deg]")
    ax.set_title("The whole sky (himmelkule.npy); white outline: what a 70$^\\circ$ picture at $\\phi_0$ = 0 covers",
                 fontsize=10)
    ax.grid(False)
    diff = (sample.astype(int) - ours.astype(int)).mean(axis=2)
    for i, (img, title) in enumerate(((ours, "Our stereographic projection"), (sample, "sample0000.png"),
                                      (None, f"sample $-$ ours: noise only (std {diff.std():.1f})"))):
        ax = fig.add_subplot(grid[1, i])
        if img is None:
            ax.imshow(diff, cmap="gray", vmin=-30, vmax=30)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("A  From the sky sphere to a flat camera picture", fontsize=12)
    save(fig, "a_stereographic_projection")


def plot_pointing(sky, library, tests):
    """Match score vs. angle for each test picture, and a panorama strip of the references."""
    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.3])
    ax = fig.add_subplot(grid[0])
    panorama = np.concatenate([library[d][:, library.shape[2] // 2:library.shape[2] // 2 + 1]
                               for d in range(360)], axis=1) / 255  # Centre column of every reference picture
    ax.imshow(panorama, extent=(0, 360, 0, 70), aspect="auto")
    ax.set_xlabel(r"$\phi_0$ [deg]")
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("360$^\\circ$ panorama along the solar system plane (centre strip of every reference picture)",
                 fontsize=10)
    ax = fig.add_subplot(grid[1])
    for (name, angle, scores), color in zip(tests, (BLUE, ORANGE, INK, MUTED)):
        ax.semilogy(np.arange(360), scores, color=color, lw=1.5, label=f"{name}: best {angle:.2f}$^\\circ$")
        ax.axvline(angle, color=color, lw=0.8, ls=":")
    ax.set_xlabel(r"reference picture $\phi_0$ [deg]")
    ax.set_ylabel("mean squared difference (8x8 blocks)")
    ax.set_xlim(0, 359)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("B  Comparing a picture with all 360 references: one sharp minimum", fontsize=10)
    fig.suptitle("B  Which way is the spacecraft pointing?", fontsize=12)
    save(fig, "b_pointing")


def plot_pointing_robustness(sky, library, noise_levels=(0, 4, 16, 32, 64, 128), n_angles=12, seed=9):
    """
    How good is the pointing software? (1) Error against picture noise, at
    random angles between the library's whole degrees. (2) Why we blur: the
    match score near the true angle for different block sizes. Raw pixels give
    a narrow, ragged valley that a whole-degree library cannot find; blocks of
    8 x 8 give one smooth, wide valley.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 360, n_angles)
    clean = [make_picture(sky, np.radians(a)) for a in angles]
    worst, median = [], []
    for sigma in noise_levels:
        errors = []
        for a, picture in zip(angles, clean):
            noisy = np.clip(picture + rng.normal(0, sigma, picture.shape), 0, 255).astype(np.uint8)
            found, _ = find_pointing(noisy, library)
            errors.append(abs((found - a + 180) % 360 - 180))
        worst.append(max(errors))
        median.append(np.median(errors))

    directions = picture_directions()
    centre = 123.0  # Any angle; the shape of the valley is what matters
    reference = make_picture(sky, np.radians(centre), directions)
    offsets = np.linspace(-6, 6, 97)
    pictures = [make_picture(sky, np.radians(centre + o), directions) for o in offsets]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.semilogy(noise_levels, np.maximum(worst, 1e-3), "o-", color=ORANGE, label="worst of the test angles")
    ax1.semilogy(noise_levels, np.maximum(median, 1e-3), "o-", color=BLUE, label="median")
    ax1.axhline(2, color=INK, ls="--", lw=1.2)
    ax1.text(noise_levels[0], 2, " API tolerance: 2$^\\circ$", va="bottom", fontsize=9)
    ax1.axvline(4, color=MUTED, ls=":", lw=1)
    ax1.text(4, ax1.get_ylim()[0] * 1.5, " noise in the\n sample pictures", fontsize=8, color=MUTED)
    ax1.set_xlabel("Gaussian noise added to every pixel (0-255 scale)")
    ax1.set_ylabel("pointing error [deg]")
    ax1.set_title(f"Robust far beyond the real noise ({n_angles} random angles per level)", fontsize=10)
    ax1.legend(fontsize=8, loc="center right")
    for block, color in zip((1, 2, 8, 32), (MUTED, AQUA, BLUE, VIOLET)):
        ref = blur(reference, block) if block > 1 else reference.astype(np.float64)
        scores = np.array([np.mean(((blur(p, block) if block > 1 else p.astype(np.float64)) - ref) ** 2)
                           for p in pictures])
        ax2.plot(offsets, scores / scores.max(), color=color, lw=2 if block == BLOCK else 1.3,
                 label=f"{block} x {block} blocks" + ("  (used)" if block == BLOCK else ""))
    for x in range(-6, 7):
        ax2.axvline(x, color=GRIDLINE, lw=0.6, zorder=0)
    ax2.set_xlabel("picture angle minus true angle [deg] (grey lines: the library's whole degrees)")
    ax2.set_ylabel("match score (normalised)")
    ax2.set_title("Why blur: raw pixels decorrelate within ~0.5$^\\circ$, blocks keep a smooth valley", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")
    fig.suptitle("B  How reliable is the pointing software?", fontsize=12)
    save(fig, "b2_pointing_robustness")
    return dict(zip(noise_levels, worst))


def main(extra_pictures=()):
    """extra_pictures: (name, path) of pictures to analyse as well (e.g. the spacecraft's own)."""
    if not have_sky():
        raise FileNotFoundError(f"{SKY_FILE} is missing, and could not be downloaded from the GitHub release")
    sky = load_sky()

    # The vectorised lookup must agree with the API exactly
    rng = np.random.default_rng(4)
    theta, phi = rng.uniform(0, np.pi, 20000), rng.uniform(-10, 10, 20000)
    api = np.array([SpaceMission.get_sky_image_pixel(t, p) for t, p in zip(theta, phi)])
    mismatches = np.count_nonzero(sky_pixel_index(theta, phi) != api)
    print(f"A     vectorised HEALPix lookup vs. the API on 20 000 random directions: {mismatches} mismatches")

    sample = np.array(Image.open(SAMPLE_FILES[0.0]).convert("RGB"))
    height, width = sample.shape[:2]
    X_max = plane_limits()
    print(f"A2    sample0000.png is {width} x {height} pixels; X and Y span +-{X_max:.4f} (FOV 70 deg)")
    ours = make_picture(sky, 0.0)
    residual = sample.astype(int) - ours.astype(int)
    print(f"      our picture at phi0 = 0 minus sample0000.png: mean {residual.mean():+.3f}, std {residual.std():.2f} "
          f"per colour: pure noise (the sample has noise added), no structure left")
    for name, flipped in (("left-right", ours[:, ::-1]), ("upside-down", ours[::-1])):
        print(f"      (flipped {name}: mean |difference| {np.abs(sample - flipped.astype(int)).mean():.1f} "
              f"instead of {np.abs(residual).mean():.1f})")
    plot_projection(sky, ours, sample)

    library = load_or_build_library(sky)
    tests = []
    pictures = [("sample0000.png", SAMPLE_FILES[0.0]), ("sample0200.png", EXTRA_SAMPLE), *extra_pictures]
    for name, path in pictures:
        if not os.path.exists(path):
            continue
        picture = np.array(Image.open(path).convert("RGB"))
        angle, scores = find_pointing(picture, library)
        tests.append((name, angle, scores))
        print(f"B     {name}: points at phi = {angle:.2f} deg")
    # Pictures at angles between the library's whole degrees, with noise like the sample's
    rng = np.random.default_rng(5)
    errors = []
    for angle_true in rng.uniform(0, 360, 20):
        noisy = np.clip(make_picture(sky, np.radians(angle_true)) + rng.normal(0, 4, (HEIGHT, WIDTH, 3)), 0, 255)
        angle, _ = find_pointing(noisy.astype(np.uint8), library)
        errors.append(abs((angle - angle_true + 180) % 360 - 180))
    print(f"      20 noisy test pictures at random angles: largest error {max(errors):.2f} deg (API allows 2 deg)")
    plot_pointing(sky, library, tests)
    worst_by_noise = plot_pointing_robustness(sky, library)
    record("hp-mismatch", mismatches, "{:d}")
    record("sample-res-std", residual.std(), "{:.1f}")
    record("pointing-max-err", max(errors), "{:.2f}")
    record("pointing-err-64", worst_by_noise[64], "{:.2f}")
    record("pointing-noise-ok", max(k for k, v in worst_by_noise.items() if v < 2), "{:d}")
    record("pointing-err-128", worst_by_noise[128], "{:.2f}")
    record("block", BLOCK, "{:d}")
    for name, angle, _ in tests:
        key = {"sample0000.png": "angle-sample0", "sample0200.png": "angle-sample200"}.get(name, "angle-craft")
        record(key, angle, "{:.2f}")
    return sky, library, {name: angle for name, angle, _ in tests}


if __name__ == "__main__":
    main()
