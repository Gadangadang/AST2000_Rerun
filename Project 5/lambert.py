"""
Lambert's problem, vectorised: which orbit around the star goes from r1 to r2 in time dt?

Universal-variable method (Curtis, "Orbital Mechanics for Engineering Students",
algorithm 5.2), for the prograde (counterclockwise) single-revolution orbit:

    A    = sin(dtheta) sqrt(r1 r2 / (1 - cos dtheta))
    y(z) = r1 + r2 + A (z S(z) - 1) / sqrt(C(z))
    F(z) = (y / C)^(3/2) S + A sqrt(y) - sqrt(mu) dt = 0

with the Stumpff functions C(z), S(z). F grows monotonically with z, so instead
of Newton's method (which needs a good start for every case) we bisect, for a
whole grid of (launch date, travel time) pairs at once: 80 halvings pin z down
to machine precision for all of them in a few milliseconds.

Then the Lagrange coefficients give the velocities at both ends:
    f = 1 - y/r1,   g = A sqrt(y/mu),   g' = 1 - y/r2
    v1 = (r2 - f r1) / g,   v2 = (g' r2 - r1) / g
"""
import numpy as np


def stumpff(z):
    """Stumpff functions C(z), S(z), with series near z = 0 to avoid 0/0."""
    z = np.asarray(z, dtype=float)
    C, S = np.empty_like(z), np.empty_like(z)
    pos, neg, small = z > 1e-6, z < -1e-6, np.abs(z) <= 1e-6
    sz = np.sqrt(z[pos])
    C[pos], S[pos] = (1 - np.cos(sz)) / z[pos], (sz - np.sin(sz)) / sz ** 3
    sz = np.sqrt(-z[neg])
    C[neg], S[neg] = (np.cosh(sz) - 1) / -z[neg], (np.sinh(sz) - sz) / sz ** 3
    C[small], S[small] = 1 / 2 - z[small] / 24, 1 / 6 - z[small] / 120
    return C, S


def lambert(r1, r2, dt, mu, n_iterations=80):
    """
    Velocities v1 (at r1) and v2 (at r2) of the prograde orbit that takes dt to go from r1 to r2.

    r1, r2: (..., 2) [AU]; dt: (...) [yr]; mu = G M [AU^3/yr^2]. All broadcast together.
    Returns v1, v2 (..., 2) [AU/yr]; NaN where no solution was found.
    """
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    r1n, r2n = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    cross = r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0]
    dtheta = np.arccos(np.clip(np.sum(r1 * r2, axis=-1) / (r1n * r2n), -1, 1))
    dtheta = np.where(cross >= 0, dtheta, 2 * np.pi - dtheta)  # Prograde: counterclockwise from r1 to r2
    A = np.sin(dtheta) * np.sqrt(r1n * r2n / (1 - np.cos(dtheta)))
    dt = np.broadcast_to(dt, A.shape)

    def y_and_F(z):
        C, S = stumpff(z)
        y = r1n + r2n + A * (z * S - 1) / np.sqrt(C)
        with np.errstate(invalid="ignore"):
            F = (y / C) ** 1.5 * S + A * np.sqrt(y) - np.sqrt(mu) * dt
        return y, np.where(y < 0, -np.inf, F)  # y < 0: z too small (travel time too short)

    low = np.full(A.shape, -4 * np.pi ** 2)
    high = np.full(A.shape, 4 * np.pi ** 2 - 1e-9)  # Single revolution: z < (2 pi)^2
    for _ in range(n_iterations):
        mid = 0.5 * (low + high)
        _, F = y_and_F(mid)
        too_short = F < 0
        low, high = np.where(too_short, mid, low), np.where(too_short, high, mid)
    z = 0.5 * (low + high)
    y, F = y_and_F(z)
    ok = np.isfinite(F) & (np.abs(F) < 1e-6 * np.sqrt(mu) * dt) & (y > 0)

    f = 1 - y / r1n
    g = A * np.sqrt(np.maximum(y, 0) / mu)
    gdot = 1 - y / r2n
    with np.errstate(invalid="ignore", divide="ignore"):
        v1 = (r2 - f[..., None] * r1) / g[..., None]
        v2 = (gdot[..., None] * r2 - r1) / g[..., None]
    v1[~ok], v2[~ok] = np.nan, np.nan
    return v1, v2
