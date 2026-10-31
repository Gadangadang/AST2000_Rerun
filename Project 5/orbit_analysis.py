"""
Part 5, challenge D: describing the orbit around the destination planet.

From one orientation (position and velocity relative to the planet) we get
r, v_r and v_theta, and then the two-body orbit (Lecture Notes 1B):

    h = r v_theta                    (angular momentum per mass)
    E = v^2 / 2 - G M / r            (energy per mass)
    a = -G M / (2 E)                 semi-major axis
    e = sqrt(1 + 2 E h^2 / (G M)^2)  eccentricity
    b = a sqrt(1 - e^2)              semi-minor axis
    P = 2 pi sqrt(a^3 / (G M))       period
    periapsis a (1 - e), apoapsis a (1 + e)

Everything is vectorised: give it many orientations (..., 2) and the elements
come out for all of them, so we can check that they stay constant orbit after orbit.
"""
import numpy as np


def polar_components(r_rel, v_rel):
    """r, v_r, v_theta for positions/velocities relative to the planet, shape (..., 2)."""
    r = np.linalg.norm(r_rel, axis=-1)
    v_r = np.sum(r_rel * v_rel, axis=-1) / r
    v_theta = (r_rel[..., 0] * v_rel[..., 1] - r_rel[..., 1] * v_rel[..., 0]) / r
    return r, v_r, v_theta


def orbital_elements(r_rel, v_rel, GM):
    """Two-body orbit from relative position and velocity (..., 2); returns a dict of arrays (units of the input)."""
    r, v_r, v_theta = polar_components(r_rel, v_rel)
    energy = 0.5 * (v_r ** 2 + v_theta ** 2) - GM / r
    h = r * v_theta
    a = -GM / (2 * energy)
    e = np.sqrt(np.maximum(1 + 2 * energy * h ** 2 / GM ** 2, 0))
    with np.errstate(invalid="ignore"):
        b = a * np.sqrt(1 - e ** 2)
        P = 2 * np.pi * np.sqrt(a ** 3 / GM)
    return {"r": r, "v_r": v_r, "v_theta": v_theta, "a": a, "b": b, "e": e, "P": P,
            "periapsis": a * (1 - e), "apoapsis": a * (1 + e), "energy": energy, "h": h}
