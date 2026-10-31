"""
Numerical integrators shared by the parts of the project.

leapfrog: kick-drift-kick, the workhorse for orbits (Parts 2 and later). It is
symplectic: energy errors stay bounded instead of drifting, and for central
forces it conserves angular momentum exactly. Every array operation acts on
all bodies at once, so the only Python loop is over time.
"""
import numpy as np


def leapfrog(acceleration, x0, v0, dt, n_out, save_every=1):
    """
    Integrate x'' = acceleration(x) with kick-drift-kick steps.

    acceleration: function of the positions, returning an array of the same shape.
    x0, v0: initial positions and velocities, any shape (e.g. (n_bodies, 2)).
    dt: time step; a scalar, or an array that broadcasts against x0 (one step per body).
    Takes n_out * save_every steps and stores every save_every-th state.
    Returns positions and velocities of shape (n_out + 1, *x0.shape), starting with the initial state.
    """
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    half_dt = 0.5 * np.asarray(dt)
    positions = np.empty((n_out + 1, *x.shape))
    velocities = np.empty_like(positions)
    positions[0], velocities[0] = x, v
    a = acceleration(x)
    for i in range(1, n_out + 1):
        for _ in range(save_every):
            v += half_dt * a
            x += dt * v
            a = acceleration(x)
            v += half_dt * a
        positions[i], velocities[i] = x, v
    return positions, velocities
