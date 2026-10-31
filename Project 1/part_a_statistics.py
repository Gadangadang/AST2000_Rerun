"""
Part 1, challenge A: Investigating Boltzmann statistics.

A1: Our own integrator (composite Simpson), tested on the 1/2/3-sigma
    probabilities and the FWHM of a Gaussian.
A2: Maxwell-Boltzmann distributions for N = 1e5 H2 molecules at T = 3000 K.
A3: <v>, P = nkT and <E> = 3/2 kT, checked by integration AND by drawing
    a finite gas sample, which tells us how accurate challenge B can be.

Run:  ../.venv/bin/python part_a_statistics.py
Figures are written to figures/a*.png.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize, special

from ast2000tools.constants import k_B, m_H2
from plot_style import AQUA, BLUE, INK, MUTED, ORANGE, SERIES, save

N = int(1e5)  # Number of H2 molecules
T = 3000.0  # Temperature [K]
L = 1e-6  # Box side from challenge B [m], only used for the number density


# ------------------------------------------------------------ A1: integration
def simpson(f, a, b, n=2000):
    """Composite Simpson's rule for the integral of f from a to b with n intervals.

    The error falls as n^-4, so a few thousand points give ~machine precision
    for smooth functions like ours.
    """
    if n % 2:
        n += 1  # Simpson needs an even number of intervals
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))


def gaussian(x, mu, sigma):
    """Normal probability density f(mu, sigma; x), eq. (1)."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def probability(a, b, mu, sigma, n=2000):
    """P(a <= x <= b), eq. (2): the chance that one draw lands in [a, b]."""
    return simpson(lambda x: gaussian(x, mu, sigma), a, b, n)


def exact_probability(a, b, mu, sigma):
    """The same probability from the error function, used only as a reference."""
    z = lambda x: (x - mu) / (sigma * np.sqrt(2))
    return 0.5 * (special.erf(z(b)) - special.erf(z(a)))


def numerical_fwhm(mu, sigma):
    """Solve f(x) = f(mu)/2 for the right half-max point; FWHM is twice its distance to mu."""
    half_max = gaussian(mu, mu, sigma) / 2
    x_half = optimize.brentq(lambda x: gaussian(x, mu, sigma) - half_max, mu, mu + 10 * sigma)
    return 2 * (x_half - mu)


# ------------------------------------------------- A2: Maxwell-Boltzmann (MB)
def sigma_v(T=T, m=m_H2):
    """Standard deviation of each velocity component: sqrt(kT/m)."""
    return np.sqrt(k_B * T / m)


def mb_1d(vx, T=T, m=m_H2):
    """1D MB distribution: a Gaussian with mu = 0 and sigma = sqrt(kT/m)."""
    return gaussian(vx, 0.0, sigma_v(T, m))


def mb_speed(v, T=T, m=m_H2):
    """MB distribution of the speed v = |v|: (m/2pi kT)^(3/2) exp(-mv^2/2kT) 4 pi v^2."""
    a = m / (2 * np.pi * k_B * T)
    return a ** 1.5 * np.exp(-m * v ** 2 / (2 * k_B * T)) * 4 * np.pi * v ** 2


def characteristic_speeds(T=T, m=m_H2):
    """Most probable, mean and rms speed of the MB speed distribution."""
    return {
        "v_mp": np.sqrt(2 * k_B * T / m),
        "<v>": np.sqrt(8 * k_B * T / (np.pi * m)),
        "v_rms": np.sqrt(3 * k_B * T / m),
    }


def sample_velocities(N, T, rng, m=m_H2):
    """Draw N velocity vectors from MB: each component independently ~ N(0, sqrt(kT/m))."""
    return rng.normal(0.0, sigma_v(T, m), size=(N, 3))


# ======================================================================= A1
def run_a1():
    mu, sigma = 0.0, 1.0
    print("A1.3  Gaussian probabilities (our Simpson | exact erf | scipy quad)")
    probs = []
    for k in (1, 2, 3):
        a, b = mu - k * sigma, mu + k * sigma
        p = probability(a, b, mu, sigma)
        probs.append(p)
        print(f"      P(|x - mu| <= {k}sigma) = {p:.6f} | {exact_probability(a, b, mu, sigma):.6f} | "
              f"{integrate.quad(gaussian, a, b, args=(mu, sigma))[0]:.6f}")

    # Figure: the Gaussian with the 1/2/3 sigma bands shaded
    x = np.linspace(-4, 4, 800)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, color, alpha in zip((3, 2, 1), (AQUA, ORANGE, BLUE), (0.25, 0.3, 0.35)):
        band = np.abs(x) <= k
        ax.fill_between(x[band], gaussian(x[band], mu, sigma), color=color, alpha=alpha, lw=0,
                        label=rf"$\pm{k}\sigma$: P = {probs[k - 1]:.4f}")
    ax.plot(x, gaussian(x, mu, sigma), color=INK)
    ax.set_xlabel(r"$(x-\mu)/\sigma$")
    ax.set_ylabel(r"$f(\mu,\sigma;x)$")
    ax.set_title("A1.3  Probability inside 1, 2 and 3 standard deviations")
    ax.legend(loc="upper right")
    save(fig, "a1_gaussian_sigma_bands")

    # Figure: how fast our integrator converges (it decides how many points we need)
    ns = np.unique(np.logspace(1, 3.5, 25).astype(int) // 2 * 2)
    exact = exact_probability(-1, 1, 0, 1)
    errors = [abs(probability(-1, 1, 0, 1, n) - exact) for n in ns]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ns, errors, "o", color=BLUE, ms=5, label="Simpson error")
    ax.loglog(ns, errors[0] * (ns / ns[0]) ** -4.0, "--", color=MUTED, lw=1.2, label=r"$\propto n^{-4}$")
    ax.axhline(1e-15, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("number of intervals n")
    ax.set_ylabel(r"$|P_{\rm Simpson} - P_{\rm exact}|$")
    ax.set_title(r"A1  Integrator convergence on $P(-1\sigma\leq x-\mu\leq 1\sigma)$")
    ax.legend()
    save(fig, "a1_integrator_convergence")

    # A1.4: FWHM
    fwhm = numerical_fwhm(mu, sigma)
    fwhm_ana = 2 * np.sqrt(2 * np.log(2)) * sigma
    print(f"A1.4  FWHM numerical = {fwhm:.8f} sigma, analytical 2 sqrt(2 ln 2) sigma = {fwhm_ana:.8f} sigma")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, gaussian(x, mu, sigma), color=INK)
    half = gaussian(mu, mu, sigma) / 2
    ax.axhline(half, color=MUTED, lw=1, ls="--")
    ax.annotate("", xy=(fwhm / 2, half), xytext=(-fwhm / 2, half),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2))
    ax.text(0, half * 1.08, f"FWHM = {fwhm:.4f}$\\sigma$", ha="center", color=INK)
    ax.text(3.9, half, "half max", va="bottom", ha="right", color=MUTED, fontsize=9)
    ax.set_xlabel(r"$(x-\mu)/\sigma$")
    ax.set_ylabel(r"$f(\mu,\sigma;x)$")
    ax.set_title(r"A1.4  Full width at half maximum = $2\sqrt{2\ln 2}\,\sigma$")
    save(fig, "a1_fwhm")


# ======================================================================= A2
def run_a2(rng):
    s = sigma_v()
    speeds = characteristic_speeds()
    print(f"A2    sigma_v = sqrt(kT/m) = {s:.2f} m/s")

    # A2.1 + A2.2: 1D distribution with the [5, 30] km/s window shaded
    vx = np.linspace(-2.5e4, 2.5e4, 1000)
    a, b = 5e3, 30e3
    p = simpson(mb_1d, a, b)
    print(f"A2.2  P(5 km/s <= vx <= 30 km/s) = {p:.6f}  ->  N*P = {N * p:.0f} molecules")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vx / 1e3, mb_1d(vx) * 1e3, color=INK)
    window = (vx >= a) & (vx <= b)
    ax.fill_between(vx[window] / 1e3, mb_1d(vx[window]) * 1e3, color=ORANGE, alpha=0.4, lw=0,
                    label=f"$5\\leq v_x\\leq 30$ km/s: P = {p:.4f}\n$\\approx${N * p:.0f} of {N:.0e} molecules")
    for k in (-1, 1):
        ax.axvline(k * s / 1e3, color=MUTED, lw=0.8, ls=":")
    ax.text(s / 1e3, 0.005, r" $+\sigma_v$", color=MUTED, fontsize=9)
    ax.set_xlabel(r"$v_x$ [km/s]")
    ax.set_ylabel(r"$P(v_x)$ [s/km]")
    ax.set_title(f"A2.1  1D Maxwell-Boltzmann, H$_2$ at T = {T:.0f} K")
    ax.legend(loc="upper left")
    save(fig, "a2_mb_vx")

    # A2.3: speed distribution with the three characteristic speeds
    v = np.linspace(0, 3e4, 1000)
    print(f"      speed distribution integrates to {simpson(mb_speed, 0, 20 * s):.8f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(v / 1e3, mb_speed(v) * 1e3, color=INK)
    labels = {"v_mp": r"$v_{mp}$", "<v>": r"$\langle v\rangle$", "v_rms": r"$v_{rms}$"}
    for (name, value), color in zip(speeds.items(), SERIES):
        ax.axvline(value / 1e3, color=color, lw=1.5, ls="--", label=f"{labels[name]} = {value / 1e3:.2f} km/s")
    ax.set_xlabel(r"$v$ [km/s]")
    ax.set_ylabel(r"$P(v)$ [s/km]")
    ax.set_title(f"A2.3  Speed distribution, H$_2$ at T = {T:.0f} K")
    ax.legend()
    save(fig, "a2_mb_speed")

    # Why A2.1 (peak at 0) and A2.3 (peak at v_mp) do not conflict:
    # velocity *vectors* are densest at the origin, but the shell of radius v has area 4 pi v^2.
    vel = sample_velocities(N, T, rng)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    lim = 4 * s / 1e3
    h = ax1.hist2d(vel[:, 0] / 1e3, vel[:, 1] / 1e3, bins=80, range=[[-lim, lim]] * 2, cmap="Blues")
    fig.colorbar(h[3], ax=ax1, label="molecules per bin")
    for r, name in ((speeds["v_mp"], r"$v_{mp}$"),):
        circle = plt.Circle((0, 0), r / 1e3, fill=False, color=ORANGE, lw=1.5, ls="--")
        ax1.add_patch(circle)
        ax1.text(0, r / 1e3 * 1.05, name, color=ORANGE, ha="center", va="bottom")
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$v_x$ [km/s]")
    ax1.set_ylabel(r"$v_y$ [km/s]")
    ax1.set_title(f"{N:.0e} sampled velocities: densest at the origin")
    vv = np.linspace(0, 3e4, 600)
    boltzmann = np.exp(-m_H2 * vv ** 2 / (2 * k_B * T))
    shell = vv ** 2
    ax2.plot(vv / 1e3, boltzmann, color=BLUE, label=r"Boltzmann factor $e^{-mv^2/2kT}$")
    ax2.plot(vv / 1e3, shell / shell.max(), color=ORANGE, label=r"shell area $4\pi v^2$ (scaled)")
    ax2.plot(vv / 1e3, mb_speed(vv) / mb_speed(vv).max(), color=INK, label=r"product $\propto P(v)$")
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel(r"$v$ [km/s]")
    ax2.set_ylabel("relative size (max = 1)")
    ax2.set_title("A2.3  Few states near v = 0, so P(v) peaks at $v_{mp}$")
    ax2.legend(loc="center right")
    save(fig, "a2_why_no_conflict")

    # How the speed distribution depends on temperature (we pick T for the engine in E)
    fig, ax = plt.subplots(figsize=(7, 4))
    for T_i, color in zip((1000, 3000, 10000), SERIES):
        ax.plot(v / 1e3, mb_speed(v, T_i) * 1e3, color=color, label=f"T = {T_i} K")
    ax.set_xlabel(r"$v$ [km/s]")
    ax.set_ylabel(r"$P(v)$ [s/km]")
    ax.set_title("A2  Hotter gas: faster and broader (engine temperature, challenge E)")
    ax.legend()
    save(fig, "a2_mb_speed_temperatures")


# ======================================================================= A3
def run_a3(rng):
    """
    Derivations (substitute x = mv^2/2kT and use eqs. (4) and (5)):
      <v> = int v P(v) dv                = sqrt(8kT/(pi m))
      P   = 1/3 int p v n(p) dp = n m <v^2>/3 = n m (3kT/m)/3 = nkT
      <E> = 1/2 m <v^2>, <v^2> = 3kT/m   -> <E> = 3/2 kT
    """
    s = sigma_v()
    v_max = 20 * s  # The integrands are ~e^-200 beyond this, so it acts as infinity
    n_density = N / L ** 3
    mean_v = simpson(lambda v: v * mb_speed(v), 0, v_max, 20000)
    mean_E = simpson(lambda v: 0.5 * m_H2 * v ** 2 * mb_speed(v), 0, v_max, 20000)
    # P = 1/3 int p v n(p) dp with p = m v and n(p) dp = n P(v) dv
    pressure = n_density / 3 * simpson(lambda v: m_H2 * v * v * mb_speed(v), 0, v_max, 20000)

    analytic = {"<v>": np.sqrt(8 * k_B * T / (np.pi * m_H2)), "P": n_density * k_B * T, "<E>": 1.5 * k_B * T}
    numeric = {"<v>": mean_v, "P": pressure, "<E>": mean_E}
    units = {"<v>": "m/s", "P": "Pa", "<E>": "J"}
    for key in analytic:
        rel = abs(numeric[key] - analytic[key]) / analytic[key]
        print(f"A3    {key:<4} numerical = {numeric[key]:.8e} {units[key]:<3}  analytical = "
              f"{analytic[key]:.8e} {units[key]:<3}  rel. diff = {rel:.1e}")

    # Figure: the integrands, whose areas are <v> and <E>
    v = np.linspace(0, 3e4, 1000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.fill_between(v / 1e3, v * mb_speed(v), color=BLUE, alpha=0.35, lw=0)
    ax1.plot(v / 1e3, v * mb_speed(v), color=BLUE)
    ax1.set_title(rf"A3.1  area under $vP(v)$ = $\langle v\rangle$ = {mean_v:.1f} m/s")
    ax1.set_xlabel(r"$v$ [km/s]")
    ax1.set_ylabel(r"$v\,P(v)$")
    E_integrand = 0.5 * m_H2 * v ** 2 * mb_speed(v)
    ax2.fill_between(v / 1e3, E_integrand, color=ORANGE, alpha=0.35, lw=0)
    ax2.plot(v / 1e3, E_integrand, color=ORANGE)
    ax2.set_title(rf"A3.3  area under $\frac{{1}}{{2}}mv^2P(v)$ = $\langle E\rangle$ = {mean_E:.3e} J")
    ax2.set_xlabel(r"$v$ [km/s]")
    ax2.set_ylabel(r"$\frac{1}{2}mv^2\,P(v)$ [J s/m]")
    save(fig, "a3_integrands")

    # A finite gas only estimates these averages. How big is the scatter for N particles?
    # For MB: std(v)/<v> = sqrt(3pi/8 - 1) = 0.422, std(v^2)/<v^2> = sqrt(2/3) = 0.816,
    # so the expected relative error of the sample mean is that / sqrt(N).
    Ns = np.logspace(1, 6, 11).astype(int)
    trials = 30
    rms_v, rms_E = [], []
    for n_i in Ns:
        err_v, err_E = [], []
        for _ in range(trials):
            speeds = np.linalg.norm(sample_velocities(n_i, T, rng), axis=1)
            err_v.append(np.mean(speeds) / analytic["<v>"] - 1)
            err_E.append(np.mean(0.5 * m_H2 * speeds ** 2) / analytic["<E>"] - 1)
        rms_v.append(np.sqrt(np.mean(np.square(err_v))))
        rms_E.append(np.sqrt(np.mean(np.square(err_E))))
    expected_v = np.sqrt(3 * np.pi / 8 - 1) / np.sqrt(Ns)
    expected_E = np.sqrt(2 / 3) / np.sqrt(Ns)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.loglog(Ns, rms_E, "o", color=ORANGE, ms=6, label=r"$\langle E\rangle$, sampled")
    ax.loglog(Ns, expected_E, "--", color=ORANGE, lw=1.2, label=r"$0.816/\sqrt{N}$")
    ax.loglog(Ns, rms_v, "o", color=BLUE, ms=6, label=r"$\langle v\rangle$, sampled")
    ax.loglog(Ns, expected_v, "--", color=BLUE, lw=1.2, label=r"$0.422/\sqrt{N}$")
    ax.axvline(N, color=MUTED, lw=0.8, ls=":")
    ax.text(N * 1.1, expected_E[0], f"N = {N:.0e}\n(challenge B)", color=MUTED, fontsize=9, va="top")
    ax.set_xlabel("number of particles N")
    ax.set_ylabel("rms relative error")
    ax.set_title(f"A3  Statistical error of a finite gas ({trials} draws per N)")
    ax.legend(loc="lower left")
    save(fig, "a3_sampling_error")
    i = np.argmin(np.abs(Ns - N))
    print(f"A3    expected 1-sigma sampling error at N = {N:.0e}: <v> {expected_v[i]:.2%}, <E> {expected_E[i]:.2%}")


def main():
    rng = np.random.default_rng(2000)  # Fixed seed so figures are reproducible
    run_a1()
    run_a2(rng)
    run_a3(rng)


if __name__ == "__main__":
    main()
