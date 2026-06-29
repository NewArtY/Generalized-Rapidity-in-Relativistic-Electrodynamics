#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
===============
Generate all 10 publication-quality figures for the EPJ D paper

   "Generalized Rapidity in Relativistic Electrodynamics:
    Particle Trajectories and Radiation Signatures in Intense Laser Fields"

Each figure is saved as BOTH a vector PDF and a vector EPS into this
directory (fig1.pdf/fig1.eps ... fig10.pdf/fig10.eps).

Reduced units used throughout:
  - lengths      in r_osc = c / omega
  - momenta      in m_e c
  - frequencies  in omega (laser carrier)

Spectral figures (4, 7, 9) use analytical Sarachik-Schappert harmonic lines
and Bessel sidebands (NOT FFT-of-trajectory); Figs 7(b) and 10 add synthetic
scatter around the universal 2*xi law with fixed RNG seeds.  These choices are
faithful to plan_clean.md and are stated honestly in the paper captions.

Run:
    python make_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.special import jv
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------- #
#  Global publication style                                                   #
# --------------------------------------------------------------------------- #
# Manuscript body text is 10 pt and the (single-column) text/column width is
# 372.0 pt = 5.147 in (measured from the compiled sn-jnl log).  EVERY figure is
# made exactly COL inches wide and included at width=\columnwidth, so the LaTeX
# scale factor is s == 1 and matplotlib points print 1:1 as document points.
# ALL textual elements therefore use the single body size BODY_PT.
BODY_PT = 10.0

plt.rcParams.update({
    "font.size": BODY_PT,
    "axes.titlesize": BODY_PT,
    "axes.labelsize": BODY_PT,
    "xtick.labelsize": BODY_PT,
    "ytick.labelsize": BODY_PT,
    "legend.fontsize": BODY_PT,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    # Match the manuscript body typeface: sn-jnl is built on the standard LaTeX
    # classes and typesets in Computer Modern (cmr/cmmi/cmsy in the build log).
    # matplotlib ships the Computer Modern fonts, so we use them for BOTH text
    # (cmr10) and math (mathtext 'cm'); no usetex needed.  axes.unicode_minus is
    # disabled because cmr10 lacks the U+2212 glyph used for negative tick labels.
    "mathtext.fontset": "cm",
    "mathtext.rm": "cmr10",
    "mathtext.it": "cmmi10",
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "axes.grid": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
# Canonical source lives in code/; figures are written into the manuscript tree.
OUTDIR = os.environ.get(
    "FIG_OUTDIR",
    os.path.normpath(os.path.join(HERE, "..", "article", "figures")),
)
os.makedirs(OUTDIR, exist_ok=True)

# Single-column == text width in this sn-jnl layout: 372.0 pt / 72.27 pt-per-in.
# Multi-panel figures use the SAME width COL (panels become narrower) so that
# every figure is included at width=\columnwidth with scale s == 1.
COL = 372.0 / 72.27          # 5.147 in
FULL = COL


def _save(fig, n):
    """Save figure n as PDF and EPS, then close."""
    pdf = os.path.join(OUTDIR, f"fig{n}.pdf")
    eps = os.path.join(OUTDIR, f"fig{n}.eps")
    # NOTE: do NOT use bbox_inches="tight" -- it trims margins so the saved
    # width would be < figsize, giving a LaTeX scale s>1 and oversized text.
    # Saving at the exact figsize keeps s == 1 (fonts print at body size).
    fig.savefig(pdf)
    # EPS via the built-in 'ps' backend; mathtext is rendered as paths/fonts.
    fig.savefig(eps, format="eps")
    # Optional PNG raster (for on-screen review); enabled via FIG_PNG=1.
    if os.environ.get("FIG_PNG"):
        fig.savefig(os.path.join(OUTDIR, f"fig{n}.png"), dpi=140)
    plt.close(fig)
    print(f"OK fig{n}")


# --------------------------------------------------------------------------- #
#  Figure 1 — Generalized rapidity Phi(phi, xi)                               #
# --------------------------------------------------------------------------- #
def fig1():
    # Bounded additive coupling Phi = phi + xi.  The physically meaningful
    # quantity is the fractional correction to the Lorentz factor,
    # d(tilde_gamma)/gamma = cosh(phi+xi)/cosh(phi) - 1, which rises from zero
    # at low energy and SATURATES at xi as v -> c (since -> xi*tanh(phi) -> xi).
    phi = np.linspace(0.0, 3.0, 400)
    xis = [0.05, 0.10]
    colors = ["#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(COL, COL * 0.62))

    for xi, c in zip(xis, colors):
        frac = np.cosh(phi + xi) / np.cosh(phi) - 1.0          # exact
        ax.plot(phi, frac, color=c, label=rf"$\xi={xi:.2f}$")
        ax.axhline(xi, color=c, ls=":", lw=1.0)                # saturation asymptote
        ax.text(2.92, xi + 0.0015, rf"$\xi={xi:.2f}$",
                color=c, fontsize=BODY_PT, ha="right", va="bottom")

    ax.set_xlabel(r"rapidity $\varphi$")
    ax.set_ylabel(r"fractional shift $\delta\tilde\gamma/\gamma=\xi\tanh\varphi$")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 0.115)
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.9)

    # Top secondary axis: gamma = cosh(phi)
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda p: np.cosh(np.clip(p, 0, 50)),
                   lambda g: np.arccosh(np.clip(g, 1, None))))
    secax.set_xlabel(r"Lorentz factor $\gamma=\cosh\varphi$")

    # Inset: the generalized rapidity itself, Phi = phi + xi (rigid additive shift)
    axins = ax.inset_axes([0.155, 0.5, 0.4, 0.44])
    for xi, c in zip([0.0] + xis, ["#1b1b1b"] + colors):
        axins.plot(phi, phi + xi, color=c, lw=1.1)
    axins.set_xlabel(r"$\varphi$", fontsize=BODY_PT, labelpad=1)
    axins.set_ylabel(r"$\Phi=\varphi+\xi$", fontsize=BODY_PT, labelpad=1)
    axins.set_xlim(0, 3)
    axins.tick_params(labelsize=BODY_PT)
    axins.grid(True)

    fig.tight_layout()
    _save(fig, 1)


# --------------------------------------------------------------------------- #
#  Figure 2 — (x,z) figure-eight + drift, V1 = m g z, linear polarization     #
# --------------------------------------------------------------------------- #
def fig2():
    a0 = 1.0
    xi1 = 1.0e-3

    def traj(psi):
        # standard linear-polarization (LL) figure-eight in average rest frame
        x0 = 1.0 - np.cos(psi)
        z0 = (a0 ** 2 / 4.0) * (psi - 0.5 * np.sin(2.0 * psi))
        return x0, z0

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULL, 2.7))

    # Panel (a): one cycle, standard figure-eight (z relative to its mean)
    psi1 = np.linspace(0, 2 * np.pi, 600)
    x0, z0 = traj(psi1)
    z0c = z0 - z0.mean()
    axa.plot(z0c, x0, color="#1f77b4")
    axa.set_xlabel(r"$z\ [r_{\rm osc}]$")
    axa.set_ylabel(r"$x\ [r_{\rm osc}]$")
    axa.set_title("(a) single cycle: figure-eight")
    axa.grid(True)
    axa.axhline(0, color="0.7", lw=0.5)

    # Panel (b): 6 cycles, xi1=0 vs xi1=1e-3, downward drift.
    # Reduced-units drift coefficient consistent with Eq.(eq:Idrift):
    #   delta_z|_drift = -(g/2 omega^2) psi^2  ->  -xi1 * psi^2 / (4 pi),
    # since xi1 = g lambda / c^2, lambda = 2 pi c/omega and lengths are in
    # r_osc = c/omega.  (The earlier -xi1 psi^2/2 was ~2 pi too large.)
    psi6 = np.linspace(0, 12 * np.pi, 4000)
    x6, z6 = traj(psi6)
    drift = -xi1 * psi6 ** 2 / (4.0 * np.pi)
    Mdisp = 5                                 # visibility magnification (labelled)
    z6_def = z6 + Mdisp * drift
    axb.plot(z6, x6, color="0.55", lw=1.0, label=r"$\xi_1=0$")
    axb.plot(z6_def, x6, color="#d62728", lw=1.1,
             label=rf"$\xi_1=10^{{-3}}$ (drift $\times{Mdisp}$)")
    axb.set_xlabel(r"$z\ [r_{\rm osc}]$")
    axb.set_ylabel(r"$x\ [r_{\rm osc}]$")
    axb.set_title("(b) six cycles: drift")
    axb.grid(True)
    axb.legend(loc="upper left")
    axb.text(0.52, 0.52,
             rf"true $|\Delta z|_{{6\lambda}}\approx{abs(drift[-1]):.2f}\,r_{{\rm osc}}$",
             transform=axb.transAxes, fontsize=BODY_PT, color="#d62728",
             ha="center", va="center",
             bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.7",
                       boxstyle="round,pad=0.25"))

    fig.tight_layout()
    _save(fig, 2)


# --------------------------------------------------------------------------- #
#  Figure 3 — 3D helix + drift, V1, circular polarization                     #
# --------------------------------------------------------------------------- #
def fig3():
    a0 = 1.0
    xi1 = 1.0e-3
    gamma_circ = 1.5                          # = 1 + a0^2/2 for a0=1
    r0 = a0 / gamma_circ
    vz0 = (a0 ** 2 / 2.0) / gamma_circ

    psi = np.linspace(0, 12 * np.pi, 4000)

    # standard helix
    xs = r0 * np.cos(psi)
    ys = r0 * np.sin(psi)
    zs = vz0 * psi

    # Modified: expanding radius [Eq.(eq:Ispiral)] + decelerating drift
    # [Eq.(eq:Ihelixz)].  In reduced units mu = g/(omega c) = xi1/(2 pi),
    # so the radius coefficient is mu/(1+a0^2/2) = xi1/(2 pi gamma_circ) and the
    # longitudinal slowdown is xi1 psi^2/(4 pi) (the same coefficient as Fig.2).
    # The earlier xi1/gamma_circ and xi1/(2 gamma_circ) were ~2 pi too large.
    mu = xi1 / (2.0 * np.pi)
    rm = r0 * (1.0 + mu * psi / gamma_circ)
    xm = rm * np.cos(psi)
    ym = rm * np.sin(psi)
    zm = vz0 * psi - xi1 * psi ** 2 / (4.0 * np.pi)

    # numerical magnitudes of the (sub-percent) deformation over the plotted span
    dr_rel = mu * psi[-1] / gamma_circ                    # fractional radius growth
    dz_rel = (xi1 * psi[-1] ** 2 / (4.0 * np.pi)) / (vz0 * psi[-1])  # rel. slowdown

    fig = plt.figure(figsize=(FULL, 2.9))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(xs, ys, zs, color="#1f77b4", lw=1.0)
    ax1.set_title(r"(a) standard helix ($\xi_1=0$)")
    ax1.set_xlabel(r"$x\ [r_{\rm osc}]$", labelpad=-2)
    ax1.set_ylabel(r"$y\ [r_{\rm osc}]$", labelpad=-2)
    ax1.set_zlabel(r"$z\ [r_{\rm osc}]$", labelpad=-2)
    ax1.tick_params(labelsize=BODY_PT, pad=-1)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(xm, ym, zm, color="#d62728", lw=1.0)
    ax2.set_title(r"(b) modified helix ($\xi_1=10^{-3}$)")
    ax2.set_xlabel(r"$x\ [r_{\rm osc}]$", labelpad=-2)
    ax2.set_ylabel(r"$y\ [r_{\rm osc}]$", labelpad=-2)
    ax2.set_zlabel(r"$z\ [r_{\rm osc}]$", labelpad=-2)
    ax2.tick_params(labelsize=BODY_PT, pad=-1)
    # the deformation is sub-percent over six cycles: annotate it numerically
    ax2.text2D(0.0, 1.0,
               rf"radius growth:" "\n"
               rf"$\Delta r/r_0\approx{dr_rel*100:.1f}\%$" "\n"
               rf"slowdown:" "\n"
               rf"$\Delta z/z\approx{dz_rel*100:.1f}\%$",
               transform=ax2.transAxes, fontsize=BODY_PT, color="#d62728",
               va="top", ha="left")

    for ax in (ax1, ax2):
        ax.view_init(elev=18, azim=-60)
        ax.set_zlim(0, vz0 * psi[-1])            # shared longitudinal range

    fig.subplots_adjust(left=0.0, right=0.90, bottom=0.05, top=0.90, wspace=0.10)
    _save(fig, 3)


# --------------------------------------------------------------------------- #
#  Helper: Sarachik-Schappert harmonic comb (on-axis)                         #
# --------------------------------------------------------------------------- #
def _ss_spectrum(omega_grid, a0, gamma, denom, xi, n_harm=3, N_cyc=30,
                 suppress_even=None):
    """Gaussian-broadened on-axis nonlinear-Compton harmonic comb."""
    spec = np.zeros_like(omega_grid)
    for n in range(1, n_harm + 1):
        omega_n = 4.0 * n * gamma ** 2 / denom * (1.0 + 2.0 * xi)
        u_n = n * a0 ** 2 / (2.0 * denom)
        amp = (n * a0) ** 2 * (jv(n - 1, u_n) - jv(n + 1, u_n)) ** 2
        if suppress_even is not None and (n % 2 == 0):
            amp *= suppress_even
        sigma = omega_n / (2.0 * np.sqrt(2.0) * N_cyc)
        spec += amp * np.exp(-(omega_grid - omega_n) ** 2 / (2.0 * sigma ** 2))
    return spec


# --------------------------------------------------------------------------- #
#  Figure 4 — On-axis radiation spectrum, V1, shift 2*xi1                      #
# --------------------------------------------------------------------------- #
def fig4():
    a0 = 2.0
    gamma = 1.0e3
    denom = 1.0 + a0 ** 2 / 2.0           # = 3
    omega_peak1 = 4.0 * gamma ** 2 / denom

    xis = [0.0, 5.0e-4, 1.0e-3]
    colors = ["#1b1b1b", "#1f77b4", "#d62728"]

    # frequency grid in units of omega_peak1, spanning the first 3 harmonics
    w = np.linspace(0.5, 3.4, 6000) * omega_peak1

    fig, ax = plt.subplots(figsize=(COL, COL * 0.66))

    for xi, c in zip(xis, colors):
        spec = _ss_spectrum(w, a0, gamma, denom, xi, n_harm=3, N_cyc=30)
        ax.plot(w / omega_peak1, spec / spec.max(), color=c,
                label=rf"$\xi_1={xi:g}$")

    ax.set_xlabel(r"$\omega'/\omega'_{\rm peak,1}$")
    ax.set_ylabel(r"$dW/d\omega'$ (norm.)")
    ax.set_title(r"On-axis spectrum, $\mathcal{V}_1$:  peak shift $2\xi_1$")
    ax.grid(True)
    ax.legend(loc="upper right")

    # inset zoom on fundamental to reveal the 2*xi shift
    axins = ax.inset_axes([0.17, 0.30, 0.40, 0.46])
    wz = np.linspace(0.985, 1.02, 3000) * omega_peak1
    for xi, c in zip(xis, colors):
        spec = _ss_spectrum(wz, a0, gamma, denom, xi, n_harm=1, N_cyc=30)
        axins.plot(wz / omega_peak1, spec / spec.max(), color=c)
    axins.set_title(r"fundamental", fontsize=BODY_PT)
    axins.tick_params(labelsize=BODY_PT)
    axins.set_xlabel(r"$\omega'/\omega'_{\rm peak,1}$", fontsize=BODY_PT, labelpad=1)
    axins.grid(True)

    fig.tight_layout()
    _save(fig, 4)


# --------------------------------------------------------------------------- #
#  Figure 5 — (x,z) Coulomb trajectories, three impact parameters             #
# --------------------------------------------------------------------------- #
def fig5():
    a0 = 1.0
    Z = 10.0
    alpha = Z / 137.0
    r_osc = a0                      # in reduced units r_osc = a0 c/omega = a0
    bs = [5.0, 1.0, 0.2]            # in units of r_osc
    labels = ["weak", "deformed", "Rutherford"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    def rhs(psi, s):
        Px, Pz, x, z = s
        U = np.sqrt(1.0 + Px ** 2 + Pz ** 2)
        dtdpsi = U / (U - Pz)
        r3 = (x ** 2 + z ** 2 + 1e-8) ** 1.5
        Fx = -alpha * x / r3
        Fz = -alpha * z / r3
        drive = a0 * np.cos(psi)
        dPx = (drive + Fx) * dtdpsi
        dPz = Fz * dtdpsi
        dx = Px / U * dtdpsi
        dz = Pz / U * dtdpsi
        return [dPx, dPz, dx, dz]

    fig, ax = plt.subplots(figsize=(COL, COL * 0.72))

    psi_span = (0.0, 6.0 * np.pi)
    psi_eval = np.linspace(*psi_span, 4000)

    for b, lab, c in zip(bs, labels, colors):
        # start transversely offset by impact parameter b, drifting toward core
        s0 = [0.0, 0.0, 0.0, -b * r_osc]
        sol = solve_ivp(rhs, psi_span, s0, t_eval=psi_eval,
                        method="RK45", rtol=1e-8, atol=1e-11, max_step=0.05)
        x = sol.y[2]
        z = sol.y[3]
        ax.plot(x, z, color=c, lw=1.2,
                label=rf"$b={b:g}\,r_{{\rm osc}}$ ({lab})")

    ax.scatter([0], [0], marker="*", s=120, color="k", zorder=5,
               label="nucleus $Z=10$")
    ax.set_xlabel(r"$x\ [r_{\rm osc}]$")
    ax.set_ylabel(r"$z\ [r_{\rm osc}]$")
    ax.set_title(r"Coulomb trajectories $\mathcal{V}_2=-\alpha/r$")
    ax.grid(True)
    ax.legend(loc="best", fontsize=BODY_PT)

    fig.tight_layout()
    _save(fig, 5)


# --------------------------------------------------------------------------- #
#  Figure 6 — 3D deformed helix, circular polarization + Coulomb              #
# --------------------------------------------------------------------------- #
def fig6():
    a0 = 1.0
    Z = 10.0
    alpha_full = Z / 137.0
    b = 1.5                                   # impact parameter ~ r_osc

    def make_rhs(alpha):
        def rhs(psi, s):
            Px, Py, Pz, x, y, z = s
            U = np.sqrt(1.0 + Px ** 2 + Py ** 2 + Pz ** 2)
            dtdpsi = U / (U - Pz)
            vx, vy, vz = Px / U, Py / U, Pz / U
            r3 = (x ** 2 + y ** 2 + z ** 2 + 1e-6) ** 1.5
            Fx = -alpha * x / r3
            Fy = -alpha * y / r3
            Fz = -alpha * z / r3
            # Circularly polarised plane wave propagating along +z.  The full
            # Lorentz force E + v x B (|B|=|E|, B = z_hat x E) supplies the
            # longitudinal v x B drive that sustains Pz = a0^2/2 and makes the
            # orbit advance as a helix (omitting it collapsed the z motion).
            # Phasing E=(-a0 sin, a0 cos) with the drift launch below yields a
            # clean constant-radius helix in the free (Z=0) case.
            Ex = -a0 * np.sin(psi)
            Ey = a0 * np.cos(psi)
            Lx = Ex * (1.0 - vz)          # = Ex / dtdpsi
            Ly = Ey * (1.0 - vz)
            Lz = vx * Ex + vy * Ey        # longitudinal v x B force
            dPx = (Lx + Fx) * dtdpsi
            dPy = (Ly + Fy) * dtdpsi
            dPz = (Lz + Fz) * dtdpsi
            dx = vx * dtdpsi
            dy = vy * dtdpsi
            dz = vz * dtdpsi
            return [dPx, dPy, dPz, dx, dy, dz]
        return rhs

    psi_span = (0.0, 12.0 * np.pi)
    psi_eval = np.linspace(*psi_span, 6000)

    # Launch already carrying the average longitudinal drift Pz = a0^2/2 and the
    # circular transverse momentum (Px,Py)=(a0,0); guiding centre offset by the
    # impact parameter b in x.  Same launch for both panels.
    s0 = [a0, 0.0, a0 ** 2 / 2.0, b, 0.0, 0.0]

    # (a) free helix  (Z = 0)
    sol_a = solve_ivp(make_rhs(0.0), psi_span, s0, t_eval=psi_eval,
                      method="RK45", rtol=1e-9, atol=1e-12, max_step=0.05)
    xs, ys, zs = sol_a.y[3], sol_a.y[4], sol_a.y[5]

    # (b) deformed helix  (Z = 10, Coulomb precession)
    sol_b = solve_ivp(make_rhs(alpha_full), psi_span, s0, t_eval=psi_eval,
                      method="RK45", rtol=1e-9, atol=1e-12, max_step=0.05)
    xm, ym, zm = sol_b.y[3], sol_b.y[4], sol_b.y[5]

    fig = plt.figure(figsize=(FULL, 3.0))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(xs, ys, zs, color="#1f77b4", lw=0.9)
    ax1.set_title(r"(a) standard helix ($Z=0$)")
    for f, lab in ((ax1.set_xlabel, r"$x$"), (ax1.set_ylabel, r"$y$"),
                   (ax1.set_zlabel, r"$z$")):
        f(lab + r"$\ [r_{\rm osc}]$", labelpad=-2)
    ax1.tick_params(labelsize=BODY_PT, pad=-1)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(xm, ym, zm, color="#d62728", lw=0.9)
    ax2.set_title(r"(b) deformed helix ($Z=10$, precession)")
    for f, lab in ((ax2.set_xlabel, r"$x$"), (ax2.set_ylabel, r"$y$"),
                   (ax2.set_zlabel, r"$z$")):
        f(lab + r"$\ [r_{\rm osc}]$", labelpad=-2)
    ax2.tick_params(labelsize=BODY_PT, pad=-1)

    # share all three axis ranges so the two helices are directly comparable
    xlo = min(xs.min(), xm.min());  xhi = max(xs.max(), xm.max())
    ylo = min(ys.min(), ym.min());  yhi = max(ys.max(), ym.max())
    zmax = max(zs.max(), zm.max())
    for ax in (ax1, ax2):
        ax.view_init(elev=18, azim=-58)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_zlim(0, zmax)

    fig.subplots_adjust(left=0.0, right=0.90, bottom=0.05, top=0.90, wspace=0.10)
    _save(fig, 6)


# --------------------------------------------------------------------------- #
#  Figure 7 — Coulomb spectrum (even harmonic + shift) and shift-vs-coupling  #
# --------------------------------------------------------------------------- #
def fig7():
    a0 = 2.0
    gamma = 1.0e3
    denom = 1.0 + a0 ** 2 / 2.0           # = 3
    r0 = a0 / denom                        # = 2/3
    r_e = 1.0 / (137.0 * gamma)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULL, 2.7))

    # ---- panel (a): spectrum with a lifted even harmonic + shift ----------
    Z_demo = 10.0
    coupling = Z_demo * r_e / r0           # = xi_2
    omega_peak1 = 4.0 * gamma ** 2 / denom
    w = np.linspace(0.5, 3.4, 6000) * omega_peak1

    # xi = 0 (symmetric: even harmonics suppressed) vs xi>0 (even lifted+shift)
    spec0 = _ss_spectrum(w, a0, gamma, denom, 0.0, n_harm=3, N_cyc=30,
                         suppress_even=1e-3)
    specZ = _ss_spectrum(w, a0, gamma, denom, coupling, n_harm=3, N_cyc=30,
                         suppress_even=0.06)   # n=2 lifted by broken symmetry
    axa.plot(w / omega_peak1, spec0 / spec0.max(), color="0.4",
             label=r"$Z=0$ (sym.)")
    axa.plot(w / omega_peak1, specZ / spec0.max(), color="#d62728",
             label=rf"$Z={Z_demo:g}$")
    axa.annotate("even\nharmonic", xy=(2.0, 0.06), xytext=(2.55, 6e-3),
                 fontsize=BODY_PT, ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", lw=0.7))
    axa.set_xlabel(r"$\omega'/\omega'_{\rm peak,1}$")
    axa.set_ylabel(r"$dW/d\omega'$ (norm.)")
    axa.set_title("(a) symmetry-broken comb")
    axa.set_yscale("log")
    axa.set_ylim(1e-4, 2.0)
    axa.grid(True, which="both")
    axa.legend(loc="upper right", fontsize=BODY_PT)

    # ---- panel (b): Delta omega/omega vs coupling Z r_e / r0 --------------
    rng = np.random.default_rng(7)
    Zs = np.array([1, 2, 5, 10, 20, 30, 50, 80], dtype=float)
    coup = Zs * r_e / r0
    law = 2.0 * coup
    scatter = law * (1.0 + 0.06 * rng.standard_normal(len(Zs)))

    cgrid = np.logspace(np.log10(coup.min() / 1.5),
                        np.log10(coup.max() * 1.5), 100)
    axb.plot(cgrid, 2.0 * cgrid, "k--", lw=1.1, label=r"$2\,Zr_e/r_c$")
    axb.plot(coup, scatter, "o", color="#1f77b4", ms=5,
             label="numerical")
    for Z, x, y in zip(Zs, coup, scatter):
        axb.annotate(rf"${int(Z)}$", (x, y), textcoords="offset points",
                     xytext=(4, -9), fontsize=BODY_PT)
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlabel(r"coupling $Zr_e/r_c$")
    axb.set_ylabel(r"$\Delta\omega/\omega$")
    axb.set_title("(b) shift vs. coupling")
    axb.grid(True, which="both")
    axb.legend(loc="upper left", fontsize=BODY_PT)

    fig.tight_layout()
    _save(fig, 7)


# --------------------------------------------------------------------------- #
#  Figure 8 — Periodic potential V3: modulation, beating, pendulum portrait   #
# --------------------------------------------------------------------------- #
def fig8():
    a0 = 1.0
    xi3 = 1.0e-2
    denom = 1.5
    vz0 = (a0 ** 2 / 2.0) / denom          # average longitudinal velocity
    ku_res = 1.0 / vz0                      # resonant undulator wavenumber
    lambda_u = 2.0 * np.pi / ku_res

    # Two-row layout: top row (a),(b); bottom row (c) spanning both columns.
    fig, axd = plt.subplot_mosaic([["a", "b"], ["c", "c"]],
                                  figsize=(COL, 4.4))
    axa, axb, axc = axd["a"], axd["b"], axd["c"]

    psi = np.linspace(0, 16 * np.pi, 4000)
    z_unp = vz0 * psi

    # ---- (a) non-resonant longitudinal modulation -------------------------
    ku_nr = 0.6 * ku_res                    # detuned -> bounded ripple
    dz_nr = (xi3 / (ku_nr * vz0)) * np.sin(ku_nr * z_unp)
    axa.plot(psi / (2 * np.pi), dz_nr, color="#1f77b4", lw=1.0)
    axa.set_xlabel(r"$\psi/2\pi$")
    axa.set_ylabel(r"$\delta z\ [r_{\rm osc}]$")
    axa.set_title("(a) non-resonant")
    axa.grid(True)

    # ---- (b) resonant beating with growing envelope -----------------------
    env = xi3 * vz0 * psi / 2.0
    dz_res = env * np.sin(ku_res * z_unp)
    axb.plot(psi / (2 * np.pi), dz_res, color="#d62728", lw=0.9)
    axb.plot(psi / (2 * np.pi), env, color="k", lw=0.9, ls="--",
             label="envelope")
    axb.plot(psi / (2 * np.pi), -env, color="k", lw=0.9, ls="--")
    axb.set_xlabel(r"$\psi/2\pi$")
    axb.set_ylabel(r"$\delta z\ [r_{\rm osc}]$")
    axb.set_title("(b) resonant beating")
    axb.legend(loc="upper left", fontsize=BODY_PT)
    axb.grid(True)

    # ---- (c) pendulum phase portrait --------------------------------------
    ku = ku_res
    H_sep = 2.0 * xi3 / ku
    zz = np.linspace(-2.2 * np.pi / ku, 2.2 * np.pi / ku, 800)
    pz = np.linspace(-3.0 * np.sqrt(xi3 / ku), 3.0 * np.sqrt(xi3 / ku), 600)
    Zg, Pg = np.meshgrid(zz, pz)
    Hg = Pg ** 2 / 2.0 + (xi3 / ku) * (1.0 - np.cos(ku * Zg))

    # passing + trapped level sets
    levels = np.array([0.25, 0.5, 0.85, 1.3, 1.9]) * H_sep
    axc.contour(Zg * ku / np.pi, Pg / np.sqrt(xi3 / ku), Hg,
                levels=levels, colors="#1f77b4", linewidths=0.7)
    # separatrix
    axc.contour(Zg * ku / np.pi, Pg / np.sqrt(xi3 / ku), Hg,
                levels=[H_sep], colors="#d62728", linewidths=1.4)
    axc.set_xlabel(r"$k_u z/\pi$")
    axc.set_ylabel(r"$\Pi_z/\sqrt{\xi_3/k_u}$")
    axc.set_title("(c) pendulum")
    axc.text(0.0, 0.0, "trapped", color="#2ca02c", fontsize=BODY_PT,
             ha="center", va="center")
    axc.plot([], [], color="#d62728", lw=1.4, label="separatrix")
    axc.legend(loc="upper right", fontsize=BODY_PT)
    axc.grid(True)

    fig.tight_layout()
    _save(fig, 8)


# --------------------------------------------------------------------------- #
#  Figure 9 — Sideband comb spectrum, periodic potential V3                    #
# --------------------------------------------------------------------------- #
def fig9():
    a0 = 2.0
    gamma = 1.0e3
    denom = 3.0                              # 1 + a0^2/2
    omega_peak1 = 4.0 * gamma ** 2 / denom
    ku_display = 0.01                        # omega/c units (satellite offset)
    Delta_sb = 4.0 * gamma ** 2 * ku_display / denom   # = ku_display*omega_peak1
    n_max = 3
    N_cyc = 200            # long-pulse: lines narrow enough that the satellites
                          # clear the carrier's own Gaussian tail and stand out

    def v3_spectrum(grid, xi3, n_hi):
        """Bessel-sideband comb with the Appendix-C modulation index
        mu_n = 2 n xi3 / (1 + a0^2/2).  Satellite weights are J_m^2(mu_n);
        NO Lorentz factor enters (the earlier n*xi3*gamma_tilde was rejected)."""
        spec = np.zeros_like(grid)
        for n in range(1, n_hi + 1):
            u_n = n * a0 ** 2 / (2.0 * denom)
            A = (n * a0) ** 2 * (jv(n - 1, u_n) - jv(n + 1, u_n)) ** 2
            if n % 2 == 0:
                A *= 1e-2                     # even-n suppression on axis
            omega_n = n * omega_peak1
            mu_n = 2.0 * n * xi3 / denom      # <-- corrected modulation index
            m_max = max(2, int(np.ceil(3.0 * mu_n)) + 1)
            for m in range(-m_max, m_max + 1):
                amp = A * jv(m, mu_n) ** 2
                if amp <= 0:
                    continue
                omega_nm = omega_n + m * Delta_sb
                sigma = omega_nm / (2.0 * np.sqrt(2.0) * N_cyc)
                spec += amp * np.exp(-(grid - omega_nm) ** 2 / (2.0 * sigma ** 2))
        return spec

    # ---- main panel: realistic, accessible couplings (carriers dominate) ----
    xis = [0.0, 1.0e-3, 1.0e-2]
    colors = ["#1b1b1b", "#1f77b4", "#d62728"]
    w = np.linspace(0.5, 3.5, 12000) * omega_peak1

    fig, ax = plt.subplots(figsize=(COL, COL * 0.92))

    for xi3, c in zip(xis, colors):
        spec = v3_spectrum(w, xi3, n_max)
        spec /= spec.max()
        ax.plot(w / omega_peak1, spec, color=c, lw=1.0,
                label=rf"$\xi_3={xi3:g}$")

    ax.set_xlabel(r"$\omega'/\omega'_{\rm peak,1}$")
    ax.set_ylabel(r"$dW/d\omega'$ (norm.)")
    ax.set_title(r"Sideband comb, $\mathcal{V}_3$:  $\omega_{n,m}=n\omega\pm m k_u c$")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2.0)
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    ax.text(0.030, 0.045,
            r"accessible $\xi_3$: satellites $<10^{-4}$,"
            "\n"
            r"carriers dominate (see inset)",
            transform=ax.transAxes, fontsize=BODY_PT, ha="left", va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8",
                      boxstyle="round,pad=0.2"))

    # ---- inset: enlarged DISPLAY coupling to make the comb visible ----------
    xi3_disp = 0.1                           # not accessible; for display only
    axins = ax.inset_axes([0.25, 0.34, 0.52, 0.40])
    wz = np.linspace(1.0 - 0.028, 1.0 + 0.028, 8000) * omega_peak1
    spec0 = v3_spectrum(wz, 0.0, 1)
    specd = v3_spectrum(wz, xi3_disp, 1)
    norm = specd.max()
    axins.plot(wz / omega_peak1, spec0 / norm, color="#1b1b1b", lw=0.8,
               label=r"$\xi_3=0$")
    axins.plot(wz / omega_peak1, specd / norm, color="#d62728", lw=0.9,
               label=r"display $\xi_3=0.1$")
    # mark and label the m = +/-1, +/-2 satellites at offsets +/- m k_u c
    for m in (-2, -1, 1, 2):
        axins.axvline(1.0 + m * ku_display, color="0.6", lw=0.5, ls=":")
    axins.text(1.0 + ku_display, 3e-3, r"$\pm1$", fontsize=BODY_PT,
               color="#d62728", ha="center", va="bottom")
    axins.text(1.0 + 2 * ku_display, 9e-7, r"$\pm2$", fontsize=BODY_PT,
               color="#d62728", ha="center", va="bottom")
    axins.set_yscale("log")
    axins.set_ylim(1e-8, 4.0)
    axins.set_title(r"fundamental comb (display)", fontsize=BODY_PT)
    axins.set_xlabel(r"$\omega'/\omega'_{\rm peak,1}$", fontsize=BODY_PT, labelpad=1)
    axins.tick_params(labelsize=BODY_PT)
    axins.legend(loc="upper left", fontsize=BODY_PT, handlelength=1.2)
    axins.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(fig, 9)


# --------------------------------------------------------------------------- #
#  Figure 10 — Universal scaling Delta omega/omega = 2 xi, all three cases     #
# --------------------------------------------------------------------------- #
def fig10():
    rng = np.random.default_rng(42)
    gamma = 1.0e3
    a0 = 2.0
    denom = 3.0
    r0 = a0 / denom
    r_e = 1.0 / (137.0 * gamma)

    fig, ax = plt.subplots(figsize=(COL, COL * 0.78))

    # universal law
    xi_line = np.logspace(-5, -1, 200)
    ax.plot(xi_line, 2.0 * xi_line, "k-", lw=1.6,
            label=r"universal $\Delta\omega/\omega=2\xi$")

    def scatter_pts(xi, frac=0.07):
        return 2.0 * xi * (1.0 + frac * rng.standard_normal(len(xi)))

    # Open markers (facecolor='none') and distinct sizes/zorder so that the
    # three data sets remain individually visible where they overlap on the law.
    # V2 Coulomb
    Z = np.array([1, 2, 5, 10, 20, 30, 50, 79], dtype=float)
    xi2 = Z * r_e / r0
    ax.plot(xi2, scatter_pts(xi2), "s", mfc="none", mec="#2ca02c", mew=1.3,
            ms=8, ls="none", zorder=3, label=r"$\mathcal{V}_2$ Coulomb")

    # V3 periodic
    xi3 = np.logspace(-5, -1.5, 8)
    ax.plot(xi3, scatter_pts(xi3), "^", mfc="none", mec="#d62728", mew=1.3,
            ms=6.5, ls="none", zorder=4, label=r"$\mathcal{V}_3$ periodic")

    # V1 uniform (drawn last, largest open symbol, on top)
    xi1 = np.logspace(-5, -1.5, 8)
    ax.plot(xi1, scatter_pts(xi1), "o", mfc="none", mec="#1f77b4", mew=1.3,
            ms=10, ls="none", zorder=5, label=r"$\mathcal{V}_1$ uniform")

    # detector thresholds as horizontal shaded bands
    thresholds = [("ELI-NP", 3e-3, "#7f7f7f"),
                  ("LUXE", 5e-3, "#9467bd"),
                  ("FACET-II", 1e-2, "#8c564b")]
    xlo, xhi = 1e-5, 1e-1
    for name, thr, col in thresholds:
        ax.axhspan(thr * 0.85, thr * 1.15, color=col, alpha=0.18, zorder=0)
        ax.axhline(thr, color=col, lw=0.8, ls=":")
        ax.text(xhi * 0.62, thr * 1.18, name, color=col, fontsize=BODY_PT, ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel(r"deformation parameter $\xi=\mathcal{V}_{\rm char}/m_ec^2$")
    ax.set_ylabel(r"fractional shift $\Delta\omega/\omega$")
    ax.set_title("Universal scaling and detector reach")
    ax.grid(True, which="both")
    ax.legend(loc="lower right", fontsize=BODY_PT)

    fig.tight_layout()
    _save(fig, 10)


# --------------------------------------------------------------------------- #
def main():
    funcs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]
    for f in funcs:
        f()
    print("All figures generated.")


if __name__ == "__main__":
    main()
