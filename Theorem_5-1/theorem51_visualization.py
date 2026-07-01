"""
Standalone visualisations for the PROOF of Theorem 5.1 (Uniqueness of
temperature) in Lieb & Yngvason, Physics Reports 310 (1999) 1-96.

Two figures, matching the aesthetics/nomenclature of the Lemma 5.1 plots:

  theorem51_kink.png
        The ASSUMPTION that the proof refutes.  At a hypothetical point Z the
        concave energy-profile S(.,V0) has a corner, so the one-sided slopes
        dS/dU disagree:  1/T_+(Z) < 1/T_-(Z).  The whole fan of supporting
        slopes is the sub-differential; inverted, it is the forbidden
        temperature interval  T(Z) = [T_-(Z), T_+(Z)].

  theorem51_contradiction.png
        The heart of Part 2, Case 1.  Transversality (T4) produces states
        X <<  Z  << Y  that are in THERMAL EQUILIBRIUM, X ~T~ Y.  But
        concavity + continuity of T_+/T_- along adiabats (Lemma 5.1) force
              T_+(X) <= T_-(Z) < T_+(Z) <= T_-(Y),
        so the temperature intervals T(X) and T(Y) are DISJOINT -- and two
        states in thermal equilibrium must share a temperature.  Contradiction.

Run directly:  python theorem51_visualization.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theorem51_model import (
    M_MINUS, M_PLUS, T_MINUS_Z, T_PLUS_Z,
    g_shape, entropy, inv_T, temperature, adiabat_U,
)

# --------------------------------------------------------------------------
# Shared style (mirrors the Lemma 5.1 figures)
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_ADIA_Z = "#2c3e50"    # the special "kink" adiabat  dA_Z          (dark slate)
C_LOW = "#2c6fbb"       # lower adiabat / state X                    (blue)
C_HIGH = "#c0392b"      # upper adiabat / state Y                    (red)
C_TEMP = "#8e44ad"      # temperature interval T(Z)                  (purple)
C_FILL = "#f0c419"      # sub-differential fan / shading             (yellow)
C_ORANGE = "#e67e22"    # emphasis                                   (orange)
C_GREEN = "#27ae60"     # thermal-equilibrium link                  (green)
C_GREY = "#7f8c8d"      # guides / envelopes                        (grey)
C_BAD = "#c0392b"       # contradiction mark                        (red)

script_dir = os.path.dirname(os.path.abspath(__file__))


# ==========================================================================
#  FIGURE 1 -- the assumption: a kink in S  <=>  a temperature interval
# ==========================================================================
def make_kink_figure():
    V0 = 1.40                      # the fixed work coordinate of the slice
    U_Z = g_shape(V0)              # kink energy = sigma = 0  at this V0
    span = 0.95
    Us = np.linspace(U_Z - span, U_Z + span, 400)

    fig = plt.figure(figsize=(11.6, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)

    # ---- Left panel: the concave entropy slice with a corner --------------
    axL = fig.add_subplot(gs[0, 0])
    S = entropy(Us, V0)
    axL.plot(Us, S, color=C_ADIA_Z, lw=2.6, zorder=4,
             label=r"$U\mapsto S(U,V_0)$  (concave, with a corner)")

    S_Z = float(entropy(U_Z, V0))          # = 0

    # the two one-sided supporting tangents at the corner
    dl = np.linspace(U_Z - span, U_Z, 60)
    dr = np.linspace(U_Z, U_Z + span, 60)
    axL.plot(dl, S_Z + M_MINUS * (dl - U_Z), color=C_LOW, lw=1.8, ls="--",
             zorder=3)
    axL.plot(dr, S_Z + M_PLUS * (dr - U_Z), color=C_HIGH, lw=1.8, ls="--",
             zorder=3)

    # the sub-differential: the whole fan of supporting slopes in [M_+, M_-]
    fan = np.linspace(M_PLUS, M_MINUS, 9)
    for m in fan:
        axL.plot([U_Z - 0.42, U_Z + 0.42],
                 [S_Z - m * 0.42, S_Z + m * 0.42],
                 color=C_FILL, lw=1.0, alpha=0.85, zorder=2)
    # shade the fan wedge
    axL.fill([U_Z, U_Z + 0.42, U_Z + 0.42],
             [S_Z, S_Z + M_PLUS * 0.42, S_Z + M_MINUS * 0.42],
             color=C_FILL, alpha=0.35, zorder=1, lw=0)
    axL.fill([U_Z, U_Z - 0.42, U_Z - 0.42],
             [S_Z, S_Z - M_PLUS * 0.42, S_Z - M_MINUS * 0.42],
             color=C_FILL, alpha=0.35, zorder=1, lw=0)

    # mark the corner
    axL.plot([U_Z], [S_Z], "o", color=C_ADIA_Z, ms=8, zorder=6)
    axL.axvline(U_Z, color=C_GREY, ls=":", lw=1.0, zorder=0)
    axL.text(U_Z, axL.get_ylim()[0], r"$U_Z$", ha="center", va="bottom",
             fontsize=11, color=C_ADIA_Z)

    # slope labels
    axL.annotate(r"slope $=1/T_-(Z)$",
                 xy=(U_Z - 0.55, S_Z + M_MINUS * (-0.55)),
                 xytext=(U_Z - 0.90, S_Z - 0.30),
                 arrowprops=dict(arrowstyle="->", color=C_LOW, lw=1.1),
                 fontsize=10, color=C_LOW, ha="left", va="center")
    axL.annotate(r"slope $=1/T_+(Z)$",
                 xy=(U_Z + 0.62, S_Z + M_PLUS * 0.62),
                 xytext=(U_Z + 0.30, S_Z - 0.34),
                 arrowprops=dict(arrowstyle="->", color=C_HIGH, lw=1.1),
                 fontsize=10, color=C_HIGH, ha="left", va="center")
    axL.annotate(r"corner of $S$", xy=(U_Z, S_Z),
                 xytext=(U_Z - 0.34, S_Z + 0.24),
                 arrowprops=dict(arrowstyle="->", color=C_ADIA_Z, lw=0.9),
                 fontsize=10, color=C_ADIA_Z, ha="center", va="bottom")

    axL.set_xlabel(r"energy  $U$   (at fixed work coordinate $V_0$)")
    axL.set_ylabel(r"entropy  $S$")
    axL.set_title(r"The assumption:  $S(\,\cdot\,,V_0)$ has a kink at $Z$")
    axL.legend(loc="upper left", fontsize=8.6, framealpha=0.9)

    # ---- Right panel: 1/T = dS/dU jumps  ->  temperature interval ----------
    axR = fig.add_subplot(gs[0, 1])
    # 1/T on each side (draw as functions of U with the jump)
    Ul = np.linspace(U_Z - span, U_Z, 200)
    Ur = np.linspace(U_Z, U_Z + span, 200)
    # left branch is entirely the sigma<=0 side; force the slope so the
    # shared endpoint at U_Z lands on 1/T_-(Z)=M_MINUS (not the np.where tie).
    axR.plot(Ul, inv_T(Ul, V0, m_plus=M_MINUS), color=C_LOW, lw=2.4)
    axR.plot(Ur, inv_T(Ur, V0), color=C_HIGH, lw=2.4)

    # the jump = the sub-differential interval at U_Z
    axR.plot([U_Z, U_Z], [M_PLUS, M_MINUS], color=C_TEMP, lw=4.0,
             solid_capstyle="round", zorder=5,
             label=r"$(\partial S/\partial U)_Z=[\,1/T_+,\,1/T_-\,]$")
    axR.plot([U_Z], [M_MINUS], "o", color=C_LOW, ms=6, zorder=6)
    axR.plot([U_Z], [M_PLUS], "o", color=C_HIGH, ms=6, zorder=6)
    axR.fill_between([U_Z - 0.02, U_Z + 0.02], M_PLUS, M_MINUS,
                     color=C_TEMP, alpha=0.15, lw=0)

    axR.annotate(r"$1/T_-(Z)$", xy=(U_Z, M_MINUS),
                 xytext=(U_Z - 0.55, M_MINUS + 0.06),
                 arrowprops=dict(arrowstyle="->", color=C_LOW, lw=1.0),
                 fontsize=10, color=C_LOW, va="center")
    axR.annotate(r"$1/T_+(Z)$", xy=(U_Z, M_PLUS),
                 xytext=(U_Z + 0.14, M_PLUS - 0.10),
                 arrowprops=dict(arrowstyle="->", color=C_HIGH, lw=1.0),
                 fontsize=10, color=C_HIGH, va="center")
    axR.axvline(U_Z, color=C_GREY, ls=":", lw=1.0, zorder=0)

    axR.text(0.03, 0.05,
             r"jump $\;\Longleftrightarrow\;$ interval $T(Z)=[\,T_-,\,T_+\,]$",
             transform=axR.transAxes, ha="left", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="#f6ecf9", ec=C_TEMP,
                       lw=1.0))

    axR.set_xlabel(r"energy  $U$   (at fixed $V_0$)")
    axR.set_ylabel(r"$\partial S/\partial U \;=\; 1/T$")
    axR.set_title(r"A corner in $S$  $\Rightarrow$  a jump in $1/T$")
    axR.set_ylim(M_PLUS - 0.28, M_MINUS + 0.30)
    axR.legend(loc="upper right", fontsize=8.6, framealpha=0.9)

    fig.suptitle(
        r"Theorem 5.1 -- the hypothesis to be refuted:  "
        r"$T_+(Z) > T_-(Z)$  (a genuine kink of the entropy)",
        fontsize=13, y=1.00)
    out = os.path.join(script_dir, "theorem51_kink.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", os.path.basename(out))


# ==========================================================================
#  FIGURE 2 -- the contradiction (Part 2, Case 1: transversality)
# ==========================================================================
def make_contradiction_figure():
    # --- geometry of the three adiabats and the marked states -------------
    Vg = np.linspace(1.0, 2.0, 400)
    c_Z, c_X, c_Y = 0.0, -0.5, +0.5           # reduced energies (sigma) of the adiabats

    V_Z, V_X, V_Y = 1.40, 1.25, 1.60          # chosen work coordinates
    U_Z = adiabat_U(V_Z, c_Z)
    U_X = adiabat_U(V_X, c_X)
    U_Y = adiabat_U(V_Y, c_Y)

    T_X = float(temperature(U_X, V_X))         # single-valued (off the kink)
    T_Y = float(temperature(U_Y, V_Y))

    fig = plt.figure(figsize=(13.6, 6.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.72, 1.0], wspace=0.24)

    # ======================  LEFT: the (V,U) plane  =======================
    ax = fig.add_subplot(gs[0, 0])

    # faint family of adiabats (level sets sigma = const)
    for c in np.linspace(-1.0, 1.0, 11):
        ax.plot(Vg, adiabat_U(Vg, c), color=C_GREY, lw=0.7, alpha=0.30,
                zorder=0)

    # the three protagonists
    ax.plot(Vg, adiabat_U(Vg, c_X), color=C_LOW, lw=2.4, zorder=3,
            label=r"adiabat $\partial A_X$:  $S(X)<S(Z)$")
    ax.plot(Vg, adiabat_U(Vg, c_Z), color=C_ADIA_Z, lw=3.0, zorder=4,
            label=r"kink adiabat $\partial A_Z$:  $S=S(Z)$")
    ax.plot(Vg, adiabat_U(Vg, c_Y), color=C_HIGH, lw=2.4, zorder=3,
            label=r"adiabat $\partial A_Y$:  $S(Y)>S(Z)$")

    # curve labels at the right edge
    ax.text(2.02, adiabat_U(2.0, c_Z), r"$\partial A_Z$", color=C_ADIA_Z,
            va="center", ha="left", fontsize=10, clip_on=False)
    ax.text(2.02, adiabat_U(2.0, c_Y), r"$\partial A_Y$", color=C_HIGH,
            va="center", ha="left", fontsize=10, clip_on=False)

    # Part 1: the temperature interval is constant along dA_Z (zeroth law).
    # Compact dots on the right of dA_Z + a short caption in the empty band,
    # kept clear of the vertical arrows.
    for Vk in [1.55, 1.72, 1.88]:
        ax.plot([Vk], [adiabat_U(Vk, c_Z)], "o", color=C_ADIA_Z, ms=4.5,
                zorder=5)
    ax.text(1.52, 1.50,
            r"$T(Y')=T(Z)$ along $\partial A_Z$" "\n" r"(Part 1, zeroth law)",
            fontsize=8.4, color=C_ADIA_Z, ha="left", va="bottom")

    # marked states X, Z, Y
    ax.plot([V_Z], [U_Z], "o", color=C_ADIA_Z, ms=9, zorder=6)
    ax.plot([V_X], [U_X], "s", color=C_LOW, ms=8, zorder=6)
    ax.plot([V_Y], [U_Y], "^", color=C_HIGH, ms=9, zorder=6)
    ax.text(V_Z + 0.03, U_Z + 0.03, r"$Z$", color=C_ADIA_Z, fontsize=13,
            ha="left", va="bottom")
    ax.text(V_X - 0.035, U_X, r"$X$", color=C_LOW, fontsize=13, ha="right",
            va="center")
    ax.text(V_Y + 0.035, U_Y + 0.02, r"$Y$", color=C_HIGH, fontsize=13,
            ha="left", va="bottom")

    # vertical-line arguments: climb from X up to dA_Z, drop from Y down to dA_Z
    UX_top = adiabat_U(V_X, c_Z)               # where X's vertical hits dA_Z
    UY_bot = adiabat_U(V_Y, c_Z)               # where Y's vertical hits dA_Z
    ax.annotate("", xy=(V_X, UX_top), xytext=(V_X, U_X),
                arrowprops=dict(arrowstyle="->", color=C_LOW, lw=1.6,
                                ls="--"), zorder=4)
    ax.annotate("", xy=(V_Y, UY_bot), xytext=(V_Y, U_Y),
                arrowprops=dict(arrowstyle="->", color=C_HIGH, lw=1.6,
                                ls="--"), zorder=4)
    ax.plot([V_X], [UX_top], "o", color=C_ADIA_Z, ms=5, zorder=6)
    ax.plot([V_Y], [UY_bot], "o", color=C_ADIA_Z, ms=5, zorder=6)
    # blue note to the LEFT of the X-vertical (climb raises T)
    ax.text(V_X - 0.055, 0.5 * (U_X + UX_top),
            r"$U\!\uparrow\,\Rightarrow\,T\!\uparrow$" "\n"
            r"$T_+(X)\leq T_-(Z)$",
            fontsize=8.4, color=C_LOW, ha="right", va="center")
    # red note to the RIGHT of the Y-vertical
    ax.text(V_Y + 0.06, 0.5 * (U_Y + UY_bot) - 0.02,
            r"$T_-(Y)\geq T_+(Z)$", fontsize=8.4, color=C_HIGH, ha="left",
            va="center")

    # transversality (T4): X ~T~ Y  (thermal equilibrium across the adiabat)
    eq = FancyArrowPatch((V_X, U_X), (V_Y, U_Y),
                         connectionstyle="arc3,rad=-0.28",
                         arrowstyle="<|-|>", mutation_scale=13,
                         color=C_GREEN, lw=1.9, zorder=5)
    ax.add_patch(eq)
    ax.text(1.005, 1.235,
            r"transversality (T4):" "\n"
            r"$X\prec\prec Z\prec\prec Y$,   $X\,\overset{T}{\sim}\,Y$",
            fontsize=8.8, color=C_GREEN, ha="left", va="bottom")

    ax.set_xlim(0.98, 2.12)
    ax.set_ylim(1.18, 2.75)
    ax.set_xlabel(r"work coordinate  $V$")
    ax.set_ylabel(r"energy  $U$")
    ax.set_title(r"(a)  Transversality straddles the kink adiabat $\partial A_Z$")
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    # ==================  RIGHT: the temperature number line  ==============
    axT = fig.add_subplot(gs[0, 1])
    axT.set_title(r"(b)  ...but their temperatures are disjoint")

    # a vertical temperature axis
    tlo, thi = 0.55, 2.48
    axT.set_ylim(tlo, thi)
    axT.set_xlim(0, 1)
    axT.spines["bottom"].set_visible(False)
    axT.set_xticks([])

    xX, xZ, xY = 0.22, 0.52, 0.80

    # the kink band + guide lines at the two kink temperatures
    axT.axhspan(T_MINUS_Z, T_PLUS_Z, color=C_TEMP, alpha=0.07, zorder=0)
    axT.axhline(T_MINUS_Z, color=C_GREY, ls=":", lw=1.0, zorder=0)
    axT.axhline(T_PLUS_Z, color=C_GREY, ls=":", lw=1.0, zorder=0)

    # T(Z) = [T_-, T_+]  (the interval that the theorem forbids)
    axT.plot([xZ, xZ], [T_MINUS_Z, T_PLUS_Z], color=C_TEMP, lw=9,
             solid_capstyle="round", zorder=4)
    axT.plot([xZ], [T_MINUS_Z], "_", color=C_TEMP, ms=20, mew=3, zorder=5)
    axT.plot([xZ], [T_PLUS_Z], "_", color=C_TEMP, ms=20, mew=3, zorder=5)
    axT.text(xZ + 0.06, T_PLUS_Z, r"$T_+(Z)$", ha="left", va="center",
             color=C_TEMP, fontsize=10)
    axT.text(xZ + 0.06, T_MINUS_Z, r"$T_-(Z)$", ha="left", va="center",
             color=C_TEMP, fontsize=10)
    axT.text(xZ - 0.06, 0.5 * (T_MINUS_Z + T_PLUS_Z), r"$T(Z)$", ha="right",
             va="center", color=C_TEMP, fontsize=11)

    # T(X), T(Y): single values, one below the band, one above it
    axT.plot([xX], [T_X], "s", color=C_LOW, ms=12, zorder=4)
    axT.text(xX, T_X - 0.09, r"$T(X)\leq T_-(Z)$", ha="center", va="top",
             color=C_LOW, fontsize=9.5)
    axT.plot([xY], [T_Y], "^", color=C_HIGH, ms=13, zorder=4)
    axT.text(xY, T_Y + 0.09, r"$T(Y)\geq T_+(Z)$", ha="center", va="bottom",
             color=C_HIGH, fontsize=9.5)

    # equilibrium would require a *shared* temperature: refute it
    xcr = 0.36
    axT.annotate("", xy=(xcr, T_Y), xytext=(xcr, T_X),
                 arrowprops=dict(arrowstyle="<->", color=C_BAD, lw=1.6),
                 zorder=3)
    ymid = 0.5 * (T_X + T_Y)
    axT.plot([xcr - 0.05, xcr + 0.05], [ymid - 0.10, ymid + 0.10],
             color=C_BAD, lw=3.0, zorder=6)
    axT.plot([xcr - 0.05, xcr + 0.05], [ymid + 0.10, ymid - 0.10],
             color=C_BAD, lw=3.0, zorder=6)
    axT.text(xcr - 0.07, ymid, "no shared\ntemperature", ha="right",
             va="center", color=C_BAD, fontsize=8.6)

    axT.text(0.5, tlo + 0.02,
             r"$X\overset{T}{\sim}Y$ needs $T(X)\cap T(Y)\neq\varnothing$"
             "\n" r"$\Rightarrow$  contradiction",
             ha="center", va="bottom", color=C_BAD, fontsize=9.2,
             bbox=dict(boxstyle="round,pad=0.35", fc="#fbecea", ec=C_BAD,
                       lw=1.1))

    axT.set_ylabel(r"temperature  $T$")

    # the inequality chain, spanning the figure
    fig.suptitle(
        r"Theorem 5.1, Part 2 (Case 1):  "
        r"$T_+(X)\ \leq\ T_-(Z)\ <\ T_+(Z)\ \leq\ T_-(Y)$  "
        r"$\Rightarrow$  $X\overset{T}{\sim}Y$ is impossible",
        fontsize=13, y=1.005)

    out = os.path.join(script_dir, "theorem51_contradiction.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", os.path.basename(out))

    # console certificate
    print(f"  T(X) = {T_X:.4f}  (should be <= T_-(Z) = {T_MINUS_Z:.4f}):",
          T_X <= T_MINUS_Z + 1e-9)
    print(f"  T(Y) = {T_Y:.4f}  (should be >= T_+(Z) = {T_PLUS_Z:.4f}):",
          T_Y >= T_PLUS_Z - 1e-9)
    print(f"  disjoint intervals: T(X) < T_-(Z) < T_+(Z) < T(Y) -> "
          f"{T_X < T_MINUS_Z < T_PLUS_Z < T_Y}")


if __name__ == "__main__":
    make_kink_figure()
    make_contradiction_figure()
