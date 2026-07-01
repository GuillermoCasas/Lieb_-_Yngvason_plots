"""
The LOGIC of Theorem 5.1 (Uniqueness of temperature) in Lieb & Yngvason,
Physics Reports 310 (1999) 1-96:

    "At every point X in the state space C of a simple system,
         T_+(X) = T_-(X),
     i.e. T(X) is the single number [(dS/dU)(X)]^{-1}."

Equivalently: the concave entropy S has NO kink in the energy direction --
the upper and lower temperatures always agree.

The proof is a reductio ad absurdum, and this figure lays out its four beats
in one 2x2 panel (mirroring the Lemma 5.1 logic figure):

  (1) THE ASSUMPTION.  Suppose some Z has T_+(Z) > T_-(Z).  Then the slice
      U |-> S(U,V0) has a corner at Z: the right/left slopes 1/T_+, 1/T_-
      disagree, so the sub-differential is a whole interval and
      T(Z) = [T_-(Z), T_+(Z)] is a genuine temperature *interval*.

  (2) PART 1 (invariance) + the T4 setup.  By the zeroth law and continuity of
      T_+/T_- along adiabats (Lemma 5.1), that interval is constant along the
      whole adiabat dA_Z:  T(Y') = T(Z) for every Y' in dA_Z.  Then -- already
      opening Part 2 -- transversality (T4) supplies states  X << Z << Y  in
      thermal equilibrium, X ~T~ Y, that straddle dA_Z.

  (3) PART 2, Case 1 (the contradiction).  Concavity (eq. 5.1) forces
          T_+(X) <= T_-(Z) < T_+(Z) <= T_-(Y),
      so the intervals T(X) and T(Y) are DISJOINT.  But two states in thermal
      equilibrium must share a temperature -- contradiction.  (Case 2 uses T5
      at the boundary of the projection and gives the same contradiction.)

  (4) THE RESOLUTION.  Hence no kink can exist: T_+ = T_- everywhere, the jump
      in dS/dU collapses to a point, and T is single-valued (and, by Thm 5.2,
      continuous) on C.

Run directly:  python theorem51_logic.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theorem51_model import (
    M_MINUS, M_PLUS, M_COMMON, T_MINUS_Z, T_PLUS_Z,
    g_shape, entropy, inv_T, temperature, adiabat_U,
)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
})

C_ADIA_Z = "#2c3e50"    # kink adiabat / point Z
C_LOW = "#2c6fbb"       # X side (lower entropy, lower T)
C_HIGH = "#c0392b"      # Y side (higher entropy, higher T)
C_TEMP = "#8e44ad"      # temperature interval
C_FILL = "#f0c419"      # sub-differential fan
C_GREEN = "#27ae60"     # thermal equilibrium
C_GREY = "#7f8c8d"      # guides
C_OK = "#1e7d45"        # resolution / conclusion

V0 = 1.40
U_Z = g_shape(V0)

fig = plt.figure(figsize=(14.5, 9.3))
gs = gridspec.GridSpec(2, 2, hspace=0.34, wspace=0.24)


# ======================================================================
#  Panel (1) -- the assumption: a kink in S  ->  a temperature interval
# ======================================================================
ax1 = fig.add_subplot(gs[0, 0])
span = 0.9
Us = np.linspace(U_Z - span, U_Z + span, 400)
ax1.plot(Us, entropy(Us, V0), color=C_ADIA_Z, lw=2.6, zorder=4)

S_Z = float(entropy(U_Z, V0))
dl = np.linspace(U_Z - span, U_Z, 40)
dr = np.linspace(U_Z, U_Z + span, 40)
ax1.plot(dl, S_Z + M_MINUS * (dl - U_Z), color=C_LOW, lw=1.8, ls="--", zorder=3)
ax1.plot(dr, S_Z + M_PLUS * (dr - U_Z), color=C_HIGH, lw=1.8, ls="--", zorder=3)

# the sub-differential fan (all supporting slopes between M_+ and M_-)
for m in np.linspace(M_PLUS, M_MINUS, 9):
    ax1.plot([U_Z - 0.40, U_Z + 0.40],
             [S_Z - m * 0.40, S_Z + m * 0.40],
             color=C_FILL, lw=1.0, alpha=0.85, zorder=2)
ax1.fill([U_Z, U_Z + 0.40, U_Z + 0.40],
         [S_Z, S_Z + M_PLUS * 0.40, S_Z + M_MINUS * 0.40],
         color=C_FILL, alpha=0.30, zorder=1, lw=0)
ax1.fill([U_Z, U_Z - 0.40, U_Z - 0.40],
         [S_Z, S_Z - M_PLUS * 0.40, S_Z - M_MINUS * 0.40],
         color=C_FILL, alpha=0.30, zorder=1, lw=0)

ax1.plot([U_Z], [S_Z], "o", color=C_ADIA_Z, ms=8, zorder=6)
ax1.annotate(r"slope $1/T_-(Z)$", xy=(U_Z - 0.6, S_Z + M_MINUS * (-0.6)),
             xytext=(U_Z - 0.88, S_Z - 0.30),
             arrowprops=dict(arrowstyle="->", color=C_LOW, lw=1.0),
             fontsize=9.5, color=C_LOW, va="center")
ax1.annotate(r"slope $1/T_+(Z)$", xy=(U_Z + 0.62, S_Z + M_PLUS * 0.62),
             xytext=(U_Z + 0.06, S_Z + 0.62),
             arrowprops=dict(arrowstyle="->", color=C_HIGH, lw=1.0),
             fontsize=9.5, color=C_HIGH, va="center")
ax1.text(U_Z, S_Z - 0.02, r"$Z$", color=C_ADIA_Z, fontsize=12, ha="right",
         va="top")
ax1.set_xlabel(r"energy  $U$   (at fixed work coordinate $V_0$)")
ax1.set_ylabel(r"entropy  $S$")
ax1.set_title(r"(1)  Assume a kink at $Z$:  $T_+(Z) > T_-(Z)$")


# ======================================================================
#  Panel (2) -- Part 1 (invariance) + transversality straddle
# ======================================================================
ax2 = fig.add_subplot(gs[0, 1])
Vg = np.linspace(1.0, 2.0, 400)
c_Z, c_X, c_Y = 0.0, -0.5, +0.5
V_Z, V_X, V_Y = 1.40, 1.25, 1.60
U_Zp, U_Xp, U_Yp = adiabat_U(V_Z, c_Z), adiabat_U(V_X, c_X), adiabat_U(V_Y, c_Y)

for c in np.linspace(-1.0, 1.0, 11):
    ax2.plot(Vg, adiabat_U(Vg, c), color=C_GREY, lw=0.7, alpha=0.28, zorder=0)
ax2.plot(Vg, adiabat_U(Vg, c_X), color=C_LOW, lw=2.2, zorder=3)
ax2.plot(Vg, adiabat_U(Vg, c_Z), color=C_ADIA_Z, lw=3.0, zorder=4)
ax2.plot(Vg, adiabat_U(Vg, c_Y), color=C_HIGH, lw=2.2, zorder=3)
ax2.text(2.02, adiabat_U(2.0, c_X), r"$\partial A_X$", color=C_LOW,
         va="center", ha="left", fontsize=9.5, clip_on=False)
ax2.text(2.02, adiabat_U(2.0, c_Z), r"$\partial A_Z$", color=C_ADIA_Z,
         va="center", ha="left", fontsize=9.5, clip_on=False)
ax2.text(2.02, adiabat_U(2.0, c_Y), r"$\partial A_Y$", color=C_HIGH,
         va="center", ha="left", fontsize=9.5, clip_on=False)

# Part 1: interval constant along dA_Z
for Vk in [1.12, 1.55, 1.82]:
    ax2.plot([Vk], [adiabat_U(Vk, c_Z)], "o", color=C_ADIA_Z, ms=4, zorder=5)
ax2.annotate("Part 1 (zeroth law):\n" r"$T(Y')=T(Z)\ \ \forall\,Y'\in\partial A_Z$",
             xy=(1.82, adiabat_U(1.82, c_Z)), xytext=(1.30, 1.42),
             arrowprops=dict(arrowstyle="->", color=C_ADIA_Z, lw=0.9,
                             connectionstyle="arc3,rad=0.22"),
             fontsize=8.4, color=C_ADIA_Z, ha="left", va="bottom")

# marked states + vertical-line arguments
ax2.plot([V_Z], [U_Zp], "o", color=C_ADIA_Z, ms=8, zorder=6)
ax2.plot([V_X], [U_Xp], "s", color=C_LOW, ms=7, zorder=6)
ax2.plot([V_Y], [U_Yp], "^", color=C_HIGH, ms=8, zorder=6)
ax2.text(V_Z + 0.03, U_Zp + 0.03, r"$Z$", color=C_ADIA_Z, fontsize=12)
ax2.text(V_X - 0.04, U_Xp, r"$X$", color=C_LOW, fontsize=12, ha="right",
         va="center")
ax2.text(V_Y + 0.04, U_Yp + 0.02, r"$Y$", color=C_HIGH, fontsize=12)
ax2.annotate("", xy=(V_X, adiabat_U(V_X, c_Z)), xytext=(V_X, U_Xp),
             arrowprops=dict(arrowstyle="->", color=C_LOW, lw=1.4, ls="--"))
ax2.annotate("", xy=(V_Y, adiabat_U(V_Y, c_Z)), xytext=(V_Y, U_Yp),
             arrowprops=dict(arrowstyle="->", color=C_HIGH, lw=1.4, ls="--"))
eq = FancyArrowPatch((V_X, U_Xp), (V_Y, U_Yp),
                     connectionstyle="arc3,rad=-0.28", arrowstyle="<|-|>",
                     mutation_scale=12, color=C_GREEN, lw=1.8, zorder=5)
ax2.add_patch(eq)
ax2.text(1.005, 1.235,
         "transversality (T4):\n"
         r"$X\prec\prec Z\prec\prec Y$,  $X\,\overset{T}{\sim}\,Y$",
         fontsize=8.4, color=C_GREEN, ha="left", va="bottom")
ax2.set_xlim(0.98, 2.12)
ax2.set_ylim(1.18, 2.75)
ax2.set_xlabel(r"work coordinate  $V$")
ax2.set_ylabel(r"energy  $U$")
ax2.set_title(r"(2)  Part 1 invariance $+$ the transversality (T4) straddle")


# ======================================================================
#  Panel (3) -- the contradiction on the temperature axis
# ======================================================================
ax3 = fig.add_subplot(gs[1, 0])
T_X = float(temperature(U_Xp, V_X))
T_Y = float(temperature(U_Yp, V_Y))
tlo, thi = 0.55, 2.5
ax3.set_ylim(tlo, thi)
ax3.set_xlim(0, 1)
ax3.spines["bottom"].set_visible(False)
ax3.set_xticks([])
xX, xZ, xY = 0.22, 0.52, 0.80

ax3.axhspan(T_MINUS_Z, T_PLUS_Z, color=C_TEMP, alpha=0.07, zorder=0)
ax3.axhline(T_MINUS_Z, color=C_GREY, ls=":", lw=1.0, zorder=0)
ax3.axhline(T_PLUS_Z, color=C_GREY, ls=":", lw=1.0, zorder=0)
ax3.plot([xZ, xZ], [T_MINUS_Z, T_PLUS_Z], color=C_TEMP, lw=9,
         solid_capstyle="round", zorder=4)
ax3.plot([xZ], [T_MINUS_Z], "_", color=C_TEMP, ms=18, mew=3, zorder=5)
ax3.plot([xZ], [T_PLUS_Z], "_", color=C_TEMP, ms=18, mew=3, zorder=5)
ax3.text(xZ + 0.06, T_PLUS_Z, r"$T_+(Z)$", ha="left", va="center",
         color=C_TEMP, fontsize=9.5)
ax3.text(xZ + 0.06, T_MINUS_Z, r"$T_-(Z)$", ha="left", va="center",
         color=C_TEMP, fontsize=9.5)
ax3.text(xZ - 0.06, 0.5 * (T_MINUS_Z + T_PLUS_Z), r"$T(Z)$", ha="right",
         va="center", color=C_TEMP, fontsize=10)
ax3.plot([xX], [T_X], "s", color=C_LOW, ms=11, zorder=4)
ax3.text(xX, T_X - 0.09, r"$T(X)\leq T_-(Z)$", ha="center", va="top",
         color=C_LOW, fontsize=9)
ax3.plot([xY], [T_Y], "^", color=C_HIGH, ms=12, zorder=4)
ax3.text(xY, T_Y + 0.09, r"$T(Y)\geq T_+(Z)$", ha="center", va="bottom",
         color=C_HIGH, fontsize=9)
xcr = 0.36
ax3.annotate("", xy=(xcr, T_Y), xytext=(xcr, T_X),
             arrowprops=dict(arrowstyle="<->", color=C_HIGH, lw=1.5))
ym = 0.5 * (T_X + T_Y)
ax3.plot([xcr - 0.05, xcr + 0.05], [ym - 0.10, ym + 0.10], color=C_HIGH,
         lw=2.8, zorder=6)
ax3.plot([xcr - 0.05, xcr + 0.05], [ym + 0.10, ym - 0.10], color=C_HIGH,
         lw=2.8, zorder=6)
ax3.text(xcr - 0.07, ym, "no shared\ntemperature", ha="right", va="center",
         color=C_HIGH, fontsize=8.4)
ax3.text(0.70, tlo + 0.05,
         r"$X\overset{T}{\sim}Y$ needs" "\n" r"$T(X)\cap T(Y)\neq\varnothing$"
         "\n" r"$\Rightarrow$  contradiction",
         ha="center", va="bottom", color=C_HIGH, fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", fc="#fbecea", ec=C_HIGH, lw=1.1))
ax3.set_ylabel(r"temperature  $T$")
ax3.set_title(r"(3)  Part 2:  $T(X)$ and $T(Y)$ are forced disjoint")


# ======================================================================
#  Panel (4) -- the resolution: the jump collapses, T is single-valued
# ======================================================================
ax4 = fig.add_subplot(gs[1, 1])
Ul = np.linspace(U_Z - span, U_Z, 200)
Ur = np.linspace(U_Z, U_Z + span, 200)
# the forbidden, kinked 1/T (faint) with its jump = the interval
ax4.plot(Ul, inv_T(Ul, V0, m_plus=M_MINUS), color=C_LOW, lw=1.4, ls="--",
         alpha=0.55)
ax4.plot(Ur, inv_T(Ur, V0), color=C_HIGH, lw=1.4, ls="--", alpha=0.55)
ax4.plot([U_Z, U_Z], [M_PLUS, M_MINUS], color=C_TEMP, lw=3.0, alpha=0.45,
         solid_capstyle="round")
ax4.text(U_Z - 0.07, 0.70,
         "forbidden\njump", color=C_TEMP, fontsize=8.2, ha="right",
         va="center", alpha=0.9)

# the resolved, single-valued 1/T (continuous through Z)
Uall = np.linspace(U_Z - span, U_Z + span, 400)
ax4.plot(Uall, inv_T(Uall, V0, m_minus=M_COMMON, m_plus=M_COMMON),
         color=C_OK, lw=2.6, zorder=4,
         label=r"$1/T$ after Thm 5.1 (single-valued)")
ax4.plot([U_Z], [M_COMMON], "o", color=C_OK, ms=8, zorder=5)
ax4.annotate(r"$T_+(Z)=T_-(Z)=T(Z)$",
             xy=(U_Z, M_COMMON), xytext=(U_Z + 0.06, M_COMMON + 0.24),
             arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.0),
             fontsize=9, color=C_OK, ha="left", va="center")
ax4.axvline(U_Z, color=C_GREY, ls=":", lw=1.0, zorder=0)
ax4.text(U_Z, ax4.get_ylim()[0], r"$U_Z$", ha="center", va="bottom",
         color=C_ADIA_Z, fontsize=10)
ax4.set_xlabel(r"energy  $U$   (at fixed $V_0$)")
ax4.set_ylabel(r"$\partial S/\partial U \;=\; 1/T$")
ax4.set_title(r"(4)  Conclusion: the jump collapses, $T$ is unique")
ax4.legend(loc="upper right", fontsize=8.4, framealpha=0.9)


# ------- master title + the logical chain between the two rows ---------
fig.suptitle(
    "The logic of Theorem 5.1: a kink in the entropy would make thermal "
    "equilibrium of straddling states impossible",
    fontsize=13.5, y=0.975)

fig.text(0.5, 0.487,
         "assume a kink  $T_+(Z)>T_-(Z)$   $\\longrightarrow$   "
         "interval constant on $\\partial A_Z$ (Part 1)   $\\longrightarrow$   "
         "transversality $\\Rightarrow$ disjoint $T(X),T(Y)$ (Part 2)   "
         "$\\longrightarrow$   contradiction $\\Rightarrow$ no kink",
         ha="center", va="center", fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbe8", ec="#cdcd80"))

script_dir = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(script_dir, "theorem51_logic.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", os.path.basename(out))

# ---------------------- console certificate ---------------------------
print(f"Model kink at Z:  T_-(Z) = {T_MINUS_Z:.4f},  T_+(Z) = {T_PLUS_Z:.4f}"
      f"  (interval width {T_PLUS_Z - T_MINUS_Z:.4f})")
print(f"T(X) = {T_X:.4f} <= T_-(Z) = {T_MINUS_Z:.4f}:  {T_X <= T_MINUS_Z+1e-9}")
print(f"T(Y) = {T_Y:.4f} >= T_+(Z) = {T_PLUS_Z:.4f}:  {T_Y >= T_PLUS_Z-1e-9}")
print(f"Disjoint  T(X) < T_-(Z) < T_+(Z) < T(Y):  "
      f"{T_X < T_MINUS_Z < T_PLUS_Z < T_Y}  ->  X ~T~ Y impossible.")
