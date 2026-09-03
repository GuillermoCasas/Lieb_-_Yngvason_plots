"""
The LOGIC of Theorem 5.2 (Continuity of temperature) in Lieb & Yngvason,
Physics Reports 310 (1999) 1-96, p. 71:

    "The temperature T(X) = T^+(X) = T^-(X) is a continuous function on
     the state space, Gamma subset R^{n+1}, of a simple system."

The proof is laid out in four interconnected logical beats, matching
the 2x2 layout of lemma51_logic.png and theorem51_logic.png:

  (1) THE GEOMETRIC SETUP.  Base state X_0 and sequence X_j -> X_0 in ball B.
      Horizontal lines l_j = {(U, V_j)} and base horizontal line l_0 = {(U, V_0)}.
      Adiabats A_j = dA_{X_j} passing through each X_j.

  (2) MANDATORY INTERSECTION.  Axiom S2 provides locally Lipschitz pressure
      0 < P_min <= P(X) <= P_max < infty.  The slope dV/dU = -1/P is strictly
      negative and bounded away from 0.  The horizontal direction (dV/dU = 0)
      is forbidden, forcing every adiabat A_j to cross the gap |V_j - V_0|
      and intersect l_0 at Y_j -> X_0.

  (3) THE TWO-LEG BRIDGE.  Decomposes X_j -> X_0 into two controlled legs:
      - Leg 1 (along adiabat A_j): |T(X_j) - T(Y_j)| <= c |X_j - Y_j| -> 0
        by Lemma 5.1 (Lipschitz continuity of T along adiabats).
      - Leg 2 (along line l_0): |T(Y_j) - T(X_0)| -> 0
        by Theorem 5.1 (1D continuity of T along horizontal line l_0).

  (4) THE RESOLUTION.  The triangle inequality
          |T(X_j) - T(X_0)| <= |T(X_j) - T(Y_j)| + |T(Y_j) - T(X_0)| -> 0
      proves that T is continuous at X_0, and therefore continuous everywhere on Gamma.

Run directly:  python theorem52_logic.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theorem52_model import (
    BALL_RADIUS, U0, V0, X0,
    temperature, pressure, adiabat_const, adiabat_U,
    intersect_adiabat_with_l0, generate_sequence, get_ball_bounds,
)

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_BASE = "#d35400"     # base state X_0                             (dark orange)
C_ADIA_0 = "#2c3e50"   # base adiabat A_0                          (dark slate)
C_ADIA_J = "#2980b9"   # sequence adiabats A_j                     (blue)
C_PTS = "#c0392b"      # sequence points X_j                       (crimson)
C_INTER = "#8e44ad"    # intersection points Y_j on l_0            (purple)
C_L0 = "#2c3e50"       # base horizontal line l_0                  (dark slate)
C_LJ = "#7f8c8d"       # horizontal lines l_j                      (grey)
C_BALL = "#16a085"     # ball B                                    (teal)
C_CONE = "#f39c12"     # transversality cone                       (amber)
C_LEG1 = "#2980b9"     # Leg 1 (along adiabat)                     (blue)
C_LEG2 = "#8e44ad"     # Leg 2 (along line l_0)                    (purple)
C_OK = "#27ae60"       # resolution / conclusion                   (green)
C_BAD = "#c0392b"      # forbidden direction                       (red)

script_dir = os.path.dirname(os.path.abspath(__file__))

seq_X = generate_sequence(4)
seq_Y = [intersect_adiabat_with_l0(pt) for pt in seq_X]

fig = plt.figure(figsize=(14.8, 9.6))
gs = gridspec.GridSpec(2, 2, hspace=0.34, wspace=0.25)


# ======================================================================
#  Panel (1) -- The Geometric Setup: Sequence X_j -> X_0 and Ball B
# ======================================================================
ax1 = fig.add_subplot(gs[0, 0])
V_span = np.linspace(0.85, 1.85, 300)
K0 = adiabat_const(U0, V0)

# Faint background adiabats
for k_val in np.linspace(K0 - 0.6, K0 + 0.6, 9):
    ax1.plot(adiabat_U(V_span, k_val), V_span, color="#d5dbdb", lw=0.6, zorder=1)

# Ball B
phi = np.linspace(0, 2 * np.pi, 150)
ax1.fill(U0 + BALL_RADIUS * np.cos(phi), V0 + BALL_RADIUS * np.sin(phi),
         color=C_BALL, alpha=0.08, zorder=2)
ax1.plot(U0 + BALL_RADIUS * np.cos(phi), V0 + BALL_RADIUS * np.sin(phi),
         color=C_BALL, ls="--", lw=1.2, zorder=2)
ax1.text(U0 - 0.36, V0 + 0.38, r"ball $B$ centered at $X_0$",
         color=C_BALL, fontsize=9.5, fontweight="bold")

# Base line l_0 (label placed cleanly below line on the right)
ax1.axhline(V0, color=C_L0, lw=2.2, zorder=3)
ax1.text(2.85, V0 - 0.035, r"$\mathbf{l_0} = \{(U, V_0)\}$",
         color=C_L0, fontsize=10, ha="right", va="top", fontweight="bold")

# Points X_j and horizontal lines l_j
for j, (Xj, Yj) in enumerate(zip(seq_X, seq_Y), 1):
    Vj = Xj[1]
    Kj = adiabat_const(*Xj)
    ax1.axhline(Vj, color=C_LJ, ls=":", lw=1.0, zorder=3)
    ax1.plot(adiabat_U(V_span, Kj), V_span, color=C_ADIA_J, lw=1.3, alpha=0.7, zorder=4)
    ax1.plot(Xj[0], Xj[1], "s", color=C_PTS, ms=6, zorder=6)
    ax1.plot(Yj[0], Yj[1], "o", color=C_INTER, ms=6, zorder=6)
    if j in [1, 2]:
        ax1.text(Xj[0] - 0.03, Xj[1] + 0.02, rf"$X_{j}$", color=C_PTS,
                 fontsize=9.5, ha="right", va="bottom")

# Label Y_1 with arrow
Y1 = seq_Y[0]
ax1.annotate(r"$Y_1$", xy=(Y1[0], Y1[1]), xytext=(Y1[0] - 0.08, Y1[1] - 0.08),
             arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.0),
             color=C_INTER, fontsize=9.5, ha="right", va="top")

# Base point X_0 and base adiabat A_0 (annotated above-right)
ax1.plot(adiabat_U(V_span, K0), V_span, color=C_ADIA_0, lw=2.2, zorder=4)
ax1.plot(X0[0], X0[1], "*", color=C_BASE, ms=12, zorder=7)
ax1.annotate(r"$\mathbf{X_0}$", xy=(X0[0], X0[1]), xytext=(X0[0] + 0.08, X0[1] + 0.08),
             arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.1),
             color=C_BASE, fontsize=11, fontweight="bold", ha="left", va="bottom")

ax1.set_xlim(1.50, 2.90)
ax1.set_ylim(0.90, 1.80)
ax1.set_xlabel(r"energy  $U$")
ax1.set_ylabel(r"work coordinate  $V$")
ax1.set_title(r"(1) Geometric Setup: $X_j \to X_0$ and Horizontal Lines $l_j$")

ax1.text(0.04, 0.06,
         r"$\bullet\ X_j = (U_j, V_j) \in \Gamma$,  $X_j \rightarrow X_0$" "\n"
         r"$\bullet\ l_j = \{(U, V_j)\}$ are horizontal lines in $(U, V)$" "\n"
         r"$\bullet\ A_j = \partial A_{X_j}$ is the adiabat through $X_j$",
         transform=ax1.transAxes, fontsize=8.8,
         bbox=dict(boxstyle="round,pad=0.35", fc="#fdfefe", ec="#bdc3c7", lw=0.9))


# ======================================================================
#  Panel (2) -- Mandatory Intersection: Axiom S2 & Transversality Cone
# ======================================================================
ax2 = fig.add_subplot(gs[0, 1])

X1, Y1 = seq_X[0], seq_Y[0]
bounds = get_ball_bounds()
s_min, s_max = bounds["slope_min"], bounds["slope_max"]

# Horizontal lines l_0 and l_1
ax2.axhline(V0, color=C_L0, lw=2.2, zorder=3)
ax2.axhline(X1[1], color=C_LJ, ls="--", lw=1.5, zorder=3)
ax2.text(2.68, V0 - 0.035, r"$l_0: V = V_0$", color=C_L0, fontsize=10, ha="right", va="top", fontweight="bold")
ax2.text(2.68, X1[1] + 0.02, r"$l_1: V = V_1$", color=C_LJ, fontsize=9.5, ha="right", va="bottom")

# Transversality cone from X_1
poly_pts = [
    [X1[0], X1[1]],
    [X1[0] + (V0 - 0.06 - X1[1]) / s_min, V0 - 0.06],
    [X1[0] + (V0 - 0.06 - X1[1]) / s_max, V0 - 0.06],
]
ax2.add_patch(Polygon(poly_pts, color=C_CONE, alpha=0.22, zorder=2, lw=0))

v_cone = np.linspace(V0 - 0.06, X1[1] + 0.06, 100)
ax2.plot(X1[0] + (v_cone - X1[1]) / s_min, v_cone, color=C_CONE, ls="--", lw=1.4, zorder=3,
         label=r"slope $-\frac{1}{P_{\min}}$")
ax2.plot(X1[0] + (v_cone - X1[1]) / s_max, v_cone, color=C_CONE, ls="-.", lw=1.4, zorder=3,
         label=r"slope $-\frac{1}{P_{\max}}$")

# True adiabat A_1
K1 = adiabat_const(*X1)
ax2.plot(adiabat_U(V_span, K1), V_span, color=C_ADIA_J, lw=2.4, zorder=5, label=r"adiabat $A_1$")

# State X_1, intersection Y_1, base X_0
ax2.plot(X1[0], X1[1], "s", color=C_PTS, ms=8, zorder=6)
ax2.text(X1[0] - 0.03, X1[1] + 0.03, r"$X_1$", color=C_PTS, fontsize=11, fontweight="bold", ha="right")
ax2.plot(Y1[0], Y1[1], "o", color=C_INTER, ms=8, zorder=6)
ax2.annotate(r"$Y_1 = A_1 \cap l_0$", xy=(Y1[0], Y1[1]),
             xytext=(Y1[0] - 0.12, Y1[1] - 0.09),
             arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.1),
             color=C_INTER, fontsize=10, fontweight="bold", ha="right", va="top")
ax2.plot(X0[0], X0[1], "*", color=C_BASE, ms=12, zorder=6)
ax2.annotate(r"$\mathbf{X_0}$", xy=(X0[0], X0[1]), xytext=(X0[0] + 0.08, X0[1] + 0.08),
             arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.1),
             color=C_BASE, fontsize=11, fontweight="bold", ha="left")

# Forbidden horizontal line
ax2.plot([X1[0] - 0.25, X1[0] + 0.35], [X1[1], X1[1]], color=C_BAD, ls=":", lw=2.0, zorder=4)
ax2.text(X1[0] + 0.20, X1[1] - 0.045, r"$\frac{dV}{dU} = 0$ (FORBIDDEN)", color=C_BAD, fontsize=8.5)

ax2.set_xlim(1.50, 2.70)
ax2.set_ylim(1.02, 1.70)
ax2.set_xlabel(r"energy  $U$")
ax2.set_ylabel(r"work coordinate  $V$")
ax2.set_title(r"(2) Mandatory Intersection: Bounded Slope $\frac{dV}{dU} = -\frac{1}{P} < 0$")
ax2.legend(loc="upper left", fontsize=8.4, framealpha=0.9)

# Placed cleanly in lower-left
ax2.text(0.04, 0.20,
         r"$\bullet\ 0 < P_{\min} \leq P(X) \leq P_{\max} < \infty$" "\n"
         r"$\bullet\ \frac{dV}{dU} \in [-\frac{1}{P_{\min}}, -\frac{1}{P_{\max}}] < 0$" "\n"
         r"$\bullet\ A_j$ cannot run parallel to $l_0$" "\n"
         r"$\Rightarrow A_j$ MUST intersect $l_0$ at $Y_j$!" "\n"
         r"$\bullet\ |Y_j - X_0| \leq (1 + P_{\max})|X_j - X_0| \rightarrow 0$",
         transform=ax2.transAxes, fontsize=8.5, ha="left", va="center",
         bbox=dict(boxstyle="round,pad=0.35", fc="#fef9e7", ec=C_CONE, lw=1.0))


# ======================================================================
#  Panel (3) -- The Two-Leg Bridge: Along Adiabat + Along Line l_0
# ======================================================================
ax3 = fig.add_subplot(gs[1, 0])

v_p1 = np.linspace(X1[1], Y1[1], 100)
u_p1 = adiabat_U(v_p1, K1)

# Leg 1
ax3.plot(u_p1, v_p1, color=C_LEG1, lw=3.6, zorder=4)
mid_i = 50
ax3.annotate("", xy=(u_p1[mid_i + 8], v_p1[mid_i + 8]),
             xytext=(u_p1[mid_i - 8], v_p1[mid_i - 8]),
             arrowprops=dict(arrowstyle="->", color=C_LEG1, lw=2.4, mutation_scale=15), zorder=5)

# Leg 2
ax3.annotate("", xy=(X0[0] - 0.015, V0), xytext=(Y1[0] + 0.015, V0),
             arrowprops=dict(arrowstyle="->", color=C_LEG2, lw=3.0, mutation_scale=16), zorder=5)

# Faint reference lines
ax3.axhline(V0, color=C_L0, lw=1.2, ls="--", alpha=0.5, zorder=2)
ax3.axhline(X1[1], color=C_LJ, lw=1.0, ls=":", alpha=0.5, zorder=2)

# Points
ax3.plot(X1[0], X1[1], "s", color=C_PTS, ms=8.5, zorder=6)
ax3.plot(Y1[0], Y1[1], "o", color=C_INTER, ms=8.5, zorder=6)
ax3.plot(X0[0], X0[1], "*", color=C_BASE, ms=13, zorder=6)

ax3.text(X1[0], X1[1] + 0.04, r"$X_j = (U_j, V_j)$", color=C_PTS, fontsize=11, fontweight="bold", ha="center")
ax3.text(Y1[0] - 0.02, Y1[1] - 0.06, r"$Y_j = (U_{Y_j}, V_0)$", color=C_INTER, fontsize=10.5, fontweight="bold", ha="center")
ax3.text(X0[0] + 0.03, X0[1] + 0.03, r"$X_0 = (U_0, V_0)$", color=C_BASE, fontsize=11, fontweight="bold", ha="left")

# Callout cards for Leg 1 and Leg 2
ax3.text(0.5 * (X1[0] + Y1[0]) - 0.18, 0.5 * (X1[1] + Y1[1]) + 0.02,
         r"$\mathbf{Leg\ 1\ (along\ A_j)}:$" "\n"
         r"$|T(X_j) - T(Y_j)| \leq c\,|X_j - Y_j|$" "\n"
         r"$\rightarrow 0$  by Lemma 5.1",
         color=C_LEG1, fontsize=9.2, ha="right", va="center",
         bbox=dict(boxstyle="round,pad=0.35", fc="#ebf5fb", ec=C_LEG1, lw=1.1))

ax3.text(0.5 * (Y1[0] + X0[0]) + 0.12, V0 - 0.14,
         r"$\mathbf{Leg\ 2\ (along\ l_0)}:$" "\n"
         r"$|T(Y_j) - T(X_0)| \rightarrow 0$" "\n"
         r"by Theorem 5.1 (1D continuity)",
         color=C_LEG2, fontsize=9.2, ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.35", fc="#f5eef8", ec=C_LEG2, lw=1.1))

ax3.set_xlim(1.50, 2.70)
ax3.set_ylim(1.00, 1.65)
ax3.set_xlabel(r"energy  $U$")
ax3.set_ylabel(r"work coordinate  $V$")
ax3.set_title(r"(3) The Two-Leg Path: Connecting $X_j \to Y_j \to X_0$")


# ======================================================================
#  Panel (4) -- The Resolution: Triangle Inequality & Continuity
# ======================================================================
ax4 = fig.add_subplot(gs[1, 1])

js = np.array([1, 2, 3, 4])
leg1 = [abs(temperature(*Xj) - temperature(*Yj)) for Xj, Yj in zip(seq_X, seq_Y)]
leg2 = [abs(temperature(*Yj) - temperature(*X0)) for Yj in seq_Y]
tot = [abs(temperature(*Xj) - temperature(*X0)) for Xj in seq_X]
dX = [np.linalg.norm(Xj - X0) for Xj in seq_X]

bw = 0.22
ax4.bar(js - bw, leg1, width=bw, color=C_LEG1, alpha=0.85, label=r"Leg 1: $|T(X_j) - T(Y_j)|$ (Lemma 5.1)")
ax4.bar(js, leg2, width=bw, color=C_LEG2, alpha=0.85, label=r"Leg 2: $|T(Y_j) - T(X_0)|$ (Thm 5.1)")
ax4.bar(js + bw, tot, width=bw, color=C_OK, alpha=0.85, label=r"Total: $|T(X_j) - T(X_0)|$")
ax4.plot(js + bw, tot, "o-", color="#196f3d", lw=1.8, ms=5, zorder=5)

ax4.set_xticks(js)
ax4.set_xticklabels([rf"$j={j}$" "\n" rf"$\Delta={dX[j-1]:.2f}$" for j in js])
ax4.set_xlabel(r"step $j$ in sequence  (distance $|X_j - X_0| \to 0$)")
ax4.set_ylabel(r"temperature error  $|\Delta T|$")
ax4.set_ylim(0, 0.54)
ax4.legend(loc="upper left", fontsize=8.2, framealpha=0.92)
ax4.set_title(r"(4) The Resolution: $|T(X_j) - T(X_0)| \to 0$ (Continuity)")

# Text box placed in upper-right with zero overlap
ax4.text(0.58, 0.62,
         r"$\mathbf{Triangle\ Inequality}:$" "\n"
         r"$|T(X_j) - T(X_0)| \;\leq\; |T(X_j) - T(Y_j)| + |T(Y_j) - T(X_0)|$" "\n\n"
         r"$\mathbf{Both\ terms\ vanish}:$" "\n"
         r"$\bullet\ |T(X_j) - T(Y_j)| \leq c|X_j - Y_j| \rightarrow 0$" "\n"
         r"$\bullet\ |T(Y_j) - T(X_0)| \rightarrow 0$" "\n\n"
         r"$\mathbf{Result}:\; T(X) \text{ is continuous on } \Gamma$!",
         transform=ax4.transAxes, ha="center", va="center", fontsize=8.8,
         bbox=dict(boxstyle="round,pad=0.45", fc="#eafaf1", ec=C_OK, lw=1.2))

fig.suptitle(
    r"Theorem 5.2 (Continuity of Temperature) -- Four-Beat Logic of the Proof",
    fontsize=13.5, y=0.99
)

out_path = os.path.join(script_dir, "theorem52_logic.png")
fig.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print("Saved", os.path.basename(out_path))


if __name__ == "__main__":
    pass
