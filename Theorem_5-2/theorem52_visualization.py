"""
Single Standalone Visualization with Accentuated Zoom Inset for the PROOF of Theorem 5.2
(Continuity of temperature) in Lieb & Yngvason, Physics Reports 310 (1999) 1-96, p. 71.

Features:
  * Main State Space (U, V):
      - Internal energy coordinate U on horizontal axis, work coordinate V on vertical axis.
      - Base state X_infinity = (U_0, V_0) and base adiabat A_infinity.
      - Ball B = B(X_infinity, r).
      - Sequence of points X_j -> X_infinity (X_1, X_2, X_3, X_4).
      - Horizontal lines l_j: V = V_j and base line l_0: V = V_0.
      - Adiabats A_j cutting transversally across the horizontal lines.
      - Intersections Y_j = A_j \\cap l_0 sliding along l_0 -> X_infinity.
      - Transversality cone from Axiom S2 (0 < P_min <= P <= P_max).
      - Two-leg decomposition for j=1 (Leg 1 along A_1, Leg 2 along l_0).
      - Triangle inequality resolution.
  * Accentuated Inset Zoom around X_infinity:
      - High magnification (~6x visual zoom) of the immediate neighborhood [2.095, 2.215] x [1.182, 1.290].
      - Clearly resolves the tight terminal cluster X_3, X_4 -> X_infinity and Y_2, Y_3, Y_4 -> X_infinity.
      - Distinct horizontal lines l_3, l_4, l_0.
      - Shows steep transversal crossing of adiabats A_3, A_4, A_infinity across l_0.
      - Micro-scale two-leg path for step j=4 (along A_4 and l_0).
      - Connected to main plot via indicate_inset_zoom.
  * Purely mathematical symbols on axes (U, V) and zero English sentences on the figure.
  * All labels equipped with protective white halo boxes in clear open whitespace.

Run directly:  python theorem52_visualization.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theorem52_model import (
    BALL_RADIUS, U0, V0, X0,
    temperature, pressure, adiabat_const, adiabat_U,
    intersect_adiabat_with_l0, generate_sequence, get_ball_bounds,
)

# --------------------------------------------------------------------------
# Styling & Palette
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_BASE = "#d35400"     # base state X_infinity                      (dark orange)
C_ADIA_0 = "#2c3e50"   # base adiabat A_infinity                    (dark slate)
C_ADIA_J = "#2980b9"   # sequence adiabats A_j                      (blue)
C_PTS = "#c0392b"      # sequence points X_j                        (crimson)
C_INTER = "#8e44ad"    # intersection points Y_j on l_0             (purple)
C_L0 = "#2c3e50"       # base horizontal line l_0                   (dark slate)
C_LJ = "#7f8c8d"       # horizontal lines l_j                       (grey)
C_BALL = "#16a085"     # neighborhood ball B                        (teal)
C_CONE = "#f39c12"     # transversality cone                        (amber)
C_LEG1 = "#2980b9"     # Leg 1 (along adiabat)                      (blue)
C_LEG2 = "#8e44ad"     # Leg 2 (along line l_0)                     (purple)
C_OK = "#27ae60"       # resolution / conclusion                    (green)
C_BAD = "#c0392b"      # forbidden direction                        (red)

script_dir = os.path.dirname(os.path.abspath(__file__))


def make_single_proof_plot_accentuated_zoom():
    fig, ax = plt.subplots(figsize=(13.2, 9.4))

    # Equal aspect ratio ensures ball B is a true geometric circle
    u_min, u_max = 1.40, 3.05
    v_min, v_max = 0.58, 1.95
    ax.set_aspect("equal", adjustable="box")

    V_grid = np.linspace(v_min, v_max, 450)

    # 1. Background faint family of adiabats
    K0 = adiabat_const(U0, V0)
    for k_val in np.linspace(K0 - 0.75, K0 + 0.75, 13):
        ax.plot(adiabat_U(V_grid, k_val), V_grid, color="#bdc3c7",
                lw=0.6, alpha=0.30, zorder=1)

    # 2. Neighborhood Ball B = B(X_infinity, r)
    phi = np.linspace(0, 2 * np.pi, 250)
    b_u = U0 + BALL_RADIUS * np.cos(phi)
    b_v = V0 + BALL_RADIUS * np.sin(phi)
    ax.fill(b_u, b_v, color=C_BALL, alpha=0.07, zorder=2, lw=0)
    ax.plot(b_u, b_v, color=C_BALL, ls="--", lw=1.5, alpha=0.75, zorder=2)
    ax.text(U0 - 0.38, V0 + 0.43,
            r"$B = B(X_\infty, r)$",
            color=C_BALL, fontsize=11.5, fontweight="bold", ha="center", va="bottom")

    # 3. Base Horizontal Line l_0: V = V_0
    ax.axhline(V0, color=C_L0, lw=2.4, zorder=3)
    ax.text(u_max - 0.03, V0 - 0.045,
            r"$l_0 = \{(U, V_0)\}$",
            color=C_L0, fontsize=11.5, ha="right", va="top", fontweight="bold")

    # 4. Base Adiabat A_infinity through X_infinity
    ax.plot(adiabat_U(V_grid, K0), V_grid, color=C_ADIA_0, lw=2.8, zorder=4,
            label=r"$A_\infty = \partial A_{X_\infty}$")

    # 5. Sequence of Points X_j -> X_infinity
    seq_X = generate_sequence(4)
    seq_Y = [intersect_adiabat_with_l0(pt) for pt in seq_X]

    # Draw horizontal lines l_j and adiabats A_j on main plot
    for j, (Xj, Yj) in enumerate(zip(seq_X, seq_Y), 1):
        Vj = Xj[1]
        Kj = adiabat_const(*Xj)

        # Horizontal line l_j: V = V_j
        ax.axhline(Vj, color=C_LJ, ls=":", lw=1.1, zorder=3)
        ax.text(u_min + 0.03, Vj + 0.015,
                rf"$l_{j}: V = V_{j}$",
                color=C_LJ, fontsize=9.2, ha="left", va="bottom")

        # Adiabat A_j through X_j
        line_w = 2.0 if j == 1 else 1.4
        alp = 0.95 if j == 1 else 0.70
        lbl = r"$A_j = \partial A_{X_j}$" if j == 2 else None
        ax.plot(adiabat_U(V_grid, Kj), V_grid, color=C_ADIA_J, lw=line_w,
                alpha=alp, zorder=4, label=lbl)

        # Plot points X_j and Y_j
        ax.plot(Xj[0], Xj[1], "s", color=C_PTS, ms=7.5, zorder=7)
        ax.plot(Yj[0], Yj[1], "o", color=C_INTER, ms=7.5, zorder=7)

        # Label points X_j on main plot with clean protective white boxes
        if j == 1:
            ax.annotate(rf"$X_1 = (U_1, V_1)$",
                        xy=(Xj[0], Xj[1]),
                        xytext=(Xj[0] - 0.08, Xj[1] + 0.06),
                        arrowprops=dict(arrowstyle="->", color=C_PTS, lw=1.2),
                        color=C_PTS, fontsize=11, fontweight="bold", ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_PTS, lw=1.0, alpha=0.95),
                        zorder=9)
        elif j == 2:
            ax.annotate(rf"$X_2$",
                        xy=(Xj[0], Xj[1]),
                        xytext=(Xj[0] - 0.06, Xj[1] + 0.04),
                        arrowprops=dict(arrowstyle="->", color=C_PTS, lw=1.0),
                        color=C_PTS, fontsize=10, fontweight="bold", ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_PTS, lw=0.8, alpha=0.95),
                        zorder=9)

    # Directional arrows for X_j -> X_infinity on main plot
    for j in range(len(seq_X) - 1):
        ax.annotate("", xy=(seq_X[j+1][0], seq_X[j+1][1]),
                    xytext=(seq_X[j][0], seq_X[j][1]),
                    arrowprops=dict(arrowstyle="->", color=C_PTS, ls="--", lw=1.2,
                                    mutation_scale=11), zorder=6)
    ax.annotate("", xy=(X0[0], X0[1]), xytext=(seq_X[-1][0], seq_X[-1][1]),
                arrowprops=dict(arrowstyle="->", color=C_PTS, ls="--", lw=1.2,
                                mutation_scale=11), zorder=6)

    # 6. Intersection points Y_j on l_0 on main plot (uncovered by any line)
    Y1 = seq_Y[0]
    ax.annotate(r"$Y_1 = A_1 \cap l_0$", xy=(Y1[0], Y1[1]),
                xytext=(Y1[0] - 0.16, Y1[1] - 0.14),
                arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.2),
                color=C_INTER, fontsize=10.5, fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_INTER, lw=1.0, alpha=0.95),
                zorder=9)

    # Arrows for Y_j -> X_infinity along line l_0
    for j in range(len(seq_Y) - 1):
        ax.annotate("", xy=(seq_Y[j+1][0], V0), xytext=(seq_Y[j][0], V0),
                    arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.4,
                                    mutation_scale=10), zorder=6)
    ax.annotate("", xy=(X0[0] - 0.015, V0), xytext=(seq_Y[-1][0], V0),
                arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.4,
                                mutation_scale=10), zorder=6)

    # 7. Base / Limit State X_infinity = (U_0, V_0) on main plot (placed safely below l_0, clear of inset frame)
    ax.plot(X0[0], X0[1], "*", color=C_BASE, ms=16, zorder=8)
    ax.annotate(r"$X_\infty = (U_0, V_0)$",
                xy=(X0[0], X0[1]),
                xytext=(X0[0] + 0.04, 1.11),
                arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.2),
                color=C_BASE, fontsize=10.5, fontweight="bold", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C_BASE, lw=1.0, alpha=0.95),
                zorder=9)

    # 8. Highlight the Two-Leg Path for j = 1 on main plot
    X1 = seq_X[0]
    K1 = adiabat_const(*X1)
    v_path = np.linspace(X1[1], Y1[1], 100)
    u_path = adiabat_U(v_path, K1)
    ax.plot(u_path, v_path, color=C_LEG1, lw=3.6, zorder=5)

    # Leg 1 arrow
    mid_idx = 50
    ax.annotate("", xy=(u_path[mid_idx + 8], v_path[mid_idx + 8]),
                xytext=(u_path[mid_idx - 8], v_path[mid_idx - 8]),
                arrowprops=dict(arrowstyle="->", color=C_LEG1, lw=2.6,
                                mutation_scale=17), zorder=6)

    # Leg 2 arrow along horizontal line l_0
    ax.annotate("", xy=(X0[0] - 0.02, V0), xytext=(Y1[0] + 0.02, V0),
                arrowprops=dict(arrowstyle="->", color=C_LEG2, lw=3.0,
                                mutation_scale=17), zorder=6)

    # Card for Leg 1 (no sentences, purely symbolic, zorder=15 in front of curves)
    ax.text(0.5 * (X1[0] + Y1[0]) - 0.18, 0.5 * (X1[1] + Y1[1]) + 0.03,
            r"$\mathbf{Leg\ 1}\ (A_1):$" "\n"
            r"$|T(X_1) - T(Y_1)| \leq c\,|X_1 - Y_1| \rightarrow 0$" "\n"
            r"$(\mathrm{Lemma\ 5.1},\; X, Y \in A_1 \cap B)$",
            color=C_LEG1, fontsize=9.5, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.40", fc="#ebf5fb", ec=C_LEG1, lw=1.2, alpha=1.0),
            zorder=15)

    # Card for Leg 2 (placed right below the purple path on l_0, zorder=15 in front of curves)
    ax.annotate(r"$\mathbf{Leg\ 2}\ (l_0):$" "\n"
                r"$|T(Y_1) - T(X_\infty)| \rightarrow 0$" "\n"
                r"$(\mathrm{Theorem\ 5.1},\; Y_1, X_\infty \in l_0)$",
                xy=(0.5 * (Y1[0] + X0[0]), V0),
                xytext=(1.96, 1.01),
                arrowprops=dict(arrowstyle="->", color=C_LEG2, lw=1.2),
                color=C_LEG2, fontsize=9.2, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.38", fc="#f5eef8", ec=C_LEG2, lw=1.2, alpha=1.0),
                zorder=15)

    # 9. Transversality Cone on main plot
    bounds = get_ball_bounds()
    s_min, s_max = bounds["slope_min"], bounds["slope_max"]

    poly_pts = [
        [X1[0], X1[1]],
        [X1[0] + (V0 - 0.06 - X1[1]) / s_min, V0 - 0.06],
        [X1[0] + (V0 - 0.06 - X1[1]) / s_max, V0 - 0.06],
    ]
    ax.add_patch(Polygon(poly_pts, color=C_CONE, alpha=0.18, zorder=2, lw=0))

    v_cone = np.linspace(V0 - 0.06, X1[1] + 0.06, 100)
    ax.plot(X1[0] + (v_cone - X1[1]) / s_min, v_cone, color=C_CONE, ls="--", lw=1.4, zorder=3)
    ax.plot(X1[0] + (v_cone - X1[1]) / s_max, v_cone, color=C_CONE, ls="-.", lw=1.4, zorder=3)

    # Forbidden zero slope
    x_cross = 2.15
    ax.plot([X1[0] - 0.22, x_cross + 0.15], [X1[1], X1[1]], color=C_BAD, ls=":", lw=1.8, zorder=3)
    ax.plot([x_cross - 0.03, x_cross + 0.03], [X1[1] - 0.03, X1[1] + 0.03], color=C_BAD, lw=2.4, zorder=6)
    ax.plot([x_cross - 0.03, x_cross + 0.03], [X1[1] + 0.03, X1[1] - 0.03], color=C_BAD, lw=2.4, zorder=6)
    ax.text(x_cross + 0.05, X1[1], r"$\frac{dV}{dU} = 0$", color=C_BAD, fontsize=9.8, va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=7)

    # 10. Summary Triangle Inequality Card (bottom left, zorder=15 in front of curves)
    ax.text(0.035, 0.04,
            r"$\mathbf{Triangle\ Inequality}:$" "\n"
            r"$|T(X_j) - T(X_\infty)| \;\leq\; |T(X_j) - T(Y_j)| + |T(Y_j) - T(X_\infty)|$" "\n\n"
            r"$|T(X_j) - T(Y_j)| \leq c\,|X_j - Y_j| \rightarrow 0 \quad (\mathrm{Lemma\ 5.1})$" "\n"
            r"$|T(Y_j) - T(X_\infty)| \rightarrow 0 \quad (\mathrm{Theorem\ 5.1})$" "\n\n"
            r"$\Longrightarrow\; |T(X_j) - T(X_\infty)| \rightarrow 0$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#eafaf1", ec=C_OK, lw=1.2, alpha=1.0),
            zorder=15)

    # 11. Transversality Card (lower right, explicitly in front of all plot curves with zorder=15)
    ax.text(0.975, 0.42,
            r"$\mathbf{Transversality\ (Axiom\ S2)}:$" "\n"
            r"$0 < P_{\min} \leq P(X) \leq P_{\max} < \infty$" "\n\n"
            r"$\frac{dV}{dU} = -\frac{1}{P(X)} \in \left[-\frac{1}{P_{\min}},\, -\frac{1}{P_{\max}}\right] < 0$" "\n\n"
            r"$\frac{dV}{dU} \neq 0 \;\Longrightarrow\; A_j \cap l_0 = \{Y_j\}$" "\n\n"
            r"$|Y_j - X_\infty| \leq (1 + P_{\max})\,|X_j - X_\infty| \rightarrow 0$",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fef9e7", ec=C_CONE, lw=1.4, alpha=1.0),
            zorder=15)

    # ======================================================================
    # 12. ACCENTUATED INSET ZOOM of the immediate neighborhood of X_infinity
    # ======================================================================
    # Large inset frame in the open upper-right quadrant: width=0.44, height=0.51
    axins = ax.inset_axes([0.53, 0.44, 0.44, 0.51])
    axins.set_facecolor("#ffffff")

    # Tight data limits: width Delta U = 0.120, height Delta V = 0.108
    # This delivers ~6.2x true visual magnification compared to the main figure!
    u_zmin, u_zmax = 2.095, 2.215
    v_zmin, v_zmax = 1.182, 1.290
    axins.set_xlim(u_zmin, u_zmax)
    axins.set_ylim(v_zmin, v_zmax)

    # Inset background grid of adiabats (terminated slightly below l_0 at 1.196)
    v_zgrid = np.linspace(1.196, v_zmax + 0.015, 180)

    # Inset horizontal line l_0
    axins.axhline(V0, color=C_L0, lw=2.4, zorder=3)
    axins.text(u_zmax - 0.003, V0 - 0.003, r"$l_0$", color=C_L0,
               fontsize=11.0, ha="right", va="top", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.90), zorder=9)

    # Base adiabat in zoom
    axins.plot(adiabat_U(v_zgrid, K0), v_zgrid, color=C_ADIA_0, lw=2.8, zorder=4)

    # Sequence adiabats and lines in zoom for j = 3, 4
    for j in [3, 4]:
        Xj = seq_X[j-1]
        Vj = Xj[1]
        Kj = adiabat_const(*Xj)

        # Horizontal line l_j
        axins.axhline(Vj, color=C_LJ, ls=":", lw=1.1, zorder=3)
        axins.text(u_zmin + 0.003, Vj + 0.0015, rf"$l_{j}$",
                   color=C_LJ, fontsize=9.5, ha="left", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=9)

        # Adiabat curve
        axins.plot(adiabat_U(v_zgrid, Kj), v_zgrid, color=C_ADIA_J, lw=1.8, alpha=0.88, zorder=4)

        # Point X_j
        axins.plot(Xj[0], Xj[1], "s", color=C_PTS, ms=8.0, zorder=7)

        # Clean label for X_j
        if j == 3:
            axins.text(Xj[0] + 0.008, Xj[1] + 0.004, rf"$X_3$",
                       color=C_PTS, fontsize=10.5, fontweight="bold", ha="left", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_PTS, lw=0.8, alpha=0.95),
                       zorder=10)
        elif j == 4:
            axins.text(Xj[0] - 0.008, Xj[1] + 0.004, rf"$X_4$",
                       color=C_PTS, fontsize=10.5, fontweight="bold", ha="right", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_PTS, lw=0.8, alpha=0.95),
                       zorder=10)

    # Also draw the tail of adiabat A_2 crossing l_0 at Y_2
    K2 = adiabat_const(*seq_X[1])
    axins.plot(adiabat_U(v_zgrid, K2), v_zgrid, color=C_ADIA_J, lw=1.4, alpha=0.55, ls="--", zorder=3)

    # Plot intersection points Y_2, Y_3, Y_4 on l_0
    for j in [2, 3, 4]:
        Yj = seq_Y[j-1]
        axins.plot(Yj[0], Yj[1], "o", color=C_INTER, ms=7.5, zorder=7)

        # Staggered labels in the clear bottom strip
        y_label_v = 1.187 if j == 3 else 1.189
        axins.text(Yj[0], y_label_v, rf"$Y_{j}$",
                   color=C_INTER, fontsize=10.0, fontweight="bold", ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_INTER, lw=0.8, alpha=0.95),
                   zorder=10)
        axins.plot([Yj[0], Yj[0]], [Yj[1] - 0.002, y_label_v + 0.004], color=C_INTER, lw=0.8, ls=":", zorder=8)

    # Directional arrow from X_2 direction -> X_3 -> X_4 -> X_infinity in zoom
    axins.annotate("", xy=(seq_X[2][0], seq_X[2][1]),
                   xytext=(seq_X[2][0] - 0.015, seq_X[2][1] + 0.013),
                   arrowprops=dict(arrowstyle="->", color=C_PTS, ls="--", lw=1.3,
                                   mutation_scale=12), zorder=6)
    axins.annotate("", xy=(seq_X[3][0], seq_X[3][1]),
                   xytext=(seq_X[2][0], seq_X[2][1]),
                   arrowprops=dict(arrowstyle="->", color=C_PTS, ls="--", lw=1.3,
                                   mutation_scale=12), zorder=6)
    axins.annotate("", xy=(X0[0], X0[1]), xytext=(seq_X[3][0], seq_X[3][1]),
                   arrowprops=dict(arrowstyle="->", color=C_PTS, ls="--", lw=1.3,
                                   mutation_scale=12), zorder=6)

    # Directional arrows Y_2 -> Y_3 -> Y_4 -> X_infinity along l_0 in zoom
    axins.annotate("", xy=(seq_Y[2][0], V0), xytext=(seq_Y[1][0], V0),
                   arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.5,
                                   mutation_scale=11), zorder=6)
    axins.annotate("", xy=(seq_Y[3][0], V0), xytext=(seq_Y[2][0], V0),
                   arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.5,
                                   mutation_scale=11), zorder=6)
    axins.annotate("", xy=(X0[0] - 0.002, V0), xytext=(seq_Y[3][0], V0),
                   arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.5,
                                   mutation_scale=11), zorder=6)

    # Base state X_infinity in zoom
    axins.plot(X0[0], X0[1], "*", color=C_BASE, ms=16, zorder=8)
    axins.text(X0[0] + 0.004, X0[1] + 0.005, r"$X_\infty$",
               color=C_BASE, fontsize=12.0, fontweight="bold", ha="left", va="bottom",
               bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_BASE, lw=0.9, alpha=0.95),
               zorder=10)

    # Highlight microscopic two-leg path for step j=4 inside the zoom
    X4 = seq_X[3]
    Y4 = seq_Y[3]
    K4 = adiabat_const(*X4)
    v_leg1_z = np.linspace(X4[1], Y4[1], 50)
    u_leg1_z = adiabat_U(v_leg1_z, K4)
    axins.plot(u_leg1_z, v_leg1_z, color=C_LEG1, lw=3.2, zorder=5)
    axins.plot([Y4[0], X0[0]], [V0, V0], color=C_LEG2, lw=3.2, zorder=5)

    # Inset styling
    axins.set_title(r"$\mathbf{Zoom\ (\approx 6\times):}\ X_j \to X_\infty \text{ and } Y_j \to X_\infty$",
                    fontsize=11.0, pad=6, color="#2c3e50")
    axins.set_xlabel(r"$U$", fontsize=10.5, labelpad=2)
    axins.set_ylabel(r"$V$", fontsize=10.5, labelpad=2)
    axins.set_xticks([2.10, 2.13, 2.16, 2.19, 2.21])
    axins.set_yticks([1.20, 1.23, 1.26, 1.28])
    axins.tick_params(axis="both", which="major", labelsize=8.8)

    # Border of inset
    for spine in axins.spines.values():
        spine.set_edgecolor(C_BASE)
        spine.set_linewidth(1.4)

    # Indicate the zoom area on the main axes with connection lines (in front of curves)
    ax.indicate_inset_zoom(axins, edgecolor=C_BASE, alpha=0.90, lw=1.5, zorder=9)

    # Main axes labels & limits
    ax.set_xlim(u_min, u_max)
    ax.set_ylim(v_min, v_max)
    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$V$")
    ax.set_title(
        r"Theorem 5.2: Sequence $X_j \to X_\infty$, Adiabats $A_j$, Ball $B$, and Intersections $Y_j = A_j \cap l_0$",
        fontsize=13.0, pad=12
    )
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)

    out_path = os.path.join(script_dir, "theorem52_visualization.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved single plot with accentuated zoom inset:", os.path.basename(out_path))


if __name__ == "__main__":
    make_single_proof_plot_accentuated_zoom()
