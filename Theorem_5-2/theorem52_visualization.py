"""
Single Standalone Visualization with Accentuated Zoom Inset for the PROOF of Theorem 5.2
(Continuity of temperature) in Lieb & Yngvason, Physics Reports 310 (1999) 1-96, p. 71.

Coordinate Convention:
  * Horizontal axis (x): Work coordinate V.
  * Vertical axis (y): Internal energy coordinate U.

Features:
  * Main State Space (V, U):
      - Work coordinate V on horizontal axis, internal energy coordinate U on vertical axis.
      - Base state X_infinity = (V_0, U_0) and base adiabat A_infinity.
      - Ball B = B(X_infinity, r).
      - Sequence of points X_j -> X_infinity (X_1, X_2, X_3, X_4).
      - Vertical lines l_j: V = V_j and base line l_0: V = V_0.
      - Adiabats A_j cutting transversally across the vertical lines.
      - Intersections Y_j = A_j \\cap l_0 sliding along l_0 -> X_infinity.
      - Transversality cone from Axiom S2 (0 < P_min <= P <= P_max).
      - Two-leg decomposition for j=1 (Leg 1 along A_1, Leg 2 along vertical line l_0).
      - Triangle inequality resolution.
  * Accentuated Inset Zoom around X_infinity:
      - High magnification (~6x visual zoom) of the immediate neighborhood around (V_0, U_0).
      - Clearly resolves the tight terminal cluster X_3, X_4 -> X_infinity and Y_2, Y_3, Y_4 -> X_infinity.
      - Distinct vertical lines l_3, l_4, l_0.
      - Shows steep transversal crossing of adiabats A_3, A_4, A_infinity across l_0.
      - Micro-scale two-leg path for step j=4 (along A_4 and vertical line l_0).
      - Connected to main plot via indicate_inset_zoom.
  * Purely mathematical symbols on axes (V, U) and zero English sentences on the figure.
  * All labels equipped with protective white halo boxes in clear open whitespace.

Run directly:  python theorem52_visualization.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, PathPatch
from matplotlib.path import Path

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
C_L0 = "#2c3e50"       # base vertical line l_0                     (dark slate)
C_LJ = "#34495e"       # vertical lines l_j                         (dark slate grey)
C_BALL = "#16a085"     # neighborhood ball B                        (teal)
C_CONE = "#f39c12"     # transversality cone                        (amber)
C_LEG1 = "#2980b9"     # Leg 1 (along adiabat)                      (blue)
C_LEG2 = "#8e44ad"     # Leg 2 (along vertical line l_0)            (purple)
C_OK = "#27ae60"       # resolution / conclusion                    (green)
C_BAD = "#c0392b"      # forbidden direction                        (red)

script_dir = os.path.dirname(os.path.abspath(__file__))


def add_curly_brace(ax, x1, x2, y, depth=0.012, color="blue", lw=1.5, zorder=20, transform=None):
    """Draws a smooth curly underbrace pointing downwards between x1 and x2 at level y."""
    mid = 0.5 * (x1 + x2)
    w = x2 - x1
    d = depth
    verts = [
        (x1, y),
        (x1, y - 0.4 * d), (x1 + 0.15 * w, y - 0.5 * d), (x1 + 0.25 * w, y - 0.5 * d),
        (mid - 0.15 * w, y - 0.5 * d), (mid - 0.08 * w, y - 0.9 * d), (mid, y - d),
        (mid + 0.08 * w, y - 0.9 * d), (mid + 0.15 * w, y - 0.5 * d), (x2 - 0.25 * w, y - 0.5 * d),
        (x2 - 0.15 * w, y - 0.5 * d), (x2, y - 0.4 * d), (x2, y)
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4
    ]
    patch = PathPatch(Path(verts, codes), facecolor="none", edgecolor=color,
                      lw=lw, zorder=zorder, transform=transform, clip_on=False)
    ax.add_patch(patch)
    return patch


def make_single_proof_plot_accentuated_zoom():
    fig, ax = plt.subplots(figsize=(13.2, 9.4))

    # Equal aspect ratio ensures ball B is a true geometric circle in (V, U)
    v_min, v_max = 0.58, 1.95
    u_min, u_max = 1.40, 3.05
    ax.set_aspect("equal", adjustable="box")

    V_grid = np.linspace(v_min, v_max, 450)

    # 1. Background faint family of adiabats
    K0 = adiabat_const(V0, U0)
    for k_val in np.linspace(K0 - 0.75, K0 + 0.75, 13):
        ax.plot(V_grid, adiabat_U(V_grid, k_val), color="#bdc3c7",
                lw=0.6, alpha=0.30, zorder=1)

    # 2. Neighborhood Ball B = B(X_infinity, r)
    phi = np.linspace(0, 2 * np.pi, 250)
    b_v = V0 + BALL_RADIUS * np.cos(phi)
    b_u = U0 + BALL_RADIUS * np.sin(phi)
    ax.fill(b_v, b_u, color=C_BALL, alpha=0.07, zorder=2, lw=0)
    ax.plot(b_v, b_u, color=C_BALL, ls="--", lw=1.5, alpha=0.75, zorder=2)
    # Curved link from ball label to the ball boundary itself
    ax.annotate(r"$B = B(X_\infty, r)$",
                xy=(1.726, 2.04),
                xytext=(1.83, 1.95),
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-0.22",
                                color=C_BALL, lw=1.5, shrinkA=0, shrinkB=0),
                color=C_BALL, fontsize=12.0, fontweight="bold", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=C_BALL, lw=1.0, alpha=0.95),
                zorder=10)

    # 3. Base Vertical Line l_0: V = V_0
    ax.axvline(V0, color=C_L0, lw=2.4, zorder=3)
    ax.text(V0 - 0.025, 2.62,
            r"$l_0 = \{(V_0, U)\}$",
            color=C_L0, fontsize=11.0, ha="right", va="top", fontweight="bold")

    # 4. Base Adiabat A_infinity through X_infinity
    ax.plot(V_grid, adiabat_U(V_grid, K0), color=C_ADIA_0, lw=2.8, zorder=4,
            label=r"$A_\infty = \partial A_{X_\infty}$")

    # 5. Sequence of Points X_j -> X_infinity
    seq_X = generate_sequence(4)
    seq_Y = [intersect_adiabat_with_l0(pt) for pt in seq_X]

    # Draw vertical lines l_j and adiabats A_j on main plot
    for j, (Xj, Yj) in enumerate(zip(seq_X, seq_Y), 1):
        Vj = Xj[0]
        Kj = adiabat_const(*Xj)

        # Vertical line l_j
        ax.axvline(Vj, color=C_LJ, ls=":", lw=1.4, zorder=3)
        ax.text(Vj + 0.012, u_min + 0.03,
                rf"$l_{j}$",
                color=C_LJ, fontsize=11.5, fontweight="bold", ha="left", va="bottom")

        # Adiabat A_j through X_j
        line_w = 2.0 if j == 1 else 1.4
        alp = 0.95 if j == 1 else 0.70
        lbl = r"$A_j = \partial A_{X_j}$" if j == 2 else None
        ax.plot(V_grid, adiabat_U(V_grid, Kj), color=C_ADIA_J, lw=line_w,
                alpha=alp, zorder=4, label=lbl)

        # Plot points X_j and Y_j
        ax.plot(Xj[0], Xj[1], "x", color=C_PTS, ms=8.5, mew=2.2, zorder=7)
        ax.plot(Yj[0], Yj[1], "o", color=C_INTER, ms=7.5, zorder=7)

        # Label points X_j on main plot with clean protective white boxes
        if j == 1:
            ax.annotate(rf"$X_1 = (V_1, U_1)$",
                        xy=(Xj[0], Xj[1]),
                        xytext=(Xj[0] + 0.08, Xj[1] - 0.06),
                        arrowprops=dict(arrowstyle="->", color=C_PTS, lw=1.2),
                        color=C_PTS, fontsize=11, fontweight="bold", ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_PTS, lw=1.0, alpha=0.95),
                        zorder=9)
        elif j == 2:
            ax.annotate(rf"$X_2$",
                        xy=(Xj[0], Xj[1]),
                        xytext=(Xj[0] + 0.05, Xj[1] - 0.04),
                        arrowprops=dict(arrowstyle="->", color=C_PTS, lw=1.0),
                        color=C_PTS, fontsize=10, fontweight="bold", ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_PTS, lw=0.8, alpha=0.95),
                        zorder=9)

    # 6. Intersection points Y_j on l_0 on main plot
    Y1 = seq_Y[0]
    ax.annotate(r"$Y_1 = A_1 \cap l_0$", xy=(Y1[0], Y1[1]),
                xytext=(Y1[0] - 0.14, Y1[1] - 0.12),
                arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.2),
                color=C_INTER, fontsize=10.5, fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_INTER, lw=1.0, alpha=0.95),
                zorder=9)

    # Arrows for Y_j -> X_infinity along vertical line l_0 (moving upward)
    for j in range(len(seq_Y) - 1):
        ax.annotate("", xy=(V0, seq_Y[j+1][1]), xytext=(V0, seq_Y[j][1]),
                    arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.4,
                                    mutation_scale=10), zorder=6)
    ax.annotate("", xy=(V0, X0[1] - 0.015), xytext=(V0, seq_Y[-1][1]),
                arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.4,
                                mutation_scale=10), zorder=6)

    # 7. Base / Limit State X_infinity = (V_0, U_0) on main plot
    ax.plot(X0[0], X0[1], "*", color=C_BASE, ms=16, zorder=8)
    ax.annotate(r"$X_\infty = (V_0, U_0)$",
                xy=(X0[0], X0[1]),
                xytext=(X0[0] - 0.06, X0[1] + 0.08),
                arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.2),
                color=C_BASE, fontsize=10.5, fontweight="bold", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C_BASE, lw=1.0, alpha=0.95),
                zorder=9)

    # 8. Highlight the Two-Leg Path for j = 1 on main plot
    X1 = seq_X[0]
    K1 = adiabat_const(*X1)
    v_path = np.linspace(X1[0], Y1[0], 100)
    u_path = adiabat_U(v_path, K1)
    ax.plot(v_path, u_path, color=C_LEG1, lw=3.6, zorder=5)

    # Leg 1 arrow (moving along curve towards Y_1)
    mid_idx = 50
    ax.annotate("", xy=(v_path[mid_idx + 8], u_path[mid_idx + 8]),
                xytext=(v_path[mid_idx - 8], u_path[mid_idx - 8]),
                arrowprops=dict(arrowstyle="->", color=C_LEG1, lw=2.6,
                                mutation_scale=17), zorder=6)

    # Leg 2 arrow along vertical line l_0 (moving upward to X_0)
    ax.annotate("", xy=(V0, X0[1] - 0.02), xytext=(V0, Y1[1] + 0.02),
                arrowprops=dict(arrowstyle="->", color=C_LEG2, lw=3.0,
                                mutation_scale=17), zorder=6)

    # Leg 1 indication with curved link below the curve
    leg1_target_idx = 45
    ax.annotate(r"$\mathbf{Leg\ 1}$",
                xy=(v_path[leg1_target_idx], u_path[leg1_target_idx]),
                xytext=(v_path[leg1_target_idx] - 0.04, u_path[leg1_target_idx] - 0.10),
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.18",
                                color=C_LEG1, lw=1.6, shrinkA=0, shrinkB=0),
                color=C_LEG1, fontsize=11.0, fontweight="bold", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="#ebf5fb", ec=C_LEG1, lw=1.2, alpha=0.95),
                zorder=15)

    # Leg 2 indication with curved link
    leg2_target_u = 2.14
    ax.annotate(r"$\mathbf{Leg\ 2}$",
                xy=(V0, leg2_target_u),
                xytext=(V0 - 0.12, leg2_target_u),
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.20",
                                color=C_LEG2, lw=1.6, shrinkA=0, shrinkB=0),
                color=C_LEG2, fontsize=11.0, fontweight="bold", ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="#f5eef8", ec=C_LEG2, lw=1.2, alpha=0.95),
                zorder=15)

    # Direct vector / displacement X_1 -> X_infinity
    ax.plot([X1[0], X0[0]], [X1[1], X0[1]], color=C_PTS, lw=3.2, ls="-", zorder=5)
    ax.annotate("", xy=(X0[0] + 0.017, X0[1] - 0.020), xytext=(X1[0] - 0.017, X1[1] + 0.020),
                arrowprops=dict(arrowstyle="->", color=C_PTS, lw=2.8,
                                mutation_scale=18), zorder=6)

    # Prominent label for X_j - X_infinity on main plot
    ax.annotate(r"$\mathbf{X_j - X_\infty}$",
                xy=(0.5 * (X1[0] + X0[0]), 0.5 * (X1[1] + X0[1])),
                xytext=(0.5 * (X1[0] + X0[0]) + 0.08, 0.5 * (X1[1] + X0[1]) + 0.06),
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-0.18",
                                color=C_PTS, lw=1.6, shrinkA=0, shrinkB=0),
                color=C_PTS, fontsize=11.5, fontweight="bold", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.28", fc="#fdf2e9", ec=C_PTS, lw=1.4, alpha=0.95),
                zorder=15)

    # 9. Transversality Cone on main plot
    bounds = get_ball_bounds()
    p_min, p_max = bounds["P_min"], bounds["P_max"]
    s_min, s_max = -p_max, -p_min  # dU/dV range

    poly_pts = [
        [X1[0], X1[1]],
        [V0 - 0.06, X1[1] + s_min * (V0 - 0.06 - X1[0])],
        [V0 - 0.06, X1[1] + s_max * (V0 - 0.06 - X1[0])],
    ]
    ax.add_patch(Polygon(poly_pts, color=C_CONE, alpha=0.18, zorder=2, lw=0))

    v_cone = np.linspace(V0 - 0.06, X1[0] + 0.06, 100)
    ax.plot(v_cone, X1[1] + s_min * (v_cone - X1[0]), color=C_CONE, ls="--", lw=1.4, zorder=3)
    ax.plot(v_cone, X1[1] + s_max * (v_cone - X1[0]), color=C_CONE, ls="-.", lw=1.4, zorder=3)

    # Direct vector entry in legend
    ax.plot([], [], color=C_PTS, lw=2.5, ls="-",
            label=r"Direct displacement $X_j - X_\infty$")

    # Forbidden vertical slope (transversality forbids vertical tangency dU/dV = -infty)
    ax.plot([X1[0], X1[0]], [X1[1] - 0.25, X1[1] + 0.25], color=C_BAD, ls=":", lw=2.2, zorder=4,
            label=r"Forbidden slope $\frac{dU}{dV} = -\infty$ (vertical)")

    # 10. Transversality Card (top left)
    ax.text(0.035, 0.960,
            r"$\mathbf{Transversality\ (Axiom\ S2)}:$" "\n"
            r"$0 < P_{\min} \leq P(X) \leq P_{\max} < \infty$" "\n\n"
            r"$\frac{dU}{dV} = -P(X) \in [-P_{\max},\, -P_{\min}] < 0$" "\n\n"
            r"$\frac{dU}{dV} \neq -\infty \;\Longrightarrow\; A_j \cap l_0 = \{Y_j\}$" "\n\n"
            r"$|Y_j - X_\infty| \leq (1 + P_{\max})\,|X_j - X_\infty| \rightarrow 0$",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fef9e7", ec=C_CONE, lw=1.4, alpha=0.98),
            zorder=15)

    # 11. Summary Triangle Inequality Card (bottom left)
    card = FancyBboxPatch((0.030, 0.032), 0.415, 0.268,
                          boxstyle="round,pad=0.012,rounding_size=0.018",
                          fc="#eafaf1", ec=C_OK, lw=1.2,
                          transform=ax.transAxes, zorder=15)
    ax.add_patch(card)

    # Title
    ax.text(0.042, 0.280, r"$\mathbf{Triangle\ Inequality}:$", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color="#1e8449", zorder=16, va="top")

    # Equation formatted cleanly
    ax.text(0.042, 0.230,
            r"$|T(X_j) - T(X_\infty)| \;\leq\; |T(X_j) - T(Y_j)| \;+\; |T(Y_j) - T(X_\infty)|$",
            transform=ax.transAxes, fontsize=8.2, color="#2c3e50", zorder=16, va="center")

    # Underbraces with matching colors (exact spans from measurement)
    add_curly_brace(ax, 0.0420, 0.1181, 0.215, depth=0.011, color=C_PTS, lw=1.5, zorder=17, transform=ax.transAxes)
    ax.text(0.0801, 0.201, r"$\mathbf{X_j - X_\infty}$", transform=ax.transAxes,
            fontsize=7.8, fontweight="bold", color=C_PTS, zorder=17, ha="center", va="top")

    add_curly_brace(ax, 0.1324, 0.2034, 0.215, depth=0.011, color=C_LEG1, lw=1.5, zorder=17, transform=ax.transAxes)
    ax.text(0.1679, 0.201, r"$\mathbf{Leg\ 1}$", transform=ax.transAxes,
            fontsize=8.0, fontweight="bold", color=C_LEG1, zorder=17, ha="center", va="top")

    add_curly_brace(ax, 0.2177, 0.2930, 0.215, depth=0.011, color=C_LEG2, lw=1.5, zorder=17, transform=ax.transAxes)
    ax.text(0.2554, 0.201, r"$\mathbf{Leg\ 2}$", transform=ax.transAxes,
            fontsize=8.0, fontweight="bold", color=C_LEG2, zorder=17, ha="center", va="top")

    # Subordinate lines (single call per line prevents any horizontal collisions)
    ax.text(0.042, 0.146,
            r"$\bullet\ \mathbf{Leg\ 1}:\; |T(X_j) - T(Y_j)| \leq c\,|X_j - Y_j| \rightarrow 0 \quad (\mathrm{Lemma\ 5.1})$",
            transform=ax.transAxes, fontsize=7.8, color="#2c3e50", zorder=16, va="center")

    ax.text(0.042, 0.104,
            r"$\bullet\ \mathbf{Leg\ 2}:\; |T(Y_j) - T(X_\infty)| \rightarrow 0 \quad (\mathrm{Theorem\ 5.1})$",
            transform=ax.transAxes, fontsize=7.8, color="#2c3e50", zorder=16, va="center")

    # Conclusion
    ax.text(0.042, 0.062,
            r"$\Longrightarrow\; |T(X_j) - T(X_\infty)| \rightarrow 0 \quad (\mathrm{Continuity\ on\ }\Gamma)$",
            transform=ax.transAxes, fontsize=8.6, fontweight="bold", color="#145a32", zorder=16, va="center")

    # ======================================================================
    # 12. ACCENTUATED INSET ZOOM of immediate neighborhood of X_infinity
    # ======================================================================
    axins = ax.inset_axes([0.53, 0.44, 0.44, 0.51])
    axins.set_facecolor("#ffffff")

    # Zoom data limits around (V_0, U_0) = (1.20, 2.20)
    v_zmin, v_zmax = 1.182, 1.290
    u_zmin, u_zmax = 2.095, 2.215
    axins.set_xlim(v_zmin, v_zmax)
    axins.set_ylim(u_zmin, u_zmax)

    v_zgrid = np.linspace(v_zmin - 0.015, v_zmax + 0.015, 180)

    # Inset vertical line l_0
    axins.axvline(V0, color=C_L0, lw=2.4, zorder=3)
    axins.text(V0 + 0.0015, u_zmax - 0.005, r"$l_0$", color=C_LJ,
               fontsize=10.5, ha="left", va="top", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=9)

    # Base adiabat in zoom
    axins.plot(v_zgrid, adiabat_U(v_zgrid, K0), color=C_ADIA_0, lw=2.8, zorder=4)

    # Sequence adiabats and lines in zoom for j = 3, 4
    for j in [3, 4]:
        Xj = seq_X[j-1]
        Vj = Xj[0]
        Kj = adiabat_const(*Xj)

        # Vertical line l_j
        axins.axvline(Vj, color=C_LJ, ls=":", lw=1.4, zorder=3)
        axins.text(Vj + 0.0015, u_zmin + 0.003, rf"$l_{j}$",
                   color=C_LJ, fontsize=10.5, ha="left", va="bottom", fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=9)

        # Adiabat curve
        axins.plot(v_zgrid, adiabat_U(v_zgrid, Kj), color=C_ADIA_J, lw=1.8, alpha=0.88, zorder=4)

        # Point X_j
        axins.plot(Xj[0], Xj[1], "x", color=C_PTS, ms=9.0, mew=2.4, zorder=7)

        # Clean label for X_j
        if j == 3:
            axins.text(Xj[0] + 0.004, Xj[1] - 0.006, rf"$X_3$",
                       color=C_PTS, fontsize=10.5, fontweight="bold", ha="left", va="top",
                       bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_PTS, lw=0.8, alpha=0.95),
                       zorder=10)
        elif j == 4:
            axins.text(Xj[0] + 0.004, Xj[1] - 0.006, rf"$X_4$",
                       color=C_PTS, fontsize=10.5, fontweight="bold", ha="left", va="top",
                       bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_PTS, lw=0.8, alpha=0.95),
                       zorder=10)

    # Tail of adiabat A_2 crossing l_0 at Y_2
    K2 = adiabat_const(*seq_X[1])
    axins.plot(v_zgrid, adiabat_U(v_zgrid, K2), color=C_ADIA_J, lw=1.4, alpha=0.55, ls="--", zorder=3)

    # Plot intersection points Y_2, Y_3, Y_4 on l_0
    for j in [2, 3, 4]:
        Yj = seq_Y[j-1]
        axins.plot(Yj[0], Yj[1], "o", color=C_INTER, ms=7.5, zorder=7)

        y_label_v = 1.189 if j == 3 else 1.187
        axins.text(y_label_v, Yj[1], rf"$Y_{j}$",
                   color=C_INTER, fontsize=10.0, fontweight="bold", ha="right", va="center",
                   bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_INTER, lw=0.8, alpha=0.95),
                   zorder=10)
        axins.plot([y_label_v + 0.004, Yj[0] - 0.002], [Yj[1], Yj[1]], color=C_INTER, lw=0.8, ls=":", zorder=8)

    # Directional arrow from X_2 -> X_3 -> X_4 -> X_infinity in zoom
    axins.annotate("", xy=(seq_X[2][0], seq_X[2][1]),
                   xytext=(seq_X[2][0] + 0.015, seq_X[2][1] - 0.015),
                   arrowprops=dict(arrowstyle="->", color=C_PTS, ls="--", lw=1.3,
                                   mutation_scale=12), zorder=6)
    axins.annotate("", xy=(seq_X[3][0], seq_X[3][1]),
                   xytext=(seq_X[2][0], seq_X[2][1]),
                   arrowprops=dict(arrowstyle="->", color=C_PTS, ls="--", lw=1.3,
                                   mutation_scale=12), zorder=6)
    axins.annotate("", xy=(X0[0], X0[1]), xytext=(seq_X[3][0], seq_X[3][1]),
                   arrowprops=dict(arrowstyle="->", color=C_PTS, ls="--", lw=2.0,
                                   mutation_scale=13), zorder=6)

    # Directional arrows Y_2 -> Y_3 -> Y_4 -> X_infinity along vertical line l_0 in zoom (upwards)
    axins.annotate("", xy=(V0, seq_Y[2][1]), xytext=(V0, seq_Y[1][1]),
                   arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.5,
                                   mutation_scale=11), zorder=6)
    axins.annotate("", xy=(V0, seq_Y[3][1]), xytext=(V0, seq_Y[2][1]),
                   arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.5,
                                   mutation_scale=11), zorder=6)
    axins.annotate("", xy=(V0, X0[1] - 0.002), xytext=(V0, seq_Y[3][1]),
                   arrowprops=dict(arrowstyle="->", color=C_INTER, lw=1.5,
                                   mutation_scale=11), zorder=6)

    # Base state X_infinity in zoom
    axins.plot(X0[0], X0[1], "*", color=C_BASE, ms=16, zorder=8)
    axins.text(X0[0] - 0.005, X0[1] + 0.005, r"$X_\infty$",
               color=C_BASE, fontsize=12.0, fontweight="bold", ha="right", va="bottom",
               bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_BASE, lw=0.9, alpha=0.95),
               zorder=10)

    # Highlight microscopic triangle for step j=4 inside the zoom
    X4 = seq_X[3]
    Y4 = seq_Y[3]
    K4 = adiabat_const(*X4)
    v_leg1_z = np.linspace(X4[0], Y4[0], 50)
    u_leg1_z = adiabat_U(v_leg1_z, K4)
    axins.plot(v_leg1_z, u_leg1_z, color=C_LEG1, lw=3.2, zorder=5)
    axins.plot([V0, V0], [Y4[1], X0[1]], color=C_LEG2, lw=3.2, zorder=5)
    # Direct path X_4 -> X_infinity
    axins.plot([X4[0], X0[0]], [X4[1], X0[1]], color=C_PTS, lw=3.2, zorder=5)
    axins.annotate("", xy=(X0[0] + 0.0025, X0[1] - 0.003), xytext=(X4[0] - 0.0025, X4[1] + 0.003),
                   arrowprops=dict(arrowstyle="->", color=C_PTS, lw=2.8,
                                   mutation_scale=16), zorder=6)
    axins.text(1.228, 2.190, r"$\mathbf{X_j - X_\infty}$",
               color=C_PTS, fontsize=9.5, fontweight="bold", ha="left", va="center",
               bbox=dict(boxstyle="round,pad=0.20", fc="#fdf2e9", ec=C_PTS, lw=1.0, alpha=0.95),
               zorder=15)

    axins.set_xticks([])
    axins.set_yticks([])

    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.4)

    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.90, lw=1.5, zorder=9)

    ax.set_xlim(v_min, v_max)
    ax.set_ylim(u_min, u_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$U$")
    ax.set_title(
        r"Theorem 5.2: Sequence $X_j \to X_\infty$, Adiabats $A_j$, Ball $B$, and Intersections $Y_j = A_j \cap l_0$",
        fontsize=13.5, pad=48
    )
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=4,
        fontsize=11.0, frameon=True, framealpha=0.95, handlelength=2.5
    )

    out_path = os.path.join(script_dir, "theorem52_visualization.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved single plot with accentuated zoom inset:", os.path.basename(out_path))


if __name__ == "__main__":
    make_single_proof_plot_accentuated_zoom()
