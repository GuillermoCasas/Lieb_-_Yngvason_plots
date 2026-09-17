"""
Theorem 5.5 Visualization Module: Isotherms cut adiabats.
Elliott H. Lieb & Jakob Yngvason, Phys. Rept. 310 (1999) 1–96, Section 5.2, pp. 73–75.

This script produces two publication-quality figures:
1. theorem55_scenario1.png: Scenario (1) Interior Transversal Crossing (T_min < T_0 < T_max).
   Stacked layout with two mutually exhaustive subplots:
   - Panel 1: Case (a) Coordinate Sections Exist: V_{X_0}, V_{X_1} in rho(A_X)
   - Panel 2: Case (b) Coordinate Line Misses Adiabat: V_{X_0} notin rho(A_X)
   Sharing the top horizontal legend and bottom Step 2 proof banner.
2. theorem55_scenario2.png: Scenario (2) Boundary Temperature Approximation (T_0 = T_max).
3. theorem55_visualization.png: Primary reference figure (synced with Scenario 1).

Coordinate Convention:
- Horizontal Axis (x): Work coordinate / volume V.
- Vertical Axis (y): Internal energy coordinate U.
- Isotherms: red and dashed, slope dU/dV|_T = a/V^2 > 0.
- Adiabats: full and black, slope dU/dV|_S = -P < 0.
"""

# Deliberate simplification: With a hard ceiling the per-column temperature supremum varies with V, so axiom T5 holds here only qualitatively; the figure illustrates the objects of the proof, like the paper's own schematic Fig. 4, not a fully axiom-faithful model.

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.legend_handler import HandlerBase
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.linewidth": 1.4,
    "axes.edgecolor": "#2c3e50",
    "axes.titlesize": 16,
    "axes.labelsize": 15,
})

cmap_w2b = LinearSegmentedColormap.from_list("w2b", ["#ffffff", "#5dade2", "#1b4f72"])
cmap_w2r = LinearSegmentedColormap.from_list("w2r", ["#ffffff", "#f5b7b1", "#922b21"])


class GradientHandler(HandlerBase):
    """Custom legend handler rendering a smooth horizontal gradient box with border using pure vector slices."""
    def __init__(self, cmap):
        super().__init__()
        self.cmap = cmap

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        n_slices = 50
        slice_w = width / n_slices
        for i in range(n_slices):
            color = self.cmap(i / (n_slices - 1))
            rect = plt.Rectangle((xdescent + i * slice_w, ydescent),
                                 slice_w * 1.02, height,
                                 facecolor=color, edgecolor="none",
                                 transform=trans)
            artists.append(rect)
        border = plt.Rectangle((xdescent, ydescent), width, height,
                               facecolor="none", edgecolor=orig_handle.get_edgecolor(),
                               lw=orig_handle.get_linewidth(), transform=trans)
        artists.append(border)
        return artists


# --------------------------------------------------------------------------
# Thermodynamic Parameters & Functions in (V, U)
# --------------------------------------------------------------------------
CV = 1.5
R_GAS = 1.0
A_PARAM = 2.0
B_PARAM = 0.5
T0_SCENARIO1 = 2.00
T_MAX_SCENARIO2 = 2.80
T_PRIME_SCENARIO2 = 2.45


def U_isotherm(V, T):
    return CV * T - A_PARAM / V


def U_adiabat(V, S_const):
    return np.exp((S_const - R_GAS * np.log(V - B_PARAM)) / CV) - A_PARAM / V


def entropy(U, V):
    return CV * np.log(U + A_PARAM / V) + R_GAS * np.log(V - B_PARAM)


def temperature(U, V):
    return (U + A_PARAM / V) / CV


# --------------------------------------------------------------------------
# Ceiling Boundary H(V) and State Space Boundary Functions
# --------------------------------------------------------------------------
H1 = 0.70

V0a = 1.05   # Case (a) anchor volume (in rho(A_X))
V0b = 0.82   # Case (b) anchor volume (trapped column, not in rho(A_X))
V1 = 2.50    # Upper anchor volume on isotherm

U0a = float(U_isotherm(V0a, T0_SCENARIO1))
S0a = float(entropy(U0a, V0a))

U0b = float(U_isotherm(V0b, T0_SCENARIO1))
S0b = float(entropy(U0b, V0b))

U1 = float(U_isotherm(V1, T0_SCENARIO1))
S1 = float(entropy(U1, V1))

# Intermediate adiabat entropy
alpha_weight = 0.35
S_MID = alpha_weight * S0a + (1.0 - alpha_weight) * S1

# Exit volume for dA_X: V_B = 0.98 solves Sigma(V_B) = S_MID
V_B = 0.98
u_b = float(U_adiabat(V_B, S_MID))
H0 = u_b - H1 * V_B


def ceiling(V):
    """Affine ceiling H(V) = h0 + h1 * V defining the upper boundary dGamma."""
    V = np.asarray(V)
    return H0 + H1 * V


def ceiling_entropy(V):
    """Entropy along the ceiling Sigma(V) = S(H(V), V). Strictly increasing."""
    return entropy(ceiling(V), V)


def rho_min(S_const):
    """
    Computes the minimum volume V_b of rho(A_X) = (V_b, infty) where the
    adiabat of entropy S_const exits through the ceiling boundary dGamma:
    Sigma(V_b) = S_const.
    """
    def obj(v):
        return ceiling_entropy(v) - S_const
    return float(brentq(obj, B_PARAM + 1e-4, 10.0))


def add_dotted_arrow(ax, p0, p1, color, lw=4.2, mutation_scale=32, head_length=0.82, head_width=0.35,
                     shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9):
    """Draws an arrow with a crisp dotted shaft and solid arrowhead."""
    head = FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
                           mutation_scale=mutation_scale, color=color, lw=0,
                           shrinkA=shrinkA, shrinkB=shrinkB, zorder=zorder)
    ax.add_patch(head)
    inv = ax.transData.inverted()
    p0_d = ax.transData.transform(p0)
    p1_d = ax.transData.transform(p1)
    v_d = p1_d - p0_d
    dist_d = np.hypot(v_d[0], v_d[1])
    if dist_d == 0:
        return head
    u_d = v_d / dist_d
    head_len_pt = head_length * mutation_scale
    start_d = p0_d + u_d * shrinkA
    end_d = p1_d - u_d * (shrinkB + head_len_pt)
    start_data = inv.transform(start_d)
    end_data = inv.transform(end_d)
    ax.plot([start_data[0], end_data[0]], [start_data[1], end_data[1]],
            color=color, lw=lw, linestyle=(0, (0.01, dot_spacing)),
            solid_capstyle="round", dash_capstyle="round", zorder=zorder)
    return head


def add_curved_callout(ax, fig, text, xy_target, xy_box, bbox_style,
                       color="#1a252f", fontsize=13.5, rad=0.18, lw=1.8,
                       preferred_corner=None, shrinkB=8):
    """Adds a curved connector without arrow from a text box to a target point."""
    t = ax.text(xy_box[0], xy_box[1], text, fontsize=fontsize, fontweight="bold",
                color=color, ha="center", va="center", zorder=100, bbox=bbox_style)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb_d = t.get_window_extent(r)
    corners_d = {
        "bl": (bb_d.x0, bb_d.y0),
        "br": (bb_d.x1, bb_d.y0),
        "tl": (bb_d.x0, bb_d.y1),
        "tr": (bb_d.x1, bb_d.y1),
    }
    bb_data = bb_d.transformed(ax.transData.inverted())
    corners_data = {
        "bl": (bb_data.x0, bb_data.y0),
        "br": (bb_data.x1, bb_data.y0),
        "tl": (bb_data.x0, bb_data.y1),
        "tr": (bb_data.x1, bb_data.y1),
    }
    if preferred_corner and preferred_corner in corners_data:
        start_pt = corners_data[preferred_corner]
    else:
        target_d = ax.transData.transform(xy_target)
        dists = {k: (c[0] - target_d[0])**2 + (c[1] - target_d[1])**2 for k, c in corners_d.items()}
        closest_key = min(dists, key=dists.get)
        start_pt = corners_data[closest_key]

    conn = FancyArrowPatch(start_pt, xy_target, arrowstyle="-",
                           connectionstyle=f"arc3,rad={rad}",
                           lw=lw, color=color, shrinkA=0, shrinkB=shrinkB, zorder=30)
    ax.add_patch(conn)
    return t, conn


def add_region_label(ax, fig, x, y, text_type, color, fontsize=33):
    """Draws mathematical region labels like X_-, X_+, or X_+ = emptyset."""
    t_main = ax.text(x, y, r"$\mathcal{X}$", fontsize=fontsize, color=color, ha="center", va="center", zorder=100)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = t_main.get_window_extent(renderer)
    inv = ax.transData.inverted()
    p0 = inv.transform((bb.x0, bb.y0))
    p1 = inv.transform((bb.x1, bb.y1))
    width = p1[0] - p0[0]
    height = p1[1] - p0[1]

    if text_type == "minus":
        x_sub = p1[0] + width * 0.14
        y_sub = y - height * 0.05
        t_sub = ax.text(x_sub, y_sub, r"$-$", fontsize=fontsize * 0.74, fontweight="bold", color=color, ha="center", va="center", zorder=100)
        return t_main, t_sub
    elif text_type == "plus":
        x_sub = p1[0] + width * 0.10
        y_sub = y - height * 0.05
        t_sub = ax.text(x_sub, y_sub, r"$+$", fontsize=fontsize * 0.74, fontweight="bold", color=color, ha="center", va="center", zorder=100)
        return t_main, t_sub
    elif text_type == "plus_empty":
        x_sub = p1[0] + width * 0.10
        y_sub = y - height * 0.05
        t_sub = ax.text(x_sub, y_sub, r"$+$", fontsize=fontsize * 0.74, fontweight="bold", color=color, ha="center", va="center", zorder=100)
        fig.canvas.draw()
        bb_sub = t_sub.get_window_extent(renderer)
        p1_sub = inv.transform((bb_sub.x1, bb_sub.y1))
        x_empty = p1_sub[0] + width * 0.15
        t_empty = ax.text(x_empty, y, r"$ = \emptyset$", fontsize=fontsize * 0.90, color=color, ha="left", va="center", zorder=100)
        return t_main, t_sub, t_empty
    elif text_type == "minus_gamma":
        x_sub = p1[0] + width * 0.14
        y_sub = y - height * 0.05
        t_sub = ax.text(x_sub, y_sub, r"$-$", fontsize=fontsize * 0.74, fontweight="bold", color=color, ha="center", va="center", zorder=100)
        fig.canvas.draw()
        bb_sub = t_sub.get_window_extent(renderer)
        p1_sub = inv.transform((bb_sub.x1, bb_sub.y1))
        x_gamma = p1_sub[0] + width * 0.15
        t_gamma = ax.text(x_gamma, y, r"$ = \Gamma \setminus I_{T_{\mathrm{max}}}$", fontsize=fontsize * 0.85, color=color, ha="left", va="center", zorder=100)
        return t_main, t_sub, t_gamma


# =============================================================================
# SCENARIO 1: Stacked Layout (Case (a) + Case (b))
# =============================================================================
def generate_scenario1(out_path):
    fig = plt.figure(figsize=(18.5, 24.5), dpi=300)
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.24, 5.0, 5.0, 1.80],
                           top=0.925, bottom=0.032, left=0.08, right=0.96, hspace=0.26)

    ax_legend = fig.add_subplot(gs[0])
    ax_legend.axis("off")
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax_banner = fig.add_subplot(gs[3])
    ax_banner.axis("off")

    T0 = T0_SCENARIO1
    v_min, v_max = 0.55, 3.05
    u_min, u_max = 0.20, 4.40

    v_grid = np.linspace(v_min, v_max, 600)
    v_grid_2d = np.linspace(v_min, v_max, 350)
    u_grid_2d = np.linspace(u_min, u_max, 350)
    V_2D, U_2D = np.meshgrid(v_grid_2d, u_grid_2d)
    T_2D = temperature(U_2D, V_2D)
    norm = TwoSlopeNorm(vmin=0.2, vcenter=T0, vmax=4.5)

    # Cutting point X'
    V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T0)) / R_GAS))
    U_PRIME = float(U_isotherm(V_PRIME, T0))

    # Curve arrays
    u_iso_vals = U_isotherm(v_grid, T0)
    u_adia0a_vals = U_adiabat(v_grid, S0a)
    u_adia0b_vals = U_adiabat(v_grid, S0b)
    u_adiaX_vals = U_adiabat(v_grid, S_MID)
    u_adia1_vals = U_adiabat(v_grid, S1)

    C_ISO = "#c0392b"
    C_ADIA = "#000000"
    C_PT_ISO = "#b94a00"
    C_PT_CUT = "#c0392b"
    C_PT_GT = "#d68910"
    C_PT_LT = "#2980b9"
    C_GAMMA = "#1a252f"

    box_curve_adia = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")
    box_curve_iso = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")
    box_white = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#95a5a6", alpha=1.0, lw=1.5)
    box_cut = dict(boxstyle="round,pad=0.24", facecolor="#ffffff", edgecolor=C_PT_CUT, alpha=1.0, lw=2.2)
    box_gt = dict(boxstyle="round,pad=0.22", facecolor="#fefbf5", edgecolor="#d4ac0d", alpha=1.0, lw=1.6)
    box_lt = dict(boxstyle="round,pad=0.22", facecolor="#f0f7fb", edgecolor="#2980b9", alpha=1.0, lw=1.6)
    box_prime0 = dict(boxstyle="round,pad=0.22", facecolor="#fdfefe", edgecolor="#c0392b", alpha=1.0, lw=1.6)
    box_prime1 = dict(boxstyle="round,pad=0.22", facecolor="#fdfefe", edgecolor="#c0392b", alpha=1.0, lw=1.6)

    # -------------------------------------------------------------------------
    # SHARED TOP LEGEND
    # -------------------------------------------------------------------------
    p_blue_s1 = Patch(edgecolor="#1b4f72", lw=1.2)
    p_red_s1 = Patch(edgecolor="#922b21", lw=1.2)
    legend_elements = [
        Line2D([0], [0], color=C_ADIA, lw=3.4, ls="-"),
        Line2D([0], [0], color=C_ISO, lw=3.4, ls=(0, (4.5, 2.5))),
        Line2D([0], [0], color=C_GAMMA, lw=2.5, ls="--"),
        p_blue_s1,
        p_red_s1,
    ]
    legend_labels = [
        r"$\mathbf{Adiabats}$",
        r"$\mathbf{Isotherms}$",
        r"$\mathbf{\partial\Gamma\ (Boundary)}$",
        r"$\mathbf{\mathcal{X}_- \ (T < T_0)}$",
        r"$\mathbf{\mathcal{X}_+ \ (T > T_0)}$",
    ]
    leg = ax_legend.legend(handles=legend_elements, labels=legend_labels,
                          handler_map={p_blue_s1: GradientHandler(cmap_w2b), p_red_s1: GradientHandler(cmap_w2r)},
                          loc="center", ncol=5, fontsize=16.0,
                          handlelength=3.0, handleheight=1.25, columnspacing=2.6,
                          framealpha=0.96, edgecolor="#2c3e50", fancybox=True, borderpad=0.7)
    leg.set_zorder(100)

    def draw_exterior(ax):
        v_c = np.linspace(v_min, v_max, 250)
        h_c = ceiling(v_c)
        ax.plot(v_c, h_c, color=C_GAMMA, lw=2.6, ls="--", zorder=8)
        verts = [(v_min, u_max), (v_min, ceiling(v_min))] + list(zip(v_c, h_c)) + [(v_max, u_max)]
        poly = Polygon(verts, closed=True, facecolor="#eaecee", edgecolor="none", hatch="//", alpha=0.55, zorder=3)
        ax.add_patch(poly)

    # =========================================================================
    # PANEL 1: CASE (a) — Coordinate Sections Exist: V_X0, V_X1 in rho(A_X)
    # =========================================================================
    ax1.set_xlim(v_min, v_max)
    ax1.set_ylim(u_min, u_max)

    T_2D_masked1 = np.copy(T_2D)
    T_2D_masked1[U_2D > ceiling(V_2D)] = np.nan
    ax1.contourf(V_2D, U_2D, T_2D_masked1, levels=80, cmap="coolwarm", norm=norm, alpha=0.22, zorder=1)
    draw_exterior(ax1)
    ax1.text(0.60, 4.15, r"$\mathbf{\partial\Gamma\ (boundary\ of\ state\ space\ —\ not\ states)}$",
             fontsize=12.5, fontweight="bold", color=C_GAMMA, ha="left", va="center", zorder=10)

    U_GT = float(U_adiabat(V0a, S_MID))
    U_LT = float(U_adiabat(V1, S_MID))

    mask_iso1 = (u_iso_vals >= u_min) & (u_iso_vals <= ceiling(v_grid)) & (u_iso_vals <= u_max)
    ax1.plot(v_grid[mask_iso1], u_iso_vals[mask_iso1], color=C_ISO, lw=3.6, ls=(0, (6, 3.5)), zorder=7)

    mask0a = (u_adia0a_vals >= u_min) & (u_adia0a_vals <= ceiling(v_grid)) & (u_adia0a_vals <= u_max)
    ax1.plot(v_grid[mask0a], u_adia0a_vals[mask0a], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    mask1a = (u_adia1_vals >= u_min) & (u_adia1_vals <= ceiling(v_grid)) & (u_adia1_vals <= u_max)
    ax1.plot(v_grid[mask1a], u_adia1_vals[mask1a], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    maskXa = (u_adiaX_vals >= u_min) & (u_adiaX_vals <= ceiling(v_grid)) & (u_adiaX_vals <= u_max)
    ax1.plot(v_grid[maskXa], u_adiaX_vals[maskXa], color=C_ADIA, lw=3.8, ls="-", zorder=6)

    # Points in Panel 1: (V, U)
    ax1.scatter([V0a], [U0a], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax1.scatter([V1], [U1], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax1.scatter([V0a], [U_GT], color=C_PT_GT, s=210, zorder=15, edgecolors="#6e4402", lw=2.2)
    ax1.scatter([V1], [U_LT], color=C_PT_LT, s=210, zorder=15, edgecolors="#104266", lw=2.2)
    ax1.scatter([V_PRIME], [U_PRIME], color=C_PT_CUT, s=320, zorder=16, marker="o", edgecolors="black", lw=3.0)

    # Region Labels in Panel 1
    add_region_label(ax1, fig, 1.70, 0.42, "minus", "#1b4f72", fontsize=33)
    add_region_label(ax1, fig, 1.45, 3.25, "plus", "#922b21", fontsize=33)

    # Proof Arrows: Vertical Heating and Cooling Slices at constant V
    add_dotted_arrow(ax1, (V0a, U0a), (V0a, U_GT), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)
    box_vol0 = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#c0392b", lw=1.5, alpha=1.0)
    ax1.text(V0a + 0.10, 0.5 * (U0a + U_GT) + 0.10, r"$\mathbf{V(X_>) = V_{X_0} \in \rho(A_X)}$" "\n" r"$\mathbf{(Heating\ Slice)}$",
             fontsize=13.5, fontweight="bold", color="#922b21", ha="left", va="center", zorder=100, bbox=box_vol0)

    add_dotted_arrow(ax1, (V1, U1), (V1, U_LT), color="#2471a3", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)
    box_vol1 = dict(boxstyle="round,pad=0.18", facecolor="#f0f7fb", edgecolor="#2471a3", lw=1.5, alpha=1.0)
    ax1.text(V1 - 0.28, 0.5 * (U_LT + U1), r"$\mathbf{V(X_<) = V_{X_1} \in \rho(A_X)}$" "\n" r"$\mathbf{(Cooling\ Slice)}$",
             fontsize=13.5, fontweight="bold", color="#154360", ha="right", va="center", zorder=100, bbox=box_vol1)

    # Projections on Panel 1: dotted lines to axes
    ax1.plot([V0a, V0a], [u_min, U0a], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax1.plot([v_min, V0a], [U0a, U0a], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax1.plot([v_min, V0a], [U_GT, U_GT], color="#784212", ls=":", lw=1.6, zorder=4)
    ax1.plot([V1, V1], [u_min, U_LT], color="#1b4f72", ls=":", lw=1.6, zorder=4)
    ax1.plot([v_min, V1], [U_LT, U_LT], color="#1b4f72", ls=":", lw=1.6, zorder=4)
    ax1.plot([v_min, V1], [U1, U1], color="#4e2103", ls=":", lw=1.6, zorder=4)

    ax1.set_xticks([V0a, V1])
    ax1.set_xticklabels([r"$\mathbf{V_{X_0}}$", r"$\mathbf{V_{X_1}}$"], fontsize=14, fontweight="bold")
    ax1.set_yticks([U0a, U_LT, U1, U_GT])
    ax1.set_yticklabels([r"$\mathbf{U(X_0)}$", r"$\mathbf{U(X_<)}$", r"$\mathbf{U(X_1)}$", r"$\mathbf{U(X_>)}$"], fontsize=14, fontweight="bold")

    # Curve Labels for Panel 1
    add_curved_callout(ax1, fig, r"$\mathbf{\partial A_X}$", (2.84, float(U_adiabat(2.84, S_MID))), (2.95, 1.28), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax1, fig, r"$\mathbf{\partial A_{X_0}}$", (2.15, float(U_adiabat(2.15, S0a))), (2.35, 0.40), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=0.12, shrinkB=0)
    add_curved_callout(ax1, fig, r"$\mathbf{\partial A_{X_1}}$", (1.95, float(U_adiabat(1.95, S1))), (2.05, 3.65), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax1, fig, r"$\mathbf{I_{T_0}}$", (2.92, float(U_isotherm(2.92, T0))), (2.96, 2.72), box_curve_iso, color=C_ISO, fontsize=18.0, lw=1.6, rad=-0.12, shrinkB=0)

    # Point Callouts for Panel 1
    add_curved_callout(ax1, fig, r"$\mathbf{X_0}$", (V0a, U0a), (1.18, 0.50), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="tl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X_1}$", (V1, U1), (2.75, 2.45), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X_<}$", (V1, U_LT), (2.75, 0.65), box_lt, color="#1b4f72", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="tl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X_>}$", (V0a, U_GT), (0.95, 3.45), box_gt, color="#784212", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X^\prime \overset{A}{\sim} X}$", (V_PRIME, U_PRIME), (1.68, 1.25), box_cut, color=C_PT_CUT, fontsize=17.5, lw=1.8, rad=-0.14, preferred_corner="tl", shrinkB=10)

    ax1.set_ylabel(r"Internal Energy  $U$", fontsize=18, fontweight="bold", labelpad=10)
    ax1.set_title(r"$\mathbf{Case\ (a):\ Coordinate\ Sections\ Exist\ —\ V_{X_0}, V_{X_1} \in \rho(A_X)}$" "\n"
                  r"$\text{Constant-volume vertical slices on }\partial A_X\text{ produce } X_> \in \mathcal{X}_+ \text{ and } X_< \in \mathcal{X}_-$",
                  fontsize=18.5, pad=12, fontweight="bold", color="#1a252f")

    # =========================================================================
    # PANEL 2: CASE (b) — Coordinate Line Misses Adiabat: V_X0 notin rho(A_X)
    # =========================================================================
    ax2.set_xlim(v_min, v_max)
    ax2.set_ylim(u_min, u_max)

    T_2D_masked2 = np.copy(T_2D)
    T_2D_masked2[U_2D > ceiling(V_2D)] = np.nan
    ax2.contourf(V_2D, U_2D, T_2D_masked2, levels=80, cmap="coolwarm", norm=norm, alpha=0.22, zorder=1)
    draw_exterior(ax2)
    ax2.text(0.60, 4.15, r"$\mathbf{\partial\Gamma\ (boundary\ of\ state\ space\ —\ not\ states)}$",
             fontsize=12.5, fontweight="bold", color=C_GAMMA, ha="left", va="center", zorder=10)

    # Isotherm clipped to Gamma
    mask_iso2 = (u_iso_vals >= u_min) & (u_iso_vals <= ceiling(v_grid)) & (u_iso_vals <= u_max)
    ax2.plot(v_grid[mask_iso2], u_iso_vals[mask_iso2], color=C_ISO, lw=3.6, ls=(0, (6, 3.5)), zorder=7)

    # dA_{X0} on Case (b)
    mask0b = (u_adia0b_vals >= u_min) & (u_adia0b_vals <= ceiling(v_grid)) & (u_adia0b_vals <= u_max)
    ax2.plot(v_grid[mask0b], u_adia0b_vals[mask0b], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    # Genuine hard case: dA_X ends exactly ON ceiling dGamma at V_B = rho_min(S_MID)
    v_grid_adiaX = np.linspace(V_B, v_max, 500)
    u_grid_adiaX = U_adiabat(v_grid_adiaX, S_MID)
    maskX_b = (u_grid_adiaX >= u_min) & (u_grid_adiaX <= ceiling(v_grid_adiaX)) & (u_grid_adiaX <= u_max)
    ax2.plot(v_grid_adiaX[maskX_b], u_grid_adiaX[maskX_b], color=C_ADIA, lw=3.8, ls="-", zorder=6)

    # Open circle marking exit of dA_X through dGamma
    U_EXIT = float(ceiling(V_B))
    ax2.scatter([V_B], [U_EXIT], s=140, facecolors="white", edgecolors=C_ADIA, lw=2.8, zorder=20)
    add_curved_callout(ax2, fig, r"$\mathbf{exit\ of\ \partial A_X\ through\ \partial\Gamma\ (limit,\ not\ a\ state)}$",
                       (V_B, U_EXIT), (1.45, 3.55), box_white, color=C_GAMMA, fontsize=12.5, lw=1.5, rad=0.10, preferred_corner="bl", shrinkB=6)

    # dA_{X1} ends on ceiling at V_B1 = rho_min(S1)
    V_B1 = float(rho_min(S1))
    v_grid_adia1 = np.linspace(V_B1, v_max, 500)
    u_grid_adia1 = U_adiabat(v_grid_adia1, S1)
    mask1_b = (u_grid_adia1 >= u_min) & (u_grid_adia1 <= ceiling(v_grid_adia1)) & (u_grid_adia1 <= u_max)
    ax2.plot(v_grid_adia1[mask1_b], u_grid_adia1[mask1_b], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    # Trapped line ell over V_X0 up to ceiling H(V0b) ONLY, with cap tick
    H_0b = float(ceiling(V0b))
    ax2.plot([V0b, V0b], [u_min, H_0b], color="#784212", ls="--", lw=2.2, zorder=4)
    tick_w = 0.03
    ax2.plot([V0b - tick_w, V0b + tick_w], [H_0b, H_0b], color="#784212", lw=3.0, zorder=12)

    # Two-line tag for trapped line ell
    box_ell_tag = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#784212", lw=1.4, alpha=1.0)
    ax2.text(V0b - 0.04, H_0b - 0.15,
             r"$\mathbf{\ell = \{(V_{X_0}, U)\}\ \subset\ \{S < S(X)\}}$" "\n"
             r"$\mathbf{max\ entropy\ over\ this\ column = \Sigma(V_{X_0}) < S(X)}$",
             fontsize=12.0, fontweight="bold", color="#784212", ha="right", va="top", zorder=100, bbox=box_ell_tag)

    # Dotted arrow at V0b with annotation box
    U0_PRIME = 1.80
    add_dotted_arrow(ax2, (V0b, U0b), (V0b, U0_PRIME), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)
    box_arrow0 = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#c0392b", lw=1.5, alpha=1.0)
    ax2.text(V0b + 0.05, 0.5 * (U0b + U0_PRIME),
             r"$\mathbf{X^\prime_0 \in \Omega_>\ (X^\prime_0 \prec\prec X)}$" "\n"
             r"$\mathbf{(Axiom\ T5:\ full\ temp.\ range\ \Rightarrow\ hot\ state\ below\ adiabat)}$",
             fontsize=11.5, fontweight="bold", color="#922b21", ha="left", va="center", zorder=100, bbox=box_arrow0)

    # Dotted arrow at V1 with annotation box
    U1_PRIME = 3.25
    add_dotted_arrow(ax2, (V1, U1), (V1, U1_PRIME), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)
    box_arrow1 = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#c0392b", lw=1.5, alpha=1.0)
    ax2.text(V1 - 0.05, 0.5 * (U1 + U1_PRIME),
             r"$\mathbf{Planck\ and\ T5:\ X^\prime_1 \in \Omega_>\ (X \prec\prec X^\prime_1)}$",
             fontsize=13.0, fontweight="bold", color="#922b21", ha="right", va="center", zorder=100, bbox=box_arrow1)

    # Interior point X_up strictly between V_B and V_PRIME
    V_UP = float(0.40 * V_B + 0.60 * V_PRIME)
    U_UP = float(U_adiabat(V_UP, S_MID))

    V_DOWN = 2.15
    U_DOWN = float(U_adiabat(V_DOWN, S_MID))

    # Scatter points for Case (b)
    ax2.scatter([V0b], [U0b], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax2.scatter([V1], [U1], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax2.scatter([V0b], [U0_PRIME], color="#e74c3c", s=210, zorder=15, edgecolors="#78281f", lw=2.2)
    ax2.scatter([V1], [U1_PRIME], color="#e74c3c", s=210, zorder=15, edgecolors="#78281f", lw=2.2)
    ax2.scatter([V_UP], [U_UP], color=C_PT_GT, s=210, zorder=15, edgecolors="#6e4402", lw=2.2)
    ax2.scatter([V_DOWN], [U_DOWN], color=C_PT_LT, s=210, zorder=15, edgecolors="#104266", lw=2.2)
    ax2.scatter([V_PRIME], [U_PRIME], color=C_PT_CUT, s=320, zorder=16, marker="o", edgecolors="black", lw=3.0)

    # Smooth curve inside Omega_> from X'_0 to X'_1 crossing dA_X exactly once, at X_up
    pts_path_v = np.array([V0b, 1.08, V_UP, 1.95, V1])
    pts_path_u = np.array([U0_PRIME, 2.65, U_UP, 2.85, U1_PRIME])
    cs_path = CubicSpline(pts_path_v, pts_path_u, bc_type="natural")
    v_path_dense = np.linspace(V0b, V1, 100)
    u_path_dense = cs_path(v_path_dense)
    ax2.plot(v_path_dense, u_path_dense, color="#e74c3c", lw=3.5, alpha=0.55, ls="-", zorder=7)

    # Verify numerically that T > T0 at sample points along the path
    v_sample = np.linspace(V0b, V1, 20)
    u_sample = cs_path(v_sample)
    t_sample = temperature(u_sample, v_sample)
    assert np.all(t_sample > T0), "Path in Omega_> must satisfy T > T0 everywhere"

    box_path = dict(boxstyle="round,pad=0.20", facecolor="#fff5f5", edgecolor="#e74c3c", lw=1.5, alpha=1.0)
    ax2.text(1.65, 3.12,
             r"$\mathbf{\Omega_>\ is\ open\ and\ connected\ and\ meets\ both\ sides\ of\ \partial A_X}$" "\n"
             r"$\mathbf{\Rightarrow\ it\ cuts\ \partial A_X\ (Step\ 1\ +\ separation)}$",
             fontsize=12.0, fontweight="bold", color="#922b21", ha="center", va="center", zorder=100, bbox=box_path)

    # Region Labels for Case (b)
    add_region_label(ax2, fig, 2.05, 0.42, "minus", "#1b4f72", fontsize=33)
    add_region_label(ax2, fig, 1.10, 3.15, "plus", "#922b21", fontsize=33)

    # Projections on Case (b)
    ax2.plot([V0b, V0b], [u_min, U0b], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([v_min, V0b], [U0b, U0b], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([v_min, V0b], [U0_PRIME, U0_PRIME], color="#922b21", ls=":", lw=1.6, zorder=4)
    ax2.plot([V1, V1], [u_min, U1], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([v_min, V1], [U1, U1], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([v_min, V1], [U1_PRIME, U1_PRIME], color="#922b21", ls=":", lw=1.6, zorder=4)

    # Bracket along x-axis for rho(A_X) starting at computed V_B
    ax2.annotate("", xy=(v_max, u_min), xytext=(V_B, u_min),
                 arrowprops=dict(arrowstyle="-", color="#2c3e50", lw=5.0, capstyle="butt"))
    ax2.plot([V_B, V_B], [u_min - 0.05, u_min + 0.05], color="#2c3e50", lw=2.5, clip_on=False, zorder=10)

    ax2.set_xticks([V0b, V_B, V1])
    ax2.set_xticklabels([r"$\mathbf{V_{X_0}}$", r"$\mathbf{\inf \rho(A_X)}$", r"$\mathbf{V_{X_1}}$"], fontsize=14, fontweight="bold")
    ax2.set_yticks([U0b, U0_PRIME, U1, U1_PRIME])
    ax2.set_yticklabels([r"$\mathbf{U(X_0)}$", r"$\mathbf{U(X^\prime_0)}$", r"$\mathbf{U(X_1)}$", r"$\mathbf{U(X^\prime_1)}$"], fontsize=14, fontweight="bold")

    # Callouts for Case (b)
    add_curved_callout(ax2, fig, r"$\mathbf{X_0}$", (V0b, U0b), (0.68, 0.40), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="tr", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X_1}$", (V1, U1), (2.75, 2.05), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X^\prime_0}$", (V0b, U0_PRIME), (0.68, U0_PRIME + 0.22), box_prime0, color="#922b21", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="tr", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X^\prime_1}$", (V1, U1_PRIME), (2.75, 3.45), box_prime1, color="#922b21", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X_\uparrow \in \partial A_X \cap \Omega_>}$", (V_UP, U_UP), (1.35, 2.05), box_gt, color="#784212", fontsize=15.0, lw=1.6, rad=-0.12, preferred_corner="tl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X_\downarrow \in \partial A_X \cap \Omega_<}$", (V_DOWN, U_DOWN), (2.25, 1.25), box_lt, color="#1b4f72", fontsize=15.0, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X^\prime \overset{A}{\sim} X}$", (V_PRIME, U_PRIME), (1.95, 1.25), box_cut, color=C_PT_CUT, fontsize=17.5, lw=1.8, rad=-0.14, preferred_corner="tl", shrinkB=10)

    # Curve labels for Case (b)
    add_curved_callout(ax2, fig, r"$\mathbf{\partial A_X}$", (2.84, float(U_adiabat(2.84, S_MID))), (2.95, 1.28), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax2, fig, r"$\mathbf{\partial A_{X_0}}$", (2.15, float(U_adiabat(2.15, S0b))), (2.35, 0.40), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=0.12, shrinkB=0)
    add_curved_callout(ax2, fig, r"$\mathbf{\partial A_{X_1}}$", (1.95, float(U_adiabat(1.95, S1))), (2.05, 3.65), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax2, fig, r"$\mathbf{I_{T_0}}$", (2.92, float(U_isotherm(2.92, T0))), (2.96, 2.72), box_curve_iso, color=C_ISO, fontsize=18.0, lw=1.6, rad=-0.12, shrinkB=0)

    ax2.set_xlabel(r"Work Coordinate  $V$", fontsize=18, fontweight="bold", labelpad=10)
    ax2.set_ylabel(r"Internal Energy  $U$", fontsize=18, fontweight="bold", labelpad=10)
    ax2.set_title(r"$\mathbf{Case\ (b):\ Coordinate\ Line\ Misses\ Adiabat\ —\ V_{X_0} \notin \rho(A_X)\ (Topological\ Separation)}$" "\n"
                  r"$\text{Vertical trapped line }\ell \text{ misses }\partial A_X\text{ + Axiom T5 }\Rightarrow X^\prime_0, X^\prime_1 \in \Omega_>\text{ straddle }\partial A_X \Rightarrow X_\uparrow \in \partial A_X \cap \Omega_>$",
                  fontsize=18.5, pad=12, fontweight="bold", color="#1a252f")

    # Main super title
    fig.suptitle("Theorem 5.5 — Step 2 (Scenario 1): Interior Transversal Crossing\n"
                 r"$T_{\min} < T_0 < T_{\max} \ \Rightarrow \ \text{Isotherm } I_{T_0} \text{ separates } \mathcal{X}_- \text{ from } \mathcal{X}_+ \ \Rightarrow \ \text{Intermediate adiabat } \partial A_X \text{ cuts } I_{T_0} \text{ at } X^\prime$",
                 fontsize=21.5, fontweight="bold", color="#1a252f", y=0.978)

    # =========================================================================
    # BANNER: UNIFIED PROOF STRUCTURE
    # =========================================================================
    ax_banner.set_xlim(0, 1)
    ax_banner.set_ylim(0, 1)
    banner_box = FancyBboxPatch((0.012, 0.03), 0.976, 0.94,
                                boxstyle="round,pad=0.01,rounding_size=0.025",
                                facecolor="#fcfdfd", edgecolor="#2c3e50", lw=2.0)
    ax_banner.add_patch(banner_box)

    lines_proof = [
        ("Step 2 Proof Structure — Comprehensive Exhaustion of All Coordinate Geometry (Lieb & Yngvason 1999):", True, 17.0),
        (r"$\mathbf{1.\ Hypotheses:}\quad \mathrm{(i)}\ T_{\min} < T_0 < T_{\max}\ \mathrm{is\ an\ interior\ isotherm;}\quad \mathrm{(ii)}\ X_0, X_1 \in I_{T_0};\quad \mathrm{(iii)}\ X_0 \prec X \prec X_1.$", False, 14.8),
        (r"$\mathbf{2.\ Construction\ of\ Straddling\ States\ on}\ \partial A_X:\quad \mathrm{Two\ mutually\ exhaustive\ geometric\ cases\ arise:}$", False, 14.8),
        (r"$\bullet\ \ \mathbf{Case\ (a)\ [V_{X_0} \in \rho(A_X)]:\ }\mathrm{Constant\ volume\ vertical\ slices\ yield}\ X_0 \prec X_>\ (V=V_{X_0})\ \Rightarrow\ X_> \in \mathcal{X}_+\ \mathrm{and}\ X_< \prec X_1\ (V=V_{X_1})\ \Rightarrow\ X_< \in \mathcal{X}_-.$", False, 14.5),
        (r"$\bullet\ \ \mathbf{Case\ (b)\ [V_{X_0} \notin \rho(A_X)]:\ }\mathrm{Vertical\ line}\ \ell\ \mathrm{over}\ V_{X_0}\ \mathrm{is\ trapped\ at}\ S < S(X).\ \mathrm{By\ T5,\ hot\ states}\ X^\prime_0, X^\prime_1 \in \Omega_>\ \mathrm{straddle}\ \partial A_X;\ \mathrm{connected}\ \Omega_>\ \mathrm{cuts}\ \partial A_X\ \mathrm{at}\ X_\uparrow \in \mathcal{X}_+.$", False, 14.5),
        (r"$\mathbf{3.\ Conclusion:}\quad \mathrm{In\ both\ cases,\ connected}\ \partial A_X\ \mathrm{joins}\ \mathcal{X}_+\ \mathrm{to}\ \mathcal{X}_-.\ \mathrm{By\ continuity\ of}\ T\ (\mathrm{IVT}),\ \partial A_X\ \mathrm{cuts}\ I_{T_0}\ \mathrm{at\ a\ state}\ X^\prime \in I_{T_0}\ \mathrm{with}\ X^\prime \overset{A}{\sim} X.$", False, 14.8)
    ]
    y_pos = [0.88, 0.72, 0.56, 0.40, 0.25, 0.09]
    for y, (line, is_bold, fs) in zip(y_pos, lines_proof):
        ax_banner.text(0.50, y, line, fontsize=fs, fontweight="bold" if is_bold else "normal",
                       color="#1a252f", ha="center", va="center", zorder=100)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated: {out_path}")


# =============================================================================
# SCENARIO 2: Boundary Temperature Approximation (T0 = T_max)
# =============================================================================
def generate_scenario2(out_path):
    fig = plt.figure(figsize=(17.5, 13.8), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[5.0, 1.75], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax_banner = fig.add_subplot(gs[1])
    ax_banner.axis("off")

    C_ADIA = "#000000"
    C_TMAX = "#922b21"
    C_TPRIME = "#e74c3c"
    C_PT_CUT = "#c0392b"
    C_PT_APPROX = "#b94a00"
    C_PT_TMAX = "#641e16"

    T_MAX = T_MAX_SCENARIO2
    T_PRIME = T_PRIME_SCENARIO2

    v_min, v_max = 0.55, 3.05
    u_min, u_max = 0.20, 4.40

    v_grid = np.linspace(v_min, v_max, 600)
    u_tmax_vals = U_isotherm(v_grid, T_MAX)
    u_tprime_vals = U_isotherm(v_grid, T_PRIME)

    # Boundary states on T_max
    V0 = 0.85
    U0 = float(U_isotherm(V0, T_MAX))
    S0 = float(entropy(U0, V0))

    V1 = 2.10
    U1 = float(U_isotherm(V1, T_MAX))
    S1 = float(entropy(U1, V1))

    S_MID = 0.52 * S0 + 0.48 * S1

    u_adia0_vals = U_adiabat(v_grid, S0)
    u_adiaX_vals = U_adiabat(v_grid, S_MID)
    u_adia1_vals = U_adiabat(v_grid, S1)

    # Intersections on approximating isotherm T'_0:
    V0_PRIME = float(B_PARAM + np.exp((S0 - CV * np.log(CV * T_PRIME)) / R_GAS))
    U0_PRIME = float(U_isotherm(V0_PRIME, T_PRIME))

    V1_PRIME = float(B_PARAM + np.exp((S1 - CV * np.log(CV * T_PRIME)) / R_GAS))
    U1_PRIME = float(U_isotherm(V1_PRIME, T_PRIME))

    V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T_PRIME)) / R_GAS))
    U_PRIME = float(U_isotherm(V_PRIME, T_PRIME))

    ax.set_xlim(v_min, v_max)
    ax.set_ylim(u_min, u_max)

    v_grid_2d = np.linspace(v_min, v_max, 350)
    u_grid_2d = np.linspace(u_min, u_max, 350)
    V_2D, U_2D = np.meshgrid(v_grid_2d, u_grid_2d)
    T_2D = temperature(U_2D, V_2D)
    norm = TwoSlopeNorm(vmin=T_2D.min(), vcenter=T_PRIME, vmax=T_MAX + 0.5)
    ax.contourf(V_2D, U_2D, T_2D, levels=80, cmap="coolwarm", norm=norm, alpha=0.22, zorder=1)

    # Forbidden region T > T_max (above u_tmax_vals)
    verts_forbid = [(v_min, u_max)]
    for v_val, u_val in zip(v_grid, u_tmax_vals):
        verts_forbid.append((v_val, min(max(u_val, u_min), u_max)))
    verts_forbid.append((v_max, u_max))
    ax.add_patch(Polygon(verts_forbid, closed=True, facecolor="#fbeee6", alpha=0.85,
                         hatch="//", edgecolor="#edbb99", zorder=2))

    # Curves: Adiabats clipped to T <= T_MAX
    t_adia0 = temperature(u_adia0_vals, v_grid)
    mask0 = (u_adia0_vals >= u_min) & (u_adia0_vals <= u_max) & (t_adia0 <= T_MAX + 1e-4)
    ax.plot(v_grid[mask0], u_adia0_vals[mask0], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    t_adia1 = temperature(u_adia1_vals, v_grid)
    mask1 = (u_adia1_vals >= u_min) & (u_adia1_vals <= u_max) & (t_adia1 <= T_MAX + 1e-4)
    ax.plot(v_grid[mask1], u_adia1_vals[mask1], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    t_adiaX = temperature(u_adiaX_vals, v_grid)
    maskX = (u_adiaX_vals >= u_min) & (u_adiaX_vals <= u_max) & (t_adiaX <= T_MAX + 1e-4)
    ax.plot(v_grid[maskX], u_adiaX_vals[maskX], color=C_ADIA, lw=3.8, ls="-", zorder=6)

    mask_prime = (u_tprime_vals >= u_min) & (u_tprime_vals <= u_max)
    ax.plot(v_grid[mask_prime], u_tprime_vals[mask_prime], color=C_TPRIME, lw=3.2, ls=(0, (6, 3.5)), zorder=7)

    mask_max = (u_tmax_vals >= u_min) & (u_tmax_vals <= u_max)
    ax.plot(v_grid[mask_max], u_tmax_vals[mask_max], color=C_TMAX, lw=4.2, ls=(0, (6, 3.5)), zorder=8)

    # Convergence Arrow: T'_0 -> T_max^- (points UPWARD toward I_Tmax)
    v_arr = 1.55
    u_arr_start = float(U_isotherm(v_arr, T_PRIME))
    u_arr_end = float(U_isotherm(v_arr, T_MAX))
    arrow_props = dict(arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#c0392b", zorder=10)
    ax.add_patch(FancyArrowPatch((v_arr, u_arr_start + 0.08), (v_arr, u_arr_end - 0.08), **arrow_props))
    ax.text(v_arr + 0.08, 0.5 * (u_arr_start + u_arr_end), r"$\mathbf{T^\prime_0 \to T_{\mathrm{max}}^-}$",
            fontsize=15.5, fontweight="bold", color="#c0392b", ha="left", va="center", zorder=100,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#fffaf5", edgecolor="#c0392b", lw=1.2, alpha=1.0))

    # Points
    ax.scatter([V0], [U0], color=C_PT_TMAX, s=190, zorder=15, edgecolors="black", lw=2.2)
    ax.scatter([V1], [U1], color=C_PT_TMAX, s=190, zorder=15, edgecolors="black", lw=2.2)
    ax.scatter([V0_PRIME], [U0_PRIME], color=C_PT_APPROX, s=200, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax.scatter([V1_PRIME], [U1_PRIME], color=C_PT_APPROX, s=200, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax.scatter([V_PRIME], [U_PRIME], color=C_PT_CUT, s=320, zorder=16, marker="o", edgecolors="black", lw=3.0)

    # Legend
    p_blue_s2 = Patch(edgecolor="#1b4f72", lw=1.2)
    p_hatch = Patch(facecolor="#fbeee6", edgecolor="#edbb99", hatch="//", lw=1.2)
    legend_elements = [
        Line2D([0], [0], color=C_ADIA, lw=3.2, ls="-"),
        Line2D([0], [0], color=C_TMAX, lw=3.8, ls=(0, (6, 3.5))),
        Line2D([0], [0], color=C_TPRIME, lw=3.0, ls=(0, (6, 3.5))),
        p_blue_s2,
        p_hatch,
    ]
    legend_labels = [
        r"$\mathbf{Adiabats}$",
        r"$\mathbf{Boundary\ Isotherm\ } I_{T_{\mathrm{max}}}$",
        r"$\mathbf{Approximating\ } I_{T^\prime_0}$",
        r"$\mathbf{\mathcal{X}_- \ (T < T_{\mathrm{max}})}$",
        r"$\mathbf{\mathcal{X}_+ = \emptyset\ (Unphysical)}$",
    ]
    leg = ax.legend(handles=legend_elements, labels=legend_labels,
                    handler_map={p_blue_s2: GradientHandler(cmap_w2b)},
                    loc="lower right", ncol=2, fontsize=13.5, framealpha=0.96,
                    edgecolor="#2c3e50", fancybox=True, borderpad=0.7)
    leg.set_zorder(100)

    # Region Labels
    add_region_label(ax, fig, 1.45, 3.90, "plus_empty", "#922b21", fontsize=28)
    add_region_label(ax, fig, 2.35, 1.00, "minus_gamma", "#1b4f72", fontsize=28)

    # Curve labels
    box_curve_adia = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")
    box_curve_iso = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")
    box_white = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#95a5a6", alpha=1.0, lw=1.5)
    box_cut = dict(boxstyle="round,pad=0.24", facecolor="#ffffff", edgecolor=C_PT_CUT, alpha=1.0, lw=2.2)
    box_prime = dict(boxstyle="round,pad=0.22", facecolor="#fef9e7", edgecolor="#b94a00", alpha=1.0, lw=1.6)

    add_curved_callout(ax, fig, r"$\mathbf{\partial A_{X_0}}$", (2.15, float(U_adiabat(2.15, S0))), (2.35, 0.40), box_curve_adia, color=C_ADIA, fontsize=17.0, lw=1.5, rad=0.12, shrinkB=0)
    add_curved_callout(ax, fig, r"$\mathbf{\partial A_X}$", (2.85, float(U_adiabat(2.85, S_MID))), (2.95, 1.35), box_curve_adia, color=C_ADIA, fontsize=17.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax, fig, r"$\mathbf{\partial A_{X_1}}$", (2.80, float(U_adiabat(2.80, S1))), (2.95, 2.65), box_curve_adia, color=C_ADIA, fontsize=17.0, lw=1.5, rad=-0.12, shrinkB=0)

    add_curved_callout(ax, fig, r"$\mathbf{I_{T^\prime_0}}$", (2.70, float(U_isotherm(2.70, T_PRIME))), (2.85, 3.10), box_curve_iso, color=C_TPRIME, fontsize=17.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax, fig, r"$\mathbf{I_{T_{\mathrm{max}}}}$", (2.55, float(U_isotherm(2.55, T_MAX))), (2.75, 3.65), box_curve_iso, color=C_TMAX, fontsize=17.0, lw=1.6, rad=-0.12, shrinkB=0)

    # Point Callouts
    add_curved_callout(ax, fig, r"$\mathbf{X_0}$", (V0, U0), (0.75, 2.25), box_white, color=C_PT_TMAX, fontsize=17.0, lw=1.6, rad=0.12, preferred_corner="br", shrinkB=8)
    add_curved_callout(ax, fig, r"$\mathbf{X_1}$", (V1, U1), (2.05, 3.65), box_white, color=C_PT_TMAX, fontsize=17.0, lw=1.6, rad=-0.12, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax, fig, r"$\mathbf{X^\prime_0}$", (V0_PRIME, U0_PRIME), (0.75, 1.25), box_prime, color=C_PT_APPROX, fontsize=17.0, lw=1.6, rad=-0.12, preferred_corner="tr", shrinkB=8)
    add_curved_callout(ax, fig, r"$\mathbf{X^\prime_1}$", (V1_PRIME, U1_PRIME), (2.65, 2.70), box_prime, color=C_PT_APPROX, fontsize=17.0, lw=1.6, rad=0.12, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax, fig, r"$\mathbf{X^\prime \overset{A}{\sim} X}$", (V_PRIME, U_PRIME), (1.25, 1.85), box_cut, color=C_PT_CUT, fontsize=17.0, lw=1.8, rad=-0.12, preferred_corner="tr", shrinkB=10)

    # Projections
    ax.plot([V0, V0], [u_min, U0], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax.plot([v_min, V0], [U0, U0], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax.plot([V1, V1], [u_min, U1], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax.plot([v_min, V1], [U1, U1], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax.plot([V_PRIME, V_PRIME], [u_min, U_PRIME], color="#922b21", ls=":", lw=1.6, zorder=4)
    ax.plot([v_min, V_PRIME], [U_PRIME, U_PRIME], color="#922b21", ls=":", lw=1.6, zorder=4)

    ax.set_xticks([V0, V_PRIME, V1])
    ax.set_xticklabels([r"$\mathbf{V(X_0)}$", r"$\mathbf{V(X^\prime)}$", r"$\mathbf{V(X_1)}$"], fontsize=14, fontweight="bold")
    ax.set_yticks([U0, U_PRIME, U1])
    ax.set_yticklabels([r"$\mathbf{U(X_0)}$", r"$\mathbf{U(X^\prime)}$", r"$\mathbf{U(X_1)}$"], fontsize=14, fontweight="bold")

    ax.set_xlabel(r"Work Coordinate  $V$", fontsize=18, fontweight="bold", labelpad=12)
    ax.set_ylabel(r"Internal Energy  $U$", fontsize=18, fontweight="bold", labelpad=12)
    ax.set_title(r"$\mathbf{Theorem\ 5.5\ —\ Step\ 3\ (Scenario\ 2):\ Boundary\ Temperature\ Approximation\ (T_0 = T_{\mathrm{max}})}$" "\n"
                 r"$\text{Since } \mathcal{X}_+ = \emptyset\text{, the boundary is analyzed via interior approximating isotherms } T^\prime_0 \to T_{\mathrm{max}}^-$",
                 fontsize=18.5, pad=14, fontweight="bold", color="#1a252f")

    # Banner
    banner_box = FancyBboxPatch((0.012, 0.03), 0.976, 0.94,
                                boxstyle="round,pad=0.01,rounding_size=0.025",
                                facecolor="#fcfdfd", edgecolor="#2c3e50", lw=2.0)
    ax_banner.add_patch(banner_box)

    lines_proof_s2 = [
        ("Step 3 Proof Structure — Upper Boundary Analysis (Lieb & Yngvason 1999):", True, 17.0),
        (r"$\mathbf{1.\ Boundary\ Structure:}\quad T_0 = T_{\max}\ \mathrm{is\ the\ supremum\ of\ physical\ temperature;}\quad \mathcal{X}_+ = \emptyset\quad \mathrm{and}\quad \mathcal{X}_- = \Gamma \setminus I_{T_{\max}}.$", False, 14.8),
        (r"$\mathbf{2.\ Approximating\ Sequence:}\quad \mathrm{Either}\ \partial A_X \cap I_{T_{\max}} \neq \emptyset\ \mathrm{(direct\ cut),\ or\ we\ choose\ interior\ isotherms}\ T^\prime_0 < T_{\max}\ \mathrm{approaching}\ T_{\max}^-:$", False, 14.8),
        (r"$\bullet\ \ \mathrm{Choose\ interior\ isotherms}\ T^\prime_0 < T_{\max}\ \mathrm{approaching}\ T_{\max}^-.\ \mathrm{Step\ 2\ applies\ to\ each\ interior}\ T^\prime_0,\ \mathrm{yielding}\ X^\prime_0, X^\prime, X^\prime_1 \in I_{T^\prime_0}.$", False, 14.5),
        (r"$\bullet\ \ \mathrm{The\ ordered\ states\ satisfy}\ X^\prime_0 \prec X^\prime \overset{A}{\sim} X \prec X^\prime_1\ \mathrm{with}\ T(X^\prime_0) = T(X^\prime) = T(X^\prime_1) = T^\prime_0.$", False, 14.5),
        (r"$\mathbf{3.\ Convergence:}\quad \mathrm{As}\ T^\prime_0 \to T_{\max}^-,\ \mathrm{we\ have}\ X^\prime_0 \to X_0,\ X^\prime_1 \to X_1,\ \mathrm{establishing\ adiabatic\ equivalence}\ X^\prime \overset{A}{\sim} X\ \mathrm{arbitrarily\ close\ to\ the\ boundary}.$", False, 14.8)
    ]
    y_pos = [0.88, 0.72, 0.56, 0.40, 0.25, 0.09]
    for y, (line, is_bold, fs) in zip(y_pos, lines_proof_s2):
        ax_banner.text(0.50, y, line, fontsize=fs, fontweight="bold" if is_bold else "normal",
                       color="#1a252f", ha="center", va="center", zorder=100)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated: {out_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    s1_path = os.path.join(script_dir, "theorem55_scenario1.png")
    s2_path = os.path.join(script_dir, "theorem55_scenario2.png")
    vis_path = os.path.join(script_dir, "theorem55_visualization.png")

    generate_scenario1(s1_path)
    generate_scenario2(s2_path)

    # Sync primary reference figure
    shutil.copyfile(s1_path, vis_path)
    print(f"Successfully synced: {vis_path}")
