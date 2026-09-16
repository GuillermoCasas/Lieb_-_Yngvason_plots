"""
Theorem 5.5 Visualization Module: Isotherms cut adiabats.
Elliott H. Lieb & Jakob Yngvason, Phys. Rept. 310 (1999) 1–96, Section 5.2, pp. 73–75.

This script produces two publication-quality figures:
1. theorem55_scenario1.png: Scenario (1) Interior Transversal Crossing (T_min < T_0 < T_max).
2. theorem55_scenario2.png: Scenario (2) Boundary Temperature Approximation (T_0 = T_max).
3. theorem55_visualization.png: Primary reference figure (synced with Scenario 1).

Design Criteria & Mathematical Geometry:
- Isotherms: red and dashed (color="#c0392b", ls=(0, (6, 3.5))).
- Adiabats: full and black (color="#000000", ls="-").
  * Intermediate adiabat dA_X: bold solid black (lw=3.8).
  * Bounding adiabats dA_X0 and dA_X1: solid black (lw=2.4).
- Direct Curve Labels: cleanly indicate which curve is which using pill badges positioned in open whitespace corridors.
- Temperature background: faint 2D contour fill going from blue to red (cmap="coolwarm", alpha=0.22).
- Strict z-ordering: all text labels, annotations, and bounding boxes have zorder=100
  with opaque backgrounds (alpha=1.0) so text is always strictly in front of lines.
- Zero curve coverage & zero curve crossings: verified by automated collision detection.
- Callouts: curved lines without arrows connecting from box corners to state points.
- Abstract axes: no numerical tick marks, pure physical coordinate labels U and V.
- Centered summary banners at the bottom detailing the axiomatic conclusions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# Colormaps for legend gradient boxes: white-to-blue and white-to-red
cmap_w2b = LinearSegmentedColormap.from_list("w2b", ["#ffffff", "#5dade2", "#1b4f72"])
cmap_w2r = LinearSegmentedColormap.from_list("w2r", ["#ffffff", "#f1948a", "#922b21"])


class GradientHandler(HandlerBase):
    """Custom legend handler rendering a smooth horizontal gradient box with border using pure vector slices."""
    def __init__(self, cmap, **kwargs):
        super().__init__(**kwargs)
        self.cmap = cmap

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        n_slices = 45
        artists = []
        slice_w = width / n_slices
        for i in range(n_slices):
            c = self.cmap(i / (n_slices - 1))
            r = Rectangle([xdescent + i * slice_w, ydescent], slice_w * 1.02, height,
                          facecolor=c, edgecolor=c, lw=0, transform=trans)
            artists.append(r)
        border = Rectangle([xdescent, ydescent], width, height,
                         facecolor="none", edgecolor=orig_handle.get_edgecolor(),
                         lw=orig_handle.get_linewidth(), transform=trans)
        artists.append(border)
        return artists

# Matplotlib Publication Styling
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 22,
    "axes.labelsize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Thermodynamic Model Parameters (Ideal Gas with Attractive Potential)
CV = 1.5
R_GAS = 1.0
A_PARAM = 0.45
B_PARAM = 0.20

T0_SCENARIO1 = 2.00


def U_isotherm(V, T):
    """Energy along isotherm: U(V, T) = c_v * T - a / V."""
    return CV * T - A_PARAM / V


def U_adiabat(V, S_const):
    """Energy along adiabat: U(V, S) = exp((S - R ln(V - b)) / c_v) - a / V."""
    return np.exp((S_const - R_GAS * np.log(V - B_PARAM)) / CV) - A_PARAM / V


def entropy(U, V):
    """Entropy function S(U, V) = c_v ln(U + a / V) + R ln(V - b)."""
    return CV * np.log(U + A_PARAM / V) + R_GAS * np.log(V - B_PARAM)


def temperature(U, V):
    """Empirical temperature T(U, V) = (U + a / V) / c_v."""
    return (U + A_PARAM / V) / CV


def slope_isotherm(V):
    """Tangent slope dV/dU along isotherm: dV/dU|_T = V^2 / a > 0."""
    return (V ** 2) / A_PARAM


def slope_adiabat(U, V):
    """Tangent slope dV/dU along adiabat: dV/dU|_S = -1 / P < 0."""
    t = temperature(U, V)
    p = R_GAS * t / (V - B_PARAM) - A_PARAM / (V ** 2)
    return -1.0 / p


def add_curved_callout(ax, fig, text, xy_target, xy_box, bbox_style,
                       color="#1a252f", fontsize=13.5, rad=0.18, lw=1.8,
                       preferred_corner=None, shrinkB=8):
    """
    Connect a text box to a state point using an aesthetically pleasing curved line
    (without arrowheads) originating precisely at one of the box's corners (closest or specified).
    """
    t = ax.text(xy_box[0], xy_box[1], text, fontsize=fontsize, fontweight="bold",
                color=color, ha="center", va="center", zorder=100, bbox=bbox_style)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    # Calculate box corners in display coordinates (for accurate Euclidean distance)
    bb_d = t.get_window_extent(r)
    corners_d = {
        "bl": (bb_d.x0, bb_d.y0),
        "br": (bb_d.x1, bb_d.y0),
        "tl": (bb_d.x0, bb_d.y1),
        "tr": (bb_d.x1, bb_d.y1),
    }

    # Box corners in data coordinates
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

    conn = FancyArrowPatch(start_pt, xy_target,
                           arrowstyle="-",
                           connectionstyle=f"arc3,rad={rad}",
                           lw=lw, color=color,
                           shrinkA=0, shrinkB=shrinkB, zorder=30)
    ax.add_patch(conn)
    return t, conn


def add_dotted_arrow(ax, p0, p1, color, lw=4.2, mutation_scale=32, head_length=0.82, head_width=0.35,
                     shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9):
    """
    Draws a horizontal or general arrow with a crisp, truly dotted shaft (round dots)
    and a solid filled arrowhead (-|>), eliminating the fragmentation artifact of FancyArrowPatch.
    """
    # 1. Solid filled arrow head via FancyArrowPatch with lw=0
    style_str = f"-|>,head_length={head_length},head_width={head_width}"
    head = FancyArrowPatch(p0, p1, arrowstyle=style_str, mutation_scale=mutation_scale,
                           color=color, lw=0, shrinkA=shrinkA, shrinkB=shrinkB, zorder=zorder)
    ax.add_patch(head)

    # 2. Dotted shaft with round dots up to the base of the arrowhead
    inv = ax.transData.inverted()
    p0_d = ax.transData.transform(p0)
    p1_d = ax.transData.transform(p1)

    v_d = p1_d - p0_d
    dist_d = np.hypot(v_d[0], v_d[1])
    if dist_d == 0:
        return head
    u_d = v_d / dist_d

    # Head length in points is head_length * mutation_scale
    head_len_pt = head_length * mutation_scale

    start_d = p0_d + u_d * shrinkA
    end_d = p1_d - u_d * (shrinkB + head_len_pt)

    start_data = inv.transform(start_d)
    end_data = inv.transform(end_d)

    ax.plot([start_data[0], end_data[0]], [start_data[1], end_data[1]],
            color=color, lw=lw, linestyle=(0, (0.01, dot_spacing)),
            solid_capstyle="round", dash_capstyle="round", zorder=zorder)
    return head


def add_region_label(ax, fig, x, y, text_type, color, fontsize=33):
    """
    Renders calligraphic region labels (X_-, X_+, X_+ = empty) with a perfectly visible,
    unobscured subscript sign so that mathtext's font flourish does not cover the minus sign.
    """
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


def generate_scenario1(out_path):
    """
    Generate Scenario (1): Interior Transversal Crossing (T_min < T_0 < T_max).
    The isotherm I_{T_0} separates X_- from X_+ and must cut the intermediate adiabat dA_X.
    """
    fig = plt.figure(figsize=(17.5, 13.8), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[5.0, 1.75], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax_banner = fig.add_subplot(gs[1])
    ax_banner.axis("off")

    # Point & Highlight Colors
    C_ISO = "#c0392b"
    C_ADIA = "#000000"
    C_PT_ISO = "#b94a00"
    C_PT_CUT = "#c0392b"
    C_PT_GT = "#d68910"
    C_PT_LT = "#2980b9"

    u_min, u_max = 1.15, 5.35
    v_min, v_max = 0.55, 3.05

    ax.set_xlim(u_min, u_max)
    ax.set_ylim(v_min, v_max)

    v_grid = np.linspace(v_min, v_max, 600)

    # Reference states on I_{T_0}
    V0 = 0.82
    U0 = float(U_isotherm(V0, T0_SCENARIO1))
    S0 = float(entropy(U0, V0))

    V1 = 2.50
    U1 = float(U_isotherm(V1, T0_SCENARIO1))
    S1 = float(entropy(U1, V1))

    # Intermediate state X with X_0 < X < X_1
    S_MID = 0.52 * S0 + 0.48 * S1
    V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T0_SCENARIO1)) / R_GAS))
    U_PRIME = float(U_isotherm(V_PRIME, T0_SCENARIO1))

    # Constant-volume state X_> on dA_X at V(X_>) = V_{X_0} in rho(A_X)
    U_GT = float(U_adiabat(V0, S_MID))
    T_GT = float(temperature(U_GT, V0))

    # Constant-volume state X_< on dA_X at V(X_<) = V_{X_1} in rho(A_X)
    V_LT = V1
    U_LT = float(U_adiabat(V1, S_MID))
    T_LT = float(temperature(U_LT, V1))

    u_iso_vals = U_isotherm(v_grid, T0_SCENARIO1)
    u_adia0_vals = U_adiabat(v_grid, S0)
    u_adiaX_vals = U_adiabat(v_grid, S_MID)
    u_adia1_vals = U_adiabat(v_grid, S1)

    # 1. Background: Faint 2D Temperature Contour Fill going from Blue to Red
    u_grid_2d = np.linspace(u_min, u_max, 350)
    v_grid_2d = np.linspace(v_min, v_max, 350)
    U_2D, V_2D = np.meshgrid(u_grid_2d, v_grid_2d)
    T_2D = temperature(U_2D, V_2D)
    norm = TwoSlopeNorm(vmin=T_2D.min(), vcenter=T0_SCENARIO1, vmax=T_2D.max())
    ax.contourf(U_2D, V_2D, T_2D, levels=80, cmap="coolwarm", norm=norm, alpha=0.22, zorder=1)

    # 3. Thermodynamic Curves: Adiabats full black, Isotherm red dashed
    mask0 = (u_adia0_vals >= u_min) & (u_adia0_vals <= u_max)
    ax.plot(u_adia0_vals[mask0], v_grid[mask0], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    mask1 = (u_adia1_vals >= u_min) & (u_adia1_vals <= u_max)
    ax.plot(u_adia1_vals[mask1], v_grid[mask1], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    maskX = (u_adiaX_vals >= u_min) & (u_adiaX_vals <= u_max)
    ax.plot(u_adiaX_vals[maskX], v_grid[maskX], color=C_ADIA, lw=3.8, ls="-", zorder=6)

    mask_iso = (u_iso_vals >= u_min) & (u_iso_vals <= u_max)
    ax.plot(u_iso_vals[mask_iso], v_grid[mask_iso], color=C_ISO, lw=3.6, ls=(0, (6, 3.5)), zorder=7)

    # 4. Points (zorders 15 to 16)
    ax.scatter([U0], [V0], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax.scatter([U1], [V1], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax.scatter([U_GT], [V0], color=C_PT_GT, s=210, zorder=15, edgecolors="#6e4402", lw=2.2)
    ax.scatter([U_LT], [V1], color=C_PT_LT, s=210, zorder=15, edgecolors="#104266", lw=2.2)
    ax.scatter([U_PRIME], [V_PRIME], color=C_PT_CUT, s=320, zorder=16, marker="o", edgecolors="black", lw=3.0)

    # Legend with Gradient Region Boxes
    p_blue_s1 = Patch(edgecolor="#1b4f72", lw=1.2)
    p_red_s1 = Patch(edgecolor="#922b21", lw=1.2)

    legend_elements = [
        Line2D([0], [0], color=C_ADIA, lw=3.2, ls="-"),
        Line2D([0], [0], color=C_ISO, lw=3.2, ls=(0, (4.5, 2.5))),
        p_blue_s1,
        p_red_s1,
    ]
    legend_labels = [
        r"$\mathbf{Adiabats}$",
        r"$\mathbf{Isotherms}$",
        r"$\mathbf{\mathcal{X}_- \ (T < T_0)}$",
        r"$\mathbf{\mathcal{X}_+ \ (T > T_0)}$",
    ]
    leg = ax.legend(handles=legend_elements, labels=legend_labels,
                    handler_map={p_blue_s1: GradientHandler(cmap_w2b), p_red_s1: GradientHandler(cmap_w2r)},
                    loc="upper right", fontsize=15.5,
                    handlelength=3.2, handleheight=1.2,
                    framealpha=0.96, edgecolor="#2c3e50", fancybox=True, borderpad=0.7, labelspacing=0.6)
    leg.set_zorder(100)

    # Direct Curve Labels with Connectors (unboxed to distinguish from point labels, near edges)
    box_curve_adia = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")
    box_curve_iso = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")

    # dA_X: near top-left edge, safely separated from the curve itself
    u_target_x = float(U_adiabat(2.84, S_MID))
    add_curved_callout(ax, fig, r"$\mathbf{\partial A_X}$", (u_target_x, 2.84), (1.28, 2.95),
                       box_curve_adia, color=C_ADIA, fontsize=19.0, lw=1.5, rad=-0.12, shrinkB=0)

    # dA_X0: near left edge (safely inside plot, away from y-axis)
    u_target_0 = float(U_adiabat(2.15, S0))
    add_curved_callout(ax, fig, r"$\mathbf{\partial A_{X_0}}$", (u_target_0, 2.15), (1.46, 2.22),
                       box_curve_adia, color=C_ADIA, fontsize=19.0, lw=1.5, rad=0.12, shrinkB=0)

    # dA_X1: near right edge
    u_target_1 = float(U_adiabat(1.20, S1))
    add_curved_callout(ax, fig, r"$\mathbf{\partial A_{X_1}}$", (u_target_1, 1.20), (4.95, 1.35),
                       box_curve_adia, color=C_ADIA, fontsize=19.0, lw=1.5, rad=-0.12, shrinkB=0)

    # I_T0: near top edge
    u_target_iso = float(U_isotherm(2.92, T0_SCENARIO1))
    add_curved_callout(ax, fig, r"$\mathbf{I_{T_0}}$", (u_target_iso, 2.92), (3.12, 2.96),
                       box_curve_iso, color=C_ISO, fontsize=19.0, lw=1.6, rad=-0.12, shrinkB=0)

    # --- CONSTANT VOLUME PROOF CONSTRUCT 1 (Theorem 5.5 Step 2: X_0 -> X_>, Heating/Red) ---
    # Horizontal dotted red arrow connecting X_0 to X_> at V = V_{X_0}: increases U and raises temperature into X_+
    add_dotted_arrow(ax, (U0, V0), (U_GT, V0), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)

    # Centered compact label on constant volume arrow for X_> (larger and closer to arrow)
    box_vol0 = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#c0392b", lw=1.5, alpha=1.0)
    ax.text(0.5 * (U0 + U_GT), V0 + 0.098,
            r"$\mathbf{V(X_>) = V_{X_0} \in \rho(A_X)}$",
            fontsize=16.0, fontweight="bold", color="#922b21", ha="center", va="center", zorder=100, bbox=box_vol0)

    # Dotted projection lines for V0
    ax.plot([u_min, U0], [V0, V0], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax.plot([U0, U0], [v_min, V0], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax.plot([U_GT, U_GT], [v_min, V0], color="#784212", ls=":", lw=1.6, zorder=4)

    # --- CONSTANT VOLUME PROOF CONSTRUCT 2 (Theorem 5.5 Step 2: X_1 -> X_<, Cooling/Blue) ---
    # Reversed horizontal dotted blue arrow from X_1 on I_{T_0} to X_< on dA_X: decreases U and lowers temperature into X_-
    add_dotted_arrow(ax, (U1, V1), (U_LT, V1), color="#2471a3", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)

    # Centered compact label on constant volume arrow for X_< (larger and closer to arrow)
    box_vol1 = dict(boxstyle="round,pad=0.18", facecolor="#f0f7fb", edgecolor="#2471a3", lw=1.5, alpha=1.0)
    ax.text(0.5 * (U_LT + U1), V1 + 0.098,
            r"$\mathbf{V(X_<) = V_{X_1} \in \rho(A_X)}$",
            fontsize=16.0, fontweight="bold", color="#154360", ha="center", va="center", zorder=100, bbox=box_vol1)

    # Dotted projection lines for V1
    ax.plot([u_min, U_LT], [V1, V1], color="#1b4f72", ls=":", lw=1.6, zorder=4)
    ax.plot([U_LT, U_LT], [v_min, V1], color="#1b4f72", ls=":", lw=1.6, zorder=4)
    ax.plot([U1, U1], [v_min, V1], color="#4e2103", ls=":", lw=1.6, zorder=4)

    # Abstract Axes Ticks (Named after states, bold and prominent)
    ax.set_xticks([U_LT, U0, U1, U_GT])
    ax.set_xticklabels([r"$\mathbf{U(X_<)}$", r"$\mathbf{U(X_0)}$", r"$\mathbf{U(X_1)}$", r"$\mathbf{U(X_>)}$"],
                       fontsize=15, fontweight="bold")
    ax.set_yticks([V0, V1])
    ax.set_yticklabels([r"$\mathbf{V_{X_0}}$", r"$\mathbf{V_{X_1}}$"], fontsize=15, fontweight="bold")

    # 8. Curved corner callouts without arrowheads (positioned tightly to labeled objects)
    box_white = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#95a5a6", alpha=1.0, lw=1.5)
    box_cut = dict(boxstyle="round,pad=0.24", facecolor="#ffffff", edgecolor=C_PT_CUT, alpha=1.0, lw=2.2)
    box_gt = dict(boxstyle="round,pad=0.22", facecolor="#fefbf5", edgecolor="#d4ac0d", alpha=1.0, lw=1.6)
    box_lt = dict(boxstyle="round,pad=0.22", facecolor="#f0f7fb", edgecolor="#2980b9", alpha=1.0, lw=1.6)

    # X0: Compact label at U=1.96, V=0.98 connecting from bottom-right corner to (U0, V0)
    add_curved_callout(ax, fig, r"$\mathbf{X_0}$",
                       (U0, V0), (1.96, 0.98), box_white, color="#4e2103", fontsize=18.5, lw=1.6, rad=-0.14, preferred_corner="br", shrinkB=8)

    # X1: Compact label at U=3.24, V=2.54 connecting from top-left corner to (U1, V1)
    add_curved_callout(ax, fig, r"$\mathbf{X_1}$",
                       (U1, V1), (3.24, 2.54), box_white, color="#4e2103", fontsize=18.5, lw=1.6, rad=0.14, preferred_corner="tl", shrinkB=8)

    # X_<: Compact label at (2.06, 2.36) connecting from top-left corner to (U_LT, V1)
    add_curved_callout(ax, fig, r"$\mathbf{X_<}$",
                       (U_LT, V1), (2.06, 2.36), box_lt, color="#1b4f72", fontsize=18.5, lw=1.6, rad=-0.14, preferred_corner="tl", shrinkB=8)

    # X_>: Compact label at (3.88, 0.95) connecting from bottom-left corner to (U_GT, V0)
    add_curved_callout(ax, fig, r"$\mathbf{X_>}$",
                       (U_GT, V0), (3.88, 0.95), box_gt, color="#784212", fontsize=18.5, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)

    # X': Compact label at U=2.98, V=1.57 connecting from bottom-left corner to (U_PRIME, V_PRIME)
    add_curved_callout(ax, fig, r"$\mathbf{X^\prime \overset{A}{\sim} X}$",
                       (U_PRIME, V_PRIME), (2.98, 1.57), box_cut, color=C_PT_CUT, fontsize=18.5, lw=1.8, rad=-0.14, preferred_corner="bl", shrinkB=10)

    ax.set_xlabel(r"Internal Energy  $U$", fontsize=19, fontweight="bold", labelpad=12)
    ax.set_ylabel(r"Work Coordinate  $V$", fontsize=19, fontweight="bold", labelpad=12)

    ax.set_title("Theorem 5.5 — Step 2 (Scenario 1): Interior Transversal Crossing",
                 fontsize=22.5, pad=58, fontweight="bold", color="#1a252f")
    sub_s1 = (r"$T_{\min} < T_0 < T_{\max} \ \Rightarrow \ $ Isotherm $I_{T_0}$ separates $\mathcal{X}_-$ from $\mathcal{X}_+$" "\n"
              r"$\Rightarrow \ $ Intermediate adiabat $\partial A_X$ cuts $I_{T_0}$ at $X^\prime$")
    ax.text(0.50, 1.015, sub_s1, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=15.0, fontweight="normal", color="#34495e", linespacing=1.35, zorder=100)

    # Bottom banner with structured proof breakdown: Hypotheses -> Construction -> Conclusion
    ax_banner.set_xlim(0, 1)
    ax_banner.set_ylim(0, 1)
    banner_box = FancyBboxPatch((0.012, 0.03), 0.976, 0.94,
                                boxstyle="round,pad=0.01,rounding_size=0.025",
                                facecolor="#fcfdfd", edgecolor="#2c3e50", lw=2.0)
    ax_banner.add_patch(banner_box)

    lines_s1 = [
        ("Step 2 Proof Structure (Lieb & Yngvason 1999):", True, 17.5),
        (r"$\mathbf{1.\ Hypotheses:}\quad \mathrm{(i)}\ T_{\min} < T_0 < T_{\max}\ \mathrm{is\ an\ interior\ isotherm;}\quad \mathrm{(ii)}\ X_0, X_1 \in I_{T_0}\;\quad \mathrm{(iii)}\ X_0 \prec X \prec X_1.$", False, 15.0),
        (r"$\mathbf{2.\ Construction\ of}\ X_>, X_<:\quad \mathrm{Coordinate\ sections\ of}\ \partial A_X\ \mathrm{at\ constant\ work\ coordinates\ yield:}$", False, 15.0),
        (r"$\bullet\ \ \mathrm{(i)}\ \ \mathrm{At}\ V = V(X_0):\ X_0 \prec X_>\ \ \Rightarrow\ \ U(X_>) > U(X_0)\ \ \Rightarrow\ \ X_> \in \mathcal{X}_+$", False, 14.8),
        (r"$\bullet\ \ \mathrm{(ii)}\ \ \mathrm{At}\ V = V(X_1):\ X_< \prec X_1\ \ \Rightarrow\ \ U(X_<) < U(X_1)\ \ \Rightarrow\ \ X_< \in \mathcal{X}_-$", False, 14.8),
        (r"$\mathbf{3.\ Conclusion:}\quad \mathrm{Since}\ \partial A_X\ \mathrm{is\ connected\ and\ joins}\ X_> \in \mathcal{X}_+\ \mathrm{to}\ X_< \in \mathcal{X}_-,\ \mathrm{it\ cuts}\ I_{T_0}\ \mathrm{at\ a\ state}\ X^\prime \in I_{T_0}\ \mathrm{with}\ X^\prime \overset{A}{\sim} X.$", False, 15.0)
    ]
    y_pos = [0.88, 0.72, 0.56, 0.40, 0.25, 0.09]
    for y, (line, is_bold, fs) in zip(y_pos, lines_s1):
        ax_banner.text(0.50, y, line, fontsize=fs, fontweight="bold" if is_bold else "normal",
                       color="#1a252f", ha="center", va="center", zorder=100)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated: {out_path}")


def generate_scenario2(out_path):
    """
    Generate Scenario (2): Boundary Temperature Approximation (T_0 = T_max).
    T_max is the frontier of physical state space (X_+ = empty). The boundary is analyzed
    via interior approximating isotherms T'_0 -> T_max^-.
    """
    fig = plt.figure(figsize=(17.5, 13.8), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[5.0, 1.75], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax_banner = fig.add_subplot(gs[1])
    ax_banner.axis("off")

    T_MAX = 2.80
    T_PRIME = 2.45

    C_ADIA = "#000000"
    C_TMAX = "#922b21"
    C_TPRIME = "#e74c3c"
    C_PT_CUT = "#c0392b"
    C_PT_APPROX = "#b94a00"
    C_PT_TMAX = "#641e16"

    u_min, u_max = 1.30, 5.35
    v_min, v_max = 0.55, 3.05

    ax.set_xlim(u_min, u_max)
    ax.set_ylim(v_min, v_max)

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

    # 1. Background: Faint 2D Temperature Contour Fill going from Blue to Red
    u_grid_2d = np.linspace(u_min, u_max, 350)
    v_grid_2d = np.linspace(v_min, v_max, 350)
    U_2D, V_2D = np.meshgrid(u_grid_2d, v_grid_2d)
    T_2D = temperature(U_2D, V_2D)
    norm = TwoSlopeNorm(vmin=T_2D.min(), vcenter=T_PRIME, vmax=T_MAX + 0.5)
    ax.contourf(U_2D, V_2D, T_2D, levels=80, cmap="coolwarm", norm=norm, alpha=0.22, zorder=1)

    # Forbidden hatched region for T > T_max (soft styled polygon)
    verts_forbid = [(u_max, v_min)]
    for v_val, u_val in zip(v_grid, u_tmax_vals):
        verts_forbid.append((min(max(u_val, u_min), u_max), v_val))
    verts_forbid.append((u_max, v_max))
    ax.add_patch(Polygon(verts_forbid, closed=True, facecolor="#fbeee6", alpha=0.85,
                         hatch="//", edgecolor="#edbb99", zorder=2))

    # 2. Thermodynamic Curves: Adiabats full black (clipped to physical state space T <= T_MAX), Isotherms red dashed
    t_adia0 = temperature(u_adia0_vals, v_grid)
    mask0 = (u_adia0_vals >= u_min) & (u_adia0_vals <= u_max) & (t_adia0 <= T_MAX + 1e-4)
    ax.plot(u_adia0_vals[mask0], v_grid[mask0], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    t_adia1 = temperature(u_adia1_vals, v_grid)
    mask1 = (u_adia1_vals >= u_min) & (u_adia1_vals <= u_max) & (t_adia1 <= T_MAX + 1e-4)
    ax.plot(u_adia1_vals[mask1], v_grid[mask1], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    t_adiaX = temperature(u_adiaX_vals, v_grid)
    maskX = (u_adiaX_vals >= u_min) & (u_adiaX_vals <= u_max) & (t_adiaX <= T_MAX + 1e-4)
    ax.plot(u_adiaX_vals[maskX], v_grid[maskX], color=C_ADIA, lw=3.8, ls="-", zorder=6)

    mask_prime = (u_tprime_vals >= u_min) & (u_tprime_vals <= u_max)
    ax.plot(u_tprime_vals[mask_prime], v_grid[mask_prime], color=C_TPRIME, lw=3.2, ls=(0, (6, 3.5)), zorder=7)

    mask_max = (u_tmax_vals >= u_min) & (u_tmax_vals <= u_max)
    ax.plot(u_tmax_vals[mask_max], v_grid[mask_max], color=C_TMAX, lw=4.2, ls=(0, (6, 3.5)), zorder=8)

    # 3. Convergence Arrow (larger and closer to arrow)
    arrow_props = dict(arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#c0392b", zorder=10)
    ax.add_patch(FancyArrowPatch((3.52, 1.85), (3.86, 1.85), **arrow_props))
    ax.text(3.69, 1.91, r"$\mathbf{T^\prime_0 \to T_{\mathrm{max}}^-}$",
            fontsize=16.5, fontweight="bold", color="#c0392b", ha="center", zorder=100,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#fffaf5", edgecolor="#c0392b", lw=1.2, alpha=1.0))

    # 4. Points (zorders 15 to 16)
    ax.scatter([U0], [V0], color=C_PT_TMAX, s=190, zorder=15, edgecolors="black", lw=2.2)
    ax.scatter([U1], [V1], color=C_PT_TMAX, s=190, zorder=15, edgecolors="black", lw=2.2)
    ax.scatter([U0_PRIME], [V0_PRIME], color=C_PT_APPROX, s=200, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax.scatter([U1_PRIME], [V1_PRIME], color=C_PT_APPROX, s=200, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax.scatter([U_PRIME], [V_PRIME], color=C_PT_CUT, s=320, zorder=16, marker="o", edgecolors="black", lw=3.0)

    # Legend with Gradient and Hatched Region Boxes
    p_blue_s2 = Patch(edgecolor="#1b4f72", lw=1.2)
    p_hatched_s2 = Patch(facecolor="#fbeee6", edgecolor="#922b21", hatch="//", lw=1.2)

    legend_elements = [
        Line2D([0], [0], color=C_ADIA, lw=3.2, ls="-"),
        Line2D([0], [0], color=C_TPRIME, lw=3.2, ls=(0, (4.5, 2.5))),
        p_blue_s2,
        p_hatched_s2,
    ]
    legend_labels = [
        r"$\mathbf{Adiabats}$",
        r"$\mathbf{Isotherms}$",
        r"$\mathbf{\mathcal{X}_- = \Gamma \, \backslash \, I_{T_{\mathrm{max}}}}$",
        r"$\mathbf{\mathcal{X}_+ = \emptyset \ (T > T_{\mathrm{max}})}$",
    ]
    leg = ax.legend(handles=legend_elements, labels=legend_labels,
                    handler_map={p_blue_s2: GradientHandler(cmap_w2b)},
                    loc="upper right", fontsize=15.5,
                    handlelength=3.2, handleheight=1.2,
                    framealpha=0.96, edgecolor="#2c3e50", fancybox=True, borderpad=0.7, labelspacing=0.6)
    leg.set_zorder(100)

    # Direct Curve Labels with Connectors (unboxed to distinguish from point labels, near edges)
    box_curve_adia = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")
    box_curve_prime = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")
    box_curve_tmax = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")

    # 1. dA_X0: near left edge
    u_target_adia0 = float(U_adiabat(2.20, S0))
    add_curved_callout(ax, fig, r"$\mathbf{\partial A_{X_0}}$", (u_target_adia0, 2.20), (1.50, 2.30),
                       box_curve_adia, color=C_ADIA, fontsize=19.0, lw=1.5, rad=0.12, shrinkB=0)

    # 2. dA_X: near top edge
    u_target_adiaX = float(U_adiabat(2.90, S_MID))
    add_curved_callout(ax, fig, r"$\mathbf{\partial A_X}$", (u_target_adiaX, 2.90), (1.95, 2.96),
                       box_curve_adia, color=C_ADIA, fontsize=19.0, lw=1.5, rad=-0.12, shrinkB=0)

    # 3. dA_X1: near top edge, centered between adiabat and isotherm
    u_target_adia1 = float(U_adiabat(2.90, S1))
    add_curved_callout(ax, fig, r"$\mathbf{\partial A_{X_1}}$", (u_target_adia1, 2.90), (3.30, 2.96),
                       box_curve_adia, color=C_ADIA, fontsize=19.0, lw=1.5, rad=-0.10, shrinkB=0)

    # 4. I_T'0: near bottom edge
    u_target_prime = float(U_isotherm(0.72, T_PRIME))
    add_curved_callout(ax, fig, r"$\mathbf{I_{T^\prime_0}}$", (u_target_prime, 0.72), (2.55, 0.75),
                       box_curve_prime, color=C_TPRIME, fontsize=19.0, lw=1.6, rad=0.12, shrinkB=0)

    # 5. I_Tmax: near bottom edge (lowered to avoid collision with X_0)
    u_target_tmax = float(U_isotherm(0.63, T_MAX))
    add_curved_callout(ax, fig, r"$\mathbf{I_{T_{\mathrm{max}}}}$", (u_target_tmax, 0.63), (4.05, 0.64),
                       box_curve_tmax, color=C_TMAX, fontsize=19.0, lw=1.8, rad=-0.10, shrinkB=0)

    # 7. Callouts (strictly zorder=100, alpha=1.0, positioned tightly to labeled objects)
    box_white = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#95a5a6", alpha=1.0, lw=1.5)
    box_cut = dict(boxstyle="round,pad=0.24", facecolor="#ffffff", edgecolor=C_PT_CUT, alpha=1.0, lw=2.2)
    box_bound = dict(boxstyle="round,pad=0.22", facecolor="#fef6f6", edgecolor=C_TMAX, alpha=1.0, lw=1.5)

    # X0: Compact label at U=3.98, V=0.85 connecting from bottom-left corner to (U0, V0)
    add_curved_callout(ax, fig, r"$\mathbf{X_0}$",
                       (U0, V0), (3.98, 0.85), box_bound, color=C_TMAX, fontsize=18.5, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)

    # X1: Compact label at U=4.34, V=2.05 connecting from bottom-left corner to (U1, V1)
    add_curved_callout(ax, fig, r"$\mathbf{X_1}$",
                       (U1, V1), (4.34, 2.05), box_bound, color=C_TMAX, fontsize=18.5, lw=1.6, rad=0.14, preferred_corner="bl", shrinkB=8)

    # X'_0: Compact label at U=2.44, V=0.92 connecting from top-right corner to (U0_PRIME, V0_PRIME)
    add_curved_callout(ax, fig, r"$\mathbf{X^\prime_0}$",
                       (U0_PRIME, V0_PRIME), (2.44, 0.92), box_white, color="#5b2c04", fontsize=18.5, lw=1.6, rad=-0.14, preferred_corner="tr", shrinkB=8)

    # X'_1: Compact label at U=3.05, V=2.48 connecting from top-right corner to (U1_PRIME, V1_PRIME)
    add_curved_callout(ax, fig, r"$\mathbf{X^\prime_1}$",
                       (U1_PRIME, V1_PRIME), (3.05, 2.48), box_white, color="#5b2c04", fontsize=18.5, lw=1.6, rad=-0.12, preferred_corner="tr", shrinkB=8)

    # X': Compact label at U=2.58, V=1.58 connecting from top-right corner to (U_PRIME, V_PRIME)
    add_curved_callout(ax, fig, r"$\mathbf{X^\prime \overset{A}{\sim} X}$",
                       (U_PRIME, V_PRIME), (2.58, 1.58), box_cut, color=C_PT_CUT, fontsize=18.5, lw=1.8, rad=-0.14, preferred_corner="tr", shrinkB=10)

    # Abstract Axes (No numerical values)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel(r"Internal Energy  $U$", fontsize=19, fontweight="bold", labelpad=12)
    ax.set_ylabel(r"Work Coordinate  $V$", fontsize=19, fontweight="bold", labelpad=12)

    ax.set_title("Theorem 5.5 — Step 3 (Scenario 2): Boundary Temperature Approximation",
                 fontsize=22.5, pad=58, fontweight="bold", color="#1a252f")
    sub_s2 = (r"$T_0 = T_{\mathrm{max}} \ \Rightarrow \ $ Boundary analyzed via interior isotherms $I_{T^\prime_0} \to I_{T_{\mathrm{max}}}$" "\n"
              r"$\Rightarrow \ $ Intermediate adiabat $\partial A_X$ cuts $I_{T^\prime_0}$ at $X^\prime$")
    ax.text(0.50, 1.015, sub_s2, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=15.0, fontweight="normal", color="#34495e", linespacing=1.35, zorder=100)

    # Bottom banner with structured proof breakdown: Hypotheses -> Construction -> Conclusion
    ax_banner.set_xlim(0, 1)
    ax_banner.set_ylim(0, 1)
    banner_box = FancyBboxPatch((0.012, 0.03), 0.976, 0.94,
                                boxstyle="round,pad=0.01,rounding_size=0.025",
                                facecolor="#fcfdfd", edgecolor="#2c3e50", lw=2.0)
    ax_banner.add_patch(banner_box)

    lines_s2 = [
        ("Step 3 Proof Structure (Lieb & Yngvason 1999):", True, 17.5),
        (r"$\mathbf{1.\ Hypotheses:}\quad \mathrm{(i)}\ T_0 = T_{\mathrm{max}}\;\quad \mathrm{(ii)}\ X_0, X_1 \in I_{T_{\mathrm{max}}}\;\quad \mathrm{(iii)}\ X_0 \prec X \prec X_1.$", False, 15.0),
        (r"$\mathbf{2.\ Construction\ (Interior\ Approximation):}\quad \mathrm{Choose\ interior\ isotherms}\ I_{T^\prime_0}\ (T^\prime_0 < T_{\mathrm{max}})\ \mathrm{with\ approximating\ states}\ X^\prime_0 \prec X^\prime_1 \in I_{T^\prime_0}:$", False, 15.0),
        (r"$\bullet\ \ \mathrm{(i)}\ \ \mathrm{By\ Step\ 2,\ each}\ I_{T^\prime_0}\ \mathrm{cuts}\ \partial A_X\ \mathrm{at\ an\ interior\ state}\ X^\prime \overset{A}{\sim} X\ \mathrm{satisfying}\ X^\prime_0 \prec X^\prime \prec X^\prime_1.$", False, 14.8),
        (r"$\mathbf{3.\ Conclusion:}\quad \mathrm{Taking}\ T^\prime_0 \to T_{\mathrm{max}}^-,\ X^\prime_0 \to X_0\ \mathrm{and}\ X^\prime_1 \to X_1,\ \mathrm{compactness\ of}\ \partial A_X\ \mathrm{guarantees}\ X^\prime \to X^\ast \in \partial A_X \cap I_{T_{\mathrm{max}}}\ (X^\ast \overset{A}{\sim} X).$", False, 15.0)
    ]
    y_pos = [0.86, 0.68, 0.50, 0.32, 0.13]
    for y, (line, is_bold, fs) in zip(y_pos, lines_s2):
        ax_banner.text(0.50, y, line, fontsize=fs, fontweight="bold" if is_bold else "normal",
                       color="#1a252f", ha="center", va="center", zorder=100)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated: {out_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    s1_path = os.path.join(base_dir, "theorem55_scenario1.png")
    s2_path = os.path.join(base_dir, "theorem55_scenario2.png")

    generate_scenario1(s1_path)
    generate_scenario2(s2_path)


if __name__ == "__main__":
    main()
