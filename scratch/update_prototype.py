import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.legend_handler import HandlerBase

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Theorem_5-5"))
from theorem55_model import (
    U_isotherm, U_adiabat, entropy, temperature,
    A_PARAM, B_PARAM, CV, R_GAS
)

cmap_w2b = LinearSegmentedColormap.from_list("w2b", ["#ffffff", "#5dade2", "#1b4f72"])
cmap_w2r = LinearSegmentedColormap.from_list("w2r", ["#ffffff", "#f1948a", "#922b21"])

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

class GradientHandler(HandlerBase):
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

def add_curved_callout(ax, fig, text, xy_target, xy_box, bbox_style,
                       color="#1a252f", fontsize=13.5, rad=0.18, lw=1.8,
                       preferred_corner=None, shrinkB=8):
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

    conn = FancyArrowPatch(start_pt, xy_target,
                           arrowstyle="-",
                           connectionstyle=f"arc3,rad={rad}",
                           lw=lw, color=color,
                           shrinkA=0, shrinkB=shrinkB, zorder=30)
    ax.add_patch(conn)
    return t, conn

def add_dotted_arrow(ax, p0, p1, color, lw=4.2, mutation_scale=32, head_length=0.82, head_width=0.35,
                     shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9):
    style_str = f"-|>,head_length={head_length},head_width={head_width}"
    head = FancyArrowPatch(p0, p1, arrowstyle=style_str, mutation_scale=mutation_scale,
                           color=color, lw=0, shrinkA=shrinkA, shrinkB=shrinkB, zorder=zorder)
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

def add_region_label(ax, fig, x, y, text_type, color, fontsize=33):
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

def build_stacked_figure(out_path):
    # Figure height and GridSpec layout
    # Top margins and spacing tuned so title, subtitle, shared legend, panels, and banner look unified
    fig = plt.figure(figsize=(18, 24.5), dpi=300)
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.22, 5.0, 5.0, 1.80],
                           top=0.925, bottom=0.032, left=0.08, right=0.96, hspace=0.26)
    
    ax_legend = fig.add_subplot(gs[0])
    ax_legend.axis("off")
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax_banner = fig.add_subplot(gs[3])
    ax_banner.axis("off")

    # Common parameters
    T0 = 2.00
    u_min, u_max = 1.15, 5.35
    v_min, v_max = 0.55, 3.05

    v_grid = np.linspace(v_min, v_max, 600)
    u_grid_2d = np.linspace(u_min, u_max, 350)
    v_grid_2d = np.linspace(v_min, v_max, 350)
    U_2D, V_2D = np.meshgrid(u_grid_2d, v_grid_2d)
    T_2D = temperature(U_2D, V_2D)
    norm = TwoSlopeNorm(vmin=T_2D.min(), vcenter=T0, vmax=T_2D.max())

    # Points on I_T0
    V0 = 0.82
    U0 = float(U_isotherm(V0, T0))
    S0 = float(entropy(U0, V0))

    V1 = 2.50
    U1 = float(U_isotherm(V1, T0))
    S1 = float(entropy(U1, V1))

    S_MID = 0.52 * S0 + 0.48 * S1
    V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T0)) / R_GAS))
    U_PRIME = float(U_isotherm(V_PRIME, T0))

    u_iso_vals = U_isotherm(v_grid, T0)
    u_adia0_vals = U_adiabat(v_grid, S0)
    u_adiaX_vals = U_adiabat(v_grid, S_MID)
    u_adia1_vals = U_adiabat(v_grid, S1)

    C_ISO = "#c0392b"
    C_ADIA = "#000000"
    C_PT_ISO = "#b94a00"
    C_PT_CUT = "#c0392b"
    C_PT_GT = "#d68910"
    C_PT_LT = "#2980b9"

    box_curve_adia = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")
    box_curve_iso = dict(boxstyle="square,pad=0.10", facecolor="none", edgecolor="none")
    box_white = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#95a5a6", alpha=1.0, lw=1.5)
    box_cut = dict(boxstyle="round,pad=0.24", facecolor="#ffffff", edgecolor=C_PT_CUT, alpha=1.0, lw=2.2)
    box_gt = dict(boxstyle="round,pad=0.22", facecolor="#fefbf5", edgecolor="#d4ac0d", alpha=1.0, lw=1.6)
    box_lt = dict(boxstyle="round,pad=0.22", facecolor="#f0f7fb", edgecolor="#2980b9", alpha=1.0, lw=1.6)
    box_prime0 = dict(boxstyle="round,pad=0.22", facecolor="#fdfefe", edgecolor="#c0392b", alpha=1.0, lw=1.6)
    box_prime1 = dict(boxstyle="round,pad=0.22", facecolor="#fdfefe", edgecolor="#c0392b", alpha=1.0, lw=1.6)

    # -------------------------------------------------------------------------
    # SHARED LEGEND AT TOP
    # -------------------------------------------------------------------------
    p_blue_s1 = Patch(edgecolor="#1b4f72", lw=1.2)
    p_red_s1 = Patch(edgecolor="#922b21", lw=1.2)
    legend_elements = [
        Line2D([0], [0], color=C_ADIA, lw=3.4, ls="-"),
        Line2D([0], [0], color=C_ISO, lw=3.4, ls=(0, (4.5, 2.5))),
        p_blue_s1,
        p_red_s1,
    ]
    legend_labels = [
        r"$\mathbf{Adiabats}$",
        r"$\mathbf{Isotherms}$",
        r"$\mathbf{\mathcal{X}_- \ (T < T_0)}$",
        r"$\mathbf{\mathcal{X}_+ \ (T > T_0)}$",
    ]
    leg = ax_legend.legend(handles=legend_elements, labels=legend_labels,
                          handler_map={p_blue_s1: GradientHandler(cmap_w2b), p_red_s1: GradientHandler(cmap_w2r)},
                          loc="center", ncol=4, fontsize=16.5,
                          handlelength=3.5, handleheight=1.25, columnspacing=3.4,
                          framealpha=0.96, edgecolor="#2c3e50", fancybox=True, borderpad=0.7)
    leg.set_zorder(100)

    # =========================================================================
    # PANEL 1: CASE (a) - Coordinate Sections Exist: V_X0, V_X1 in rho(A_X)
    # =========================================================================
    ax1.set_xlim(u_min, u_max)
    ax1.set_ylim(v_min, v_max)
    ax1.contourf(U_2D, V_2D, T_2D, levels=80, cmap="coolwarm", norm=norm, alpha=0.22, zorder=1)

    U_GT = float(U_adiabat(V0, S_MID))
    U_LT = float(U_adiabat(V1, S_MID))

    mask0 = (u_adia0_vals >= u_min) & (u_adia0_vals <= u_max)
    ax1.plot(u_adia0_vals[mask0], v_grid[mask0], color=C_ADIA, lw=2.4, ls="-", zorder=5)
    mask1 = (u_adia1_vals >= u_min) & (u_adia1_vals <= u_max)
    ax1.plot(u_adia1_vals[mask1], v_grid[mask1], color=C_ADIA, lw=2.4, ls="-", zorder=5)
    maskX = (u_adiaX_vals >= u_min) & (u_adiaX_vals <= u_max)
    ax1.plot(u_adiaX_vals[maskX], v_grid[maskX], color=C_ADIA, lw=3.8, ls="-", zorder=6)
    mask_iso = (u_iso_vals >= u_min) & (u_iso_vals <= u_max)
    ax1.plot(u_iso_vals[mask_iso], v_grid[mask_iso], color=C_ISO, lw=3.6, ls=(0, (6, 3.5)), zorder=7)

    # Points
    ax1.scatter([U0], [V0], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax1.scatter([U1], [V1], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax1.scatter([U_GT], [V0], color=C_PT_GT, s=210, zorder=15, edgecolors="#6e4402", lw=2.2)
    ax1.scatter([U_LT], [V1], color=C_PT_LT, s=210, zorder=15, edgecolors="#104266", lw=2.2)
    ax1.scatter([U_PRIME], [V_PRIME], color=C_PT_CUT, s=320, zorder=16, marker="o", edgecolors="black", lw=3.0)

    # Regions
    add_region_label(ax1, fig, 1.80, 1.45, "minus", "#1b4f72", fontsize=33)
    add_region_label(ax1, fig, 4.35, 1.95, "plus", "#922b21", fontsize=33)

    # Arrows
    add_dotted_arrow(ax1, (U0, V0), (U_GT, V0), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)
    box_vol0 = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#c0392b", lw=1.5, alpha=1.0)
    ax1.text(0.5 * (U0 + U_GT), V0 + 0.098, r"$\mathbf{V(X_>) = V_{X_0} \in \rho(A_X)}$",
             fontsize=15.0, fontweight="bold", color="#922b21", ha="center", va="center", zorder=100, bbox=box_vol0)

    add_dotted_arrow(ax1, (U1, V1), (U_LT, V1), color="#2471a3", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)
    box_vol1 = dict(boxstyle="round,pad=0.18", facecolor="#f0f7fb", edgecolor="#2471a3", lw=1.5, alpha=1.0)
    ax1.text(0.5 * (U_LT + U1), V1 + 0.098, r"$\mathbf{V(X_<) = V_{X_1} \in \rho(A_X)}$",
             fontsize=15.0, fontweight="bold", color="#154360", ha="center", va="center", zorder=100, bbox=box_vol1)

    # Projections
    ax1.plot([u_min, U0], [V0, V0], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax1.plot([U0, U0], [v_min, V0], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax1.plot([U_GT, U_GT], [v_min, V0], color="#784212", ls=":", lw=1.6, zorder=4)
    ax1.plot([u_min, U_LT], [V1, V1], color="#1b4f72", ls=":", lw=1.6, zorder=4)
    ax1.plot([U_LT, U_LT], [v_min, V1], color="#1b4f72", ls=":", lw=1.6, zorder=4)
    ax1.plot([U1, U1], [v_min, V1], color="#4e2103", ls=":", lw=1.6, zorder=4)

    ax1.set_xticks([U_LT, U0, U1, U_GT])
    ax1.set_xticklabels([r"$\mathbf{U(X_<)}$", r"$\mathbf{U(X_0)}$", r"$\mathbf{U(X_1)}$", r"$\mathbf{U(X_>)}$"], fontsize=14, fontweight="bold")
    ax1.set_yticks([V0, V1])
    ax1.set_yticklabels([r"$\mathbf{V_{X_0}}$", r"$\mathbf{V_{X_1}}$"], fontsize=14, fontweight="bold")

    # Curve labels
    u_target_x = float(U_adiabat(2.84, S_MID))
    add_curved_callout(ax1, fig, r"$\mathbf{\partial A_X}$", (u_target_x, 2.84), (1.28, 2.95), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    u_target_0 = float(U_adiabat(2.15, S0))
    add_curved_callout(ax1, fig, r"$\mathbf{\partial A_{X_0}}$", (u_target_0, 2.15), (1.46, 2.22), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=0.12, shrinkB=0)
    u_target_1 = float(U_adiabat(1.20, S1))
    add_curved_callout(ax1, fig, r"$\mathbf{\partial A_{X_1}}$", (u_target_1, 1.20), (4.95, 1.35), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    u_target_iso = float(U_isotherm(2.92, T0))
    add_curved_callout(ax1, fig, r"$\mathbf{I_{T_0}}$", (u_target_iso, 2.92), (3.12, 2.96), box_curve_iso, color=C_ISO, fontsize=18.0, lw=1.6, rad=-0.12, shrinkB=0)

    # Point callouts
    add_curved_callout(ax1, fig, r"$\mathbf{X_0}$", (U0, V0), (1.96, 0.98), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="br", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X_1}$", (U1, V1), (3.24, 2.54), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="tl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X_<}$", (U_LT, V1), (2.06, 2.36), box_lt, color="#1b4f72", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="tl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X_>}$", (U_GT, V0), (3.88, 0.95), box_gt, color="#784212", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X^\prime \overset{A}{\sim} X}$", (U_PRIME, V_PRIME), (2.98, 1.57), box_cut, color=C_PT_CUT, fontsize=17.5, lw=1.8, rad=-0.14, preferred_corner="bl", shrinkB=10)

    ax1.set_ylabel(r"Work Coordinate  $V$", fontsize=18, fontweight="bold", labelpad=10)
    ax1.set_title(r"$\mathbf{Case\ (a):\ Coordinate\ Sections\ Exist\ —\ V_{X_0}, V_{X_1} \in \rho(A_X)}$" "\n"
                  r"$\text{Direct coordinate slices on }\partial A_X\text{ produce } X_> \in \mathcal{X}_+ \text{ and } X_< \in \mathcal{X}_-$",
                  fontsize=18.5, pad=12, fontweight="bold", color="#1a252f")

    # =========================================================================
    # PANEL 2: CASE (b) - Coordinate Line Misses Adiabat: V_X0 notin rho(A_X)
    # =========================================================================
    ax2.set_xlim(u_min, u_max)
    ax2.set_ylim(v_min, v_max)
    ax2.contourf(U_2D, V_2D, T_2D, levels=80, cmap="coolwarm", norm=norm, alpha=0.22, zorder=1)

    # Truncated adiabat dA_X on rho(A_X) = [1.22, 3.05] (so V0 = 0.82 notin rho(A_X))
    v_rho_min = 1.22
    mask_rho = (v_grid >= v_rho_min)
    ax2.plot(u_adiaX_vals[mask_rho], v_grid[mask_rho], color=C_ADIA, lw=3.8, ls="-", zorder=6)
    ax2.plot(u_adia0_vals[mask0], v_grid[mask0], color=C_ADIA, lw=2.4, ls="-", zorder=5)
    ax2.plot(u_adia1_vals[mask1], v_grid[mask1], color=C_ADIA, lw=2.4, ls="-", zorder=5)
    ax2.plot(u_iso_vals[mask_iso], v_grid[mask_iso], color=C_ISO, lw=3.6, ls=(0, (6, 3.5)), zorder=7)

    # Sub-step (a): Energy line ell over V_X0 is trapped below dA_X (S < S(X))
    ax2.plot([u_min, u_max], [V0, V0], color="#784212", ls="--", lw=2.0, zorder=4)
    
    # Label for line ell at right edge of plot
    box_ell_tag = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#784212", lw=1.4, alpha=1.0)
    ax2.text(u_max - 0.08, V0 - 0.075,
             r"$\mathbf{\ell = \{(U, V_{X_0})\}\ \subset\ \{S < S(X)\}\ (Trapped\ Line)}$",
             fontsize=13.0, fontweight="bold", color="#784212", ha="right", va="top", zorder=100, bbox=box_ell_tag)

    # Sub-step (b): Hot state X'_0 on ell (T5 + T0 < Tmax gives T(X'_0) > T0, yet X'_0 << X)
    U0_PRIME = 3.35
    add_dotted_arrow(ax2, (U0, V0), (U0_PRIME, V0), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)

    box_arrow0 = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#c0392b", lw=1.5, alpha=1.0)
    ax2.text(0.5 * (U0 + U0_PRIME), V0 + 0.10,
             r"$\mathbf{Axiom\ T5:\ X^\prime_0 \in \Omega_>\ (X^\prime_0 \prec\prec X)}$",
             fontsize=14.0, fontweight="bold", color="#922b21", ha="center", va="center", zorder=100, bbox=box_arrow0)

    # Sub-step (c): Hot state X'_1 above X_1 on line V_X1 (Planck + Monotonicity + T5)
    U1_PRIME = 3.65
    add_dotted_arrow(ax2, (U1, V1), (U1_PRIME, V1), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)
    box_arrow1 = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#c0392b", lw=1.5, alpha=1.0)
    ax2.text(0.5 * (U1 + U1_PRIME), V1 + 0.10,
             r"$\mathbf{Planck\ and\ T5:\ X^\prime_1 \in \Omega_>\ (X \prec\prec X^\prime_1)}$",
             fontsize=14.0, fontweight="bold", color="#922b21", ha="center", va="center", zorder=100, bbox=box_arrow1)

    # Sub-step (e): Points X_uparrow in Omega_> cap dA_X and X_downarrow in Omega_< cap dA_X
    U_UP = float(U_adiabat(v_rho_min, S_MID))
    V_UP = v_rho_min
    V_DOWN = 2.15
    U_DOWN = float(U_adiabat(V_DOWN, S_MID))

    # Scatter points for Case (b)
    ax2.scatter([U0], [V0], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax2.scatter([U1], [V1], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax2.scatter([U0_PRIME], [V0], color="#e74c3c", s=210, zorder=15, edgecolors="#78281f", lw=2.2)
    ax2.scatter([U1_PRIME], [V1], color="#e74c3c", s=210, zorder=15, edgecolors="#78281f", lw=2.2)
    ax2.scatter([U_UP], [V_UP], color=C_PT_GT, s=210, zorder=15, edgecolors="#6e4402", lw=2.2)
    ax2.scatter([U_DOWN], [V_DOWN], color=C_PT_LT, s=210, zorder=15, edgecolors="#104266", lw=2.2)
    ax2.scatter([U_PRIME], [V_PRIME], color=C_PT_CUT, s=320, zorder=16, marker="o", edgecolors="black", lw=3.0)

    # Region Labels for Case (b)
    add_region_label(ax2, fig, 1.80, 1.45, "minus", "#1b4f72", fontsize=33)
    add_region_label(ax2, fig, 4.35, 1.95, "plus", "#922b21", fontsize=33)

    # Projections on Case (b)
    ax2.plot([u_min, U0], [V0, V0], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([U0, U0], [v_min, V0], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([U0_PRIME, U0_PRIME], [v_min, V0], color="#922b21", ls=":", lw=1.6, zorder=4)
    ax2.plot([u_min, U1], [V1, V1], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([U1, U1], [v_min, V1], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([U1_PRIME, U1_PRIME], [v_min, V1], color="#922b21", ls=":", lw=1.6, zorder=4)

    # Tick marks for Case (b)
    ax2.set_xticks([U0, U0_PRIME, U1, U1_PRIME])
    ax2.set_xticklabels([r"$\mathbf{U(X_0)}$", r"$\mathbf{U(X^\prime_0)}$", r"$\mathbf{U(X_1)}$", r"$\mathbf{U(X^\prime_1)}$"], fontsize=14, fontweight="bold")
    ax2.set_yticks([V0, v_rho_min, V1])
    ax2.set_yticklabels([r"$\mathbf{V_{X_0} \notin \rho(A_X)}$", r"$\mathbf{\inf \rho(A_X)}$", r"$\mathbf{V_{X_1}}$"], fontsize=14, fontweight="bold")

    # Visual bracket / range of rho(A_X) along the y-axis spine
    ax2.annotate("", xy=(u_min, v_max), xytext=(u_min, v_rho_min),
                 arrowprops=dict(arrowstyle="-", color="#2c3e50", lw=5.0, capstyle="butt"))
    # Small tick caps at inf rho(A_X)
    ax2.plot([u_min - 0.04, u_min + 0.04], [v_rho_min, v_rho_min], color="#2c3e50", lw=2.5, clip_on=False, zorder=10)

    # Callouts for Case (b)
    # X0: to the left of X0
    add_curved_callout(ax2, fig, r"$\mathbf{X_0}$", (U0, V0), (1.92, 0.98), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="br", shrinkB=8)
    # X1: above X1
    add_curved_callout(ax2, fig, r"$\mathbf{X_1}$", (U1, V1), (2.72, 2.72), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=0.10, preferred_corner="bl", shrinkB=8)
    # X'_0: to the right of X'_0
    add_curved_callout(ax2, fig, r"$\mathbf{X^\prime_0}$", (U0_PRIME, V0), (3.95, 0.95), box_prime0, color="#922b21", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)
    # X'_1: to the right of X'_1
    add_curved_callout(ax2, fig, r"$\mathbf{X^\prime_1}$", (U1_PRIME, V1), (4.20, 2.65), box_prime1, color="#922b21", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="tl", shrinkB=8)
    # X_uparrow: on bottom tip of dA_X
    add_curved_callout(ax2, fig, r"$\mathbf{X_\uparrow \in \partial A_X \cap \Omega_>}$", (U_UP, V_UP), (3.65, 1.32), box_gt, color="#784212", fontsize=15.5, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)
    # X_downarrow: in open corridor between dA_X and I_T0
    add_curved_callout(ax2, fig, r"$\mathbf{X_\downarrow \in \partial A_X \cap \Omega_<}$", (U_DOWN, V_DOWN), (2.12, 2.32), box_lt, color="#1b4f72", fontsize=15.0, lw=1.6, rad=0.12, preferred_corner="bl", shrinkB=8)
    # X' cut point
    add_curved_callout(ax2, fig, r"$\mathbf{X^\prime \overset{A}{\sim} X}$", (U_PRIME, V_PRIME), (2.98, 1.57), box_cut, color=C_PT_CUT, fontsize=17.5, lw=1.8, rad=-0.14, preferred_corner="bl", shrinkB=10)

    # Curve labels for Case (b)
    add_curved_callout(ax2, fig, r"$\mathbf{\partial A_X}$", (u_target_x, 2.84), (1.28, 2.95), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax2, fig, r"$\mathbf{\partial A_{X_0}}$", (u_target_0, 2.15), (1.46, 2.22), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=0.12, shrinkB=0)
    add_curved_callout(ax2, fig, r"$\mathbf{\partial A_{X_1}}$", (u_target_1, 1.20), (4.95, 1.35), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax2, fig, r"$\mathbf{I_{T_0}}$", (u_target_iso, 2.92), (3.12, 2.96), box_curve_iso, color=C_ISO, fontsize=18.0, lw=1.6, rad=-0.12, shrinkB=0)

    ax2.set_xlabel(r"Internal Energy  $U$", fontsize=18, fontweight="bold", labelpad=10)
    ax2.set_ylabel(r"Work Coordinate  $V$", fontsize=18, fontweight="bold", labelpad=10)
    ax2.set_title(r"$\mathbf{Case\ (b):\ Coordinate\ Line\ Misses\ Adiabat\ —\ V_{X_0} \notin \rho(A_X)\ (Topological\ Separation)}$" "\n"
                  r"$\text{Trapped line }\ell \text{ below }\partial A_X\text{ + Axiom T5 }\Rightarrow X^\prime_0, X^\prime_1 \in \Omega_>\text{ straddle }\partial A_X \Rightarrow X_\uparrow \in \partial A_X \cap \Omega_>$",
                  fontsize=18.5, pad=12, fontweight="bold", color="#1a252f")

    # Main super title
    fig.suptitle("Theorem 5.5 — Step 2 (Scenario 1): Interior Transversal Crossing\n"
                 r"$T_{\min} < T_0 < T_{\max} \ \Rightarrow \ \text{Isotherm } I_{T_0} \text{ separates } \mathcal{X}_- \text{ from } \mathcal{X}_+ \ \Rightarrow \ \text{Intermediate adiabat } \partial A_X \text{ cuts } I_{T_0} \text{ at } X^\prime$",
                 fontsize=21.5, fontweight="bold", color="#1a252f", y=0.978)

    # =========================================================================
    # BANNER: UNIFIED PROOF STRUCTURE (Sharing both cases)
    # =========================================================================
    ax_banner.set_xlim(0, 1)
    ax_banner.set_ylim(0, 1)
    banner_box = FancyBboxPatch((0.012, 0.03), 0.976, 0.94,
                                boxstyle="round,pad=0.01,rounding_size=0.025",
                                facecolor="#fcfdfd", edgecolor="#2c3e50", lw=2.0)
    ax_banner.add_patch(banner_box)

    lines_proof = [
        ("Step 2 Proof Structure — Comprehensive Exhaustion of All Coordinate Geometry (Lieb & Yngvason 1999):", True, 17.0),
        (r"$\mathbf{1.\ Hypotheses:}\quad \mathrm{(i)}\ T_{\min} < T_0 < T_{\max}\ \mathrm{is\ an\ interior\ isotherm;}\quad \mathrm{(ii)}\ X_0, X_1 \in I_{T_0}\;\quad \mathrm{(iii)}\ X_0 \prec X \prec X_1.$", False, 14.8),
        (r"$\mathbf{2.\ Construction\ of\ Straddling\ States\ on}\ \partial A_X:\quad \mathrm{Two\ mutually\ exhaustive\ geometric\ cases\ arise:}$", False, 14.8),
        (r"$\bullet\ \ \mathbf{Case\ (a)\ [V_{X_0} \in \rho(A_X)]:\ }\mathrm{Constant\ volume\ slices\ yield}\ X_0 \prec X_>\ (V=V_{X_0})\ \Rightarrow\ X_> \in \mathcal{X}_+\ \mathrm{and}\ X_< \prec X_1\ (V=V_{X_1})\ \Rightarrow\ X_< \in \mathcal{X}_-.$", False, 14.5),
        (r"$\bullet\ \ \mathbf{Case\ (b)\ [V_{X_0} \notin \rho(A_X)]:\ }\mathrm{Line}\ \ell\ \mathrm{over}\ V_{X_0}\ \mathrm{is\ trapped\ at}\ S < S(X).\ \mathrm{By\ T5,\ hot\ states}\ X^\prime_0, X^\prime_1 \in \Omega_>\ \mathrm{straddle}\ \partial A_X;\ \mathrm{connected}\ \Omega_>\ \mathrm{cuts}\ \partial A_X\ \mathrm{at}\ X_\uparrow \in \mathcal{X}_+.$", False, 14.5),
        (r"$\mathbf{3.\ Conclusion:}\quad \mathrm{In\ both\ cases,\ connected}\ \partial A_X\ \mathrm{joins}\ \mathcal{X}_+\ \mathrm{to}\ \mathcal{X}_-.\ \mathrm{By\ continuity\ of}\ T\ (\mathrm{IVT}),\ \partial A_X\ \mathrm{cuts}\ I_{T_0}\ \mathrm{at\ a\ state}\ X^\prime \in I_{T_0}\ \mathrm{with}\ X^\prime \overset{A}{\sim} X.$", False, 14.8)
    ]
    y_pos = [0.88, 0.72, 0.56, 0.40, 0.25, 0.09]
    for y, (line, is_bold, fs) in zip(y_pos, lines_proof):
        ax_banner.text(0.50, y, line, fontsize=fs, fontweight="bold" if is_bold else "normal",
                       color="#1a252f", ha="center", va="center", zorder=100)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated stacked test plot: {out_path}")

build_stacked_figure("scratch/test_stacked_scenario1.png")
