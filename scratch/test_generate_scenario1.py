import os
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

CV = 1.5
R_GAS = 1.0
A_PARAM = 2.0
B_PARAM = 0.5
T0 = 2.00

def entropy(U, V):
    U = np.asarray(U)
    V = np.asarray(V)
    return CV * np.log(U + A_PARAM / V) + R_GAS * np.log(V - B_PARAM)

def temperature(U, V):
    U = np.asarray(U)
    V = np.asarray(V)
    return (U + A_PARAM / V) / CV

def U_isotherm(V, T=T0):
    V = np.asarray(V)
    return CV * T - A_PARAM / V

def U_adiabat(V, S_const):
    V = np.asarray(V)
    return np.exp((S_const - R_GAS * np.log(V - B_PARAM)) / CV) - A_PARAM / V

# Ceiling parameters
H0 = 2.1646
H1 = 0.70

def ceiling(V):
    V = np.asarray(V)
    return H0 + H1 * V

def ceiling_entropy(V):
    return entropy(ceiling(V), V)

def rho_min(S_const):
    def obj(v):
        return ceiling_entropy(v) - S_const
    return brentq(obj, B_PARAM + 1e-4, 10.0)

# Anchor states
V0b = 0.82
U0b = float(U_isotherm(V0b))
S0b = float(entropy(U0b, V0b))

V1 = 2.50
U1 = float(U_isotherm(V1))
S1 = float(entropy(U1, V1))

V0a = 1.15
U0a = float(U_isotherm(V0a))
S0a = float(entropy(U0a, V0a))

S_MID = 0.52 * S0a + 0.48 * S1
V_B = float(rho_min(S_MID))
V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T0)) / R_GAS))
U_PRIME = float(U_isotherm(V_PRIME))

# Verification of V_B
print(f"Computed V_B = {V_B:.4f}, expected ~ 1.05")
assert abs(ceiling_entropy(V_B) - S_MID) < 1e-9

def add_dotted_arrow(ax, p0, p1, color, lw=4.2, mutation_scale=32, head_length=0.82, head_width=0.35,
                     shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9):
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

print("Setup completed successfully.")

def generate_test_figure(out_path):
    fig = plt.figure(figsize=(18, 24.5), dpi=300)
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.22, 5.0, 5.0, 1.80],
                           top=0.925, bottom=0.032, left=0.08, right=0.96, hspace=0.26)

    ax_legend = fig.add_subplot(gs[0])
    ax_legend.axis("off")
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax_banner = fig.add_subplot(gs[3])
    ax_banner.axis("off")

    v_min, v_max = 0.55, 3.05
    u_min, u_max = 0.20, 3.80

    v_grid = np.linspace(v_min, v_max, 600)
    v_grid_2d = np.linspace(v_min, v_max, 350)
    u_grid_2d = np.linspace(u_min, u_max, 350)
    V_2D, U_2D = np.meshgrid(v_grid_2d, u_grid_2d)
    T_2D = temperature(U_2D, V_2D)
    norm = TwoSlopeNorm(vmin=T_2D.min(), vcenter=T0, vmax=T_2D.max())

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

    # SHARED LEGEND
    p_blue_s1 = Patch(edgecolor="#1b4f72", lw=1.2)
    p_red_s1 = Patch(edgecolor="#922b21", lw=1.2)
    p_gamma = Line2D([0], [0], color=C_GAMMA, lw=2.8, ls="--")
    legend_elements = [
        Line2D([0], [0], color=C_ADIA, lw=3.4, ls="-"),
        Line2D([0], [0], color=C_ISO, lw=3.4, ls=(0, (4.5, 2.5))),
        p_gamma,
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
                          loc="center", ncol=5, fontsize=15.5,
                          handlelength=3.0, handleheight=1.25, columnspacing=2.6,
                          framealpha=0.96, edgecolor="#2c3e50", fancybox=True, borderpad=0.7)
    leg.set_zorder(100)

    # -------------------------------------------------------------------------
    # Helper to draw exterior Gamma hatch
    # -------------------------------------------------------------------------
    def draw_exterior(ax):
        v_ext = np.linspace(v_min, v_max, 200)
        h_ext = ceiling(v_ext)
        verts = [(v_min, u_max)]
        for vv, hh in zip(v_ext, h_ext):
            verts.append((vv, min(hh, u_max)))
        verts.append((v_max, u_max))
        poly = Polygon(verts, closed=True, facecolor="#eaecee", edgecolor="#7f8c8d",
                       hatch="//", alpha=0.60, lw=1.0, zorder=3)
        ax.add_patch(poly)
        # Plot ceiling boundary
        mask_ceil = (h_ext <= u_max + 0.5)
        ax.plot(v_ext[mask_ceil], h_ext[mask_ceil], color=C_GAMMA, lw=3.0, ls="--", zorder=8)

    # =========================================================================
    # PANEL 1: CASE (a) — Coordinate Sections Exist: V_X0a, V_X1 in rho(A_X)
    # =========================================================================
    ax1.set_xlim(v_min, v_max)
    ax1.set_ylim(u_min, u_max)

    # Shading clipped to U < H(V)
    T_2D_masked1 = np.copy(T_2D)
    T_2D_masked1[U_2D > ceiling(V_2D)] = np.nan
    ax1.contourf(V_2D, U_2D, T_2D_masked1, levels=80, cmap="coolwarm", norm=norm, alpha=0.22, zorder=1)
    draw_exterior(ax1)

    # Ceil label in Panel 1
    v_clabel = 2.05
    ax1.text(v_clabel, min(ceiling(v_clabel) + 0.08, u_max - 0.12),
             r"$\mathbf{\partial\Gamma\ (boundary\ of\ state\ space\ —\ not\ states)}$",
             fontsize=13.0, fontweight="bold", color=C_GAMMA, ha="center", va="bottom", zorder=10)

    # Level curves in Panel 1: clipped to U < H(V)
    u_iso = U_isotherm(v_grid, T0)
    u_adia0a = U_adiabat(v_grid, S0a)
    u_adia1 = U_adiabat(v_grid, S1)
    u_adiaX = U_adiabat(v_grid, S_MID)

    mask_iso = (u_iso >= u_min) & (u_iso <= ceiling(v_grid))
    ax1.plot(v_grid[mask_iso], u_iso[mask_iso], color=C_ISO, lw=3.6, ls=(0, (6, 3.5)), zorder=7)

    mask0a = (u_adia0a >= u_min) & (u_adia0a <= ceiling(v_grid)) & (u_adia0a <= u_max)
    ax1.plot(v_grid[mask0a], u_adia0a[mask0a], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    mask1 = (u_adia1 >= u_min) & (u_adia1 <= ceiling(v_grid)) & (u_adia1 <= u_max)
    ax1.plot(v_grid[mask1], u_adia1[mask1], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    maskX = (u_adiaX >= u_min) & (u_adiaX <= ceiling(v_grid)) & (u_adiaX <= u_max)
    ax1.plot(v_grid[maskX], u_adiaX[maskX], color=C_ADIA, lw=3.8, ls="-", zorder=6)

    # Panel 1 Points
    U_GT_a = float(U_adiabat(V0a, S_MID))
    U_LT_a = float(U_adiabat(V1, S_MID))

    ax1.scatter([V0a], [U0a], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax1.scatter([V1], [U1], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax1.scatter([V0a], [U_GT_a], color=C_PT_GT, s=210, zorder=15, edgecolors="#6e4402", lw=2.2)
    ax1.scatter([V1], [U_LT_a], color=C_PT_LT, s=210, zorder=15, edgecolors="#104266", lw=2.2)
    ax1.scatter([V_PRIME], [U_PRIME], color=C_PT_CUT, s=320, zorder=16, marker="o", edgecolors="black", lw=3.0)

    # Labels
    add_region_label(ax1, fig, 1.85, 0.42, "minus", "#1b4f72", fontsize=33)
    add_region_label(ax1, fig, 1.55, 3.10, "plus", "#922b21", fontsize=33)

    # Arrows
    add_dotted_arrow(ax1, (V0a, U0a), (V0a, U_GT_a), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)
    box_vol0a = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#c0392b", lw=1.5, alpha=1.0)
    ax1.text(V0a + 0.08, 0.5 * (U0a + U_GT_a), r"$\mathbf{V(X_>) = V_{X_0} \in \rho(A_X)}$" "\n" r"$\mathbf{(Heating\ Slice)}$",
             fontsize=13.0, fontweight="bold", color="#922b21", ha="left", va="center", zorder=100, bbox=box_vol0a)

    add_dotted_arrow(ax1, (V1, U1), (V1, U_LT_a), color="#2471a3", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)
    box_vol1 = dict(boxstyle="round,pad=0.18", facecolor="#f0f7fb", edgecolor="#2471a3", lw=1.5, alpha=1.0)
    ax1.text(V1 - 0.28, 0.5 * (U_LT_a + U1), r"$\mathbf{V(X_<) = V_{X_1} \in \rho(A_X)}$" "\n" r"$\mathbf{(Cooling\ Slice)}$",
             fontsize=13.5, fontweight="bold", color="#154360", ha="right", va="center", zorder=100, bbox=box_vol1)

    # Projections
    ax1.plot([V0a, V0a], [u_min, U0a], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax1.plot([v_min, V0a], [U0a, U0a], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax1.plot([v_min, V0a], [U_GT_a, U_GT_a], color="#784212", ls=":", lw=1.6, zorder=4)
    ax1.plot([V1, V1], [u_min, U_LT_a], color="#1b4f72", ls=":", lw=1.6, zorder=4)
    ax1.plot([v_min, V1], [U_LT_a, U_LT_a], color="#1b4f72", ls=":", lw=1.6, zorder=4)
    ax1.plot([v_min, V1], [U1, U1], color="#4e2103", ls=":", lw=1.6, zorder=4)

    ax1.set_xticks([V0a, V1])
    ax1.set_xticklabels([r"$\mathbf{V_{X_0}}$", r"$\mathbf{V_{X_1}}$"], fontsize=14, fontweight="bold")
    ax1.set_yticks([U0a, U_LT_a, U1, U_GT_a])
    ax1.set_yticklabels([r"$\mathbf{U(X_0)}$", r"$\mathbf{U(X_<)}$", r"$\mathbf{U(X_1)}$", r"$\mathbf{U(X_>)}$"], fontsize=14, fontweight="bold")

    # Callouts
    add_curved_callout(ax1, fig, r"$\mathbf{\partial A_X}$", (2.84, float(U_adiabat(2.84, S_MID))), (2.95, 1.28), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax1, fig, r"$\mathbf{\partial A_{X_0}}$", (2.15, float(U_adiabat(2.15, S0a))), (2.35, 0.40), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=0.12, shrinkB=0)
    add_curved_callout(ax1, fig, r"$\mathbf{\partial A_{X_1}}$", (1.70, float(U_adiabat(1.70, S1))), (1.85, 3.65), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax1, fig, r"$\mathbf{I_{T_0}}$", (2.92, float(U_isotherm(2.92, T0))), (2.96, 2.72), box_curve_iso, color=C_ISO, fontsize=18.0, lw=1.6, rad=-0.12, shrinkB=0)

    add_curved_callout(ax1, fig, r"$\mathbf{X_0}$", (V0a, U0a), (1.00, 0.55), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="tr", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X_1}$", (V1, U1), (2.75, 2.45), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X_<}$", (V1, U_LT_a), (2.75, 0.65), box_lt, color="#1b4f72", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="tl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X_>}$", (V0a, U_GT_a), (0.95, 3.10), box_gt, color="#784212", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax1, fig, r"$\mathbf{X^\prime \overset{A}{\sim} X}$", (V_PRIME, U_PRIME), (1.80, 1.25), box_cut, color=C_PT_CUT, fontsize=17.5, lw=1.8, rad=-0.14, preferred_corner="tl", shrinkB=10)

    ax1.set_ylabel(r"Internal Energy  $U$", fontsize=18, fontweight="bold", labelpad=10)
    ax1.set_title(r"$\mathbf{Case\ (a):\ Coordinate\ Sections\ Exist\ —\ V_{X_0}, V_{X_1} \in \rho(A_X)}$" "\n"
                  r"$\text{Constant-volume vertical slices on }\partial A_X\text{ produce } X_> \in \mathcal{X}_+ \text{ and } X_< \in \mathcal{X}_-$",
                  fontsize=18.5, pad=12, fontweight="bold", color="#1a252f")

    # =========================================================================
    # PANEL 2: CASE (b) — Coordinate Line Misses Adiabat: V_X0b notin rho(A_X)
    # =========================================================================
    ax2.set_xlim(v_min, v_max)
    ax2.set_ylim(u_min, u_max)

    # Contour clipped to U < H(V)
    T_2D_masked2 = np.copy(T_2D)
    T_2D_masked2[U_2D > ceiling(V_2D)] = np.nan
    ax2.contourf(V_2D, U_2D, T_2D_masked2, levels=80, cmap="coolwarm", norm=norm, alpha=0.22, zorder=1)
    draw_exterior(ax2)

    # Ceil label in Panel 2
    ax2.text(v_clabel, min(ceiling(v_clabel) + 0.08, u_max - 0.12),
             r"$\mathbf{\partial\Gamma\ (boundary\ of\ state\ space\ —\ not\ states)}$",
             fontsize=13.0, fontweight="bold", color=C_GAMMA, ha="center", va="bottom", zorder=10)

    # Level curves in Panel 2:
    # Isotherm I_T0 clipped to U < H(V)
    ax2.plot(v_grid[mask_iso], u_iso[mask_iso], color=C_ISO, lw=3.6, ls=(0, (6, 3.5)), zorder=7)

    # Adiabats clipped to U < H(V)
    u_adia0b = U_adiabat(v_grid, S0b)
    mask0b = (u_adia0b >= u_min) & (u_adia0b <= ceiling(v_grid)) & (u_adia0b <= u_max)
    ax2.plot(v_grid[mask0b], u_adia0b[mask0b], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    # dA_X ends exactly ON the dashed ceiling at V_B
    v_grid_adiaX = np.linspace(V_B, v_max, 500)
    u_grid_adiaX = U_adiabat(v_grid_adiaX, S_MID)
    maskX_b = (u_grid_adiaX >= u_min) & (u_grid_adiaX <= ceiling(v_grid_adiaX)) & (u_grid_adiaX <= u_max)
    ax2.plot(v_grid_adiaX[maskX_b], u_grid_adiaX[maskX_b], color=C_ADIA, lw=3.8, ls="-", zorder=6)

    # Open circle at exit of dA_X through dGamma
    U_EXIT = float(ceiling(V_B))
    ax2.scatter([V_B], [U_EXIT], s=140, facecolors="white", edgecolors=C_ADIA, lw=2.8, zorder=20)
    # Annotated exit
    add_curved_callout(ax2, fig, r"$\mathbf{exit\ of\ \partial A_X\ through\ \partial\Gamma\ (limit,\ not\ a\ state)}$",
                       (V_B, U_EXIT), (1.48, 3.35), box_white, color=C_GAMMA, fontsize=12.5, lw=1.5, rad=0.10, preferred_corner="bl", shrinkB=6)

    # dA_X1 ends on ceiling at larger V
    V_B1 = float(rho_min(S1))
    v_grid_adia1 = np.linspace(V_B1, v_max, 500)
    u_grid_adia1 = U_adiabat(v_grid_adia1, S1)
    mask1_b = (u_grid_adia1 >= u_min) & (u_grid_adia1 <= ceiling(v_grid_adia1)) & (u_grid_adia1 <= u_max)
    ax2.plot(v_grid_adia1[mask1_b], u_grid_adia1[mask1_b], color=C_ADIA, lw=2.4, ls="-", zorder=5)

    # Trapped line ell over V0b from floor (u_min) up to ceiling H(V0b) ONLY
    H_0b = float(ceiling(V0b))
    ax2.plot([V0b, V0b], [u_min, H_0b], color="#784212", ls="--", lw=2.2, zorder=4)
    # Cap tick where ell meets dGamma
    tick_w = 0.03
    ax2.plot([V0b - tick_w, V0b + tick_w], [H_0b, H_0b], color="#784212", lw=3.0, zorder=12)

    # Label for trapped line ell
    box_ell_tag = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#784212", lw=1.4, alpha=1.0)
    ax2.text(V0b - 0.04, H_0b - 0.15,
             r"$\mathbf{\ell = \{(V_{X_0}, U)\}\ \subset\ \{S < S(X)\}}$" "\n"
             r"$\mathbf{max\ entropy\ over\ this\ column = \Sigma(V_{X_0}) < S(X)}$",
             fontsize=12.0, fontweight="bold", color="#784212", ha="right", va="top", zorder=100, bbox=box_ell_tag)

    # Points on ell: X0b and X'_0
    U0_prime = float(0.5 * (U0b + H_0b))
    add_dotted_arrow(ax2, (V0b, U0b), (V0b, U0_prime), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)

    # Points on V1: X1 and X'_1
    U1_prime = float(0.5 * (U1 + ceiling(V1)))
    add_dotted_arrow(ax2, (V1, U1), (V1, U1_prime), color="#c0392b", lw=4.2, mutation_scale=32,
                     head_length=0.82, head_width=0.35, shrinkA=9.0, shrinkB=12.0, dot_spacing=2.0, zorder=9)

    box_arrow1 = dict(boxstyle="round,pad=0.18", facecolor="#fffcfc", edgecolor="#c0392b", lw=1.5, alpha=1.0)
    ax2.text(V1 - 0.05, 0.5 * (U1 + U1_prime),
             r"$\mathbf{Planck\ and\ T5:\ X^\prime_1 \in \Omega_>\ (X \prec\prec X^\prime_1)}$",
             fontsize=13.0, fontweight="bold", color="#922b21", ha="right", va="center", zorder=100, bbox=box_arrow1)

    # Points on dA_X:
    # X_up at INTERIOR point strictly between V_B and V_PRIME
    V_UP = float(0.45 * V_B + 0.55 * V_PRIME)
    U_UP = float(U_adiabat(V_UP, S_MID))

    # X_down on other side of X'
    V_DOWN = 2.15
    U_DOWN = float(U_adiabat(V_DOWN, S_MID))

    # Scatter points in Panel 2
    ax2.scatter([V0b], [U0b], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax2.scatter([V1], [U1], color=C_PT_ISO, s=190, zorder=15, edgecolors="#4e2103", lw=2.2)
    ax2.scatter([V0b], [U0_prime], color="#e74c3c", s=210, zorder=15, edgecolors="#78281f", lw=2.2)
    ax2.scatter([V1], [U1_prime], color="#e74c3c", s=210, zorder=15, edgecolors="#78281f", lw=2.2)
    ax2.scatter([V_UP], [U_UP], color=C_PT_GT, s=210, zorder=15, edgecolors="#6e4402", lw=2.2)
    ax2.scatter([V_DOWN], [U_DOWN], color=C_PT_LT, s=210, zorder=15, edgecolors="#104266", lw=2.2)
    ax2.scatter([V_PRIME], [U_PRIME], color=C_PT_CUT, s=320, zorder=16, marker="o", edgecolors="black", lw=3.0)

    # Light semi-transparent red curve inside Omega_> from X'_0 to X'_1 crossing dA_X at X_up
    # Control points for smooth Bezier / spline hugging ceiling
    pts_path_v = np.array([V0b, 0.95, V_UP, 1.85, V1])
    pts_path_u = np.array([U0_prime, 2.50, U_UP, 2.65, U1_prime])
    cs_path = CubicSpline(pts_path_v, pts_path_u, bc_type='natural')
    v_path_dense = np.linspace(V0b, V1, 100)
    u_path_dense = cs_path(v_path_dense)
    ax2.plot(v_path_dense, u_path_dense, color="#e74c3c", lw=3.5, alpha=0.55, ls="-", zorder=7)

    # Verify T > T0 at 20 sample points along the path
    v_sample = np.linspace(V0b, V1, 20)
    u_sample = cs_path(v_sample)
    t_sample = temperature(u_sample, v_sample)
    assert np.all(t_sample > T0), f"Sample points must be in Omega_> (T > T0), got min T = {t_sample.min()}"

    # Path annotation
    box_path = dict(boxstyle="round,pad=0.20", facecolor="#fff5f5", edgecolor="#e74c3c", lw=1.5, alpha=1.0)
    ax2.text(1.70, 2.80,
             r"$\mathbf{\Omega_>\ is\ open\ and\ connected\ and\ meets\ both\ sides\ of\ \partial A_X\ \Rightarrow\ it\ cuts\ \partial A_X\ (Step\ 1\ +\ separation)}$",
             fontsize=12.2, fontweight="bold", color="#922b21", ha="center", va="center", zorder=100, bbox=box_path)

    # Region labels for Panel 2
    add_region_label(ax2, fig, 1.85, 0.42, "minus", "#1b4f72", fontsize=33)
    add_region_label(ax2, fig, 1.55, 3.10, "plus", "#922b21", fontsize=33)

    # Projections in Panel 2
    ax2.plot([V0b, V0b], [u_min, U0b], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([v_min, V0b], [U0b, U0b], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([v_min, V0b], [U0_prime, U0_prime], color="#922b21", ls=":", lw=1.6, zorder=4)
    ax2.plot([V1, V1], [u_min, U1], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([v_min, V1], [U1, U1], color="#4e2103", ls=":", lw=1.6, zorder=4)
    ax2.plot([v_min, V1], [U1_prime, U1_prime], color="#922b21", ls=":", lw=1.6, zorder=4)

    # Bracket along x-axis for rho(A_X) starting at V_B
    ax2.annotate("", xy=(v_max, u_min), xytext=(V_B, u_min),
                 arrowprops=dict(arrowstyle="-", color="#2c3e50", lw=5.0, capstyle="butt"))
    ax2.plot([V_B, V_B], [u_min - 0.05, u_min + 0.05], color="#2c3e50", lw=2.5, clip_on=False, zorder=10)

    ax2.set_xticks([V0b, V_B, V1])
    ax2.set_xticklabels([r"$\mathbf{V_{X_0}}$", r"$\mathbf{\inf \rho(A_X)}$", r"$\mathbf{V_{X_1}}$"], fontsize=14, fontweight="bold")
    ax2.set_yticks([U0b, U1, U0_prime, U1_prime])
    ax2.set_yticklabels([r"$\mathbf{U(X_0)}$", r"$\mathbf{U(X_1)}$", r"$\mathbf{U(X^\prime_0)}$", r"$\mathbf{U(X^\prime_1)}$"], fontsize=14, fontweight="bold")

    # Callouts in Panel 2
    add_curved_callout(ax2, fig, r"$\mathbf{X_0}$", (V0b, U0b), (0.98, 0.35), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=-0.14, preferred_corner="tl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X_1}$", (V1, U1), (2.75, 2.05), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X^\prime_0 \in \Omega_>\ (X^\prime_0 \prec\prec X)}$" "\n" r"$\mathbf{(Axiom\ T5:\ full\ temperature\ range\ on\ every\ column\ \Rightarrow\ hot\ state\ below\ the\ adiabat\ level)}$",
                       (V0b, U0_prime), (1.20, 2.05), box_prime0, color="#922b21", fontsize=11.5, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X^\prime_1}$", (V1, U1_prime), (2.75, 3.45), box_prime1, color="#922b21", fontsize=17.5, lw=1.6, rad=0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X_\uparrow \in \partial A_X \cap \Omega_>}$", (V_UP, U_UP), (1.30, 2.70), box_gt, color="#784212", fontsize=15.0, lw=1.6, rad=-0.12, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X_\downarrow \in \partial A_X \cap \Omega_<}$", (V_DOWN, U_DOWN), (2.25, 1.25), box_lt, color="#1b4f72", fontsize=15.0, lw=1.6, rad=-0.14, preferred_corner="bl", shrinkB=8)
    add_curved_callout(ax2, fig, r"$\mathbf{X^\prime \overset{A}{\sim} X}$", (V_PRIME, U_PRIME), (1.80, 1.25), box_cut, color=C_PT_CUT, fontsize=17.5, lw=1.8, rad=-0.14, preferred_corner="tl", shrinkB=10)

    # Curve labels
    add_curved_callout(ax2, fig, r"$\mathbf{\partial A_X}$", (2.84, float(U_adiabat(2.84, S_MID))), (2.95, 1.28), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax2, fig, r"$\mathbf{\partial A_{X_0}}$", (2.15, float(U_adiabat(2.15, S0b))), (2.35, 0.40), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=0.12, shrinkB=0)
    add_curved_callout(ax2, fig, r"$\mathbf{\partial A_{X_1}}$", (2.05, float(U_adiabat(2.05, S1))), (2.15, 3.65), box_curve_adia, color=C_ADIA, fontsize=18.0, lw=1.5, rad=-0.12, shrinkB=0)
    add_curved_callout(ax2, fig, r"$\mathbf{I_{T_0}}$", (2.92, float(U_isotherm(2.92, T0))), (2.96, 2.72), box_curve_iso, color=C_ISO, fontsize=18.0, lw=1.6, rad=-0.12, shrinkB=0)

    ax2.set_xlabel(r"Work Coordinate  $V$", fontsize=18, fontweight="bold", labelpad=10)
    ax2.set_ylabel(r"Internal Energy  $U$", fontsize=18, fontweight="bold", labelpad=10)
    ax2.set_title(r"$\mathbf{Case\ (b):\ Coordinate\ Line\ Misses\ Adiabat\ —\ V_{X_0} \notin \rho(A_X)\ (Topological\ Separation)}$" "\n"
                  r"$\text{Vertical trapped line }\ell \text{ misses }\partial A_X\text{ + Axiom T5 }\Rightarrow X^\prime_0, X^\prime_1 \in \Omega_>\text{ straddle }\partial A_X \Rightarrow X_\uparrow \in \partial A_X \cap \Omega_>$",
                  fontsize=18.5, pad=12, fontweight="bold", color="#1a252f")

    # Super title
    fig.suptitle("Theorem 5.5 — Step 2 (Scenario 1): Interior Transversal Crossing\n"
                 r"$T_{\min} < T_0 < T_{\max} \ \Rightarrow \ \text{Isotherm } I_{T_0} \text{ separates } \mathcal{X}_- \text{ from } \mathcal{X}_+ \ \Rightarrow \ \text{Intermediate adiabat } \partial A_X \text{ cuts } I_{T_0} \text{ at } X^\prime$",
                 fontsize=21.5, fontweight="bold", color="#1a252f", y=0.978)

    # Banner
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
    print(f"Successfully generated test figure: {out_path}")

generate_test_figure("scratch/test_scenario1_rendered.png")
