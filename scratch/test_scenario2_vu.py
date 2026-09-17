import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.legend_handler import HandlerBase

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
T_MAX = 2.80
T_PRIME = 2.45

def U_isotherm(V, T):
    return CV * T - A_PARAM / V

def U_adiabat(V, S_const):
    return np.exp((S_const - R_GAS * np.log(V - B_PARAM)) / CV) - A_PARAM / V

def entropy(U, V):
    return CV * np.log(U + A_PARAM / V) + R_GAS * np.log(V - B_PARAM)

def temperature(U, V):
    return (U + A_PARAM / V) / CV

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

    if text_type == "plus_empty":
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

generate_scenario2("scratch/test_scenario2_vu.png")
