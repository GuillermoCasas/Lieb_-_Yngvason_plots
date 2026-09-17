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

print("Setup completed.")
