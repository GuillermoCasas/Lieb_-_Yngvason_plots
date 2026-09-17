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

print("Setup complete.")
