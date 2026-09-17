import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.legend_handler import HandlerBase
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

# Setup styling
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
    return CV * np.log(U + A_PARAM / V) + R_GAS * np.log(V - B_PARAM)

def temperature(U, V):
    return (U + A_PARAM / V) / CV

def U_isotherm(V, T=T0):
    return CV * T - A_PARAM / V

def U_adiabat(V, S_const):
    return np.exp((S_const - R_GAS * np.log(V - B_PARAM)) / CV) - A_PARAM / V

print("Ready to implement full test rendering.")
