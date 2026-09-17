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

# Common thermodynamic functions
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

print("Helper definitions complete.")
