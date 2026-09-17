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

# Let's test:
# V0a = 0.82
# V1 = 2.50
# V_b = 0.78
# V0b = 0.68
# h1 = 0.5
# u_b = U_adiabat(V_b, S_MID)
# h0 = u_b - h1 * V_b
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

V0a = 0.82
V1 = 2.50
V0b = 0.68
V_B = 0.78

U0a = float(U_isotherm(V0a))
S0a = float(entropy(U0a, V0a))
U0b = float(U_isotherm(V0b))
S0b = float(entropy(U0b, V0b))
U1 = float(U_isotherm(V1))
S1 = float(entropy(U1, V1))

S_MID = 0.52 * S0a + 0.48 * S1
u_b = float(U_adiabat(V_B, S_MID))
h1 = 0.5
h0 = u_b - h1 * V_B

def ceiling(V):
    return h0 + h1 * np.asarray(V)

def ceiling_entropy(V):
    return entropy(ceiling(V), V)

def rho_min(S_const):
    f = lambda v: ceiling_entropy(v) - S_const
    return brentq(f, B_PARAM + 1e-4, 10.0)

v_b_calc = float(rho_min(S_MID))
V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T0)) / R_GAS))
U_PRIME = float(U_isotherm(V_PRIME))

print(f"V0b={V0b}, V_B={v_b_calc:.4f}, V0a={V0a}, V_PRIME={V_PRIME:.4f}, V1={V1}")
print(f"h0={h0:.4f}, h1={h1:.4f}")
print(f"U0b={U0b:.4f}, H(V0b)={ceiling(V0b):.4f}")
print(f"U0a={U0a:.4f}, U_GT={U_adiabat(V0a, S_MID):.4f}, H(V0a)={ceiling(V0a):.4f}")

# Check all assertions
assert S0b < ceiling_entropy(V0b) < S_MID
assert abs(ceiling_entropy(v_b_calc) - S_MID) < 1e-9
assert U_adiabat(V0b, S_MID) > ceiling(V0b)
assert U_adiabat(V0b, S1) > ceiling(V0b)
U0_prime = 0.5 * (U0b + ceiling(V0b))
assert temperature(U0_prime, V0b) > T0 and entropy(U0_prime, V0b) < S_MID
U1_prime = 0.5 * (U1 + ceiling(V1))
assert temperature(U1_prime, V1) > T0 and entropy(U1_prime, V1) > S_MID
assert np.all(U_isotherm(np.linspace(0.55, 3.05, 50)) < ceiling(np.linspace(0.55, 3.05, 50)))
assert V0b < v_b_calc < V_PRIME < V1
assert V0a > v_b_calc
assert U_adiabat(V0a, S_MID) < ceiling(V0a)
assert U_adiabat(V1, S_MID) < ceiling(V1)
print("ALL ASSERTIONS PASSED!")
