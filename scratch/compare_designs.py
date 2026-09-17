import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

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

# Let's inspect Design A (V0a=0.82, V0b=0.68, Vb=0.76)
# and Design B (V0b=0.82, Vb=1.05, V0a=1.15)

print("Writing comparison script...")
