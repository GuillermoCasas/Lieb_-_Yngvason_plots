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

# Let's test Design 1:
# V0a = 0.82, Vb = 0.78, V0b = 0.70, V1 = 2.50
# H(V) = h0 + h1 * V with Vb = 0.78
# Let's see:
S0a = float(entropy(U_isotherm(0.82), 0.82))
S1 = float(entropy(U_isotherm(2.50), 2.50))
S_MID = 0.52 * S0a + 0.48 * S1

u_b1 = float(U_adiabat(0.78, S_MID)) # ~3.33
h1_1 = 0.4
h0_1 = u_b1 - h1_1 * 0.78 # ~3.02

# Design 2:
# V0b = 0.82, Vb = 1.05, V0a = 1.15, V1 = 2.50
S0a_2 = float(entropy(U_isotherm(1.15), 1.15))
S_MID_2 = 0.52 * S0a_2 + 0.48 * S1
u_b2 = float(U_adiabat(1.05, S_MID_2))
h1_2 = 0.7
h0_2 = u_b2 - h1_2 * 1.05

fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot Design 1 Panel 1 & 2
v = np.linspace(0.55, 3.05, 300)
# D1 P1
axs[0, 0].set_title("Design 1: Panel 1 (V0a = 0.82)")
axs[0, 0].plot(v, U_isotherm(v), 'r--', label='I_T0')
axs[0, 0].plot(v, U_adiabat(v, S_MID), 'k-', label='dA_X')
axs[0, 0].plot(v, h0_1 + h1_1 * v, 'g--', label='Ceiling H(V)')
axs[0, 0].scatter([0.82], [U_isotherm(0.82)], color='orange', s=80, label='X0a')
axs[0, 0].scatter([0.82], [U_adiabat(0.82, S_MID)], color='gold', s=80, label='X_>')
axs[0, 0].set_ylim(0, 4)
axs[0, 0].legend()

# D1 P2
axs[0, 1].set_title("Design 1: Panel 2 (V0b = 0.70, Vb = 0.78)")
axs[0, 1].plot(v, U_isotherm(v), 'r--', label='I_T0')
mask_d1 = v >= 0.78
axs[0, 1].plot(v[mask_d1], U_adiabat(v[mask_d1], S_MID), 'k-', label='dA_X')
axs[0, 1].plot(v, h0_1 + h1_1 * v, 'g--', label='Ceiling H(V)')
axs[0, 1].scatter([0.78], [h0_1 + h1_1 * 0.78], facecolors='none', edgecolors='k', s=80, label='Exit')
axs[0, 1].scatter([0.70], [U_isotherm(0.70)], color='orange', s=80, label='X0b')
axs[0, 1].plot([0.70, 0.70], [0, h0_1 + h1_1 * 0.70], 'brown', ls='--', label='ell')
axs[0, 1].set_ylim(0, 4)
axs[0, 1].legend()

# Plot Design 2 Panel 1 & 2
# D2 P1
axs[1, 0].set_title("Design 2: Panel 1 (V0a = 1.15)")
axs[1, 0].plot(v, U_isotherm(v), 'r--', label='I_T0')
axs[1, 0].plot(v, U_adiabat(v, S_MID_2), 'k-', label='dA_X')
axs[1, 0].plot(v, h0_2 + h1_2 * v, 'g--', label='Ceiling H(V)')
axs[1, 0].scatter([1.15], [U_isotherm(1.15)], color='orange', s=80, label='X0a')
axs[1, 0].scatter([1.15], [U_adiabat(1.15, S_MID_2)], color='gold', s=80, label='X_>')
axs[1, 0].set_ylim(0, 4)
axs[1, 0].legend()

# D2 P2
axs[1, 1].set_title("Design 2: Panel 2 (V0b = 0.82, Vb = 1.05)")
axs[1, 1].plot(v, U_isotherm(v), 'r--', label='I_T0')
mask_d2 = v >= 1.05
axs[1, 1].plot(v[mask_d2], U_adiabat(v[mask_d2], S_MID_2), 'k-', label='dA_X')
axs[1, 1].plot(v, h0_2 + h1_2 * v, 'g--', label='Ceiling H(V)')
axs[1, 1].scatter([1.05], [h0_2 + h1_2 * 1.05], facecolors='none', edgecolors='k', s=80, label='Exit')
axs[1, 1].scatter([0.82], [U_isotherm(0.82)], color='orange', s=80, label='X0b')
axs[1, 1].plot([0.82, 0.82], [0, h0_2 + h1_2 * 0.82], 'brown', ls='--', label='ell')
axs[1, 1].set_ylim(0, 4)
axs[1, 1].legend()

plt.tight_layout()
plt.savefig("scratch/candidates.png", dpi=150)
print("Saved scratch/candidates.png")
