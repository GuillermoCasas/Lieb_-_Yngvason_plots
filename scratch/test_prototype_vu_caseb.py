import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

CV = 1.5
R_GAS = 1.0
A_PARAM = 2.0
B_PARAM = 0.5
T0 = 2.00

def U_isotherm(V, T):
    return CV * T - A_PARAM / V

def U_adiabat(V, S_const):
    return np.exp((S_const - R_GAS * np.log(V - B_PARAM)) / CV) - A_PARAM / V

def entropy(U, V):
    return CV * np.log(U + A_PARAM / V) + R_GAS * np.log(V - B_PARAM)

def temperature(U, V):
    return (U + A_PARAM / V) / CV

V0 = 0.82
U0 = float(U_isotherm(V0, T0))
S0 = float(entropy(U0, V0))

V1 = 2.50
U1 = float(U_isotherm(V1, T0))
S1 = float(entropy(U1, V1))

S_MID = 0.52 * S0 + 0.48 * S1
V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T0)) / R_GAS))
U_PRIME = float(U_isotherm(V_PRIME, T0))

v_min, v_max = 0.55, 3.05
u_min, u_max = 0.20, 3.80

v_rho_min = 1.15
v_grid = np.linspace(v_min, v_max, 500)
v_rho_grid = np.linspace(v_rho_min, v_max, 400)

u_iso = U_isotherm(v_grid, T0)
u_adia0 = U_adiabat(v_grid, S0)
u_adiaX_trunc = U_adiabat(v_rho_grid, S_MID)
u_adia1 = U_adiabat(v_grid, S1)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(v_min, v_max)
ax.set_ylim(u_min, u_max)

v_2d = np.linspace(v_min, v_max, 200)
u_2d = np.linspace(u_min, u_max, 200)
V_2D, U_2D = np.meshgrid(v_2d, u_2d)
T_2D = temperature(U_2D, V_2D)
norm = TwoSlopeNorm(vmin=T_2D.min(), vcenter=T0, vmax=T_2D.max())
ax.contourf(V_2D, U_2D, T_2D, levels=50, cmap="coolwarm", norm=norm, alpha=0.25)

ax.plot(v_grid, u_iso, 'r--', lw=2.5, label=r"$I_{T_0}$")
ax.plot(v_grid, u_adia0, 'k-', lw=1.5, label=r"$\partial A_{X_0}$")
ax.plot(v_rho_grid, u_adiaX_trunc, 'k-', lw=3.2, label=r"$\partial A_X$")
ax.plot(v_grid, u_adia1, 'k-', lw=1.5, label=r"$\partial A_{X_1}$")

# Trapped line ell over V0 (vertical dashed line)
ax.plot([V0, V0], [u_min, u_max], color="#784212", ls="--", lw=2.0)

# Hot states X'0 and X'1
U0_PRIME = 2.40
U1_PRIME = 3.20

# Endpoints / states on dA_X
U_UP = float(U_adiabat(v_rho_min, S_MID))
V_UP = v_rho_min
V_DOWN = 2.15
U_DOWN = float(U_adiabat(V_DOWN, S_MID))

ax.scatter([V0, V1], [U0, U1], c='orange', s=100, zorder=5)
ax.scatter([V0, V1], [U0_PRIME, U1_PRIME], c='red', s=100, zorder=5)
ax.scatter([V_UP], [U_UP], c='gold', s=100, zorder=5)
ax.scatter([V_DOWN], [U_DOWN], c='blue', s=100, zorder=5)
ax.scatter([V_PRIME], [U_PRIME], c='red', s=150, zorder=6)

# Arrow from X0 to X'0 (upward along ell)
ax.annotate("", xy=(V0, U0_PRIME), xytext=(V0, U0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2))
# Arrow from X1 to X'1 (upward along line V1)
ax.annotate("", xy=(V1, U1_PRIME), xytext=(V1, U1),
            arrowprops=dict(arrowstyle="->", color="red", lw=2))

# rho(A_X) range along x-axis
ax.annotate("", xy=(v_max, u_min), xytext=(v_rho_min, u_min),
            arrowprops=dict(arrowstyle="-", color="#2c3e50", lw=5.0))

ax.set_xlabel("Work Coordinate V")
ax.set_ylabel("Internal Energy U")
plt.savefig("scratch/test_vu_caseb.png", bbox_inches="tight")
print("Saved scratch/test_vu_caseb.png")
