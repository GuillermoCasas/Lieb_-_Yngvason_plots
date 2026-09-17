import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm

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

v_min, v_max = 0.55, 3.05
u_min, u_max = 0.20, 4.40

v_grid = np.linspace(v_min, v_max, 500)
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

# Intersections on approximating isotherm T'_0:
V0_PRIME = float(B_PARAM + np.exp((S0 - CV * np.log(CV * T_PRIME)) / R_GAS))
U0_PRIME = float(U_isotherm(V0_PRIME, T_PRIME))

V1_PRIME = float(B_PARAM + np.exp((S1 - CV * np.log(CV * T_PRIME)) / R_GAS))
U1_PRIME = float(U_isotherm(V1_PRIME, T_PRIME))

V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T_PRIME)) / R_GAS))
U_PRIME = float(U_isotherm(V_PRIME, T_PRIME))

u_adia0_vals = U_adiabat(v_grid, S0)
u_adiaX_vals = U_adiabat(v_grid, S_MID)
u_adia1_vals = U_adiabat(v_grid, S1)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(v_min, v_max)
ax.set_ylim(u_min, u_max)

v_2d = np.linspace(v_min, v_max, 200)
u_2d = np.linspace(u_min, u_max, 200)
V_2D, U_2D = np.meshgrid(v_2d, u_2d)
T_2D = temperature(U_2D, V_2D)
norm = TwoSlopeNorm(vmin=T_2D.min(), vcenter=T_PRIME, vmax=T_MAX + 0.5)
ax.contourf(V_2D, U_2D, T_2D, levels=50, cmap="coolwarm", norm=norm, alpha=0.25)

# Forbidden region T > T_max (polygon ABOVE u_tmax_vals)
verts_forbid = [(v_min, u_max)]
for v_val, u_val in zip(v_grid, u_tmax_vals):
    verts_forbid.append((v_val, min(max(u_val, u_min), u_max)))
verts_forbid.append((v_max, u_max))
ax.add_patch(Polygon(verts_forbid, closed=True, facecolor="#fbeee6", alpha=0.85,
                     hatch="//", edgecolor="#edbb99", zorder=2))

# Plot isotherms
ax.plot(v_grid, u_tmax_vals, color="#922b21", lw=3.5, ls="--", label=r"$I_{T_{\max}}$")
ax.plot(v_grid, u_tprime_vals, color="#e74c3c", lw=2.5, ls="--", label=r"$I_{T'_0}$")

# Plot adiabats clipped to T <= T_MAX
t0_mask = temperature(u_adia0_vals, v_grid) <= T_MAX + 1e-4
ax.plot(v_grid[t0_mask], u_adia0_vals[t0_mask], 'k-', lw=1.5)
tX_mask = temperature(u_adiaX_vals, v_grid) <= T_MAX + 1e-4
ax.plot(v_grid[tX_mask], u_adiaX_vals[tX_mask], 'k-', lw=3)
t1_mask = temperature(u_adia1_vals, v_grid) <= T_MAX + 1e-4
ax.plot(v_grid[t1_mask], u_adia1_vals[t1_mask], 'k-', lw=1.5)

# Scatter points
ax.scatter([V0, V1], [U0, U1], c='darkred', s=100, zorder=5)
ax.scatter([V0_PRIME, V1_PRIME], [U0_PRIME, U1_PRIME], c='orange', s=100, zorder=5)
ax.scatter([V_PRIME], [U_PRIME], c='red', s=150, zorder=6)

# Arrow from I_{T'_0} to I_{T_MAX} pointing UPWARD
v_arrow = 1.45
u_arrow_start = float(U_isotherm(v_arrow, T_PRIME))
u_arrow_end = float(U_isotherm(v_arrow, T_MAX))
ax.annotate("", xy=(v_arrow, u_arrow_end), xytext=(v_arrow, u_arrow_start),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.5))

ax.set_xlabel("Work Coordinate V")
ax.set_ylabel("Internal Energy U")
plt.savefig("scratch/test_vu_scen2.png", bbox_inches="tight")
print("Saved scratch/test_vu_scen2.png")
