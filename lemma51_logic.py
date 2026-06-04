"""
The LOGIC of Lemma 5.1 (Lieb & Yngvason, Physics Reports 310 (1999) 1-96):

    "The temperatures T+ and T- are locally Lipschitz continuous along each
     adiabat dA_X.  For each X and each closed ball B_{X,r} subset Gamma of
     radius r centered at X there is a constant c(X,r) such that
         |T+(X) - T+(Y)| <= c(X,r) |X - Y|
     for all Y in dA_X with |X - Y| < r.  Furthermore c(X,r) is continuous
     in X on any domain D with B_{X,2r} subset Gamma for all X in D."

The proof is a transfer of regularity:

    (S2)  P is locally Lipschitz:  |P(Z) - P(Z')| <= C |Z - Z'|  on a bounded
          region  (take the region to be the ball B_{X,2r}, giving a constant
          C = C(X, 2r)).
        |
        v  the adiabat is the graph of u_X solving  du/dV_i = P_i(u(V), V);
           a Gronwall estimate (Lemma 5.1's core, the G_eps argument) makes
           the energy gap between nearby adiabats Lipschitz in V.
        |
        v  1/T = dS/dU and dS/dV_i = P_i / T, so T is built from the same
           solution and inherits a Lipschitz bound with constant proportional
           to C.
        |
        v  RESULT: T+ (and T-) are Lipschitz along dA_X with constant c(X,r),
           and because C is taken uniformly on B_{X,2r}, c(X,r) is continuous
           in X.

This script draws that chain with a concrete van-der-Waals-like model
(S(U,V) = ln(U - a/V) + (gamma-1) ln V, so T(U,V) = U - a/V) in which T
genuinely varies along an adiabat, making the Lipschitz "cones" meaningful.

Panels:
  (1) state space Gamma with the ball B_{X,r}, B_{X,2r}, and the adiabat dA_X.
  (2) the pressure constraint: P is trapped inside a Lipschitz cone of slope C.
  (3) the temperature conclusion: T(Y) for Y on dA_X is trapped inside a
      Lipschitz cone of slope c(X,r) around T(X).
  (4) numerical certificate: the ratio |T(X)-T(Y)| / |X-Y| stays below c(X,r).
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle, FancyArrowPatch

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
A = 0.3            # attractive coefficient (vdW-like)
GAMMA = 1.6        # adiabatic exponent
X0 = np.array([2.0, 1.0])   # base state X = (U0, V0)
R = 0.30           # ball radius r
ARC = 0.34         # how far along the adiabat (in V) we explore


def T_field(U, V):
    """Temperature  T = (dS/dU)^{-1} = U - A/V."""
    return U - A / V


def pressure_along_adiabat(V, U):
    """dU/dV at constant entropy for the model (defines the adiabat)."""
    U = U[0]
    return [-(A / V**2 + (GAMMA - 1.0) * (U - A / V) / V)]


def P_field(U, V):
    """The (generalized) pressure P = T * dS/dV as a state function.
    For this model this equals  -dU/dV along the adiabat (same expression)."""
    return -(A / V**2 + (GAMMA - 1.0) * (U - A / V) / V) * (-1.0) * (-1.0)
    # i.e. P(U,V) = A/V^2*... ; we only need its Lipschitz behaviour below.


def P_state(U, V):
    """Clean form of P as a function of state, used for the slope cone."""
    return A / V**2 + (GAMMA - 1.0) * (U - A / V) / V


# --------------------------------------------------------------------------
# Lipschitz constants on the ball B_{X, 2r}  (this is the (S2) input)
# --------------------------------------------------------------------------
def lipschitz_const(func_grad_norm, center, radius, npts=300):
    Us = np.linspace(center[0] - radius, center[0] + radius, npts)
    Vs = np.linspace(center[1] - radius, center[1] + radius, npts)
    UU, VV = np.meshgrid(Us, Vs)
    return func_grad_norm(UU, VV).max()


def gradnorm_P(U, V):
    dU = (GAMMA - 1.0) / V
    dV = -2 * A / V**3 + (GAMMA - 1.0) * (A / V**2) / V - (GAMMA - 1.0) * (U - A / V) / V**2
    return np.sqrt(dU**2 + dV**2)


def gradnorm_T(U, V):
    # grad T = (1, A/V^2)
    return np.sqrt(1.0 + (A / V**2) ** 2)


C_press = lipschitz_const(gradnorm_P, X0, 2 * R)     # constant C in (S2)
c_temp = lipschitz_const(gradnorm_T, X0, 2 * R)      # resulting c(X,r) for T

# --------------------------------------------------------------------------
# The adiabat through X, and temperature along it
# --------------------------------------------------------------------------
Vg = np.linspace(X0[1], X0[1] + ARC, 200)
sol = solve_ivp(pressure_along_adiabat, (X0[1], Vg[-1]), [X0[0]],
                t_eval=Vg, rtol=1e-11, atol=1e-13)
U_ad = sol.y[0]
adiabat = np.column_stack([U_ad, Vg])          # points (U, V) on dA_X
T_ad = T_field(U_ad, Vg)
dist = np.linalg.norm(adiabat - X0, axis=1)    # |X - Y|
dT = np.abs(T_ad - T_field(*X0))               # |T(X) - T(Y)|
ratio = np.divide(dT, dist, out=np.zeros_like(dT), where=dist > 0)

# Also integrate the adiabat the other direction for a fuller curve in panel 1
Vg_back = np.linspace(X0[1], X0[1] - ARC, 200)
sol_b = solve_ivp(pressure_along_adiabat, (X0[1], Vg_back[-1]), [X0[0]],
                  t_eval=Vg_back, rtol=1e-11, atol=1e-13)
adiabat_full_V = np.concatenate([Vg_back[::-1], Vg])
adiabat_full_U = np.concatenate([sol_b.y[0][::-1], U_ad])

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
})

fig = plt.figure(figsize=(14.5, 9.2))
gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.26)

COL_BALL = "#dfe9f3"
COL_BALL2 = "#eef3f8"
COL_ADIA = "#2c6fbb"
COL_P = "#c0392b"
COL_T = "#8e44ad"
COL_CONE = "#f0c419"
COL_OK = "#27ae60"

# ===== Panel 1: state space, the two balls, and the adiabat ===============
ax1 = fig.add_subplot(gs[0, 0])
# state space backdrop (just a generous open region)
ax1.add_patch(Circle((X0[1], X0[0]), 2 * R, transform=ax1.transData,
                      facecolor=COL_BALL2, edgecolor="#9fb4cc",
                      lw=1.0, ls="--", zorder=0))
ax1.add_patch(Circle((X0[1], X0[0]), R, transform=ax1.transData,
                      facecolor=COL_BALL, edgecolor="#5a7fb0",
                      lw=1.4, zorder=1))
ax1.plot(adiabat_full_V, adiabat_full_U, color=COL_ADIA, lw=2.4, zorder=2,
         label=r"adiabat $\partial A_X$")
ax1.plot([X0[1]], [X0[0]], "o", color="black", ms=8, zorder=5)
ax1.text(X0[1] + 0.015, X0[0] + 0.04, r"$X$", fontsize=13)

# label the balls
ax1.annotate(r"$B_{X,r}$", xy=(X0[1] + R * 0.55, X0[0] + R * 0.55),
             fontsize=11, color="#345")
ax1.annotate(r"$B_{X,2r}$", xy=(X0[1] + 2 * R * 0.72, X0[0] - 2 * R * 0.78),
             fontsize=11, color="#678")

# mark a representative Y on the adiabat inside the ball
iY = np.argmin(np.abs(dist - 0.6 * R))
Y = adiabat[iY]
ax1.plot([Y[1]], [Y[0]], "s", color=COL_ADIA, ms=8, zorder=5)
ax1.text(Y[1] + 0.012, Y[0] - 0.05, r"$Y\in\partial A_X$", fontsize=11,
         color=COL_ADIA)
ax1.annotate("", xy=(Y[1], Y[0]), xytext=(X0[1], X0[0]),
             arrowprops=dict(arrowstyle="-", color="black", lw=0.9, ls=":"))

ax1.set_aspect("equal")
ax1.set_xlabel(r"work coordinate  $V$")
ax1.set_ylabel(r"energy  $U$")
ax1.set_title(r"(1)  Ball $B_{X,r}$ and the adiabat through $X$")
ax1.legend(loc="lower right", fontsize=9)
pad = 2.4 * R
ax1.set_xlim(X0[1] - pad, X0[1] + pad)
ax1.set_ylim(X0[0] - pad, X0[0] + pad)

# ===== Panel 2: the pressure Lipschitz cone (the S2 input) ================
ax2 = fig.add_subplot(gs[0, 1])
# evaluate P along the adiabat and show it is trapped in a cone of slope C
P_on_adiabat = P_state(U_ad, Vg)
P0 = P_state(*X0)
ax2.plot(dist, P_on_adiabat, color=COL_P, lw=2.6,
         label=r"$P$ evaluated along $\partial A_X$")
ax2.fill_between(dist, P0 - C_press * dist, P0 + C_press * dist,
                 color=COL_CONE, alpha=0.35, lw=0,
                 label=r"Lipschitz cone, slope $\pm C$")
ax2.plot(dist, P0 + C_press * dist, color="#b8950f", lw=1.0, ls="--")
ax2.plot(dist, P0 - C_press * dist, color="#b8950f", lw=1.0, ls="--")
ax2.plot([0], [P0], "o", color="black", ms=6, zorder=5)
ax2.text(0.005, P0, r"$P(X)$", fontsize=11, va="bottom")
ax2.set_xlabel(r"$|X-Y|$  ($Y$ on the adiabat)")
ax2.set_ylabel(r"pressure  $P$")
ax2.set_title(r"(2)  Axiom S2: $|P(Z)-P(Z')|\leq C\,|Z-Z'|$ on $B_{X,2r}$")
ax2.legend(loc="upper left", fontsize=9)

# ===== Panel 3: the temperature Lipschitz cone (the conclusion) ===========
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(dist, T_ad, color=COL_T, lw=2.6,
         label=r"$T^+$ along $\partial A_X$")
ax3.fill_between(dist, T_field(*X0) - c_temp * dist,
                 T_field(*X0) + c_temp * dist,
                 color=COL_CONE, alpha=0.35, lw=0,
                 label=r"Lipschitz cone, slope $\pm c(X,r)$")
ax3.plot(dist, T_field(*X0) + c_temp * dist, color="#b8950f", lw=1.0, ls="--")
ax3.plot(dist, T_field(*X0) - c_temp * dist, color="#b8950f", lw=1.0, ls="--")
ax3.plot([0], [T_field(*X0)], "o", color="black", ms=6, zorder=5)
ax3.text(0.005, T_field(*X0), r"$T^+(X)$", fontsize=11, va="bottom")
# mark the representative Y again
ax3.plot([dist[iY]], [T_ad[iY]], "s", color=COL_T, ms=8, zorder=5)
ax3.text(dist[iY] + 0.004, T_ad[iY], r"$T^+(Y)$", fontsize=10,
         color=COL_T, va="top")
ax3.set_xlabel(r"$|X-Y|$  ($Y$ on the adiabat)")
ax3.set_ylabel(r"temperature  $T^+$")
ax3.set_title(r"(3)  Conclusion: $|T^+(X)-T^+(Y)|\leq c(X,r)\,|X-Y|$")
ax3.legend(loc="upper left", fontsize=9)

# ===== Panel 4: numerical certificate -- ratio stays below c(X,r) =========
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(dist[1:], ratio[1:], color=COL_OK, lw=2.4,
         label=r"$\dfrac{|T^+(X)-T^+(Y)|}{|X-Y|}$")
ax4.axhline(c_temp, color=COL_P, lw=1.8, ls="--",
            label=r"$c(X,r)$ (slope bound)")
ax4.fill_between(dist[1:], 0, c_temp, color=COL_OK, alpha=0.08, lw=0)
ax4.set_ylim(0, c_temp * 1.25)
ax4.set_xlabel(r"$|X-Y|$  ($Y$ on the adiabat)")
ax4.set_ylabel("difference quotient")
ax4.set_title("(4)  The quotient never exceeds the bound")
ax4.legend(loc="lower right", fontsize=9)

# ----- master title and the logical arrow chain --------------------------
fig.suptitle(
    "The logic of Lemma 5.1: local Lipschitz continuity of the pressure "
    "transfers to the temperature along an adiabat",
    fontsize=13.5, y=0.975)

# a thin annotation linking panel 2 -> panel 3 (the transfer of regularity)
fig.text(0.5, 0.485,
         "$P$ Lipschitz (S2)   $\\longrightarrow$   "
         "[ adiabat ODE  $u_X' = P$ ,   $1/T=\\partial S/\\partial U$ ]   "
         "$\\longrightarrow$   $T^\\pm$ Lipschitz",
         ha="center", va="center", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbe8", ec="#cdcd80"))

fig.savefig("/home/claude/lemma51_logic.png", dpi=150, bbox_inches="tight")
print("Saved figure.")

# --------------------------------------------------------------------------
# Console certificate
# --------------------------------------------------------------------------
print(f"Model: T(U,V) = U - A/V   with A={A}, gamma={GAMMA}")
print(f"Base state X = (U0, V0) = ({X0[0]}, {X0[1]}),  ball radius r = {R}")
print(f"Pressure Lipschitz constant on B(X,2r):  C      = {C_press:.4f}")
print(f"Temperature Lipschitz constant c(X,r):   c(X,r) = {c_temp:.4f}")
print(f"T varies along adiabat:  {T_ad[0]:.4f} -> {T_ad[-1]:.4f}")
print(f"Max difference quotient |dT|/|dX| observed: {ratio[1:].max():.4f}")
print(f"Bound c(X,r) = {c_temp:.4f}  -> respected: {ratio[1:].max() <= c_temp}")
