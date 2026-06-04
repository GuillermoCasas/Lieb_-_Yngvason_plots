"""
Graphical representation of the worked-out part of Lemma 5.1 in
Lieb & Yngvason, "The physics and mathematics of the second law of
thermodynamics" (Physics Reports 310 (1999) 1-96).

The proof shows that the upper temperature T^+ is locally Lipschitz
continuous along an adiabat. The geometric heart of it:

  * The adiabat through X = (U0, V0) is the graph (W0(V), V), solving
        dW/dV_i = P_i(W(V), V),   W0(V0) = U0.
  * The eps-shifted adiabat starts higher,  We(V0) = U0 + eps, and
    (this is the omitted step)  STAYS higher everywhere:
        h(V) := We(V) - W0(V) > 0   for all V,
    equivalently  G_eps(V) + 1 = h(V)/eps > 0.
  * A Gronwall / integrating-factor argument with the Lipschitz
    constant C of the pressure bounds h, hence bounds G_eps linearly
    in |V - V0|, which yields the Lipschitz estimate for T^+.

We use an ideal-gas-style model (constant heat capacity) so the
adiabat has the closed form  W(V) = W(V0) * (V0/V)^(gamma-1),
letting us validate the ODE integration.

Author: produced for an Obsidian note on the proof.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --------------------------------------------------------------------------
# Model and parameters
# --------------------------------------------------------------------------
GAMMA = 1.4          # adiabatic index -> P(W,V) = -(gamma-1) W / V
V0, U0 = 1.0, 2.0    # base point X = (U0, V0)
EPS = 0.15           # vertical shift of the second adiabat at V0
V_MAX = 2.2          # how far along V we integrate
# Lipschitz constant of P on the working region (sup of |dP/dW| = (gamma-1)/V).
# Smallest V in the region is V0, so:
C_LIP = (GAMMA - 1.0) / V0


def pressure(W, V):
    """P = dU/dV along an adiabat for the constant-heat-capacity model."""
    return -(GAMMA - 1.0) * W / V


def adiabat_rhs(V, W):
    return [pressure(W[0], V)]


def closed_form(W_start, Vgrid):
    """Exact adiabat: U * V^(gamma-1) = const."""
    return W_start * (V0 / Vgrid) ** (GAMMA - 1.0)


# --------------------------------------------------------------------------
# Integrate the two adiabats
# --------------------------------------------------------------------------
Vgrid = np.linspace(V0, V_MAX, 400)

sol0 = solve_ivp(adiabat_rhs, (V0, V_MAX), [U0],
                 t_eval=Vgrid, rtol=1e-10, atol=1e-12)
sole = solve_ivp(adiabat_rhs, (V0, V_MAX), [U0 + EPS],
                 t_eval=Vgrid, rtol=1e-10, atol=1e-12)

W0 = sol0.y[0]               # lower adiabat  W0(V)
We = sole.y[0]               # shifted adiabat We(V)

# The gap and the positivity quantity
h = We - W0                  # h(V) = We(V) - W0(V),  h(V0) = eps
G_plus_1 = h / EPS           # G_eps(V) + 1 = h(V)/eps  (must stay > 0)

# Parametrize by arclength t = |V - V0| along the line (1-D V here)
t = Vgrid - V0

# --------------------------------------------------------------------------
# Gronwall envelopes (the bound established in the proof)
# --------------------------------------------------------------------------
# Lower/upper comparison bounds on h(t):  eps*exp(-C t) <= h(t) <= eps*exp(C t).
# The lower bound is exactly what proves positivity (h > 0).
h_lower = EPS * np.exp(-C_LIP * t)
h_upper = EPS * np.exp(+C_LIP * t)

# In terms of G+1 = h/eps:
Gp1_lower = np.exp(-C_LIP * t)   # stays strictly positive -> the omitted step
Gp1_upper = np.exp(+C_LIP * t)

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(14, 5.2))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)

C_LOW = "#2c6fbb"     # lower adiabat
C_HIGH = "#c0392b"    # shifted adiabat
C_FILL = "#f0c419"    # gap shading
C_ENV = "#7f8c8d"     # envelopes

# ---- Panel 1: the two adiabats in the (V, U) plane ----------------------
ax1 = fig.add_subplot(gs[0])
ax1.plot(Vgrid, W0, color=C_LOW, lw=2.2,
         label=r"$\mathcal{W}_0(V)$  (adiabat through $X$)")
ax1.plot(Vgrid, We, color=C_HIGH, lw=2.2,
         label=r"$\mathcal{W}_\varepsilon(V)$  (starts at $U_0+\varepsilon$)")
ax1.fill_between(Vgrid, W0, We, color=C_FILL, alpha=0.45, lw=0,
                 label=r"gap $h(V)=\mathcal{W}_\varepsilon-\mathcal{W}_0>0$")

# Mark the base point and the initial shift eps
ax1.plot([V0], [U0], "o", color=C_LOW, ms=7, zorder=5)
ax1.plot([V0], [U0 + EPS], "o", color=C_HIGH, ms=7, zorder=5)
ax1.annotate("", xy=(V0, U0 + EPS), xytext=(V0, U0),
             arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
ax1.text(V0 + 0.012, U0 + EPS / 2, r"$\varepsilon$",
         va="center", ha="left", fontsize=12)
ax1.text(V0, U0 - 0.10, r"$X=(U_0,V_0)$", ha="center", va="top",
         fontsize=10)

# Show a couple of vertical "gap" measurements downstream
for Vmark in [1.5, 2.0]:
    i = np.argmin(np.abs(Vgrid - Vmark))
    ax1.plot([Vgrid[i], Vgrid[i]], [W0[i], We[i]],
             color="black", lw=0.8, ls=":")

ax1.set_xlabel(r"work coordinate  $V$")
ax1.set_ylabel(r"energy  $U$")
ax1.set_title("Two adiabats: the higher one stays higher")
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

# ---- Panel 2: the gap h(t) with Gronwall envelopes ----------------------
ax2 = fig.add_subplot(gs[1])
ax2.plot(t, h, color=C_FILL, lw=2.4, label=r"$h(t)=\mathcal{W}_\varepsilon-\mathcal{W}_0$")
ax2.plot(t, h_upper, color=C_ENV, lw=1.4, ls="--",
         label=r"$\varepsilon\,e^{+Ct}$")
ax2.plot(t, h_lower, color=C_ENV, lw=1.4, ls="-.",
         label=r"$\varepsilon\,e^{-Ct}$")
ax2.fill_between(t, h_lower, h_upper, color=C_ENV, alpha=0.12, lw=0)
ax2.axhline(0, color="black", lw=0.8)
ax2.plot([0], [EPS], "o", color="black", ms=6, zorder=5)
ax2.text(0.02, EPS + 0.004, r"$h(0)=\varepsilon$", fontsize=10)

ax2.set_xlabel(r"distance along the line  $t=|V-V_0|$")
ax2.set_ylabel(r"gap  $h(t)$")
ax2.set_title(r"Comparison bounds trap $h(t)$")
ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax2.set_ylim(bottom=0)

# ---- Panel 3: the positivity quantity G_eps + 1 -------------------------
ax3 = fig.add_subplot(gs[2])
ax3.plot(t, G_plus_1, color=C_HIGH, lw=2.4,
         label=r"$G_\varepsilon(V)+1=\dfrac{h}{\varepsilon}$")
ax3.plot(t, Gp1_upper, color=C_ENV, lw=1.4, ls="--", label=r"$e^{+Ct}$")
ax3.plot(t, Gp1_lower, color=C_ENV, lw=1.4, ls="-.", label=r"$e^{-Ct}$")
ax3.fill_between(t, Gp1_lower, Gp1_upper, color=C_ENV, alpha=0.12, lw=0)

# Emphasise the strictly-positive lower envelope = the omitted step
ax3.axhline(0, color="black", lw=0.8)
ax3.fill_between(t, 0, Gp1_lower, color="#27ae60", alpha=0.10, lw=0)
ax3.text(t[-1] * 0.5, 0.18,
         r"$e^{-Ct}>0\ \Rightarrow\ G_\varepsilon+1>0$",
         color="#1e7d45", ha="center", fontsize=9.5)
ax3.plot([0], [1.0], "o", color="black", ms=6, zorder=5)
ax3.text(0.02, 1.02, r"$=1$ at $t=0$", fontsize=10)

ax3.set_xlabel(r"distance along the line  $t=|V-V_0|$")
ax3.set_ylabel(r"$G_\varepsilon+1$")
ax3.set_title("The omitted step: positivity")
ax3.set_ylim(0, max(Gp1_upper) * 1.08)
ax3.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

fig.suptitle(
    "Lemma 5.1 (Lieb\u2013Yngvason): the $\\varepsilon$-shifted adiabat stays above, "
    "so $G_\\varepsilon+1=h/\\varepsilon>0$, and Gr\u00f6nwall bounds give Lipschitz $T^+$",
    fontsize=12.5, y=1.02)

fig.savefig("lemma51_visualization.png", dpi=150,
            bbox_inches="tight")
print("Saved figure.")

# --------------------------------------------------------------------------
# Numerical sanity checks printed to console
# --------------------------------------------------------------------------
print(f"Lipschitz constant C = (gamma-1)/V0 = {C_LIP:.4f}")
print(f"h(0) = {h[0]:.6f}  (eps = {EPS})")
print(f"min h = {h.min():.6f}  -> positive: {np.all(h > 0)}")
print(f"min(G+1) = {G_plus_1.min():.6f}")
print("Closed-form check, max |num - exact|:",
      f"{np.max(np.abs(W0 - closed_form(U0, Vgrid))):.2e},",
      f"{np.max(np.abs(We - closed_form(U0 + EPS, Vgrid))):.2e}")
# Envelope containment check
inside = np.all((h >= h_lower - 1e-9) & (h <= h_upper + 1e-9))
print("h within [eps e^{-Ct}, eps e^{+Ct}] everywhere:", inside)
