import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle

# Model and parameters
GAMMA = 1.4          # adiabatic index
V0, U0 = 1.0, 2.0    # base point X = (V0, U0)
EPS = 0.15           # vertical shift
TAU = 0.5            # radius of ball (and integration length tau)
V_END = V0 + TAU     # V0 + tau = 1.5
C_LIP = (GAMMA - 1.0) / V0 # Lipschitz constant of P

def pressure(W, V):
    return -(GAMMA - 1.0) * W / V

def adiabat_rhs(V, W):
    return [pressure(W[0], V)]

def closed_form(W_start, Vgrid):
    return W_start * (V0 / Vgrid) ** (GAMMA - 1.0)

# Integrate the adiabats for the actual distance t in [0, TAU]
Vgrid = np.linspace(V0, V_END, 400)
sol0 = solve_ivp(adiabat_rhs, (V0, V_END), [U0], t_eval=Vgrid, rtol=1e-10, atol=1e-12)
sole = solve_ivp(adiabat_rhs, (V0, V_END), [U0 + EPS], t_eval=Vgrid, rtol=1e-10, atol=1e-12)

W0 = sol0.y[0]
We = sole.y[0]
h = We - W0
G_plus_1 = h / EPS
t = Vgrid - V0

h_lower = EPS * np.exp(-C_LIP * t)
h_upper = EPS * np.exp(+C_LIP * t)
Gp1_lower = np.exp(-C_LIP * t)
Gp1_upper = np.exp(+C_LIP * t)

# Figure setup
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
ax1.set_aspect('equal')

# Wide grid to plot adiabats across the entire ball
Vgrid_wide = np.linspace(0.05, 1.95, 400)
W0_wide = closed_form(U0, Vgrid_wide)
We_wide = closed_form(U0 + EPS, Vgrid_wide)

# Plot the adiabats
ax1.plot(Vgrid_wide, W0_wide, color=C_LOW, lw=2.2,
         label=r"adiabat through $X$: $S = S_0$")
ax1.plot(Vgrid_wide, We_wide, color=C_HIGH, lw=2.2,
         label=r"shifted adiabat: $S = S_\varepsilon$")

# Labels directly next to the curves in matching colors
V_lbl = 1.75
W0_lbl = closed_form(U0, V_lbl)
We_lbl = closed_form(U0 + EPS, V_lbl)
ax1.text(V_lbl, W0_lbl - 0.05, r"$S = S_0$", color=C_LOW, fontsize=12, va="top", ha="center")
ax1.text(V_lbl, We_lbl + 0.05, r"$S = S_\varepsilon$", color=C_HIGH, fontsize=12, va="bottom", ha="center")

# Draw the ball B_{X, 2\tau}
ball = Circle((V0, U0), 2 * TAU, facecolor='#eef3f8', edgecolor='#7f8c8d',
              lw=1.3, ls='--', alpha=0.6, zorder=1,
              label=r"neighborhood ball $B_{X, 2\tau}$ (radius $2\tau$)")
ax1.add_patch(ball)

# Fill the gap h(V) only in the integration interval [V0, V_END]
Vgrid_fill = np.linspace(V0, V_END, 200)
W0_fill = closed_form(U0, Vgrid_fill)
We_fill = closed_form(U0 + EPS, Vgrid_fill)
ax1.fill_between(Vgrid_fill, W0_fill, We_fill, color=C_FILL, alpha=0.45, lw=0,
                 label=r"gap $h(V) = \mathcal{W}_\varepsilon(V) - \mathcal{W}_0(V) > 0$", zorder=2)

# Mark the base point X and the shifted point
ax1.plot([V0], [U0], "o", color=C_LOW, ms=7, zorder=5)
ax1.plot([V0], [U0 + EPS], "o", color=C_HIGH, ms=7, zorder=5)
ax1.annotate("", xy=(V0, U0 + EPS), xytext=(V0, U0),
             arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))

# Calculate midpoint vertically between curves at V0 - 0.04 to prevent overlap
x_eps = V0 - 0.04
y_eps = (closed_form(U0, x_eps) + closed_form(U0 + EPS, x_eps)) / 2
ax1.text(x_eps, y_eps, r"$\varepsilon$",
         va="center", ha="right", fontsize=12)
ax1.text(V0 - 0.06, U0 - 0.04, r"$X=(U_0,V_0)$", ha="right", va="center",
         fontsize=10, zorder=6)

# Generic point V (previously Vt, taking the place of V as the generic variable)
V_gen = V0 + 0.35  # 1.35
W0_Vgen = closed_form(U0, V_gen)
We_Vgen = closed_form(U0 + EPS, V_gen)
ax1.plot([V_gen], [W0_Vgen], "o", color="black", ms=5, zorder=5)

# Graphical representation of Pi(U, t)
ax1.annotate("", xy=(V_gen, 1.0), xytext=(V_gen, W0_Vgen),
             arrowprops=dict(arrowstyle="->", color="purple", lw=1.5, ls=":", shrinkA=0, shrinkB=0),
             zorder=4)
ax1.text(V_gen + 0.03, (W0_Vgen + 1.0) / 2, r"$\Pi(U, t)$", color="purple", fontsize=11, va="center")

# Plot points on the V-axis to indicate V0, V (generic), and V0 + tau without clipping
ax1.plot([V0], [1.0], "o", color="black", ms=4, zorder=5, clip_on=False)
ax1.plot([V_gen], [1.0], "x", color="purple", ms=6, mew=1.5, zorder=5, clip_on=False)
ax1.plot([V_END], [1.0], "o", color="black", ms=4, zorder=5, clip_on=False)

# Vertical coordinates at V0 (endpoints) and V_gen (intermediate values)
W0_V0 = U0
We_V0 = U0 + EPS

# Horizontal dashed lines from points on the curves to the vertical axis
# Projection of initial values at V0
ax1.plot([0.0, V0], [W0_V0, W0_V0], color='gray', ls=':', lw=1.0)
ax1.plot([0.0, V0], [We_V0, We_V0], color='gray', ls=':', lw=1.0)
# Projection of intermediate values at generic point V
ax1.plot([0.0, V_gen], [W0_Vgen, W0_Vgen], color='gray', ls=':', lw=1.0)
ax1.plot([0.0, V_gen], [We_Vgen, We_Vgen], color='gray', ls=':', lw=1.0)

# Vertical dashed lines from V0, V (generic), and V0 + tau on the x-axis to the curves
ax1.plot([V0, V0], [1.0, We_V0], color='gray', ls=':', lw=1.0)
ax1.plot([V_gen, V_gen], [1.0, We_Vgen], color='gray', ls=':', lw=1.0)
ax1.plot([V_END, V_END], [1.0, closed_form(U0 + EPS, V_END)], color='gray', ls=':', lw=1.0)

# Plot points on the U-axis to indicate coordinates without clipping
ax1.plot([0.0], [W0_V0], "o", color="black", ms=4, zorder=5, clip_on=False)
ax1.plot([0.0], [We_V0], "o", color="black", ms=4, zorder=5, clip_on=False)
ax1.plot([0.0], [W0_Vgen], "o", color="black", ms=4, zorder=5, clip_on=False)
ax1.plot([0.0], [We_Vgen], "o", color="black", ms=4, zorder=5, clip_on=False)

# Add dummy plot for Pi(U, t) legend label
ax1.plot([], [], color="purple", ls=":", lw=1.5, label=r"projection $\Pi(U, V) = V$")

# Set limits and tick labels to indicate V0, V, and V0 + tau
ax1.set_xlim(0.0, 2.0)
ax1.set_ylim(1.0, 3.0)

ax1.set_xticks([V0, V_gen, V_END])
ax1.set_xticklabels([r"$V_0$", r"$V$", r"$V_0 + \tau$"], fontsize=11)

ax1.set_yticks([W0_Vgen, We_Vgen, W0_V0, We_V0])
ax1.set_yticklabels([
    r"$\mathcal{W}_0(V)$", 
    r"$\mathcal{W}_\varepsilon(V)$", 
    r"$U_0$", 
    r"$U_0 + \varepsilon$"
], fontsize=11)

ax1.set_xlabel(r"work coordinate  $V$")
ax1.set_ylabel(r"energy  $U$")
ax1.set_title("Two adiabats & local neighborhood")
ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

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
ax2.text(0.02, EPS + 0.01, r"$h(0)=\varepsilon$", fontsize=10)

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

fig.savefig("Lemma_5-1/test_lemma51_final_3.png", dpi=150, bbox_inches="tight")
print("Saved test figure final 3.")
