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
from matplotlib.patches import Circle, Arc

# --------------------------------------------------------------------------
# Model and parameters
# --------------------------------------------------------------------------
# We use a model where the gap grows and the curves are strongly convex:
# dW/dV = -B * (M - W) / V^2.
# The exact solution is W(V) = M + (W_start - M) * exp(B * (1 - 1/V)).
# We choose B = 1.2 and M = 3.5 to get a strongly curved convex downwards shape.
B_PARAM = 1.2
M_PARAM = 3.5
V0, U0 = 1.0, 2.0    # base point X = (U0, V0)
EPS = 0.375          # vertical shift of the second adiabat at V0 (1.5x of 0.25)
Vt_val = 1.50        # generic point V(t), which is also the endpoint of the shaded segment

# Lipschitz constant of P on the working region (sup of |dP/dW| = B_PARAM / V^2).
# Smallest V in the region is V0, so:
C_LIP = B_PARAM / (V0 ** 2)


def pressure(W, V):
    """P = dU/dV along an adiabat for the new model where the gap grows and curves are convex."""
    return -B_PARAM * (M_PARAM - W) / (V ** 2)


def adiabat_rhs(V, W):
    return [pressure(W[0], V)]


def closed_form(W_start, Vgrid):
    """Exact adiabat: W(V) = M + (W_start - M) * exp(B * (1 - 1/V))."""
    return M_PARAM + (W_start - M_PARAM) * np.exp(B_PARAM * (1.0 - 1.0 / Vgrid))


# --------------------------------------------------------------------------
# Integrate the two adiabats
# --------------------------------------------------------------------------
Vgrid = np.linspace(V0, Vt_val, 400)

sol0 = solve_ivp(adiabat_rhs, (V0, Vt_val), [U0],
                 t_eval=Vgrid, rtol=1e-10, atol=1e-12)
sole = solve_ivp(adiabat_rhs, (V0, Vt_val), [U0 + EPS],
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

C_LOW = "#2c6fbb"     # lower adiabat
C_HIGH = "#c0392b"    # shifted adiabat
C_FILL = "#f0c419"    # gap shading
C_ENV = "#7f8c8d"     # envelopes

# ---- Standalone Plot 1: the two adiabats in the (V, U) plane -------------
fig1 = plt.figure(figsize=(6.5, 6.5))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')

# Margins to shift axes (horizontal axis down to 0.8, vertical axis left to -0.2)
X_LEFT = -0.2
Y_BOTTOM = 0.8

# Calculate the exact starting V where each curve enters the top boundary (U = 3.0) to prevent upward prolongation
def find_V_start(W_start):
    ratio = (3.0 - M_PARAM) / (W_start - M_PARAM)
    return 1.0 / (1.0 - np.log(ratio) / B_PARAM)

# Wide grids to plot adiabats starting exactly at the top boundary (U=3.0) and extending to V=2.00 (just before labels at V=2.02)
Vgrid_W0 = np.linspace(find_V_start(U0), 2.00, 400)
Vgrid_We = np.linspace(find_V_start(U0 + EPS), 2.00, 400)
W0_wide = closed_form(U0, Vgrid_W0)
We_wide = closed_form(U0 + EPS, Vgrid_We)

# Plot the adiabats without clipping so the blue curve extends below the bottom axis to reach near its label
ax1.plot(Vgrid_W0, W0_wide, color=C_LOW, lw=2.2,
         label=r"adiabat through $X$: $S = S_0$", clip_on=False)
ax1.plot(Vgrid_We, We_wide, color=C_HIGH, lw=2.2,
         label=r"shifted adiabat: $S = S_\varepsilon$", clip_on=False)

# Labels directly next to the curves in matching colors, placed just outside the plot border to avoid clutter
V_lbl = 2.02
W0_lbl = closed_form(U0, V_lbl)
We_lbl = closed_form(U0 + EPS, V_lbl)
ax1.text(V_lbl, W0_lbl, r"$S = S_0$", color=C_LOW, fontsize=11, va="center", ha="left", clip_on=False)
ax1.text(V_lbl, We_lbl, r"$S = S_\varepsilon$", color=C_HIGH, fontsize=11, va="center", ha="left", clip_on=False)

# Draw the ball B_{X, 2T}
ball = Circle((V0, U0), 1.0, facecolor='#eef3f8', edgecolor='#7f8c8d',
              lw=1.3, ls='--', alpha=0.6, zorder=1,
              label=r"neighborhood ball $B_{X, 2T}$ (radius $2T$)")
ax1.add_patch(ball)

# Fill the gap h(V) only in the integration interval [V0, Vt_val]
Vgrid_fill = np.linspace(V0, Vt_val, 200)
W0_fill = closed_form(U0, Vgrid_fill)
We_fill = closed_form(U0 + EPS, Vgrid_fill)
W_eps_const = W0_fill + EPS  # Constant shift distance uniformly equal to epsilon

# 1. Constant part of the gap (uniformly equal to epsilon)
ax1.fill_between(Vgrid_fill, W0_fill, W_eps_const, color='#f0c419', alpha=0.40, lw=0,
                 label=r"constant shift $\varepsilon$", zorder=2)

# 2. Remaining growth part of the gap (divergence of the curves)
ax1.fill_between(Vgrid_fill, W_eps_const, We_fill, color='#e67e22', alpha=0.45, lw=0,
                 label=r"growth part $h(V) - \varepsilon$", zorder=2)

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

# Points on adiabats at generic point Vt_val
W0_Vt = closed_form(U0, Vt_val)
We_Vt = closed_form(U0 + EPS, Vt_val)
ax1.plot([Vt_val], [W0_Vt], "o", color="black", ms=5, zorder=5)
ax1.plot([Vt_val], [We_Vt], "o", color="black", ms=5, zorder=5)

# Dimension lines (arrows) and labels at V(t) to indicate the constant and growth parts of the gap
x_dim = Vt_val
# Constant part arrow of height EPS
ax1.annotate("", xy=(x_dim, W0_Vt + EPS), xytext=(x_dim, W0_Vt),
             arrowprops=dict(arrowstyle="<->", color="black", lw=1.0, shrinkA=0, shrinkB=0), zorder=4)

# Growth part arrow of height h(V(t)) - EPS = EPS * G_eps(V(t))
ax1.annotate("", xy=(x_dim, We_Vt), xytext=(x_dim, W0_Vt + EPS),
             arrowprops=dict(arrowstyle="<->", color="black", lw=1.0, shrinkA=0, shrinkB=0), zorder=4)

# Label the constant shift part directly next to the segment (no link)
ax1.text(Vt_val - 0.04, W0_Vt + EPS / 2, r"$\varepsilon$",
         color="black", fontsize=11.0, va="center", ha="right", zorder=6)

# Label the growth part using a leader line below the top Pi label
ax1.annotate(r"$\varepsilon G_\varepsilon(V(t))$",
             xy=(x_dim, W0_Vt + EPS + (We_Vt - W0_Vt - EPS) / 2),
             xytext=(1.80, 1.62),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.0, connectionstyle="arc3,rad=-0.15"),
             fontsize=11.0, color="black", ha="left", va="center", zorder=6)

# Slopes (derivatives dU/dV = -P) at the 4 points
m_A = pressure(U0 + EPS, V0)      # at (V0, U0 + eps)
m_B = pressure(U0, V0)            # at (V0, U0)
m_C = pressure(We_Vt, Vt_val)     # at (Vt_val, We_Vt)
m_D = pressure(W0_Vt, Vt_val)     # at (Vt_val, W0_Vt)

# Draw tangent line segments (hypotenuses) and slope triangles at the 4 points
dx_t = 0.18    # horizontal run size of triangle
dx_back = 0.05 # extend tangent line slightly backwards for tangency context

def draw_slope_triangle(ax, V_p, U_p, m, va_horiz="bottom"):
    # Tangent line (hypotenuse)
    ax.plot([V_p - dx_back, V_p + dx_t], [U_p - dx_back * m, U_p + dx_t * m],
            color='#2c3e50', lw=1.8, zorder=6)
    # Horizontal leg (run = 1)
    ax.plot([V_p, V_p + dx_t], [U_p, U_p],
            color='#7f8c8d', lw=1.2, ls='--', zorder=6)
    # Vertical leg (rise = slope) - highlighted in bright orange/pumpkin color
    ax.plot([V_p + dx_t, V_p + dx_t], [U_p, U_p + dx_t * m],
            color='#e67e22', lw=2.2, ls='-', zorder=7)
    # Circular arc representing the angle
    theta_deg = np.degrees(np.arctan(m))
    arc = Arc((V_p, U_p), 0.08, 0.08, angle=0, theta1=360+theta_deg, theta2=360,
              color='#2c3e50', lw=1.0, zorder=7)
    ax.add_patch(arc)
    # Label '1' for the equal horizontal run
    y_text_offset = 0.045 if va_horiz == "bottom" else -0.045
    ax.text(V_p + dx_t / 2, U_p + y_text_offset, r"$1$",
            color='#555555', fontsize=8.5, ha="center", va="center")

# Draw the 4 slope triangles
draw_slope_triangle(ax1, V0, U0 + EPS, m_A, va_horiz="bottom")
draw_slope_triangle(ax1, V0, U0, m_B, va_horiz="top")
draw_slope_triangle(ax1, Vt_val, We_Vt, m_C, va_horiz="bottom")
draw_slope_triangle(ax1, Vt_val, W0_Vt, m_D, va_horiz="top")

# Annotate slopes with arrows pointing to the vertical leg (rise) in matching orange
# Point A
ax1.annotate(r"$-\Pi(U_0+\varepsilon, V_0)$",
             xy=(V0 + dx_t, U0 + EPS + dx_t * m_A / 2),
             xytext=(1.38, 2.22),
             arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.0, connectionstyle="arc3,rad=-0.15"),
             fontsize=9.5, color="#d35400", ha="left", va="center")

# Point B
ax1.annotate(r"$-\Pi(U_0, V_0)$",
             xy=(V0 + dx_t, U0 + dx_t * m_B / 2),
             xytext=(V0 - 0.42, U0 - 0.35),
             arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.0, connectionstyle="arc3,rad=0.15"),
             fontsize=9.5, color="#d35400", ha="right", va="center")

# Point C
ax1.annotate(r"$-\Pi(\mathcal{W}_\varepsilon(V(t)), V(t))$",
             xy=(Vt_val + dx_t, We_Vt + dx_t * m_C / 2),
             xytext=(Vt_val + dx_t + 0.12, We_Vt + 0.15),
             arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.0, connectionstyle="arc3,rad=0.15"),
             fontsize=9.5, color="#d35400", ha="left", va="center")

# Point D
ax1.annotate(r"$-\Pi(\mathcal{W}_0(V(t)), V(t))$",
             xy=(Vt_val + dx_t, W0_Vt + dx_t * m_D / 2),
             xytext=(Vt_val + dx_t + 0.12, W0_Vt - 0.25),
             arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.0, connectionstyle="arc3,rad=-0.15"),
             fontsize=9.5, color="#d35400", ha="left", va="center")

# Plot points on the V-axis to indicate V0 and V(t) without clipping
ax1.plot([V0], [Y_BOTTOM], "o", color="black", ms=4, zorder=5, clip_on=False)
ax1.plot([Vt_val], [Y_BOTTOM], "o", color="black", ms=4, zorder=5, clip_on=False)

# Vertical coordinates at V0 (endpoints) and Vt_val (generic values)
W0_V0 = U0
We_V0 = U0 + EPS

# Horizontal dashed lines from points on the curves to the vertical axis
# Projection of initial values at V0
ax1.plot([X_LEFT, V0], [W0_V0, W0_V0], color='gray', ls=':', lw=1.0)
ax1.plot([X_LEFT, V0], [We_V0, We_V0], color='gray', ls=':', lw=1.0)
# Projection of generic values at generic point V(t)
ax1.plot([X_LEFT, Vt_val], [W0_Vt, W0_Vt], color='gray', ls=':', lw=1.0)
ax1.plot([X_LEFT, Vt_val], [We_Vt, We_Vt], color='gray', ls=':', lw=1.0)

# Vertical dashed lines from V0 and V(t) on the x-axis to the curves
ax1.plot([V0, V0], [Y_BOTTOM, W0_V0], color='gray', ls=':', lw=1.0)
ax1.plot([Vt_val, Vt_val], [Y_BOTTOM, W0_Vt], color='gray', ls=':', lw=1.0)

# Plot points on the U-axis to indicate coordinates without clipping
ax1.plot([X_LEFT], [W0_V0], "o", color="black", ms=4, zorder=5, clip_on=False)
ax1.plot([X_LEFT], [We_V0], "o", color="black", ms=4, zorder=5, clip_on=False)
ax1.plot([X_LEFT], [W0_Vt], "o", color="black", ms=4, zorder=5, clip_on=False)
ax1.plot([X_LEFT], [We_Vt], "o", color="black", ms=4, zorder=5, clip_on=False)



# Set limits and tick labels to indicate V0 and V(t)
ax1.set_xlim(X_LEFT, 2.0)
ax1.set_ylim(Y_BOTTOM, 3.0)

ax1.set_xticks([V0, Vt_val])
ax1.set_xticklabels([r"$V_0$", r"$V(t)$"], fontsize=11)

ax1.set_yticks([W0_Vt, We_Vt, W0_V0, We_V0])
ax1.set_yticklabels([
    r"$\mathcal{W}_0(V(t))$", 
    r"$\mathcal{W}_\varepsilon(V(t))$", 
    r"$U_0$", 
    r"$U_0 + \varepsilon$"
], fontsize=11)

ax1.set_xlabel(r"work coordinate  $V$")
ax1.set_ylabel(r"energy  $U$")
# No title for Standalone Plot 1 to keep it clean and focused
ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

# Save Standalone Plot 1
fig1.savefig("lemma51_visualization.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved lemma51_visualization.png (standalone adiabats plot).")

# ---- Standalone Plot 2: the gap h(t) with Gronwall envelopes -------------
fig2 = plt.figure(figsize=(6.0, 4.6))
ax2 = fig2.add_subplot(111)
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
ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax2.set_ylim(bottom=0)

fig2.savefig("lemma51_gap_bounds.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved lemma51_gap_bounds.png (standalone gap bounds plot).")

# ---- Standalone Plot 3: the positivity quantity G_eps + 1 ----------------
fig3 = plt.figure(figsize=(6.0, 4.6))
ax3 = fig3.add_subplot(111)
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

fig3.savefig("lemma51_positivity.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved lemma51_positivity.png (standalone positivity plot).")

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
