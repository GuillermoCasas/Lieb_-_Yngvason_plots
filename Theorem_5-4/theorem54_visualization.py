"""
Clean, Abstract Two-Panel Visualization for Theorem 5.4: Carnot Efficiency
(Geometric Foundation: Strict Concavity of Entropy Fixes Exchange Rates for Both Reservoirs)

Reference:
  Lieb & Yngvason, "The physics and mathematics of the second law of
  thermodynamics", Physics Reports 310 (1999) 1-96, Section 5, pp. 72-74.

The main argument:
  Each reservoir is a separate simple system with its own concave entropy profile
  U -> S(U, V) and initial temperature:
  - Hot Reservoir 1 (delivering heat Q_1 > 0 at initial temperature T_1):
      Slope of tangent = 1 / T_1 (shallower slope, high T_1).
      U_{1f} = U_{1i} - Q_1 < U_{1i}.
      By concavity: S_1(U_{1f}) <= S_1(U_{1i}) - Q_1 / T_1
      => -Delta S_1 >= Q_1 / T_1  (Entropy loss is AT LEAST Q_1 / T_1).

  - Cold Reservoir 0 (absorbing heat |Q_0| > 0 at initial temperature T_0 < T_1):
      Slope of tangent = 1 / T_0 (steeper slope, 1/T_0 > 1/T_1).
      U_{0f} = U_{0i} + |Q_0| > U_{0i}.
      By concavity: S_0(U_{0f}) <= S_0(U_{0i}) + |Q_0| / T_0
      => Delta S_0 <= |Q_0| / T_0  (Entropy gain is AT MOST |Q_0| / T_0).

  - Combined Entropy Budget & Carnot Bound:
      0 <= Delta S_total = Delta S_1 + Delta S_0 <= -Q_1 / T_1 + |Q_0| / T_0
      => |Q_0| / T_0 >= Q_1 / T_1  =>  |Q_0| >= Q_1 (T_0 / T_1).
      Since work W = Q_1 - |Q_0|:
         W <= Q_1 (1 - T_0 / T_1)  =>  eta = W / Q_1 <= 1 - T_0 / T_1.

Run directly:
  python Theorem_5-4/theorem54_visualization.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theorem54_model import (
    entropy, temperature, dS_dU, tangent_line,
)

# --------------------------------------------------------------------------
# Styling & Palette
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13.5,
    "axes.labelsize": 12.5,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_HOT_CURVE = "#c0392b"     # Hot reservoir curve (crimson)
C_HOT_TAN = "#78281f"       # Hot tangent (dark red)
C_HOT_FILL = "#f9d5d5"      # Hot deficit fill (soft red)

C_COLD_CURVE = "#2980b9"    # Cold reservoir curve (blue)
C_COLD_TAN = "#1a5276"      # Cold tangent (dark blue)
C_COLD_FILL = "#d4e6f1"     # Cold deficit fill (soft blue)


def make_abstract_two_reservoir_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.2, 10.2), dpi=300)

    # ======================================================================
    # PANEL 1: Hot Reservoir (Left Panel)
    # ======================================================================
    U_min1, U_max1 = 160.0, 520.0
    u1_pts = np.linspace(U_min1, U_max1, 500)
    C1 = 1.0
    S1 = lambda u: entropy(u, C=C1, U_ref=100.0)
    U1i = 400.0
    S1i = float(S1(U1i))
    slope1 = float(dS_dU(U1i, C=C1))  # 1 / T1 = 1 / 400 = 0.0025 (shallower)
    T1 = 1.0 / slope1
    Q1 = 120.0
    U1f = U1i - Q1                    # 280.0
    S1f = float(S1(U1f))
    tan1 = lambda u: tangent_line(u, U1i, C=C1)
    S1f_tan = float(tan1(U1f))

    # 1. Plot Hot Curve & Tangent
    ax1.plot(u1_pts, S1(u1_pts), color=C_HOT_CURVE, lw=3.4, zorder=4,
             label=r"$\mathrm{Hot\ Profile\ } S_1(U_1, V_1) \quad (S_1'' < 0)$")
    tan1_domain = np.linspace(U_min1 + 10, U_max1 - 10, 300)
    ax1.plot(tan1_domain, tan1(tan1_domain), color=C_HOT_TAN, lw=2.4, ls="--", zorder=3,
             label=r"$\mathrm{Initial\ Tangent\ (Slope\ } \frac{1}{T_1}\mathrm{)}$")

    # 2. Hot Deficit Wedge (concavity deficit)
    u1_wedge = np.linspace(U1f, U1i, 150)
    ax1.fill_between(u1_wedge, S1(u1_wedge), tan1(u1_wedge), color=C_HOT_FILL, alpha=0.55, zorder=2,
                     label=r"$\mathrm{Hot\ Deficit\ } (-\Delta S_1 \geq Q_1 / T_1)$")

    # 3. Markers & Deficit Double Arrow
    ax1.plot([U1i], [S1i], "o", color=C_HOT_TAN, ms=9.5, zorder=6)
    ax1.plot([U1f], [S1f], "o", color=C_HOT_CURVE, ms=9.5, zorder=6)
    ax1.plot([U1f], [S1f_tan], "s", color=C_HOT_TAN, ms=7.5, zorder=6)
    arr1 = FancyArrowPatch((U1f, S1f_tan), (U1f, S1f), arrowstyle="<->", color=C_HOT_CURVE,
                           lw=2.0, mutation_scale=13, zorder=5)
    ax1.add_patch(arr1)

    # 4. Projections & Energy Axis Flow Arrow
    y_level1 = 0.18
    ax1.plot([U1f, U1f], [y_level1, S1f_tan], color=C_HOT_CURVE, ls=":", lw=1.2, alpha=0.6, zorder=1)
    ax1.plot([U1i, U1i], [y_level1, S1i], color=C_HOT_TAN, ls=":", lw=1.2, alpha=0.6, zorder=1)
    ax1.plot([U_min1, U1f], [S1f, S1f], color=C_HOT_CURVE, ls=":", lw=1.1, alpha=0.5, zorder=1)
    ax1.plot([U_min1, U1f], [S1f_tan, S1f_tan], color=C_HOT_TAN, ls=":", lw=1.1, alpha=0.5, zorder=1)
    ax1.plot([U_min1, U1i], [S1i, S1i], color=C_HOT_TAN, ls=":", lw=1.1, alpha=0.5, zorder=1)

    arr_q1 = FancyArrowPatch((U1i, y_level1), (U1f, y_level1), arrowstyle="->", color=C_HOT_CURVE,
                             lw=2.2, mutation_scale=15, zorder=5)
    ax1.add_patch(arr_q1)
    ax1.text((U1i + U1f) / 2, y_level1 + 0.05, r"$\Delta U_1 = -Q_1 < 0$", ha="center", va="bottom",
             fontsize=12.5, color=C_HOT_CURVE, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_HOT_CURVE, lw=1.0))

    # 5. Slope Callout (vertically above initial state)
    ax1.annotate(
        r"$\mathrm{Initial\ Slope} = \left.\frac{\partial S_1}{\partial U_1}\right|_{U_{1i}} = \frac{1}{T_1}$" "\n"
        r"$(\mathbf{Shallower\ Slope},\ \text{high } T_1)$",
        xy=(U1i, S1i), xytext=(U1i - 5, 1.88),
        arrowprops=dict(arrowstyle="->", color=C_HOT_TAN, lw=1.5),
        fontsize=12.2, color=C_HOT_TAN, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_HOT_TAN, lw=1.2, zorder=7)
    )

    # 6. Hot Inequality Callout (upper-left open space)
    ax1.annotate(
        r"$\mathbf{Hot\ Reservoir\ (Delivering\ Energy\ } Q_1 > 0\mathbf{):}$" "\n"
        r"$S_1(U_{1f}) \leq S_1(U_{1i}) - \frac{Q_1}{T_1}$" "\n"
        r"$\Rightarrow \Delta S_1 \leq -\frac{Q_1}{T_1}$" "\n"
        r"$\mathbf{\Rightarrow -\Delta S_1 \geq \frac{Q_1}{T_1}}$" "\n"
        r"(Entropy loss is AT LEAST $Q_1/T_1$)",
        xy=(U1f, S1f_tan), xytext=(U_min1 + 10, 1.40),
        arrowprops=dict(arrowstyle="->", color=C_HOT_CURVE, lw=1.5),
        fontsize=12.5, color=C_HOT_CURVE,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff9f9", ec=C_HOT_CURVE, lw=1.4, zorder=7)
    )

    # 7. Axes Ticks & Labels (Purely Symbolic)
    ax1.set_xticks([U1f, U1i])
    ax1.set_xticklabels([r"$U_{1f} = U_{1i} - Q_1$", r"$U_{1i}$"], fontsize=13.0)
    ax1.set_yticks([S1f, S1f_tan, S1i])
    ax1.set_yticklabels([r"$S_1(U_{1f})$", r"$S_1(U_{1i}) - \frac{Q_1}{T_1}$", r"$S_1(U_{1i})$"], fontsize=12.2)

    ax1.set_title(r"$\mathbf{Hot\ Reservoir\ 1\ (Temperature\ } T_1\mathbf{,\ Delivering\ } Q_1 > 0\mathbf{)}$",
                  fontsize=14.0, pad=16, color=C_HOT_CURVE)
    ax1.set_xlabel(r"$\mathrm{Internal\ Energy\ } U_1$", labelpad=12, fontsize=13.5)
    ax1.set_ylabel(r"$\mathrm{Entropy\ } S_1(U_1, V_1)$", labelpad=12, fontsize=13.5)
    ax1.set_xlim(U_min1, U_max1)
    ax1.set_ylim(0.08, 2.25)
    ax1.legend(loc="center left", bbox_to_anchor=(0.02, 0.22), fontsize=10.5, framealpha=0.95, edgecolor="#bdc3c7")
    ax1.grid(True, linestyle=":", alpha=0.30, color="#95a5a6")

    # ======================================================================
    # PANEL 2: Cold Reservoir (Right Panel)
    # ======================================================================
    U_min0, U_max0 = 50.0, 260.0
    u0_pts = np.linspace(U_min0, U_max0, 500)
    C0 = 1.0
    S0 = lambda u: entropy(u, C=C0, U_ref=50.0)
    U0i = 120.0
    S0i = float(S0(U0i))
    slope0 = float(dS_dU(U0i, C=C0))  # 1 / T0 = 1 / 120 = 0.00833 (3.3x STEEPER than hot)
    T0 = 1.0 / slope0
    Q0_abs = 60.0
    U0f = U0i + Q0_abs                 # 180.0
    S0f = float(S0(U0f))
    tan0 = lambda u: tangent_line(u, U0i, C=C0, U_ref=50.0)
    S0f_tan = float(tan0(U0f))

    # 1. Plot Cold Curve & Tangent
    ax2.plot(u0_pts, S0(u0_pts), color=C_COLD_CURVE, lw=3.4, zorder=4,
             label=r"$\mathrm{Cold\ Profile\ } S_0(U_0, V_0) \quad (S_0'' < 0)$")
    tan0_domain = np.linspace(U_min0 + 10, U_max0 - 10, 300)
    ax2.plot(tan0_domain, tan0(tan0_domain), color=C_COLD_TAN, lw=2.4, ls="--", zorder=3,
             label=r"$\mathrm{Initial\ Tangent\ (Slope\ } \frac{1}{T_0}\mathrm{)}$")

    # 2. Cold Deficit Wedge (concavity deficit)
    u0_wedge = np.linspace(U0i, U0f, 150)
    ax2.fill_between(u0_wedge, S0(u0_wedge), tan0(u0_wedge), color=C_COLD_FILL, alpha=0.55, zorder=2,
                     label=r"$\mathrm{Cold\ Deficit\ } (\Delta S_0 \leq |Q_0| / T_0)$")

    # 3. Markers & Deficit Double Arrow
    ax2.plot([U0i], [S0i], "o", color=C_COLD_TAN, ms=9.5, zorder=6)
    ax2.plot([U0f], [S0f], "o", color=C_COLD_CURVE, ms=9.5, zorder=6)
    ax2.plot([U0f], [S0f_tan], "s", color=C_COLD_TAN, ms=7.5, zorder=6)
    arr0 = FancyArrowPatch((U0f, S0f_tan), (U0f, S0f), arrowstyle="<->", color=C_COLD_CURVE,
                           lw=2.0, mutation_scale=13, zorder=5)
    ax2.add_patch(arr0)

    # 4. Projections & Energy Axis Flow Arrow
    y_level0 = -0.15
    ax2.plot([U0i, U0i], [y_level0, S0i], color=C_COLD_TAN, ls=":", lw=1.2, alpha=0.6, zorder=1)
    ax2.plot([U0f, U0f], [y_level0, S0f_tan], color=C_COLD_CURVE, ls=":", lw=1.2, alpha=0.6, zorder=1)
    ax2.plot([U_min0, U0i], [S0i, S0i], color=C_COLD_TAN, ls=":", lw=1.1, alpha=0.5, zorder=1)
    ax2.plot([U_min0, U0f], [S0f, S0f], color=C_COLD_CURVE, ls=":", lw=1.1, alpha=0.5, zorder=1)
    ax2.plot([U_min0, U0f], [S0f_tan, S0f_tan], color=C_COLD_TAN, ls=":", lw=1.1, alpha=0.5, zorder=1)

    arr_q0 = FancyArrowPatch((U0i, y_level0), (U0f, y_level0), arrowstyle="->", color=C_COLD_CURVE,
                             lw=2.2, mutation_scale=15, zorder=5)
    ax2.add_patch(arr_q0)
    ax2.text((U0i + U0f) / 2, y_level0 + 0.05, r"$\Delta U_0 = +|Q_0| > 0$", ha="center", va="bottom",
             fontsize=12.5, color=C_COLD_CURVE, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_COLD_CURVE, lw=1.0))

    # 5. Slope Callout (vertically above initial state)
    ax2.annotate(
        r"$\mathrm{Initial\ Slope} = \left.\frac{\partial S_0}{\partial U_0}\right|_{U_{0i}} = \frac{1}{T_0}$" "\n"
        r"$(\mathbf{Steeper\ Slope},\ \frac{1}{T_0} > \frac{1}{T_1})$",
        xy=(U0i, S0i), xytext=(U0i - 5, 1.88),
        arrowprops=dict(arrowstyle="->", color=C_COLD_TAN, lw=1.5),
        fontsize=12.2, color=C_COLD_TAN, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_COLD_TAN, lw=1.2, zorder=7)
    )

    # 6. Cold Inequality Callout (lower-right open space)
    ax2.annotate(
        r"$\mathbf{Cold\ Reservoir\ (Absorbing\ Energy\ } |Q_0| > 0\mathbf{):}$" "\n"
        r"$S_0(U_{0f}) \leq S_0(U_{0i}) + \frac{|Q_0|}{T_0}$" "\n"
        r"$\mathbf{\Rightarrow \Delta S_0 \leq \frac{|Q_0|}{T_0}}$" "\n"
        r"(Entropy gain is AT MOST $|Q_0|/T_0$)",
        xy=(U0f, S0f), xytext=(U0f + 14, 0.40),
        arrowprops=dict(arrowstyle="->", color=C_COLD_CURVE, lw=1.5),
        fontsize=12.5, color=C_COLD_CURVE,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f9fd", ec=C_COLD_CURVE, lw=1.4, zorder=7)
    )

    # 7. Axes Ticks & Labels (Purely Symbolic)
    ax2.set_xticks([U0i, U0f])
    ax2.set_xticklabels([r"$U_{0i}$", r"$U_{0f} = U_{0i} + |Q_0|$"], fontsize=13.0)
    ax2.set_yticks([S0i, S0f, S0f_tan])
    ax2.set_yticklabels([r"$S_0(U_{0i})$", r"$S_0(U_{0f})$", r"$S_0(U_{0i}) + \frac{|Q_0|}{T_0}$"], fontsize=12.2)

    ax2.set_title(r"$\mathbf{Cold\ Reservoir\ 0\ (Temperature\ } T_0 < T_1\mathbf{,\ Absorbing\ } |Q_0| > 0\mathbf{)}$",
                  fontsize=14.0, pad=16, color=C_COLD_CURVE)
    ax2.set_xlabel(r"$\mathrm{Internal\ Energy\ } U_0$", labelpad=12, fontsize=13.5)
    ax2.set_ylabel(r"$\mathrm{Entropy\ } S_0(U_0, V_0)$", labelpad=12, fontsize=13.5)
    ax2.set_xlim(U_min0, U_max0)
    ax2.set_ylim(-0.30, 2.25)
    ax2.legend(loc="center left", bbox_to_anchor=(0.02, 0.22), fontsize=10.5, framealpha=0.95, edgecolor="#bdc3c7")
    ax2.grid(True, linestyle=":", alpha=0.30, color="#95a5a6")

    # Overall Figure Super-Title
    fig.suptitle(r"$\mathbf{Theorem\ 5.4:\ Concavity\ and\ Tangent\ Bounds\ for\ Both\ Reservoirs\ (T_1 > T_0 \Rightarrow \frac{1}{T_0} > \frac{1}{T_1})}$",
                 fontsize=15.5, y=0.985, color="#1a252f")

    plt.tight_layout(rect=[0, 0.10, 1, 0.96])

    # ---------------- CONCLUSION BANNER BELOW PLOTS ----------------
    banner_text = (
        r"$\mathbf{Conclusion\ (Entropy\ Budget\ and\ Carnot\ Bound):}\quad "
        r"0 \leq \Delta S_{\mathrm{total}} \leq \frac{|Q_0|}{T_0} - \frac{Q_1}{T_1} "
        r"\quad\Rightarrow\quad \frac{|Q_0|}{T_0} \geq \frac{Q_1}{T_1} "
        r"\quad\Rightarrow\quad |Q_0| \geq Q_1\left(\frac{T_0}{T_1}\right) "
        r"\quad\Rightarrow\quad \mathbf{\eta = \frac{W}{Q_1} \leq 1 - \frac{T_0}{T_1} = \eta_{\mathrm{Carnot}}}$"
    )
    fig.text(0.5, 0.040, banner_text, ha="center", va="center", fontsize=12.2, color="#0f172a",
             bbox=dict(boxstyle="round,pad=0.55", fc="#f8fafc", ec="#2563eb", lw=1.6))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theorem54_visualization.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure successfully saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    make_abstract_two_reservoir_plot()
