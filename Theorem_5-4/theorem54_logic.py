"""
The LOGIC of Theorem 5.4 (Carnot Efficiency via the Entropy Budget)
in Lieb & Yngvason, Physics Reports 310 (1999) 1-96, Section 5, pp. 72-74.

    "Between temperature and entropy we can now derive the usual formula
     for the Carnot efficiency."

The proof is laid out in four interconnected logical beats, matching the
2x2 layout of lemma51_logic.png and theorem51_logic.png:

  (1) THE EXCHANGE RATES (Concavity of Entropy).
      A reservoir has fixed work coordinates, so its state moves along the
      concave graph U -> S(U, V). Since a concave function lies below its
      tangents anchored at the initial state (U_i, S(U_i)):
          Delta S_res <= -Q / T_i
      - Delivering energy Q_1 > 0 costs the hot reservoir at least Q_1 / T_1:
            -Delta S_1 >= Q_1 / T_1.
      - Absorbing energy |Q_0| > 0 gains the cold reservoir at most |Q_0| / T_0:
            Delta S_0 <= |Q_0| / T_0.

  (2) THE CYCLIC ENGINE (Entropy as a State Function).
      The working machine operates in a cycle. Because entropy is a single-valued
      state function on state space:
          Delta S_machine = 0.
      No quasi-static paths, cycles, or ideal gas assumptions are needed.

  (3) THE ENTROPY BUDGET (Forced Heat Disposal).
      By the Entropy Principle for products of simple systems (Theorem 4.8):
          Delta S_total = Delta S_machine + Delta S_hot + Delta S_cold >= 0
          => Delta S_cold >= -Delta S_hot.
      Combining with the concavity exchange rates:
          Q_1 / T_1 <= -Delta S_1 <= Delta S_0 <= |Q_0| / T_0
      Forces a minimum disposal of heat into the cold sink:
          |Q_0| >= (T_0 / T_1) * Q_1.

  (4) THE CARNOT BOUND & KELVIN-PLANCK STATEMENT.
      The extractable work is what remains after disposal:
          W = Q_1 - |Q_0| <= Q_1 * (1 - T_0 / T_1)
          => eta = W / Q_1 <= 1 - T_0 / T_1 = eta_C.
      If T_0 >= T_1, then eta <= 0 (Kelvin-Planck statement as a direct corollary).

Run directly:  python Theorem_5-4/theorem54_logic.py
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12.0,
    "axes.labelsize": 11.0,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_HOT = "#c0392b"       # Hot reservoir / heat extraction Q1          (crimson)
C_COLD = "#2980b9"      # Cold reservoir / heat absorption |Q0|       (blue)
C_ENGINE = "#2c3e50"    # Cyclic machine                              (dark slate)
C_WORK = "#27ae60"      # Work output W                               (emerald green)
C_CARD_BG = "#fdfefe"   # Card background


def make_logic_plot():
    fig = plt.figure(figsize=(15.2, 10.8), dpi=300)
    gs = gridspec.GridSpec(2, 2, wspace=0.20, hspace=0.24,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    fig.suptitle(
        "Theorem 5.4: The Logic of Carnot Efficiency as an Entropy Budget",
        fontsize=15.5, y=0.975, fontweight="bold", color="#1a252f"
    )

    # ======================================================================
    # BEAT 1: The Exchange Rates (Concavity of Entropy)
    # ======================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    ax1.set_title("(1) The Exchange Rates: Concavity Fixes Cost & Gain",
                  pad=10, fontsize=12.0, fontweight="bold", color="#2c3e50")

    card1 = FancyBboxPatch((0.02, 0.02), 0.96, 0.94, boxstyle="round,pad=0.03",
                           facecolor=C_CARD_BG, edgecolor="#bdc3c7", lw=1.2)
    ax1.add_patch(card1)

    text1 = (
        "A reservoir is a simple system with fixed work coordinates:\n"
        r"Its state moves along the concave energy curve $U \mapsto S(U, V)$." "\n"
        r"Differentiability (Theorem 5.3) gives initial tangent slope $1/T_i = \partial S / \partial U$." "\n\n"
        r"$\mathbf{Since\ a\ concave\ function\ lies\ below\ its\ tangents:}$" "\n"
        r"   $S(U_f) \leq S(U_i) + \frac{1}{T_i}(U_f - U_i)$" "\n\n"
        r"With energy change $U_f - U_i = -Q$ (energy delivered by reservoir):" "\n"
        r"   $\mathbf{\Delta S_{\mathrm{res}} \leq -\frac{Q}{T_i}}$" "\n\n"
        "Both directions of exchange are unfavourable to the reservoir:\n\n"
        r"• Delivering energy $Q_1 > 0$ (cooling):" "\n"
        r"   $\mathbf{-\Delta S_1 \geq \frac{Q_1}{T_1}} \quad (\mathrm{Costs\ at\ least\ } Q_1/T_1 \mathrm{\ of\ entropy})$" "\n\n"
        r"• Absorbing energy $|Q_0| > 0$ (warming):" "\n"
        r"   $\mathbf{\Delta S_0 \leq \frac{|Q_0|}{T_0}} \quad (\mathrm{Gains\ at\ most\ } |Q_0|/T_0 \mathrm{\ of\ entropy})$"
    )
    ax1.text(0.06, 0.94, text1, ha="left", va="top", fontsize=9.8, color="#1c2833")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # ======================================================================
    # BEAT 2: The Cyclic Engine (Entropy is a State Function)
    # ======================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("(2) The Cyclic Engine: Entropy as a State Function",
                  pad=10, fontsize=12.0, fontweight="bold", color="#2c3e50")

    card2 = FancyBboxPatch((0.02, 0.02), 0.96, 0.94, boxstyle="round,pad=0.03",
                           facecolor=C_CARD_BG, edgecolor="#bdc3c7", lw=1.2)
    ax2.add_patch(card2)

    # Diagram of Machine Coupling
    rect_hot = FancyBboxPatch((0.06, 0.65), 0.25, 0.25, boxstyle="round,pad=0.03",
                              facecolor="#fdeeed", edgecolor=C_HOT, lw=1.8)
    ax2.add_patch(rect_hot)
    ax2.text(0.185, 0.80, "Hot Reservoir", ha="center", va="center",
             fontsize=10.5, color=C_HOT, fontweight="bold")
    ax2.text(0.185, 0.71, r"$T_1, \quad U_{1i} \to U_{1f}$" "\n" r"Delivers $Q_1 > 0$",
             ha="center", va="center", fontsize=9.0, color="#78281f")

    rect_eng = FancyBboxPatch((0.40, 0.65), 0.22, 0.25, boxstyle="round,pad=0.03",
                              facecolor="#f4f6f7", edgecolor=C_ENGINE, lw=1.8)
    ax2.add_patch(rect_eng)
    ax2.text(0.51, 0.80, "Cyclic Engine", ha="center", va="center",
             fontsize=10.5, color=C_ENGINE, fontweight="bold")
    ax2.text(0.51, 0.71, r"$\mathbf{\Delta S_{\mathrm{mach}} = 0}$" "\n" r"(State function)",
             ha="center", va="center", fontsize=9.0, color=C_ENGINE)

    rect_cold = FancyBboxPatch((0.71, 0.65), 0.25, 0.25, boxstyle="round,pad=0.03",
                               facecolor="#ebf5fb", edgecolor=C_COLD, lw=1.8)
    ax2.add_patch(rect_cold)
    ax2.text(0.835, 0.80, "Cold Reservoir", ha="center", va="center",
             fontsize=10.5, color=C_COLD, fontweight="bold")
    ax2.text(0.835, 0.71, r"$T_0, \quad U_{0i} \to U_{0f}$" "\n" r"Absorbs $|Q_0| > 0$",
             ha="center", va="center", fontsize=9.0, color="#1b4f72")

    # Connecting Arrows
    arr_q1 = FancyArrowPatch((0.31, 0.77), (0.40, 0.77), arrowstyle="->",
                             color=C_HOT, lw=2.0, mutation_scale=13)
    ax2.add_patch(arr_q1)
    ax2.text(0.355, 0.81, r"$Q_1$", ha="center", va="bottom", fontsize=10.5, color=C_HOT, fontweight="bold")

    arr_q0 = FancyArrowPatch((0.62, 0.77), (0.71, 0.77), arrowstyle="->",
                             color=C_COLD, lw=2.0, mutation_scale=13)
    ax2.add_patch(arr_q0)
    ax2.text(0.665, 0.81, r"$|Q_0|$", ha="center", va="bottom", fontsize=10.5, color=C_COLD, fontweight="bold")

    # Work Arrow
    arr_w = FancyArrowPatch((0.51, 0.65), (0.51, 0.50), arrowstyle="->",
                            color=C_WORK, lw=2.2, mutation_scale=14)
    ax2.add_patch(arr_w)
    ax2.text(0.51, 0.45, r"Work Extracted: $W = Q_1 - |Q_0|$",
             ha="center", va="top", fontsize=10.5, color=C_WORK, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="#eafaf1", ec=C_WORK, lw=1.1))

    text2 = (
        "Key Insights of the Lieb & Yngvason Construction:\n\n"
        r"• Entropy is a state function: Because the machine returns to its" "\n"
        r"  initial state at the end of a cycle, $\Delta S_{\mathrm{machine}} = 0$ exactly," "\n"
        r"  without evaluating any path integrals." "\n\n"
        r"• Black-box machine: No assumptions about working fluids, ideal gases," "\n"
        r"  frictionlessness, or quasi-static execution are required." "\n\n"
        r"• Energy conservation (First Law): $W = Q_1 - |Q_0|$."
    )
    ax2.text(0.06, 0.35, text2, ha="left", va="top", fontsize=9.6, color="#1c2833")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # ======================================================================
    # BEAT 3: The Entropy Budget (Forced Minimum Heat Disposal)
    # ======================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    ax3.set_title("(3) The Entropy Budget: Forced Minimum Heat Disposal",
                  pad=10, fontsize=12.0, fontweight="bold", color="#2c3e50")

    card3 = FancyBboxPatch((0.02, 0.02), 0.96, 0.94, boxstyle="round,pad=0.03",
                           facecolor=C_CARD_BG, edgecolor="#bdc3c7", lw=1.2)
    ax3.add_patch(card3)

    text3 = (
        "1. Total Compound is Adiabatically Isolated:\n"
        "The machine + hot reservoir + cold reservoir exchange no heat with\n"
        r"the outside world. By the Entropy Principle (Theorem 4.8):" "\n"
        r"   $\Delta S_{\mathrm{total}} = \Delta S_{\mathrm{machine}} + \Delta S_{\mathrm{hot}} + \Delta S_{\mathrm{cold}} \geq 0$" "\n\n"
        r"Since $\Delta S_{\mathrm{machine}} = 0$, this enforces:" "\n"
        r"   $\mathbf{\Delta S_{\mathrm{cold}} \geq -\Delta S_{\mathrm{hot}}}$" "\n\n"
        "2. The Golden Inequality Chain (The Entropy Budget):\n"
        r"Combining the hot and cold concavity exchange rates:" "\n"
        r"   $\mathbf{\frac{Q_1}{T_1} \;\leq\; -\Delta S_{\mathrm{hot}} \;\leq\; \Delta S_{\mathrm{cold}} \;\leq\; \frac{|Q_0|}{T_0}}$" "\n\n"
        "3. Forced Minimum Disposal:\n"
        r"The extreme ends of the inequality chain force a lower bound on $|Q_0|$:" "\n"
        r"   $\frac{Q_1}{T_1} \;\leq\; \frac{|Q_0|}{T_0} \quad \Rightarrow \quad \mathbf{|Q_0| \;\geq\; \frac{T_0}{T_1}\,Q_1}$" "\n\n"
        "The cold reservoir must absorb at least this much energy to dispose\n"
        "of the entropy withdrawn from the hot reservoir."
    )
    ax3.text(0.06, 0.94, text3, ha="left", va="top", fontsize=9.7, color="#1c2833")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # ======================================================================
    # BEAT 4: The Carnot Bound & Kelvin-Planck Statement
    # ======================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_title("(4) The Carnot Bound & Kelvin-Planck Statement",
                  pad=10, fontsize=12.0, fontweight="bold", color="#2c3e50")

    card4 = FancyBboxPatch((0.02, 0.02), 0.96, 0.94, boxstyle="round,pad=0.03",
                           facecolor=C_CARD_BG, edgecolor="#bdc3c7", lw=1.2)
    ax4.add_patch(card4)

    text4 = (
        "1. The Carnot Efficiency Bound:\n"
        "Work is what remains after mandatory heat disposal:\n"
        r"   $W = Q_1 - |Q_0| \leq Q_1 - \frac{T_0}{T_1}Q_1 = Q_1\left(1 - \frac{T_0}{T_1}\right)$" "\n\n"
        "Thermal efficiency is bounded by the Carnot factor:\n"
        r"   $\mathbf{\eta = \frac{W}{Q_1} \leq 1 - \frac{T_0}{T_1} = \eta_C}$" "\n\n"
        "2. Kelvin-Planck Statement as a One-Line Corollary:\n"
        r"If $T_0 \geq T_1$ (cold reservoir equal or hotter than source):" "\n"
        r"   $\eta \leq 1 - \frac{T_0}{T_1} \leq 0 \quad \Rightarrow \quad \mathbf{W \leq 0}$" "\n\n"
        "No net positive work can be extracted using a single reservoir\n"
        "or a colder heat source!\n\n"
        "3. Reservoir Generality:\n"
        r"Because the bound anchors to initial temperatures $T_1$ and $T_0$," "\n"
        "it holds rigorously for both finite and infinite thermal reservoirs."
    )
    ax4.text(0.06, 0.94, text4, ha="left", va="top", fontsize=9.7, color="#1c2833")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # Save output plot
    output_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(output_dir, "theorem54_logic.png")
    plt.savefig(out_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close()
    print(f"[SUCCESS] High-resolution logic plot saved to: {out_path}")


if __name__ == "__main__":
    make_logic_plot()
