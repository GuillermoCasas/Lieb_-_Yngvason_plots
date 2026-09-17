import numpy as np
import matplotlib.pyplot as plt

fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6))

# Case A: V0b = 0.68, V_B = 0.78
axA.set_xlim(0.55, 3.05)
axA.set_ylim(0.0, 3.80)
axA.set_xticks([0.68, 0.78, 2.50])
axA.set_xticklabels([r"$\mathbf{V_{X_0}}$", r"$\mathbf{\inf \rho(A_X)}$", r"$\mathbf{V_{X_1}}$"], fontsize=14, fontweight="bold")
axA.set_title("Case A: V0b = 0.68, V_B = 0.78")

# Case B: V0b = 0.82, V_B = 1.05
axB.set_xlim(0.55, 3.05)
axB.set_ylim(0.2, 3.80)
axB.set_xticks([0.82, 1.05, 2.50])
axB.set_xticklabels([r"$\mathbf{V_{X_0}}$", r"$\mathbf{\inf \rho(A_X)}$", r"$\mathbf{V_{X_1}}$"], fontsize=14, fontweight="bold")
axB.set_title("Case B: V0b = 0.82, V_B = 1.05")

plt.tight_layout()
plt.savefig("scratch/panel2_ticks.png", dpi=150)
print("Saved scratch/panel2_ticks.png")
