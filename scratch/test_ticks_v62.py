import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0.55, 3.05)
ax.set_ylim(0.0, 3.80)
ax.set_xticks([0.62, 0.80, 2.50])
ax.set_xticklabels([r"$\mathbf{V_{X_0}}$", r"$\mathbf{\inf \rho(A_X)}$", r"$\mathbf{V_{X_1}}$"], fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("scratch/ticks_62_80.png", dpi=150)
print("Saved scratch/ticks_62_80.png")
