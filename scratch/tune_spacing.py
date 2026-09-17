with open("scratch/update_prototype.py") as f:
    code = f.read()

# 1. Update GridSpec and suptitle spacing
old_gs = """    fig = plt.figure(figsize=(18, 26.0), dpi=300)
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.28, 5.0, 5.0, 1.85],
                           top=0.92, bottom=0.035, left=0.08, right=0.96, hspace=0.28)"""

new_gs = """    fig = plt.figure(figsize=(18, 24.5), dpi=300)
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.22, 5.0, 5.0, 1.80],
                           top=0.925, bottom=0.032, left=0.08, right=0.96, hspace=0.26)"""
code = code.replace(old_gs, new_gs)

# 2. Update suptitle
old_title = """    # Main super title
    fig.suptitle("Theorem 5.5 — Step 2 (Scenario 1): Interior Transversal Crossing\\n"
                 r"$T_{\\min} < T_0 < T_{\\max} \\ \\Rightarrow \\ \\text{Isotherm } I_{T_0} \\text{ separates } \\mathcal{X}_- \\text{ from } \\mathcal{X}_+ \\ \\Rightarrow \\ \\text{Intermediate adiabat } \\partial A_X \\text{ cuts } I_{T_0} \\text{ at } X^\\prime$",
                 fontsize=22.0, fontweight="bold", color="#1a252f", y=0.985)"""

new_title = """    # Main super title
    fig.suptitle("Theorem 5.5 — Step 2 (Scenario 1): Interior Transversal Crossing\\n"
                 r"$T_{\\min} < T_0 < T_{\\max} \\ \\Rightarrow \\ \\text{Isotherm } I_{T_0} \\text{ separates } \\mathcal{X}_- \\text{ from } \\mathcal{X}_+ \\ \\Rightarrow \\ \\text{Intermediate adiabat } \\partial A_X \\text{ cuts } I_{T_0} \\text{ at } X^\\prime$",
                 fontsize=21.5, fontweight="bold", color="#1a252f", y=0.978)"""
code = code.replace(old_title, new_title)

# 3. Position region labels so they never touch curves:
# In ax1:
code = code.replace("""    add_region_label(ax1, fig, 2.05, 1.85, "minus", "#1b4f72", fontsize=33)""",
                    """    add_region_label(ax1, fig, 1.80, 1.45, "minus", "#1b4f72", fontsize=33)""")
# In ax2:
code = code.replace("""    add_region_label(ax2, fig, 2.05, 1.85, "minus", "#1b4f72", fontsize=33)""",
                    """    add_region_label(ax2, fig, 1.80, 1.45, "minus", "#1b4f72", fontsize=33)""")

# 4. Fix X_\downarrow in ax2 so callout is close and clear:
old_down = """    # X_downarrow: on upper segment of dA_X
    add_curved_callout(ax2, fig, r"$\\mathbf{X_\\downarrow \\in \\partial A_X \\cap \\Omega_<}$", (U_DOWN, V_DOWN), (1.35, 2.45), box_lt, color="#1b4f72", fontsize=15.5, lw=1.6, rad=0.12, preferred_corner="tr", shrinkB=8)"""

new_down = """    # X_downarrow: on upper segment of dA_X
    add_curved_callout(ax2, fig, r"$\\mathbf{X_\\downarrow \\in \\partial A_X \\cap \\Omega_<}$", (U_DOWN, V_DOWN), (1.50, 2.38), box_lt, color="#1b4f72", fontsize=15.0, lw=1.6, rad=-0.14, preferred_corner="tr", shrinkB=8)"""
code = code.replace(old_down, new_down)

# 5. Fix X_1 callout in ax2 so connector doesn't cross I_T0:
old_x1_ax2 = """    # X1: to the left of X1
    add_curved_callout(ax2, fig, r"$\\mathbf{X_1}$", (U1, V1), (2.36, 2.65), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=-0.12, preferred_corner="br", shrinkB=8)"""

new_x1_ax2 = """    # X1: above X1
    add_curved_callout(ax2, fig, r"$\\mathbf{X_1}$", (U1, V1), (2.72, 2.72), box_white, color="#4e2103", fontsize=17.5, lw=1.6, rad=0.10, preferred_corner="bl", shrinkB=8)"""
code = code.replace(old_x1_ax2, new_x1_ax2)

with open("scratch/update_prototype.py", "w") as f:
    f.write(code)

