with open("scratch/update_prototype.py") as f:
    code = f.read()

old_down = """    # X_downarrow: on upper segment of dA_X
    add_curved_callout(ax2, fig, r"$\\mathbf{X_\\downarrow \\in \\partial A_X \\cap \\Omega_<}$", (U_DOWN, V_DOWN), (1.50, 2.38), box_lt, color="#1b4f72", fontsize=15.0, lw=1.6, rad=-0.14, preferred_corner="tr", shrinkB=8)"""

new_down = """    # X_downarrow: on upper segment of dA_X
    add_curved_callout(ax2, fig, r"$\\mathbf{X_\\downarrow \\in \\partial A_X \\cap \\Omega_<}$", (U_DOWN, V_DOWN), (1.38, 2.36), box_lt, color="#1b4f72", fontsize=15.0, lw=1.6, rad=-0.12, preferred_corner="br", shrinkB=8)"""

code = code.replace(old_down, new_down)

with open("scratch/update_prototype.py", "w") as f:
    f.write(code)

