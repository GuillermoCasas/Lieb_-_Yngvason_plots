import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json

with open('config.json', 'r') as f:
    config = json.load(f)

phys_cfg = config['physics']
plot_cfg = config['plotting']

fig, ax = plt.subplots(figsize=tuple(plot_cfg['figure']['figsize']))

# Plot the diagonal U1 = U2 (The Isotherms)
ax.plot([0, 6], [0, 6], linestyle='-', color='black', linewidth=2.5, label='Isotherm ($U_1 = U_2$)')

# Define a function to visualize the "forward sector" (reachable states) for an initial non-equilibrium state
def plot_sector(u2, u1, lambda1=1, lambda2=1, color='blue', label_pt='X', u1_str='U_1', u2_str='U_2', step_size=0.15):
    # Calculate the total energy weighted by the capacities (lambda) of the two interacting systems.
    # When thermal contact is made and the systems equilibrate, they will both reach this energy density, u_bar.
    u_bar = (lambda1 * u1 + lambda2 * u2) / (lambda1 + lambda2)
    
    # The original state is represented as a point (U_2, U_1) before interaction
    x_pt, y_pt = u2, u1
    # The equilibrated state inevitably falls on the diagonal (U_1 = U_2) after interaction
    x_bar, y_bar = u_bar, u_bar
    
    # Plot both initial state (X) and final equilibrated state (X')
    ax.plot(x_pt, y_pt, 'ko', zorder=5) 
    ax.plot(x_bar, y_bar, 'ko', zorder=5)
    
    # Define terminology for legend grouping: bounding sets representing physical bounds on achievable energy
    set_a_label = "A'" if label_pt == 'Y' else "A"
    set_b_label = "B'" if label_pt == 'Y' else "B"
    
    # Annotate states algebraically using their parameter constraints
    label_X_coord = rf'${label_pt} = (\lambda_1 {u1_str}, \lambda_2 {u2_str})$'
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.5)
    ax.text(x_pt, y_pt - 0.35, label_X_coord, fontsize=12, ha='center', bbox=bbox_props)
    ax.text(x_bar - 0.35, y_bar, rf'${label_pt}^\prime$', fontsize=12, ha='right', bbox=bbox_props)
    
    # Draw Adiabatic Boundaries: these lines encapsulate all thermodynamically reachable
    # states (the forward sector) based on the Second Law of Thermodynamics.
    # 1. Vertical boundary: represents adiabatic work extraction/insertion limits from system 1
    ax.plot([x_bar, x_bar], [y_bar, 6], color=color, linewidth=2, linestyle='--')
    
    # 2. Reversible equilibrium path segment associating X with equilibrium point X'
    ax.plot([x_bar, x_pt], [y_bar, y_pt], color=color, linewidth=2, linestyle='--')
    
    # 3. Horizontal boundary: represents adiabatic work extraction/insertion limits from system 2
    ax.plot([x_pt, 6], [y_pt, y_pt], color=color, linewidth=2, linestyle='--')
    
    # Fill Set A with vertical lines using step_size
    for x_line in np.arange(x_pt, 6.0, step_size):
        ax.plot([x_line, x_line], [y_pt, 6], color=color, alpha=0.5, linewidth=1.0)

    # Create proxy artist for Set A legend (vertical hatching)
    patch_A = mpatches.Rectangle((0, 0), 0, 0, facecolor='none', edgecolor=color, hatch='|||', linewidth=0, label=rf'${set_a_label}$')
    ax.add_patch(patch_A)
    
    # Fill Set B with horizontal lines using step_size
    for y_line in np.arange(y_bar, 6.0, step_size):
        ax.plot([x_bar, 6], [y_line, y_line], color=color, alpha=0.5, linewidth=1.0)
    
    # Create proxy artist for Set B legend (horizontal hatching)
    # Using Rectangle instead of Patch to satisfy matplotlib's get_path requirement
    patch_B = mpatches.Rectangle((0, 0), 0, 0, facecolor='none', edgecolor=color, hatch='---', linewidth=0, label=rf'${set_b_label}$')
    ax.add_patch(patch_B)
    
    # Indicate u_bar with a vertical line to the axis
    u_bar_label = r"$\overline{U}'$" if label_pt == 'Y' else r"$\overline{U}$"
    ax.plot([x_bar, x_bar], [0, y_bar], color=color, linewidth=1.5, linestyle=':')
    
    # Label the equilibrium temperature/energy intersection on the independent axis
    ax.text(x_bar, -0.2, u_bar_label, color=color, fontsize=12, ha='center', va='top')
    
    # Plot an intersection node visualizing the final energy position mapped against U_2
    ax.plot(x_bar, 0, 'o', color=color, markersize=6, zorder=6)

    # Visually fill the generated convex subspace which constitutes the forward sector (accessible states)
    hull_label = rf'$A_{{{label_pt}}} = \mathrm{{Hull}}({set_a_label} \cup {set_b_label})$'
    ax.fill_between([x_bar, x_pt, 6], [y_bar, y_pt, y_pt], [6, 6, 6], color=color, alpha=0.1, label=hull_label)
    
    # Inject a non-visible reference trace to act as the legend handle for the collective boundary
    ax.plot([], [], color=color, linewidth=2, linestyle='--', label=rf'$\partial A_{{{label_pt}}}$')
    
    return ((x_bar + x_pt) / 2, (y_bar + y_pt) / 2)

# --- Define the thermodynamic system scenario ---
# Heat capacities of the two sub-systems (Lambda_1 and Lambda_2) dictating trajectory slope.
lambda_1 = phys_cfg['lambda_1']
lambda_2 = phys_cfg['lambda_2']

step_size = plot_cfg['styling']['step_size']

# Render the forward sector regions for two distinct initial non-equilibrium situations.
sX = phys_cfg['state_X']
stylX = plot_cfg['styling']['state_X']
mid_X = plot_sector(u2=sX['u2'], u1=sX['u1'], lambda1=lambda_1, lambda2=lambda_2, step_size=step_size, **stylX)

sY = phys_cfg['state_Y']
stylY = plot_cfg['styling']['state_Y']
mid_Y = plot_sector(u2=sY['u2'], u1=sY['u1'], lambda1=lambda_1, lambda2=lambda_2, step_size=step_size, **stylY)

# Parameters determining spatial placement of the unified slope label Box
slope_label_x = plot_cfg['styling']['slope_annotation']['position_x']
slope_label_y = plot_cfg['styling']['slope_annotation']['position_y']

# Insert unified annotation illustrating the slope of the constant-energy equilibration path ($-\lambda_2 / \lambda_1$)
text_obj = ax.text(slope_label_x, slope_label_y, rf'slope $= -\frac{{\lambda_2}}{{\lambda_1}} = {-lambda_2/lambda_1:g}$', 
        fontsize=11, color='black', ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='black', alpha=1.0, pad=5.0), zorder=6)

# Formatting
fig_cfg = plot_cfg['figure']
ax.set_xlim(*fig_cfg['xlim'])
ax.set_ylim(*fig_cfg['ylim'])
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(fig_cfg['xlabel'], fontsize=14)
ax.set_ylabel(fig_cfg['ylabel'], fontsize=14)

title_str = fig_cfg['title_template'].replace('{lambda_1}', str(lambda_1)).replace('{lambda_2}', str(lambda_2))
ax.set_title(title_str, fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Execute layout algorithm before determining graphical annotation bounds to prevent PDF drift.
fig.canvas.draw()

# Extract precisely bounded mathematical corners from strictly formatted layout transforms.
bbox = text_obj.get_bbox_patch().get_window_extent()
bbox_data = ax.transData.inverted().transform(bbox.get_points())

corners_data = [
    (bbox_data[0,0], bbox_data[0,1]), # Bottom-Left
    (bbox_data[1,0], bbox_data[0,1]), # Bottom-Right
    (bbox_data[0,0], bbox_data[1,1]), # Top-Left
    (bbox_data[1,0], bbox_data[1,1])  # Top-Right
]

# Calculate geometric minimum spanning distances to accurately anchor auxiliary visual lines from label
def dist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

best_corner = min(corners_data, key=lambda c: dist(c, mid_X) + dist(c, mid_Y))

# Render indicator lines routing from the computed layout corner text node linking directly toward the equilibration path
ax.annotate('', xy=mid_X, xytext=best_corner, textcoords='data',
            arrowprops=dict(arrowstyle="-", color='black', linestyle=':', alpha=1.0), zorder=4)
ax.annotate('', xy=mid_Y, xytext=best_corner, textcoords='data',
            arrowprops=dict(arrowstyle="-", color='black', linestyle=':', alpha=1.0), zorder=4)
import os
os.makedirs('output', exist_ok=True)
plt.savefig(f'output/forward_sectors_{lambda_1:g}_{lambda_2:g}.png', bbox_inches='tight')
plt.close()