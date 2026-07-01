# Lieb & Yngvason Plot Reproductions

This repository contains Python code to reproduce and enrich plots depicting the physics and mathematics of the second law of thermodynamics, based on the foundational 1999 paper by Elliott H. Lieb and Jakob Yngvason: ["The Physics and Mathematics of the Second Law of Thermodynamics"](https://arxiv.org/abs/cond-mat/9708200) (Phys. Rept. 310, 1-96, 1999).

## Theory Background

The plot visualizes the **forward sector** of states accessible from an initial non-equilibrium state, according to the second law of thermodynamics. 
* We consider a state space $\Gamma^{(2)} \times \Gamma^{(1)}$, representing the energy coordinates $(U_2, U_1)$ of two mutually interacting thermodynamic systems.
* The two systems have heat capacities $\lambda_1$ and $\lambda_2$.
* When a thermal contact is established between an initial non-equilibrium state (e.g., $X$ or $Y$), the systems exchange heat until they reach an equilibrium temperature. This final equilibrium state lies on the diagonal where $U_1 = U_2$ (the isothermal line).
* During this equilibration, energy is conserved: $\lambda_1 U_1 + \lambda_2 U_2 = (\lambda_1 + \lambda_2) \overline{U}$. This reversible equilibration path is represented by a line of slope $-\lambda_2 / \lambda_1$.
* The accessible states under adiabatic processes (the "forward sector" $A_X$ or $A_Y$) are bounded by the adiabatic work extraction/insertion limits (vertical and horizontal dashed lines) and the reversible equilibration path. The thermodynamically forward sector mathematically corresponds to the convex hull of these boundary sets.

By reproducing this figure, we enrich the original paper's schematic with mathematically precise geometric bounds, algebraically labeled regions (sets $A$ and $B$), and concrete energy scales parameterized by $\lambda_1$ and $\lambda_2$.

## Inputs and Configuration

The [plot_forward_sectors.py](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/World_of_thermometers/plot_forward_sectors.py) script is driven by [config.json](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/World_of_thermometers/config.json), which defines the thermodynamic parameters, initial conditions, and visual styling:

### `physics`
- `lambda_1`, `lambda_2`: Heat capacities (or sizes) of the two sub-systems. This ratio determines the slope ($-\lambda_2/\lambda_1$) of the equilibration path in the $(U_2, U_1)$ phase space.
- `state_X`, `state_Y`: The energy coordinates `(u2, u1)` of initial non-equilibrium states $X$ and $Y$ before thermal contact.

### `plotting`
- Parameters specifying plot dimensions (`xlim`, `ylim`, `figsize`), axes labels, titles, colors, annotation positions, and step size for shading the regions.

## Outputs

When you run the script:

```bash
python World_of_thermometers/plot_forward_sectors.py
```

It parses `config.json` and generates a precise phase-space diagram illustrating the forward sectors for the given states. The resulting plot is automatically saved into the `World_of_thermometers/output/` directory (e.g., `World_of_thermometers/output/forward_sectors_1_1.png`).
