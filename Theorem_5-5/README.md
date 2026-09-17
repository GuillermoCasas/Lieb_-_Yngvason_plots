# Theorem 5.5: Isotherms Cut Adiabats

Reference:  
**Elliott H. Lieb and Jakob Yngvason**, *"The physics and mathematics of the second law of thermodynamics"*, *Physics Reports* **310** (1999) 1–96, Section 5.2, pp. 73–75.

---

## 1. Overview & Core Physical Idea

In axiomatic thermodynamics, textbook phase diagrams frequently depict adiabats and isotherms intersecting cleanly to form grids (such as in Carnot cycles). However, in an axiomatic foundation, an essential question arises:
> *Could an isotherm terminate prematurely, run parallel to an adiabat without ever crossing it, or become trapped within a subregion of the state space?*

If an isotherm could not cross from one side of an adiabat to the other, the state space would effectively be fractured into non-communicating thermodynamic sectors, making it impossible to traverse between certain states at constant temperature.

**Theorem 5.5** proves that such pathologies are strictly impossible:
Every isotherm passing through two states $X_0$ and $X_1$ at temperature $T_0$ **must cut every intermediate adiabat** $\partial A_X$ situated between $X_0$ and $X_1$.

To provide optimal clarity without visual clutter, this behavior is illustrated in two separate publication figures:
- **Scenario (1): Interior Transversal Crossing ($T_{\min} < T_0 < T_{\max}$)**
- **Scenario (2): Boundary Temperature Approximation ($T_0 = T_{\max}$)**

Both figures are designed as **abstract theoretical diagrams** without numerical coordinates or tick values on the axes. The focus is placed entirely on the symbolic mathematical entities, order relations ($X_0 \prec X' \prec X_1$), equivalence classes ($X' \overset{A}{\sim} X$), and topological boundaries.

---

## 2. Theorem Statement

> **Theorem 5.5 (Isotherms cut adiabats).**  
> Suppose $X_0 \prec X \prec X_1$ and $X_0$ and $X_1$ have equal temperatures:
> $$T(X_0) = T(X_1) = T_0.$$
>
> 1. **Interior Temperature Range ($T_{\min} < T_0 < T_{\max}$):**  
>    There exists a state $X' \overset{A}{\sim} X$ with $T(X') = T_0$.  
>    *In other words: The isotherm through $X_0$ and $X_1$ cuts every intermediate adiabat between $X_0$ and $X_1$.*
>
> 2. **Upper Boundary Case ($T_0 = T_{\max}$):**  
>    Either there is an $X' \overset{A}{\sim} X$ with $T(X') = T_0$, or for any $T'_0 < T_0$ there exist points $X'_0, X', X'_1$ with:
>    $$X'_0 \prec X' \overset{A}{\sim} X \prec X'_1 \quad \text{and} \quad T(X'_0) = T(X') = T(X'_1) = T'_0.$$
>
> 3. **Lower Boundary Case ($T_0 = T_{\min}$):**  
>    Either there is an $X' \overset{A}{\sim} X$ with $T(X') = T_0$, or for any $T'_0 > T_0$ there exist points $X'_0, X', X'_1$ with:
>    $$X'_0 \prec X' \overset{A}{\sim} X \prec X'_1 \quad \text{and} \quad T(X'_0) = T(X') = T(X'_1) = T'_0.$$

---

## 3. Geometric & Topological Mechanisms

### Scenario (1): Interior Transversal Crossing ($T_{\min} < T_0 < T_{\max}$)

1. **Separating Surface:**  
   The isotherm $I_{T_0} = \{ Y \in \Gamma : T(Y) = T_0 \}$ partitions $\Gamma$ into two open connected sets:
   - **Cooler domain:** $\mathcal{X}_- = \{ Y \in \Gamma : T(Y) < T_0 \}$ (states below $I_{T_0}$)
   - **Warmer domain:** $\mathcal{X}_+ = \{ Y \in \Gamma : T(Y) > T_0 \}$ (states above $I_{T_0}$)

2. **Straddling States on $\partial A_X$ (Step 2 Geometric Exhaustion):**  
   Lieb & Yngvason prove the existence of states on $\partial A_X$ in both regimes via two mutually exhaustive geometric cases:
   - **Case (a) [$V_{X_0}, V_{X_1} \in \rho(A_X)$ — Constant-Volume Vertical Slices]:**  
     - **$X_> \in \partial A_X \cap \mathcal{X}_+$ at fixed volume $V = V_{X_0}$:**  
       Since $X_0 \prec X_>$ along the vertical line $V = V_{X_0}$, we have $U(X_>) > U(X_0) \implies T(X_>) > T(X_0) = T_0$, placing $X_> \in \mathcal{X}_+$.
     - **$X_< \in \partial A_X \cap \mathcal{X}_-$ at fixed volume $V = V_{X_1}$:**  
       Since $X_< \prec X_1$ along the vertical line $V = V_{X_1}$, we have $U(X_<) < U(X_1) \implies T(X_<) < T(X_1) = T_0$, placing $X_< \in \mathcal{X}_-$.
   - **Case (b) [$V_{X_0} \notin \rho(A_X)$ — Topological Separation via Axiom T5]:**  
     - The entire vertical energy line $\ell = \{ (V_{X_0}, U) \in \Gamma \}$ misses $\partial A_X$ and is trapped in the lower-entropy sector $\{ S < S(X) \}$.
     - By Axiom T5, there exists a hot state $X'_0 \in \ell$ with $T(X'_0) > T_0$ despite $X'_0 \prec\prec X$.
     - Similarly, there exists $X'_1$ with $X \prec\prec X'_1$ and $T(X'_1) > T_0$.
     - Thus $X'_0$ and $X'_1$ belong to the connected open warmer component $\Omega_> \subset \mathcal{X}_+$ and straddle the boundary $\partial A_X$. Because $\partial A_X$ separates the state space, the connected path connecting $X'_0$ to $X'_1$ in $\Omega_>$ must cut $\partial A_X$, producing $X_\uparrow \in \partial A_X \cap \Omega_> \subset \mathcal{X}_+$.

3. **Mandatory Cut (Intermediate Value Theorem):**  
   Since $T$ is continuous (Theorem 5.2) and $\partial A_X$ is connected (Axiom S3), $T$ must attain the value $T_0$ somewhere along $\partial A_X$. This guarantees an intersection state:
   $$X' = \partial A_X \cap I_{T_0} \implies X' \overset{A}{\sim} X \quad \text{and} \quad T(X') = T_0.$$

4. **Transversal Slopes:**  
   In $(V, U)$ coordinates (where $V$ is on the horizontal axis and $U$ is on the vertical axis):
   $$\left.\frac{dU}{dV}\right|_T = \frac{a}{V^2} > 0 > -P = \left.\frac{dU}{dV}\right|_S$$
   The isotherm has positive slope (curves upward), whereas adiabats have strictly negative slope (curve downward). The tangent slopes have strictly opposite signs, ensuring a clean, transversal intersection.

---

### Scenario (2): Boundary Temperature Approximation ($T_0 = T_{\max}$)

1. **Boundary Geometry:**  
   $T_{\max}$ is the upper boundary of the temperature range on the state space. Therefore:
   $$\mathcal{X}_+ = \emptyset \quad \text{and} \quad \mathcal{X}_- = \Gamma \setminus I_{T_{\max}}$$
   No physical states exist with $T > T_{\max}$, and all states strictly below the boundary isotherm belong to the cooler domain $\mathcal{X}_-$.

2. **Interior Approximation:**  
   For any interior temperature $T'_0 < T_{\max}$, the isotherm $I_{T'_0}$ lies strictly in the interior where Scenario (1) applies.

3. **Ordering and Convergence:**  
   $I_{T'_0}$ intersects the three adiabats at:
   $$X'_0 \prec X' \prec X'_1 \quad \text{along } I_{T'_0}$$
   As $T'_0 \to T_{\max}^-$, the approximating points converge continuously:
   $$X'_0 \to X_0 \quad \text{and} \quad X'_1 \to X_1$$
   This establishes the adiabatic equivalence $X' \overset{A}{\sim} X$ arbitrary close to the boundary.

---

## 4. Significance for Theorem 5.6 (Uniqueness of Entropy)

Theorem 5.5 is the linchpin needed to prove **Theorem 5.6 (Adiabats and isotherms determine the entropy)**:
- If two candidate entropy functions $S$ and $S^*$ share the same adiabats and isotherms on $\Gamma$, then $S^* = f(S)$.
- The derivative $f'(S)$ is constant on both adiabats and isotherms.
- Because Theorem 5.5 ensures that isotherms cut across all intermediate adiabats, the constant value of $f'(S)$ propagates across the entire range of entropy.
- Hence, $f'(S) = \text{const}$, proving that entropy is **uniquely determined up to an affine scale transformation**:
  $$S^*(X) = a\,S(X) + b, \quad a > 0.$$

---

## 5. Repository Structure

| File | Description |
|---|---|
| [`theorem55_model.py`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-5/theorem55_model.py) | Thermodynamic model for a fluid system ($S(U, V)$, $T(U, V)$, $P(U, V)$), exact inverted level curves, and assertion verification suite. |
| [`theorem55_visualization.py`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-5/theorem55_visualization.py) | Dual visualization generator for Scenario (1) and Scenario (2). |
| [`theorem55_scenario1.png`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-5/theorem55_scenario1.png) | High-resolution publication plot for Scenario (1): Interior Transversal Crossing ($T_{\min} < T_0 < T_{\max}$). |
| [`theorem55_scenario2.png`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-5/theorem55_scenario2.png) | High-resolution publication plot for Scenario (2): Boundary Temperature Approximation ($T_0 = T_{\max}$). |
| [`theorem55_visualization.png`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-5/theorem55_visualization.png) | Primary reference plot (Scenario 1). |
| [`README.md`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-5/README.md) | Complete mathematical and axiomatic documentation. |

---

## 6. Running the Code

To run the thermodynamic model verification suite:
```bash
python3 Theorem_5-5/theorem55_model.py
```

To regenerate all figures:
```bash
python3 Theorem_5-5/theorem55_visualization.py
```
