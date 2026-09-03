# Theorem 5.2: Continuity of Temperature

This directory contains the standalone implementation and single comprehensive illustration of the proof of **Theorem 5.2 (Continuity of temperature)** from Elliott H. Lieb and Jakob Yngvason, *"The Physics and Mathematics of the Second Law of Thermodynamics"*, Physics Reports 310 (1999) 1–96 (Section 5, p. 71).

---

## 1. Theorem Statement

> **Theorem 5.2 (Continuity of temperature).**  
> The temperature $T(X) = T^+(X) = T^-(X)$ is a continuous function on the state space, $\Gamma \subset \mathbb{R}^{n+1}$, of a simple system.

---

## 2. Theoretical Background & Proof Mechanism

Let $X_\infty \in \Gamma$ (denoted $X_0$ in the original paper) be an arbitrary base state, and let $X_1, X_2, X_3, \dots$ be any sequence in $\Gamma$ such that $X_j \to X_\infty$ as $j \to \infty$.
We parametrize states by $X = (U, V)$, where:
- $U \in \mathbb{R}$ is the **internal energy** (plotted on the horizontal axis).
- $V \in \mathbb{R}^n$ is the **work coordinate** / volume (plotted on the vertical axis).

Under this coordinate convention:
- $l_j = \{ (U, V_j) : (U, V_j) \in \Gamma \}$ are **horizontal lines** corresponding to constant work coordinate $V = V_j$.
- $l_0 \equiv l_\infty = \{ (U, V_0) : (U, V_0) \in \Gamma \}$ is the base **horizontal line** passing through $X_\infty = (U_0, V_0)$.
- $A_j = \partial A_{X_j}$ is the **adiabat** passing through $X_j$.
- $B = B(X_\infty, r)$ is a small neighborhood ball centered at $X_\infty$ containing the sequence.

### Why Each Adiabat Must Intersect the Horizontal Line $l_0$
Along an adiabat $A_j$, the internal energy satisfies the Pfaffian differential relation:
$$dU = -P(U, V)\,dV \quad \Longleftrightarrow \quad \frac{dV}{dU} = -\frac{1}{P(U, V)}.$$

1. **Axiom S2 (Locally Lipschitz Pressure):**  
   The pressure $P(X) > 0$ is strictly positive and locally Lipschitz continuous. Within a small ball $B$ centered at $X_\infty$, the pressure is bounded:
   $$0 < P_{\min} \le P(U, V) \le P_{\max} < \infty.$$

2. **Transversality Cone:**  
   In the $(U, V)$ plane, the tangent slope of the adiabat is strictly negative and bounded away from zero:
   $$-\frac{1}{P_{\min}} \le \frac{dV}{dU} \le -\frac{1}{P_{\max}} < 0.$$
   - The horizontal direction ($\frac{dV}{dU} = 0$) is **strictly forbidden**: the adiabat can never run parallel to the horizontal lines $l_j$ and $l_0$.
   - The vertical direction ($\frac{dV}{dU} = -\infty$) is **strictly forbidden**: $P$ is bounded above.

3. **Mandatory Transversal Intersection:**  
   Because the vertical slope is non-zero, as the work coordinate traverses the gap $|V_j - V_0|$, the solution curve must cross the horizontal line $l_0: V = V_0$ at a unique point:
   $$Y_j = A_j \cap l_0 = (U(Y_j), V_0).$$
   Moreover, the horizontal displacement is bounded:
   $$|U(Y_j) - U_j| \le P_{\max} |V_0 - V_j| \le P_{\max} |X_j - X_\infty|.$$
   Therefore, $|Y_j - X_\infty| \to 0$ as $j \to \infty$; the intersection points $Y_j$ slide along $l_0$ directly toward $X_\infty$.

---

### The Two-Leg Temperature Bound
To evaluate $|T(X_j) - T(X_\infty)|$, the proof routes the comparison through the intermediate intersection point $Y_j$:
$$|T(X_j) - T(X_\infty)| \le \underbrace{|T(X_j) - T(Y_j)|}_{\text{Leg 1 (along adiabat } A_j\text{)}} + \underbrace{|T(Y_j) - T(X_\infty)|}_{\text{Leg 2 (along line } l_0\text{)}}.$$

- **Leg 1 (Along adiabat $A_j$ from $X_j$ to $Y_j$):**  
  Both $X_j$ and $Y_j$ lie on the same adiabat $A_j$ within ball $B$. By **Lemma 5.1**, temperature is locally Lipschitz continuous along adiabats with a uniform constant $c < \infty$ on $B$ (instantiating points $X \equiv X_j$ and $Y \equiv Y_j$):
  $$|T(X_j) - T(Y_j)| \le c\,|X_j - Y_j| \le c\,(|X_j - X_\infty| + |Y_j - X_\infty|) \to 0.$$

- **Leg 2 (Along horizontal line $l_0$ from $Y_j$ to $X_\infty$):**  
  Both $Y_j$ and $X_\infty$ have the exact same work coordinate $V_0$, lying on the 1D slice $l_0$. By **Theorem 5.1** ($T^+ = T^-$ everywhere), the temperature function $U \mapsto T(U, V_0)$ is single-valued, continuous, and strictly monotone. Since $Y_j \to X_\infty$ on $l_0$:
  $$|T(Y_j) - T(X_\infty)| \to 0.$$

**Conclusion:** Both legs vanish, so $|T(X_j) - T(X_\infty)| \to 0$, establishing that $T$ is continuous at $X_\infty$, and hence continuous on all of $\Gamma$.

---

## 3. Directory Contents

| File | Description |
|---|---|
| `theorem52_visualization.png` | **Single comprehensive plot** illustrating the proof: sequence $X_j \to X_\infty$, horizontal lines $l_j$, base line $l_0 \equiv l_\infty$, adiabats $A_j$, intersections $Y_j \in l_0$, ball $B(X_\infty, r)$, cited points from Lemma 5.1, transversality cone, and the two-leg proof bridge. |
| `theorem52_visualization.py` | Standalone Python script that computes the geometry and generates `theorem52_visualization.png`. |
| `theorem52_model.py` | Analytical and numerical thermodynamic model of a simple system ($S(U, V)$, $T(U, V)$, $P(U, V)$, adiabats $A_j$, intersections $Y_j$, Lipschitz constants, and sequence convergence). |
| `README.md` | Theoretical and mathematical documentation. |

---

## 4. How to Run

```bash
# Generate the single proof visualization plot
python Theorem_5-2/theorem52_visualization.py

# Verify the thermodynamic model and numerical bounds
python Theorem_5-2/theorem52_model.py
```
