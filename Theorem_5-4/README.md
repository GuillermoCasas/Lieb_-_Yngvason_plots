# Theorem 5.4: Carnot Efficiency via Concavity of Entropy and the Entropy Budget

Reference:  
**Elliott H. Lieb and Jakob Yngvason**, *"The physics and mathematics of the second law of thermodynamics"*, *Physics Reports* **310** (1999) 1–96, Section 5, pp. 72–74.

---

## 1. Overview and Core Physical Idea

In the Lieb–Yngvason axiomatic formulation of thermodynamics, the classical **Carnot efficiency formula** does not require Carnot cycles, ideal gas equations of state, reversible quasi-static paths, or microscopic models. Instead, it emerges as a direct consequence of:
1. **The Concavity of Entropy** ($U \mapsto S(U, V)$ is strictly concave, Theorems 2.8 & 4.3).
2. **Entropy as a State Function** (a cyclic machine returns to its initial state, so $\Delta S_{\text{machine}} = 0$).
3. **The Entropy Principle** (the total entropy of an adiabatically isolated compound system cannot decrease, Theorem 4.8).

When viewed together, the derivation is an **entropy budget**:
- The heat engine withdraws entropy from the hot reservoir when extracting energy $Q_1$.
- Because the engine operates in a cycle ($\Delta S_{\text{machine}} = 0$), it cannot store any net entropy.
- Therefore, all entropy withdrawn must be deposited into the cold reservoir.
- Strict concavity fixes the unfavourable exchange rates between energy and entropy at each reservoir, forcing a minimum amount of energy $|Q_0|$ that must be discarded to the cold sink.

---

## 2. Mathematical Derivation

### 2.1 The Exchange Rates (Concavity Tangent Bounds)
A thermal reservoir is modeled as a simple system with fixed work coordinates (e.g., constant volume $V$), so its internal state moves along the concave energy curve:
$$U \mapsto S(U, V), \quad \text{with } S'' = \frac{\partial^2 S}{\partial U^2} < 0.$$

By differentiability (Theorem 5.3), the slope of the tangent line anchored at an initial state $(U_i, S(U_i))$ is the reciprocal of the absolute initial temperature:
$$\left.\frac{\partial S}{\partial U}\right|_{U_i} = \frac{1}{T_i}.$$

Because any concave function lies strictly below its tangent lines:
$$S(U_f) \leq S(U_i) + \frac{1}{T_i}(U_f - U_i).$$

If the reservoir delivers net energy $Q$ to the engine, its internal energy changes by:
$$U_f - U_i = -Q \quad \implies \quad \Delta S_{\text{res}} \leq -\frac{Q}{T_i}.$$

This inequality governs both directions of heat transfer:
- **Hot Reservoir (delivering energy $Q_1 > 0$, cooling $\Delta U_1 = -Q_1 < 0$):**
  $$\Delta S_1 \leq -\frac{Q_1}{T_1} \quad \implies \quad \mathbf{-\Delta S_1 \geq \frac{Q_1}{T_1}}.$$
  *Delivering energy costs the hot reservoir at least $Q_1 / T_1$ in entropy loss.*

- **Cold Reservoir (absorbing energy $|Q_0| > 0$, warming $\Delta U_0 = +|Q_0| > 0$):**
  $$\Delta S_0 \leq +\frac{|Q_0|}{T_0}.$$
  *Absorbing energy gains the cold reservoir at most $|Q_0| / T_0$ in entropy.*

---

### 2.2 The Cyclic Machine
The cyclic machine couples the hot and cold reservoirs to extract work $W$:
- **First Law (Energy Conservation):**
  $$W = Q_1 - |Q_0|.$$
- **State Function Property:**
  Because the engine returns to its initial state at the completion of a cycle:
  $$\Delta S_{\text{machine}} = 0.$$
  This holds without any assumptions about internal reversibility, friction, or working substances.

---

### 2.3 The Entropy Budget Inequality Chain
The composite system (hot reservoir + cyclic machine + cold reservoir) is adiabatically isolated from the rest of the universe. By the **Entropy Principle** (Theorem 4.8):
$$\Delta S_{\text{total}} = \Delta S_{\text{machine}} + \Delta S_1 + \Delta S_0 \geq 0.$$

Substituting $\Delta S_{\text{machine}} = 0$:
$$\Delta S_0 \geq -\Delta S_1.$$

Chaining this with the concavity exchange rates:
$$\frac{Q_1}{T_1} \;\leq\; -\Delta S_1 \;\leq\; \Delta S_0 \;\leq\; \frac{|Q_0|}{T_0}.$$

Comparing the outermost terms gives the **mandatory heat disposal**:
$$\frac{Q_1}{T_1} \leq \frac{|Q_0|}{T_0} \quad \implies \quad \mathbf{|Q_0| \geq \frac{T_0}{T_1} Q_1}.$$

---

### 2.4 The Carnot Bound & Kelvin–Planck Statement
The work extracted is whatever remains after mandatory disposal:
$$W = Q_1 - |Q_0| \leq Q_1 - \frac{T_0}{T_1} Q_1 = Q_1 \left(1 - \frac{T_0}{T_1}\right).$$

The thermal efficiency $\eta = W / Q_1$ is therefore bounded by the **Carnot efficiency**:
$$\mathbf{\eta = \frac{W}{Q_1} \leq 1 - \frac{T_0}{T_1} = \eta_C}.$$

#### Corollary: Kelvin–Planck Statement
If $T_0 \geq T_1$ (a single reservoir, or drawing heat from a colder source to dump into a hotter sink):
$$\eta \leq 1 - \frac{T_0}{T_1} \leq 0 \quad \implies \quad \mathbf{W \leq 0}.$$
*No cyclic machine can produce positive work from a single thermal reservoir or by transferring heat from cold to hot.*

---

## 3. Repository Structure

| File | Purpose |
|---|---|
| [`theorem54_model.py`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-4/theorem54_model.py) | Mathematical and thermodynamic model: concave entropy, tangent equations, exchange rate bounds, and Carnot budget verification. |
| [`theorem54_visualization.py`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-4/theorem54_visualization.py) | Standalone abstract visualization featuring two distinct panels/curves—one for Hot Reservoir 1 (slope $1/T_1$, shallower) and one for Cold Reservoir 0 (slope $1/T_0$, steeper)—illustrating concavity tangent bounds, deficit wedges, and symbolic equations. |
| [`theorem54_visualization.png`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-4/theorem54_visualization.png) | High-resolution rendered 300 DPI publication plot (two curves with different tangent slopes, zero numeric labels, collision-free layout, enlarged equations). |
| [`theorem54_logic.py`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-4/theorem54_logic.py) | Generates the 4-beat proof logic diagram matching repository standards (`lemma51_logic.py`, `theorem51_logic.py`). |
| [`theorem54_logic.png`](file:///Users/guillermocasasgonzalez/repos/Lieb_-_Yngvason_plots/Theorem_5-4/theorem54_logic.png) | High-resolution 2x2 logical card diagram illustrating the four steps of the proof. |

---

## 4. Running the Code

To execute the thermodynamic model test suite:
```bash
python3 Theorem_5-4/theorem54_model.py
```

To regenerate the standalone abstract geometric plot:
```bash
python3 Theorem_5-4/theorem54_visualization.py
```

To regenerate the 4-beat logical diagram:
```bash
python3 Theorem_5-4/theorem54_logic.py
```
