"""
Mathematical and thermodynamic model for Theorem 5.5 figures
(Lieb & Yngvason, "The physics and mathematics of the second law of
thermodynamics", Physics Reports 310 (1999) 1-96, Section 5.2, pp. 73-75).

------------------------------------------------------------------------
Theorem 5.5 (Isotherms cut adiabats).
    "Suppose X_0 < X < X_1 and X_0 and X_1 have equal temperatures,
     T(X_0) = T(X_1) = T_0.
     (1) If T_min < T_0 < T_max then there is a point X' ~_A X with T(X') = T_0.
         In other words: The isotherm through X_0 cuts every adiabat
         between X_0 and X_1.
     (2) If T_0 = T_max, then either there is an X' ~_A X with T(X') = T_0,
         or, for any T'_0 < T_0 there exist points X'_0, X' and X'_1 with
         X'_0 < X' ~_A X < X'_1 and T(X'_0) = T(X') = T(X'_1) = T'_0.
     (3) If T_0 = T_min, then either there is an X' ~_A X with T(X') = T_0,
         or, for any T'_0 > T_0 there exist points X'_0, X' and X'_1 with
         X'_0 < X' ~_A X < X'_1 and T(X'_0) = T(X') = T(X'_1) = T'_0."
------------------------------------------------------------------------

Thermodynamic Framework:
We model a simple fluid system in the state space Gamma subset R^2:
  * Horizontal coordinate: Work coordinate / volume V
  * Vertical coordinate: Internal energy U
  * State space: Gamma = { (V, U) : V > b, U + a/V > 0, U < H(V) }
    with an affine ceiling H(V) = h0 + h1 * V (h1 > 0), giving Gamma a genuine boundary.
  * Entropy function: S(U, V) = C_V * ln(U + a/V) + R * ln(V - b)
  * Temperature function: T(U, V) = (U + a/V) / C_V
  * Pressure function: P(U, V) = R * T(U, V) / (V - b) - a / (V^2)

Geometric and Topological Structure of Theorem 5.5:
1. Open connected regions:
     X_+ = { (V, U) in Gamma : T(U, V) > T_0 }   (hotter region)
     X_- = { (V, U) in Gamma : T(U, V) < T_0 }   (colder region)
   The isotherm I_{T_0} = { (V, U) in Gamma : T(U, V) = T_0 } separates X_+ from X_-.
2. Slope of the isotherm:
     Along I_{T_0}: U(V) = C_V * T_0 - a/V => dU/dV|_T = a / V^2 > 0.
3. Slope of the adiabats:
     Along dA_X: dU = -P dV => dU/dV|_S = -P < 0.
   The isotherm and adiabats intersect transversally with opposite signs of slope.
4. Intermediate adiabat dA_X:
     For any X with X_0 < X < X_1 (entropy S_0 < S(X) < S_1), dA_X connects
     a compressed state in X_+ (where T > T_0) to an expanded state in X_- (where T < T_0).
5. Intermediate Value Theorem:
     Since dA_X is connected (Axiom S3) and T is continuous (Theorem 5.2),
     there exists a point X' in dA_X with T(X') = T_0, proving that
     the isotherm I_{T_0} cuts dA_X.
"""

# Deliberate simplification: With a hard ceiling the per-column temperature supremum varies with V, so axiom T5 holds here only qualitatively; the figure illustrates the objects of the proof, like the paper's own schematic Fig. 4, not a fully axiom-faithful model.

import numpy as np
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Model Parameters (Van der Waals / real gas fluid)
# --------------------------------------------------------------------------
CV = 1.5           # isochoric heat capacity (dimensionless units)
R_GAS = 1.0        # gas constant
A_PARAM = 2.0      # attractive potential parameter
B_PARAM = 0.5      # excluded volume parameter
T0 = 2.00          # base isotherm temperature


# --------------------------------------------------------------------------
# Fundamental State Functions
# --------------------------------------------------------------------------
def entropy(U, V):
    """
    Entropy S(U, V) = C_V * ln(U + a/V) + R * ln(V - b).
    Strictly concave on Gamma = { (V, U) : V > b, U + a/V > 0, U < H(V) }.
    """
    U = np.asarray(U)
    V = np.asarray(V)
    return CV * np.log(U + A_PARAM / V) + R_GAS * np.log(V - B_PARAM)


def temperature(U, V):
    """
    Absolute temperature T(U, V) = (dS/dU)^{-1} = (U + a/V) / C_V.
    Strictly positive and continuous everywhere on Gamma (Theorem 5.2).
    """
    U = np.asarray(U)
    V = np.asarray(V)
    return (U + A_PARAM / V) / CV


def pressure(U, V):
    """
    Pressure P(U, V) = T * (dS/dV) = R * T / (V - b) - a / V^2.
    Locally Lipschitz and strictly positive on the working domain.
    """
    t = temperature(U, V)
    V = np.asarray(V)
    return R_GAS * t / (V - B_PARAM) - A_PARAM / (V ** 2)


# --------------------------------------------------------------------------
# Analytic Level Curves (Isotherms & Adiabats)
# --------------------------------------------------------------------------
def U_isotherm(V, T=T0):
    """
    Internal energy U as a function of V along the isotherm T = const:
    U(V; T) = C_V * T - a / V.
    """
    V = np.asarray(V)
    return CV * T - A_PARAM / V


def V_isotherm(U, T=T0):
    """
    Inverse: Volume V as a function of U along the isotherm T = const:
    V(U; T) = a / (C_V * T - U).
    """
    U = np.asarray(U)
    return A_PARAM / (CV * T - U)


def U_adiabat(V, S_const):
    """
    Internal energy U as a function of V along an adiabat S(U, V) = S_const:
    U(V; S) = exp((S - R * ln(V - b)) / C_V) - a / V.
    """
    V = np.asarray(V)
    return np.exp((S_const - R_GAS * np.log(V - B_PARAM)) / CV) - A_PARAM / V


def slope_isotherm(V):
    """
    Tangent slope dU/dV along the isotherm in (V, U) plane:
    dU/dV|_T = a / V^2 > 0.
    """
    return A_PARAM / (V ** 2)


def slope_adiabat(U, V):
    """
    Tangent slope dU/dV along an adiabat in (V, U) plane:
    dU/dV|_S = -P(U, V) < 0.
    """
    p = pressure(U, V)
    return -float(p)


# --------------------------------------------------------------------------
# Ceiling Boundary H(V) and State Space Boundary Functions
# --------------------------------------------------------------------------
# Ceiling slope h1 > 0 and intercept h0 chosen so that the adiabat of level
# S_MID exits Gamma through dGamma at volume V_B = 0.98.
H1 = 0.70

# We first define the shared panel anchors to fix S_MID
V0a = 1.05   # Case (a) anchor volume (in rho(A_X))
V0b = 0.82   # Case (b) anchor volume (trapped column, not in rho(A_X))
V1 = 2.50    # Upper anchor volume on isotherm

U0a = float(U_isotherm(V0a, T0))
S0a = float(entropy(U0a, V0a))

U0b = float(U_isotherm(V0b, T0))
S0b = float(entropy(U0b, V0b))

U1 = float(U_isotherm(V1, T0))
S1 = float(entropy(U1, V1))

# Intermediate adiabat entropy
alpha_weight = 0.35
S_MID = alpha_weight * S0a + (1.0 - alpha_weight) * S1

# Exit volume for dA_X: V_B = 0.98 solves Sigma(V_B) = S_MID
V_B = 0.98
u_b = float(U_adiabat(V_B, S_MID))
H0 = u_b - H1 * V_B


def ceiling(V):
    """Affine ceiling H(V) = h0 + h1 * V defining the upper boundary dGamma."""
    V = np.asarray(V)
    return H0 + H1 * V


def ceiling_entropy(V):
    """Entropy along the ceiling Sigma(V) = S(H(V), V). Strictly increasing."""
    return entropy(ceiling(V), V)


def rho_min(S_const):
    """
    Computes the minimum volume V_b of rho(A_X) = (V_b, infty) where the
    adiabat of entropy S_const exits through the ceiling boundary dGamma:
    Sigma(V_b) = S_const.
    """
    def obj(v):
        return ceiling_entropy(v) - S_const
    return float(brentq(obj, B_PARAM + 1e-4, 10.0))


# --------------------------------------------------------------------------
# Key States for Theorem 5.5
# --------------------------------------------------------------------------
X0a = np.array([V0a, U0a])
X0b = np.array([V0b, U0b])
X1 = np.array([V1, U1])

# Guaranteed intersection point X' = dA_X \cap I_{T_0}
# At X', T = T0 and S = S_MID
V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T0)) / R_GAS))
U_PRIME = float(U_isotherm(V_PRIME, T0))
X_PRIME = np.array([V_PRIME, U_PRIME])

# Hot trapped state X'_0 on column V0b (T5 gives T(X'_0) > T0 while S(X'_0) < S_MID)
# Placed between U_isotherm(V0b) and H(V0b)
U0_PRIME = 1.80
X0_PRIME = np.array([V0b, U0_PRIME])

# Hot state X'_1 on column V1 (Planck + T5 gives T(X'_1) > T0 and S(X'_1) > S_MID)
U1_PRIME = 3.25
X1_PRIME = np.array([V1, U1_PRIME])

# Case (a) straddling points on dA_X
U_GT = float(U_adiabat(V0a, S_MID))
X_GT = np.array([V0a, U_GT])

U_LT = float(U_adiabat(V1, S_MID))
X_LT = np.array([V1, U_LT])

# Case (b) straddling points on dA_X
V_UP = float(0.40 * V_B + 0.60 * V_PRIME)
U_UP = float(U_adiabat(V_UP, S_MID))
X_UP = np.array([V_UP, U_UP])

V_DOWN = 2.15
U_DOWN = float(U_adiabat(V_DOWN, S_MID))
X_DOWN = np.array([V_DOWN, U_DOWN])


# --------------------------------------------------------------------------
# Verification Suite
# --------------------------------------------------------------------------
def run_model_checks():
    """Validates all mathematical and thermodynamic conditions of Theorem 5.5."""
    # 1. Temperature equality on anchor states
    assert np.isclose(temperature(U0a, V0a), T0), "T(X0a) must equal T0"
    assert np.isclose(temperature(U0b, V0b), T0), "T(X0b) must equal T0"
    assert np.isclose(temperature(U1, V1), T0), "T(X1) must equal T0"

    # 2. Strict adiabatic accessibility ordering
    assert S0a < S_MID < S1, f"Expected S0a < S_MID < S1, got {S0a:.3f} < {S_MID:.3f} < {S1:.3f}"
    assert S0b < S_MID < S1, f"Expected S0b < S_MID < S1, got {S0b:.3f} < {S_MID:.3f} < {S1:.3f}"

    # 3. Intersection state X' properties
    assert np.isclose(temperature(U_PRIME, V_PRIME), T0), "T(X') must equal T0"
    assert np.isclose(entropy(U_PRIME, V_PRIME), S_MID), "S(X') must equal S_MID"

    # 4. Transversality of slopes at X'
    slope_iso = slope_isotherm(V_PRIME)
    slope_adia = slope_adiabat(U_PRIME, V_PRIME)
    assert slope_iso > 0, "Isotherm slope dU/dV must be positive"
    assert slope_adia < 0, "Adiabat slope dU/dV must be negative"
    assert slope_iso != slope_adia, "Slopes must not coincide (transversal crossing)"

    # 5. Extended Prompt Assertions (a)-(g):
    # a. Sigma(V0b) < S_MID and entropy(U0b, V0b) < Sigma(V0b) [trapped column]
    Sigma_V0b = float(ceiling_entropy(V0b))
    assert Sigma_V0b < S_MID, f"Expected Sigma(V0b) < S_MID, got {Sigma_V0b:.4f} >= {S_MID:.4f}"
    assert entropy(U0b, V0b) < Sigma_V0b, "Expected entropy(U0b, V0b) < Sigma(V0b)"

    # b. abs(entropy(H(Vb), Vb) - S_MID) < 1e-9 for Vb = rho_min(S_MID) [adiabat exits ON dGamma]
    Vb = rho_min(S_MID)
    assert abs(entropy(ceiling(Vb), Vb) - S_MID) < 1e-9, "Expected adiabat to exit ON dGamma"

    # c. U_adiabat(V0b, S_MID) > H(V0b) and U_adiabat(V0b, S1) > H(V0b)
    #    [neither dA_X nor dA_{X1} intersects the trapped column in Gamma]
    assert U_adiabat(V0b, S_MID) > ceiling(V0b), "dA_X must lie above ceiling on column V0b"
    assert U_adiabat(V0b, S1) > ceiling(V0b), "dA_{X1} must lie above ceiling on column V0b"

    # d. T(X'_0) > T0, S(X'_0) < S_MID; T(X'_1) > T0, S(X'_1) > S_MID
    assert temperature(U0_PRIME, V0b) > T0, "T(X'_0) must be > T0"
    assert entropy(U0_PRIME, V0b) < S_MID, "S(X'_0) must be < S_MID"
    assert temperature(U1_PRIME, V1) > T0, "T(X'_1) must be > T0"
    assert entropy(U1_PRIME, V1) > S_MID, "S(X'_1) must be > S_MID"

    # e. U_isotherm(V) < H(V) on the plotted V-window [isotherm stays inside Gamma]
    v_window = np.linspace(0.55, 3.05, 500)
    assert np.all(U_isotherm(v_window) < ceiling(v_window)), "Isotherm must stay inside Gamma"

    # f. V0b < rho_min(S_MID) < V_PRIME < V_{X1}
    assert V0b < Vb < V_PRIME < V1, f"Expected {V0b} < {Vb:.3f} < {V_PRIME:.3f} < {V1}"

    # g. Case (a): V0a > rho_min(S_MID), and Case (a) points X_>, X_< lie strictly below ceiling
    assert V0a > Vb, f"Expected V0a > Vb, got {V0a} <= {Vb:.3f}"
    assert U_GT < ceiling(V0a), "X_> must lie strictly below the ceiling"
    assert U_LT < ceiling(V1), "X_< must lie strictly below the ceiling"

    print("All Theorem 5.5 model verification checks passed successfully!")
    print(f"  Ceiling dGamma: H(V) = {H0:.4f} + {H1:.4f} * V")
    print(f"  Exit volume V_b = rho_min(S_MID) = {Vb:.4f} with H(V_b) = {ceiling(Vb):.4f}")
    print(f"  Case (a) Anchor X_0a: V={V0a:.3f}, U={U0a:.3f} => S_0a={S0a:.3f}, T={T0:.3f}")
    print(f"  Case (b) Anchor X_0b: V={V0b:.3f}, U={U0b:.3f} => S_0b={S0b:.3f}, T={T0:.3f}")
    print(f"  Upper Anchor X_1:     V={V1:.3f}, U={U1:.3f} => S_1={S1:.3f}, T={T0:.3f}")
    print(f"  Cutting State X':     V={V_PRIME:.3f}, U={U_PRIME:.3f} => S={S_MID:.3f}, T={T0:.3f}")
    print(f"  Trapped Column V0b:   Sigma(V0b)={Sigma_V0b:.3f} < S_MID={S_MID:.3f}")
    print(f"  Hot Trapped State X'_0: V={V0b:.3f}, U={U0_PRIME:.3f} => T={temperature(U0_PRIME, V0b):.3f} > T0, S={entropy(U0_PRIME, V0b):.3f} < S_MID")
    print(f"  Hot State X'_1:         V={V1:.3f}, U={U1_PRIME:.3f} => T={temperature(U1_PRIME, V1):.3f} > T0, S={entropy(U1_PRIME, V1):.3f} > S_MID")


if __name__ == "__main__":
    run_model_checks()
