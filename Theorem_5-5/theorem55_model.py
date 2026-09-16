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
  * Horizontal coordinate: Internal energy U
  * Vertical coordinate: Work coordinate / volume V
  * State space: Gamma = { (U, V) : V > b, U + a/V > 0 }
  * Entropy function: S(U, V) = C_V * ln(U + a/V) + R * ln(V - b)
  * Temperature function: T(U, V) = (U + a/V) / C_V
  * Pressure function: P(U, V) = R * T(U, V) / (V - b) - a / (V^2)

Geometric and Topological Structure of Theorem 5.5:
1. Open connected regions:
     X_+ = { (U, V) in Gamma : T(U, V) > T_0 }   (hotter region)
     X_- = { (U, V) in Gamma : T(U, V) < T_0 }   (colder region)
   The isotherm I_{T_0} = { (U, V) : T(U, V) = T_0 } separates X_+ from X_-.
2. Slope of the isotherm:
     Along I_{T_0}: U(V) = C_V * T_0 - a/V => dV/dU|_T = V^2 / a > 0.
3. Slope of the adiabats:
     Along dA_X: dU = -P dV => dV/dU|_S = -1/P < 0.
   The isotherm and adiabats intersect transversally with opposite signs of slope.
4. Intermediate adiabat dA_X:
     For any X with X_0 < X < X_1 (entropy S_0 < S(X) < S_1), dA_X connects
     a compressed state X_+ in X_+ (where T > T_0) to an expanded state X_-
     in X_- (where T < T_0).
5. Intermediate Value Theorem:
     Since dA_X is connected (Axiom S4) and T is continuous (Theorem 5.2),
     there exists a point X' in dA_X with T(X') = T_0, proving that
     the isotherm I_{T_0} cuts dA_X.
"""

import numpy as np

# --------------------------------------------------------------------------
# Model Parameters (Van der Waals / real gas fluid)
# --------------------------------------------------------------------------
CV = 1.5           # isochoric heat capacity (dimensionless units)
R_GAS = 1.0        # gas constant
A_PARAM = 0.45     # attractive potential parameter
B_PARAM = 0.20     # excluded volume parameter
T0 = 2.00          # base isotherm temperature


# --------------------------------------------------------------------------
# Fundamental State Functions
# --------------------------------------------------------------------------
def entropy(U, V):
    """
    Entropy S(U, V) = C_V * ln(U + a/V) + R * ln(V - b).
    Strictly concave on Gamma = { (U, V) : V > b, U + a/V > 0 }.
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
    Tangent slope dV/dU along the isotherm in (U, V) plane:
    dV/dU|_T = V^2 / a > 0.
    """
    return (V ** 2) / A_PARAM


def slope_adiabat(U, V):
    """
    Tangent slope dV/dU along an adiabat in (U, V) plane:
    dV/dU|_S = -1 / P(U, V) < 0.
    """
    p = pressure(U, V)
    return -1.0 / p


# --------------------------------------------------------------------------
# Key States for Theorem 5.5
# --------------------------------------------------------------------------
# State X_0 on isotherm T0 (lower entropy anchor)
V0 = 0.82
U0 = float(U_isotherm(V0, T0))
S0 = float(entropy(U0, V0))
X0 = np.array([U0, V0])

# State X_1 on isotherm T0 (higher entropy anchor, X_0 < X_1)
V1 = 2.50
U1 = float(U_isotherm(V1, T0))
S1 = float(entropy(U1, V1))
X1 = np.array([U1, V1])

# Intermediate adiabat dA_X with entropy S_mid strictly between S_0 and S_1
# (X_0 < X < X_1)
S_MID = 0.52 * S0 + 0.48 * S1

# Guaranteed intersection point X' = dA_X \cap I_{T_0}
# At X', T = T0 and S = S_MID
V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T0)) / R_GAS))
U_PRIME = float(U_isotherm(V_PRIME, T0))
X_PRIME = np.array([U_PRIME, V_PRIME])

# Proof test points along the connected curve dA_X:
# Compressed state X_+ in hotter region X_+ (T > T0)
V_PLUS = 0.80
U_PLUS = float(U_adiabat(V_PLUS, S_MID))
T_PLUS = float(temperature(U_PLUS, V_PLUS))
X_PLUS = np.array([U_PLUS, V_PLUS])

# Expanded state X_- in colder region X_- (T < T0)
V_MINUS = 2.40
U_MINUS = float(U_adiabat(V_MINUS, S_MID))
T_MINUS = float(temperature(U_MINUS, V_MINUS))
X_MINUS = np.array([U_MINUS, V_MINUS])


# --------------------------------------------------------------------------
# Verification Suite
# --------------------------------------------------------------------------
def run_model_checks():
    """Validates all mathematical and thermodynamic conditions of Theorem 5.5."""
    # 1. Temperature equality on anchor states
    assert np.isclose(temperature(U0, V0), T0), "T(X0) must equal T0"
    assert np.isclose(temperature(U1, V1), T0), "T(X1) must equal T0"

    # 2. Strict adiabatic accessibility ordering X0 < X < X1
    assert S0 < S_MID < S1, f"Expected S0 < S_mid < S1, got {S0:.3f} < {S_MID:.3f} < {S1:.3f}"

    # 3. Intersection state X' properties
    assert np.isclose(temperature(U_PRIME, V_PRIME), T0), "T(X') must equal T0"
    assert np.isclose(entropy(U_PRIME, V_PRIME), S_MID), "S(X') must equal S_mid"
    assert V0 < V_PRIME < V1, "V' must lie strictly between V0 and V1 along the isotherm"

    # 4. Proof points X+ and X- along dA_X
    assert np.isclose(entropy(U_PLUS, V_PLUS), S_MID), "X+ must lie on adiabat dA_X"
    assert np.isclose(entropy(U_MINUS, V_MINUS), S_MID), "X- must lie on adiabat dA_X"
    assert T_PLUS > T0, f"X+ must lie in X_+ (hotter region), got T_PLUS={T_PLUS:.3f} > {T0}"
    assert T_MINUS < T0, f"X- must lie in X_- (colder region), got T_MINUS={T_MINUS:.3f} < {T0}"

    # 5. Transversality of slopes at X'
    slope_iso = slope_isotherm(V_PRIME)
    slope_adia = slope_adiabat(U_PRIME, V_PRIME)
    assert slope_iso > 0, "Isotherm slope dV/dU must be positive"
    assert slope_adia < 0, "Adiabat slope dV/dU must be negative"
    assert slope_iso != slope_adia, "Slopes must not coincide (transversal crossing)"

    print("All Theorem 5.5 model verification checks passed successfully!")
    print(f"  Anchor State X_0: U={U0:.3f}, V={V0:.3f} => S_0={S0:.3f}, T={T0:.3f}")
    print(f"  Anchor State X_1: U={U1:.3f}, V={V1:.3f} => S_1={S1:.3f}, T={T0:.3f}")
    print(f"  Cutting State X': U={U_PRIME:.3f}, V={V_PRIME:.3f} => S={S_MID:.3f}, T={T0:.3f}")
    print(f"  Proof Point X_+:  U={U_PLUS:.3f}, V={V_PLUS:.3f} => T={T_PLUS:.3f} > T_0")
    print(f"  Proof Point X_-:  U={U_MINUS:.3f}, V={V_MINUS:.3f} => T={T_MINUS:.3f} < T_0")
    print(f"  Slopes at X':     dV/dU|_T = +{slope_iso:.3f}, dV/dU|_S = {slope_adia:.3f}")


if __name__ == "__main__":
    run_model_checks()
