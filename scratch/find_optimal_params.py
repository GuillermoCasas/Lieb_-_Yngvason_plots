import numpy as np
from scipy.optimize import brentq

CV = 1.5
R_GAS = 1.0
A_PARAM = 2.0
B_PARAM = 0.5
T0 = 2.00

def entropy(U, V):
    return CV * np.log(U + A_PARAM / V) + R_GAS * np.log(V - B_PARAM)

def temperature(U, V):
    return (U + A_PARAM / V) / CV

def U_isotherm(V, T=T0):
    return CV * T - A_PARAM / V

def U_adiabat(V, S_const):
    return np.exp((S_const - R_GAS * np.log(V - B_PARAM)) / CV) - A_PARAM / V

# We want:
# 1. V0b < V_b < V0a < V_PRIME < V1
# 2. In Panel 1:
#    U(X0a) and U(X_<) well separated: |U(X0a) - U(X_<)| >= 0.35
#    U(X_<) and U(X1) well separated: |U(X1) - U(X_<)| >= 0.5
#    U(X_>) and U(X1) well separated: |U(X_>) - U(X1)| >= 0.5
# 3. In Panel 2:
#    V_b - V0b >= 0.20 (ample room for trapped line, annotations, and tick labels)
#    V_PRIME - V_b >= 0.35 (ample room for X_up, Omega_> path, X')
#    U(X0b) >= 0.35 (well above u_min = 0.20)
#    H(V0b) - U(X0b) >= 1.5 (ample room for X'_0 and dotted arrow)
#    H(V1) - U(X1) >= 1.0 (ample room for X'_1 and dotted arrow)
# 4. All assertions (a)-(g) pass!

results = []

for V0b in [0.75, 0.78, 0.80, 0.82]:
    U0b = float(U_isotherm(V0b))
    S0b = float(entropy(U0b, V0b))
    for V1 in [2.50, 2.60, 2.70]:
        U1 = float(U_isotherm(V1))
        S1 = float(entropy(U1, V1))
        for V_b in np.linspace(V0b + 0.18, V0b + 0.35, 10):
            # For each V_b, we can choose S_MID
            # What S_MID makes V_b the exit?
            # Along the ceiling H(V) = h0 + h1 * V
            for h1 in [0.4, 0.5, 0.6, 0.7, 0.8]:
                # Let's test different S_MID
                for alpha in np.linspace(0.40, 0.65, 10):
                    # We will test candidate V0a
                    for V0a in np.linspace(V_b + 0.08, V_b + 0.25, 6):
                        U0a = float(U_isotherm(V0a))
                        S0a = float(entropy(U0a, V0a))
                        S_MID = alpha * S0a + (1 - alpha) * S1
                        V_PRIME = float(B_PARAM + np.exp((S_MID - CV * np.log(CV * T0)) / R_GAS))
                        if not (V_b < V0a < V_PRIME < V1):
                            continue
                        if V_PRIME - V_b < 0.25:
                            continue
                        # Determine h0 so that Sigma(V_b) = S_MID
                        # That means H(V_b) = U_adiabat(V_b, S_MID)
                        u_b = float(U_adiabat(V_b, S_MID))
                        h0 = u_b - h1 * V_b
                        
                        def ceiling(v):
                            return h0 + h1 * v
                        def ceiling_entropy(v):
                            return entropy(ceiling(v), v)
                        
                        # Check Sigma strictly increasing
                        v_test = np.linspace(0.55, 3.05, 50)
                        sig_test = ceiling_entropy(v_test)
                        if not np.all(np.diff(sig_test) > 0):
                            continue
                        # Check isotherm inside Gamma
                        if np.any(ceiling(v_test) <= U_isotherm(v_test)):
                            continue
                        # Check assertion a: Sigma(V0b) < S_MID and entropy(U0b, V0b) < Sigma(V0b)
                        sig_0b = float(ceiling_entropy(V0b))
                        if not (S0b < sig_0b < S_MID):
                            continue
                        # Check assertion c:
                        if U_adiabat(V0b, S_MID) <= ceiling(V0b) or U_adiabat(V0b, S1) <= ceiling(V0b):
                            continue
                        # Check assertion d:
                        U0_prime = 0.5 * (U0b + ceiling(V0b))
                        t0_prime = temperature(U0_prime, V0b)
                        s0_prime = entropy(U0_prime, V0b)
                        if not (t0_prime > T0 and s0_prime < S_MID):
                            continue
                        U1_prime = 0.5 * (U1 + ceiling(V1))
                        t1_prime = temperature(U1_prime, V1)
                        s1_prime = entropy(U1_prime, V1)
                        if not (t1_prime > T0 and s1_prime > S_MID):
                            continue
                        # Check assertion g: X_>, X_< below ceiling
                        U_GT_a = float(U_adiabat(V0a, S_MID))
                        U_LT_a = float(U_adiabat(V1, S_MID))
                        if U_GT_a >= ceiling(V0a) or U_LT_a >= ceiling(V1):
                            continue
                        # Now check visual spacing in Panel 1:
                        # y ticks in Panel 1: U0a, U_LT_a, U1, U_GT_a
                        # We want all adjacent pairs well separated:
                        y_panel1 = sorted([U0a, U_LT_a, U1, U_GT_a])
                        min_y_diff_p1 = min(np.diff(y_panel1))
                        if min_y_diff_p1 < 0.25:
                            continue
                        # Visual spacing in Panel 2:
                        y_panel2 = sorted([U0b, U1, U0_prime, U1_prime])
                        min_y_diff_p2 = min(np.diff(y_panel2))
                        if min_y_diff_p2 < 0.25:
                            continue
                        
                        score = min_y_diff_p1 + min_y_diff_p2 + (V_b - V0b) + (V_PRIME - V_b)
                        results.append((score, V0b, V_b, V0a, V_PRIME, V1, h0, h1, S_MID, min_y_diff_p1, min_y_diff_p2))

results.sort(key=lambda x: x[0], reverse=True)
print(f"Total valid parameter configurations found: {len(results)}")
for r in results[:5]:
    score, V0b, V_b, V0a, V_PRIME, V1, h0, h1, S_MID, dy1, dy2 = r
    print(f"Score: {score:.2f} | V0b={V0b:.2f}, V_b={V_b:.2f}, V0a={V0a:.2f}, V'={V_PRIME:.2f}, V1={V1:.2f}")
    print(f"   h0={h0:.3f}, h1={h1:.2f}, dy_p1={dy1:.2f}, dy_p2={dy2:.2f}")
