import numpy as np

CV = 1.5
R_GAS = 1.0
A_PARAM = 2.0
B_PARAM = 0.5
T0 = 2.00

def U_isotherm(V, T=T0):
    return CV * T - A_PARAM / V

def U_adiabat(V, S_const):
    return np.exp((S_const - R_GAS * np.log(V - B_PARAM)) / CV) - A_PARAM / V

def entropy(U, V):
    return CV * np.log(U + A_PARAM / V) + R_GAS * np.log(V - B_PARAM)

# V0b = 0.82, V1 in [2.4, 2.6]
V0b = 0.82
for V1 in [2.40, 2.50, 2.60]:
    U1 = float(U_isotherm(V1))
    S1 = float(entropy(U1, V1))
    for V0a in [1.10, 1.15, 1.20, 1.25]:
        U0a = float(U_isotherm(V0a))
        S0a = float(entropy(U0a, V0a))
        for s_mid in np.linspace(S0a + 0.1, S1 - 0.2, 50):
            v_prime = float(B_PARAM + np.exp((s_mid - CV * np.log(CV * T0)) / R_GAS))
            if not (V0a < v_prime < V1):
                continue
            u_gt = float(U_adiabat(V0a, s_mid))
            u_lt = float(U_adiabat(V1, s_mid))
            
            # Y ticks in Panel 1: U0a, u_lt, U1, u_gt
            y_ticks = sorted([U0a, u_lt, U1, u_gt])
            dy = np.diff(y_ticks)
            if np.min(dy) >= 0.28:
                print(f"FOUND: V1={V1}, V0a={V0a}, s_mid={s_mid:.3f}, v_prime={v_prime:.3f}")
                print(f"  y_ticks: {[round(y, 3) for y in y_ticks]}")
                print(f"  min_dy: {np.min(dy):.3f}")
                break
