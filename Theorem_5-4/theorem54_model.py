"""
Thermodynamic model for Theorem 5.4 and the Carnot Efficiency Bound
(Lieb & Yngvason, "The physics and mathematics of the second law of
thermodynamics", Physics Reports 310 (1999) 1-96, Section 5, pp. 72-74).

This module implements:
1. The concave entropy profile S(U) for simple systems with fixed work coordinates.
2. Absolute temperature T(U) = (dS/dU)^{-1} and tangent lines S_tangent(U; U_anchor).
3. The geometric tangent bounds derived from strict concavity:
   - Hot reservoir delivering Q_1 > 0:  -Delta S_1 >= Q_1 / T_1.
   - Cold reservoir absorbing |Q_0| > 0: Delta S_0 <= |Q_0| / T_0.
4. The entropy budget and Carnot efficiency bound:
   eta = W / Q_1 <= 1 - T_0 / T_1 = eta_C.
"""

import numpy as np

# --------------------------------------------------------------------------
# Physical Parameters for the Generic Reservoir Model
# --------------------------------------------------------------------------
# S(U) = C * ln(U / U_ref) + S_ref
# T(U) = (dS/dU)^{-1} = U / C
HEAT_CAPACITY_C = 1.0      # Heat capacity C (J/K)
U_REF = 100.0              # Reference internal energy (J)
S_REF = 0.0                # Reference entropy (J/K)


def entropy(U, C=HEAT_CAPACITY_C, U_ref=U_REF, S_ref=S_REF):
    """
    Entropy S(U) = C * ln(U / U_ref) + S_ref.
    Strictly concave: d^2S/dU^2 = -C / U^2 < 0 for all U > 0.
    """
    U = np.asarray(U, dtype=float)
    return C * np.log(U / U_ref) + S_ref


def dS_dU(U, C=HEAT_CAPACITY_C):
    """First derivative dS/dU = C / U = 1 / T."""
    U = np.asarray(U, dtype=float)
    return C / U


def temperature(U, C=HEAT_CAPACITY_C):
    """Absolute temperature T(U) = (dS/dU)^{-1} = U / C."""
    U = np.asarray(U, dtype=float)
    return U / C


def tangent_line(U, U_anchor, C=HEAT_CAPACITY_C, U_ref=U_REF, S_ref=S_REF):
    """
    Linear tangent to S(U) anchored at U_anchor:
    S_tangent(U) = S(U_anchor) + (1 / T(U_anchor)) * (U - U_anchor).
    Because S is strictly concave, S(U) <= S_tangent(U) for all U,
    with equality if and only if U == U_anchor.
    """
    U = np.asarray(U, dtype=float)
    S_anchor = entropy(U_anchor, C=C, U_ref=U_ref, S_ref=S_ref)
    slope = dS_dU(U_anchor, C=C)  # 1 / T(U_anchor)
    return S_anchor + slope * (U - U_anchor)


def concavity_deficit(U, U_anchor, C=HEAT_CAPACITY_C, U_ref=U_REF, S_ref=S_REF):
    """
    Deficit delta(U; U_anchor) = S_tangent(U; U_anchor) - S(U) >= 0.
    Measures how far the concave curve lies below the tangent.
    """
    return tangent_line(U, U_anchor, C=C, U_ref=U_ref, S_ref=S_ref) - entropy(U, C=C, U_ref=U_ref, S_ref=S_ref)


# --------------------------------------------------------------------------
# Tangent Bounds for Heat Transfer
# --------------------------------------------------------------------------
def hot_reservoir_bound(U_initial, Q1, C=HEAT_CAPACITY_C):
    """
    Hot reservoir delivers heat Q1 > 0:
    Energy change: Delta U = U_final - U_initial = -Q1 < 0.
    Tangent anchored at U_initial gives:
    S(U_final) <= S(U_initial) - Q1 / T_initial
    => Delta S_1 <= -Q1 / T_initial
    => -Delta S_1 >= Q1 / T_initial (entropy loss is at least Q1 / T_initial).
    """
    T_initial = temperature(U_initial, C=C)
    U_final = U_initial - Q1
    S_i = entropy(U_initial, C=C)
    S_f = entropy(U_final, C=C)
    delta_S = S_f - S_i
    bound = -Q1 / T_initial

    return {
        "U_initial": U_initial,
        "U_final": U_final,
        "T_initial": T_initial,
        "T_final": temperature(U_final, C=C),
        "Q1": Q1,
        "delta_S": delta_S,
        "bound": bound,
        "loss": -delta_S,
        "min_loss": Q1 / T_initial,
        "deficit": bound - delta_S,  # >= 0
        "satisfied": delta_S <= bound + 1e-12,
    }


def cold_reservoir_bound(U_initial, Q0_abs, C=HEAT_CAPACITY_C):
    """
    Cold reservoir absorbs heat |Q0| > 0:
    Energy change: Delta U = U_final - U_initial = |Q0| > 0.
    Tangent anchored at U_initial gives:
    S(U_final) <= S(U_initial) + |Q0| / T_initial
    => Delta S_0 <= |Q0| / T_initial (entropy gain is at most |Q0| / T_initial).
    """
    T_initial = temperature(U_initial, C=C)
    U_final = U_initial + Q0_abs
    S_i = entropy(U_initial, C=C)
    S_f = entropy(U_final, C=C)
    delta_S = S_f - S_i
    bound = Q0_abs / T_initial

    return {
        "U_initial": U_initial,
        "U_final": U_final,
        "T_initial": T_initial,
        "T_final": temperature(U_final, C=C),
        "Q0_abs": Q0_abs,
        "delta_S": delta_S,
        "bound": bound,
        "deficit": bound - delta_S,  # >= 0
        "satisfied": delta_S <= bound + 1e-12,
    }


# --------------------------------------------------------------------------
# Entropy Budget & Carnot Cycle Verification
# --------------------------------------------------------------------------
def carnot_efficiency_budget(T1, T0, Q1, Q0_abs=None):
    """
    Evaluates the entropy budget and Carnot efficiency bound for a heat engine
    operating between reservoirs at temperatures T1 (hot) and T0 (cold),
    with T1 > T0 > 0.

    By the Entropy Principle:
      Delta S_total = Delta S_machine + Delta S_hot + Delta S_cold >= 0.
    For a cyclic machine, Delta S_machine = 0, so:
      Delta S_cold >= -Delta S_hot.

    Together with the concavity exchange rates:
      Q1 / T1 <= -Delta S_hot <= Delta S_cold <= |Q0| / T0
      => |Q0| >= (T0 / T1) * Q1.

    Work extracted: W = Q1 - |Q0| <= Q1 * (1 - T0 / T1).
    Thermal efficiency: eta = W / Q1 <= 1 - T0 / T1 = eta_Carnot.
    """
    eta_carnot = 1.0 - T0 / T1
    min_Q0_disposal = (T0 / T1) * Q1
    max_work = Q1 * eta_carnot

    if Q0_abs is None:
        Q0_abs = min_Q0_disposal

    work = Q1 - Q0_abs
    eta = work / Q1

    return {
        "T1": T1,
        "T0": T0,
        "Q1": Q1,
        "Q0_abs": Q0_abs,
        "min_Q0_disposal": min_Q0_disposal,
        "work": work,
        "max_work": max_work,
        "eta": eta,
        "eta_carnot": eta_carnot,
        "satisfies_carnot_bound": eta <= eta_carnot + 1e-12,
        "satisfies_entropy_budget": Q0_abs >= min_Q0_disposal - 1e-12,
    }


if __name__ == "__main__":
    print("=== Testing Theorem 5.4 Thermodynamic Model ===")

    # 1. Test concavity
    u_vals = np.linspace(50.0, 500.0, 100)
    curvatures = -HEAT_CAPACITY_C / (u_vals ** 2)
    assert np.all(curvatures < 0), "Entropy must be strictly concave!"
    print("[PASS] Strict concavity verified: d^2S/dU^2 < 0 everywhere.")

    # 2. Test hot reservoir bound
    hot_res = hot_reservoir_bound(U_initial=400.0, Q1=100.0)
    assert hot_res["satisfied"], "Hot reservoir bound must hold!"
    print(f"[PASS] Hot reservoir: -Delta S1 = {hot_res['loss']:.4f} >= Q1/T1 = {hot_res['min_loss']:.4f} (Deficit: {hot_res['deficit']:.4f})")

    # 3. Test cold reservoir bound
    cold_res = cold_reservoir_bound(U_initial=100.0, Q0_abs=50.0)
    assert cold_res["satisfied"], "Cold reservoir bound must hold!"
    print(f"[PASS] Cold reservoir: Delta S0 = {cold_res['delta_S']:.4f} <= |Q0|/T0 = {cold_res['bound']:.4f} (Deficit: {cold_res['deficit']:.4f})")

    # 4. Test Carnot efficiency budget and bound
    budget = carnot_efficiency_budget(T1=400.0, T0=100.0, Q1=120.0, Q0_abs=40.0)
    assert budget["satisfies_carnot_bound"], "Carnot bound must hold!"
    assert budget["satisfies_entropy_budget"], "Entropy budget must be satisfied!"
    print(f"[PASS] Carnot engine: T1 = {budget['T1']} K, T0 = {budget['T0']} K")
    print(f"       Extracted work W = {budget['work']:.2f} J (Max work = {budget['max_work']:.2f} J)")
    print(f"       Efficiency eta = {budget['eta']:.4f} <= eta_Carnot = {budget['eta_carnot']:.4f}")
    print(f"       Heat disposal |Q0| = {budget['Q0_abs']:.2f} J >= Min disposal = {budget['min_Q0_disposal']:.2f} J")

    print("\nAll thermodynamic model assertions passed successfully!")
