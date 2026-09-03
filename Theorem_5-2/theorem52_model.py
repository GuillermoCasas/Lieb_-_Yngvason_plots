"""
Shared thermodynamic model for the Theorem 5.2 figures
(Lieb & Yngvason, "The physics and mathematics of the second law of
thermodynamics", Physics Reports 310 (1999) 1-96, p. 71).

------------------------------------------------------------------------
Theorem 5.2 (Continuity of temperature).
    "The temperature T(X) = T^+(X) = T^-(X) is a continuous function on
     the state space, Gamma subset R^{n+1}, of a simple system."
------------------------------------------------------------------------

Proof mechanism:
Let X_0, X_1, X_2, ... in Gamma such that X_j -> X_0 as j -> infty.
Write X_j = (U_j, V_j), let A_j denote the adiabat dA_{X_j}, and set
l_j = { (U, V_j) : (U, V_j) in Gamma }.

In coordinates (U, V) where internal energy U is the horizontal axis
and work coordinate V is the vertical axis:
  * l_j is the HORIZONTAL LINE V = V_j passing through X_j.
  * l_0 is the base HORIZONTAL LINE V = V_0 passing through X_0.
  * Each adiabat A_j satisfies dU = -P(U, V) dV, or dV/dU = -1/P(U, V).
  * Since pressure P(X) is locally Lipschitz continuous and bounded
    (0 < P_min <= P <= P_max < infty) on a small ball B centered at X_0,
    the slope dV/dU is bounded away from 0:
        -1/P_min <= dV/dU <= -1/P_max < 0.
    In particular, the adiabat cannot be horizontal (dV/dU != 0).
    Therefore, each adiabat A_j MUST intersect the horizontal line l_0
    at some point Y_j = (U(Y_j), V_0).
  * As j -> infty, |X_j - X_0| -> 0 forces Y_j -> X_0 along l_0.
  * The temperature difference decomposes via the triangle inequality:
        |T(X_j) - T(X_0)| <= |T(X_j) - T(Y_j)| + |T(Y_j) - T(X_0)|.
    - Leg 1: |T(X_j) - T(Y_j)| <= c |X_j - Y_j| -> 0
      by Lemma 5.1 (T is locally Lipschitz along each adiabat A_j).
    - Leg 2: |T(Y_j) - T(X_0)| -> 0
      by Theorem 5.1 (T is continuous and monotone along the line l_0).
    Hence T(X_j) -> T(X_0), proving continuity on Gamma.
"""

import numpy as np

# --------------------------------------------------------------------------
# Model Parameters (van der Waals / simple gas system)
# --------------------------------------------------------------------------
A_PARAM = 0.50          # molecular attraction parameter (A/V potential energy)
GAMMA_PARAM = 1.40      # adiabatic index
BALL_RADIUS = 0.55      # radius of the local neighborhood ball B centered at X_0

# Base state X_0 = (U_0, V_0)
U0 = 2.20
V0 = 1.20
X0 = np.array([U0, V0])


# --------------------------------------------------------------------------
# Thermodynamic State Functions
# --------------------------------------------------------------------------
def entropy(U, V):
    """
    Entropy S(U, V) = ln(U - A/V) + (gamma - 1) ln(V).
    Concave on Gamma = { (U, V) : U > A/V, V > 0 }.
    """
    return np.log(U - A_PARAM / V) + (GAMMA_PARAM - 1.0) * np.log(V)


def temperature(U, V):
    """
    Temperature T(U, V) = (dS/dU)^{-1} = U - A/V.
    Single-valued and strictly positive everywhere on Gamma (Theorem 5.1).
    Along any line l_j (constant V_j), T is linear and strictly increasing in U.
    """
    return U - A_PARAM / V


def pressure(U, V):
    """
    Pressure P(U, V) = T * (dS/dV) = A/V^2 + (gamma - 1) * T(U, V) / V.
    Smooth (locally Lipschitz) and strictly positive on Gamma.
    """
    t = temperature(U, V)
    return A_PARAM / (V ** 2) + (GAMMA_PARAM - 1.0) * t / V


def adiabat_const(U, V):
    """
    Adiabatic invariant K = exp(S) = (U - A/V) * V^(gamma - 1) = T * V^(gamma - 1).
    Level sets S = const are level sets K = const.
    """
    return (U - A_PARAM / V) * (V ** (GAMMA_PARAM - 1.0))


def adiabat_U(V, K):
    """
    Energy U as a function of V along the adiabat with constant K:
    U(V) = A/V + K * V^{-(gamma - 1)}.
    """
    return A_PARAM / V + K * (V ** (-(GAMMA_PARAM - 1.0)))


def adiabat_slope_dV_dU(U, V):
    """
    Slope of the adiabat in the (U, V) plane:
    dV/dU = -1 / P(U, V) < 0.
    """
    p = pressure(U, V)
    return -1.0 / p


# --------------------------------------------------------------------------
# Intersections with the Horizontal Line l_0: V = V_0
# --------------------------------------------------------------------------
def intersect_adiabat_with_l0(X_j):
    """
    Given a state X_j = (U_j, V_j), find the intersection point Y_j = (U_Yj, V_0)
    of the adiabat A_j passing through X_j with the horizontal line l_0: V = V_0.

    Since S is constant along A_j:
      K_j = adiabat_const(U_j, V_j)
      U(Y_j) = adiabat_U(V_0, K_j) = A/V_0 + K_j * V_0^{-(gamma - 1)}.
    """
    K_j = adiabat_const(X_j[0], X_j[1])
    U_Yj = adiabat_U(V0, K_j)
    return np.array([U_Yj, V0])


# --------------------------------------------------------------------------
# Sequence Generation: X_j -> X_0
# --------------------------------------------------------------------------
def generate_sequence(n_points=4):
    """
    Generate a clean sequence of points X_1, X_2, ..., X_n converging to X_0.
    Each X_j = (U_j, V_j) has V_j > V_0, so l_j: V = V_j is a horizontal line above l_0.
    """
    # Well-spaced geometric convergence for clear visual annotation
    offsets = [
        (-0.35, 0.30),    # j=1: outer state
        (-0.18, 0.15),    # j=2: intermediate
        (-0.09, 0.075),   # j=3: closer
        (-0.04, 0.033),   # j=4: very close
    ]
    if n_points > len(offsets):
        # generate further points if requested
        for j in range(len(offsets), n_points):
            factor = 0.5 ** (j - len(offsets) + 1)
            offsets.append((-0.04 * factor, 0.033 * factor))

    points = [X0 + np.array(off) for off in offsets[:n_points]]
    return points


# --------------------------------------------------------------------------
# Verification of Bounds for Ball B(X_0, BALL_RADIUS)
# --------------------------------------------------------------------------
def get_ball_bounds(radius=BALL_RADIUS):
    """
    Compute extremal values of P(U, V) and slope dV/dU on the ball B(X_0, r).
    Guarantees transversality and existence of intersection with l_0.
    """
    thetas = np.linspace(0, 2 * np.pi, 200)
    rs = np.linspace(0, radius, 20)
    P_vals = []
    slope_vals = []

    for r in rs:
        for th in thetas:
            u = U0 + r * np.cos(th)
            v = V0 + r * np.sin(th)
            if v > 0 and u > A_PARAM / v:
                p = pressure(u, v)
                P_vals.append(p)
                slope_vals.append(-1.0 / p)

    P_min, P_max = np.min(P_vals), np.max(P_vals)
    slope_min, slope_max = np.min(slope_vals), np.max(slope_vals)
    return {
        "P_min": P_min, "P_max": P_max,
        "slope_min": slope_min, "slope_max": slope_max,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Thermodynamic Model for Theorem 5.2 (Continuity of Temperature)")
    print("=" * 70)
    print(f"Base State X_0 = (U_0={U0:.2f}, V_0={V0:.2f})")
    print(f"Temperature T(X_0) = {temperature(U0, V0):.4f}")
    print(f"Pressure P(X_0)    = {pressure(U0, V0):.4f}")
    print(f"Adiabat slope dV/dU at X_0 = {adiabat_slope_dV_dU(U0, V0):.4f}")

    bounds = get_ball_bounds()
    print("\nBounds inside Ball B(X_0, r):")
    print(f"  P_min = {bounds['P_min']:.4f},  P_max = {bounds['P_max']:.4f}")
    print(f"  Adiabat slope dV/dU in [{bounds['slope_min']:.4f}, {bounds['slope_max']:.4f}] < 0")
    print("  -> Slope is strictly negative and bounded away from 0 (never horizontal).")
    print("  -> Transversality forces every adiabat A_j to intersect l_0!")

    print("\nSequence X_j -> X_0 and Triangle Inequality Decomposition:")
    seq = generate_sequence(4)
    for j, Xj in enumerate(seq, 1):
        Yj = intersect_adiabat_with_l0(Xj)
        dist_X = np.linalg.norm(Xj - X0)
        dist_Y = np.linalg.norm(Yj - X0)
        dist_XY = np.linalg.norm(Xj - Yj)
        T_Xj = temperature(*Xj)
        T_Yj = temperature(*Yj)
        T_X0 = temperature(*X0)
        dT_adia = abs(T_Xj - T_Yj)   # Leg 1 (Lemma 5.1)
        dT_line = abs(T_Yj - T_X0)   # Leg 2 (Theorem 5.1)
        dT_tot  = abs(T_Xj - T_X0)   # Total error

        print(f"  Point X_{j}: ({Xj[0]:.3f}, {Xj[1]:.3f}) -> Y_{j}: ({Yj[0]:.3f}, {Yj[1]:.3f})")
        print(f"    |X_{j}-X_0| = {dist_X:.4f},  |Y_{j}-X_0| = {dist_Y:.4f}")
        print(f"    Leg 1 (along A_{j}): |T(X_{j}) - T(Y_{j})| = {dT_adia:.4f}")
        print(f"    Leg 2 (along l_0):  |T(Y_{j}) - T(X_0)| = {dT_line:.4f}")
        print(f"    Total |T(X_{j}) - T(X_0)| = {dT_tot:.4f} <= {dT_adia + dT_line:.4f}  [OK]")
