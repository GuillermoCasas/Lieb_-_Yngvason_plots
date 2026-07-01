"""
Shared thermodynamic model for the Theorem 5.1 figures
(Lieb & Yngvason, "The physics and mathematics of the second law of
thermodynamics", Physics Reports 310 (1999) 1-96).

------------------------------------------------------------------------
Theorem 5.1 (Uniqueness of temperature).  At every point X of the state
space C of a simple system,  T_+(X) = T_-(X),  i.e.  T(X) = [(dS/dU)(X)]^{-1}
is a single number:  the concave entropy S has NO kink in the energy
direction.
------------------------------------------------------------------------

The proof is by contradiction: one *assumes* a point Z where the one-sided
energy derivatives of S disagree,

        1/T_+(Z) = (dS/dU)_+  <  (dS/dU)_- = 1/T_-(Z)      (concavity)
   <=>          T_-(Z)        <        T_+(Z),

so that the "temperature" at Z is really a whole interval  T(Z)=[T_-,T_+].
The proof then shows this contradicts the zeroth law, transversality (T4)
and axiom T5.

To DRAW that hypothetical situation honestly we need a concave S(U,V) that
actually has such a kink along one adiabat.  We use

        sigma(U,V) = U - g(V),     g(V) = A/V + U_off        (reduced energy)
        S(U,V)     = f( sigma ),

with adiabats = level sets  sigma = const  =>  U = A/V + U_off + c,
downward-sloping and convex in the (V,U) plane, exactly like the Lemma 5.1
figures.  The one-variable profile f is concave with a corner at sigma = 0:

        f'(sigma) = 1/T =  M_minus - beta*sigma   for sigma < 0
                           M_plus  - beta*sigma   for sigma > 0     (M_minus>M_plus)

so f' (= 1/T) jumps DOWN by (M_minus-M_plus) across the kink adiabat
sigma = 0.  That jump is precisely the temperature interval
[T_-, T_+] = [1/M_minus, 1/M_plus] that the theorem forbids.  Away from the
kink adiabat S is smooth and T is single-valued, as in a real simple system.
"""

import numpy as np

# ----- model parameters ---------------------------------------------------
A_SHAPE = 1.0     # adiabat shape:  g(V) = A/V + U_off
U_OFF = 1.2       # vertical offset placing the kink adiabat mid-plot
BETA = 0.25       # curvature of 1/T in sigma (mild variation of T with energy)
M_MINUS = 1.00    # left  slope dS/dU at the kink = 1/T_-(Z)  (steeper)
M_PLUS = 0.60     # right slope dS/dU at the kink = 1/T_+(Z)  (shallower)
M_COMMON = 0.80   # slope used for the "resolved" (no-kink) picture
# concavity  <=>  M_MINUS > M_PLUS  and  BETA > 0

# temperatures of the kink adiabat (sigma = 0)
T_MINUS_Z = 1.0 / M_MINUS   # lower temperature T_-(Z)   (smaller)
T_PLUS_Z = 1.0 / M_PLUS     # upper temperature T_+(Z)   (larger)


def g_shape(V):
    """Adiabat backbone.  The kink adiabat (sigma=0) is U = g(V)."""
    return A_SHAPE / V + U_OFF


def sigma(U, V):
    """Reduced energy.  Adiabats are the level sets sigma = const."""
    return U - g_shape(V)


def entropy(U, V, m_minus=M_MINUS, m_plus=M_PLUS, beta=BETA):
    """Concave entropy S(U,V) with a corner along sigma = 0."""
    s = sigma(U, V)
    slope = np.where(s < 0, m_minus, m_plus)
    return slope * s - 0.5 * beta * s ** 2


def inv_T(U, V, m_minus=M_MINUS, m_plus=M_PLUS, beta=BETA):
    """dS/dU = 1/T.  Two-valued exactly on the kink adiabat sigma = 0."""
    s = sigma(U, V)
    slope = np.where(s < 0, m_minus, m_plus)
    return slope - beta * s


def temperature(U, V, **kw):
    """T = (dS/dU)^{-1}, single-valued off the kink adiabat."""
    return 1.0 / inv_T(U, V, **kw)


def adiabat_U(V, c):
    """Adiabat with reduced energy c:  U = A/V + U_off + c."""
    return g_shape(V) + c


def pressure(V):
    """P = -dU/dV along an adiabat = A/V^2  (independent of the adiabat c)."""
    return A_SHAPE / V ** 2
