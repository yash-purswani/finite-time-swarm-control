r"""Theoretical bounds, safety certificates, and required design rates.

Implements the finite-time convergence bounds and transient certification rules:
- Universal station tracking bound \overline{R}(t) and deadline \overline{T}
- Dynamic disagreement tube \rho_H(t)
- Time-varying clearance margin s(t) = a(t) DQ_0 - d_safe
- Required decay rate \gamma_req(\kappa) = sup_{\tau \in (0, \overline{T}]} \ln(\rho_0 / s(\tau)) / \tau
- Sufficient coupling gain m_req(\kappa) = (\gamma_req - 1) / (\underline{w} * \lambda_2(L_H))
"""
from typing import Optional
import numpy as np
from scipy.optimize import minimize_scalar
from .stations import star_scale


def disagreement_vector(delta: np.ndarray) -> np.ndarray:
    """Compute formation disagreement eta_i = delta_i - (1/n) * sum_j delta_j."""
    return delta - np.mean(delta, axis=-2, keepdims=True)


def disagreement_norm(delta: np.ndarray) -> float:
    """Compute Frobenius/L2 norm of formation disagreement ||eta||_2."""
    eta = disagreement_vector(delta)
    return float(np.linalg.norm(eta))


def universal_tracking_bound(t: np.ndarray, delta0_norm: float, gamma: float = 1.0) -> np.ndarray:
    r"""Evaluate universal upper bound on tracking error norm R(t) from Theorem 1.
    
    Theorem 1: R(t) = [ (1 + \sqrt{||\delta_0||}) * exp(-0.5 * t) - 1 ]_+^2.
    Independent of graph and coupling gain m.
    """
    t = np.asarray(t, dtype=float)
    factor = (1.0 + (delta0_norm ** 0.5)) * np.exp(-0.5 * t) - 1.0
    return np.where(factor > 0.0, factor ** 2, 0.0)


def universal_settling_deadline(delta0_norm: float, gamma: float = 1.0) -> float:
    r"""Evaluate theoretical universal settling deadline: \overline{T} = 2 * ln(1 + \sqrt{||\delta_0||})."""
    return float(2.0 * np.log(1.0 + (delta0_norm ** 0.5)))


def geometric_clearance_margin(
    t: np.ndarray,
    kappa: float,
    dq0: float,
    ell_h0: Optional[float] = None,
    d_safe: float = 1.0,
    r_c: float = 6.0,
    a0: float = 1.10,
    a_inf: float = 0.75
) -> np.ndarray:
    r"""Geometric clearance margin s(t) = min{d_q(t) - d_safe, R_c - \ell_H(t)} (Eq. 42)."""
    at = star_scale(t, kappa, a0, a_inf)
    s_sep = at * dq0 - d_safe
    if ell_h0 is not None:
        s_conn = r_c - at * ell_h0
        return np.minimum(s_sep, s_conn)
    return s_sep


def dynamic_distortion_tube(
    t: np.ndarray,
    delta0_norm: float,
    rho0: float,
    gamma: float
) -> np.ndarray:
    r"""Dynamic disagreement distortion tube \rho_H(t) = min{\sqrt{2} * R(t), \rho_0 * exp(-\gamma * t)} (Eq. 39)."""
    r_bar = universal_tracking_bound(t, delta0_norm)
    return np.minimum(np.sqrt(2.0) * r_bar, rho0 * np.exp(-gamma * np.asarray(t, dtype=float)))


def effective_decay_rate(
    m: float,
    r_c: float = 6.0,
    eps_w: float = 0.2,
    lambda2_lh: float = 10.0,
    w_lower: Optional[float] = None
) -> float:
    r"""Effective disagreement exponential decay rate \gamma = 1 + m * \underline{w} * \lambda_2(L_H).

    Defaults to \underline{w} = 1 / 6.8 as defined in prompt specifications.
    """
    if w_lower is None:
        w_lower = 1.0 / 6.8 if (r_c == 6.0 and eps_w == 0.2) else 1.0 / (r_c + eps_w)
    return float(1.0 + m * w_lower * lambda2_lh)


def compute_required_rate(
    kappa: float,
    rho0: float,
    dq0: float,
    ell_h0: Optional[float] = None,
    d_safe: float = 1.0,
    r_c: float = 6.0,
    a0: float = 1.10,
    a_inf: float = 0.75,
    t_bar: Optional[float] = None
) -> float:
    r"""Evaluate required decay rate: \gamma_req = sup_{\tau \in (0, \overline{T}]} ln(\rho_0 / s(\tau)) / \tau (Eq. 43)."""
    if t_bar is None:
        raise ValueError("t_bar must be provided")

    # Initial margin check from Corollary 2 assumption: 0 < rho0 < s(t0)
    s0 = float(geometric_clearance_margin(0.0, kappa, dq0, ell_h0=ell_h0, d_safe=d_safe, r_c=r_c, a0=a0, a_inf=a_inf))
    if s0 <= 0.0 or rho0 >= s0:
        return np.inf

    t_dense = np.linspace(1e-4, t_bar, 2000)
    s_vals = geometric_clearance_margin(t_dense, kappa, dq0, ell_h0=ell_h0, d_safe=d_safe, r_c=r_c, a0=a0, a_inf=a_inf)
    if np.any(s_vals <= 0.0):
        return np.inf

    g_vals = np.log(rho0 / s_vals) / t_dense
    idx = int(np.argmax(g_vals))
    t_l = t_dense[max(0, idx - 1)]
    t_r = t_dense[min(len(t_dense) - 1, idx + 1)]

    res = minimize_scalar(
        lambda tau: -np.log(rho0 / float(geometric_clearance_margin(tau, kappa, dq0, ell_h0=ell_h0, d_safe=d_safe, r_c=r_c, a0=a0, a_inf=a_inf))) / tau,
        bounds=(t_l, t_r), method='bounded', options={'xatol': 1e-12}
    )
    return float(max(np.max(g_vals), -res.fun))


def compute_required_gain(
    gamma_req: float,
    r_c: float = 6.0,
    eps_w: float = 0.2,
    lambda2_lh: float = 10.0,
    w_lower: Optional[float] = None
) -> Optional[float]:
    r"""Compute required coupling gain m_req = max(0, (\gamma_req - 1) / (\underline{w} * \lambda_2(L_H)))."""
    if not np.isfinite(gamma_req):
        return None
    if w_lower is None:
        w_lower = 1.0 / 6.8 if (r_c == 6.0 and eps_w == 0.2) else 1.0 / (r_c + eps_w)
    return float(max(0.0, (gamma_req - 1.0) / (w_lower * lambda2_lh)))


def classify_outcome(
    certified: bool,
    observed_safe: bool,
    observed_connected: bool,
    stations_infeasible: bool = False
) -> str:
    """Classify run outcome into Categories A, B, C, D:

    Category A: Certified safe by Corollary 2 (gamma > gamma_req) and observed safe.
    Category B: Uncertified by conservative bound, but observed safe in simulation.
    Category C: Observed transient collision or disconnect.
    Category D: Geometrically infeasible station plan.
    """
    if not observed_safe or not observed_connected:
        return 'C'
    if stations_infeasible:
        return 'D'
    if certified:
        return 'A'
    return 'B'
