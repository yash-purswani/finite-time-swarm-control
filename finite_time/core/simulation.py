"""Numerical integration and metrics extraction using scipy.integrate.solve_ivp.

Executes adaptive-step integration with DOP853 (dense output enabled),
records internal solver nodes and dense trajectory points, and extracts
collision/connectivity statistics and certification metrics.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from .dynamics import (
    closed_loop_ode_rhs,
    evaluate_control_command,
    pairwise_distance_matrix,
    proximity_weights,
    laplacian_matrix,
    sig_beta,
)
from .stations import (
    build_star_template,
    template_geometry_metrics,
    backbone_geometry_metrics,
    star_stations,
    star_stations_batch,
    star_scale,
    _BACKBONE_ELL_H0,
    _BACKBONE_LAMBDA2,
    BACKBONE_EDGES,
)
from .spawn import (
    generate_random_ring_perturbation,
)
from .theory import (
    disagreement_vector,
    universal_tracking_bound,
    universal_settling_deadline,
    geometric_clearance_margin,
    dynamic_distortion_tube,
    effective_decay_rate,
    compute_required_rate,
    compute_required_gain,
    classify_outcome,
)


@dataclass
class SimulationResult:
    """Encapsulates continuous and sampled trajectories and summary metrics."""
    t: np.ndarray
    x: np.ndarray             # (T, n, dim)
    q: np.ndarray             # (T, n, dim)
    delta: np.ndarray         # (T, n, dim)
    eta: np.ndarray           # (T, n, dim)
    d_x: np.ndarray           # (T,)
    ell_x: np.ndarray         # (T,)
    u_norm: np.ndarray        # (T,)
    delta_norm: np.ndarray    # (T,)
    eta_norm: np.ndarray      # (T,)
    eta_bound: np.ndarray     # (T,)
    lam2: np.ndarray          # (T,)
    lam2_integral: np.ndarray # (T,)
    active_edges: np.ndarray  # (T, n, n)
    s: np.ndarray             # (T,)
    rho_h: np.ndarray         # (T,)
    r_bar: np.ndarray         # (T,)
    d_q: np.ndarray           # (T,)
    ell_q: np.ndarray         # (T,)
    sol: Any                  # scipy OdeResult
    metrics: Dict[str, Any]

    @property
    def settings(self) -> Dict[str, Any]:
        """Solver settings used for integration."""
        return self.metrics.get('solver_settings', {})

    @property
    def X(self) -> np.ndarray:
        """Alias for states array."""
        return self.x

    @property
    def cfg(self) -> Any:
        """Legacy configuration placeholder for compatibility."""
        class _LegacyCfg:
            n = self.x.shape[1]
            d = self.x.shape[2]
            M = np.eye(2)
        return _LegacyCfg()

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert arrays to dictionary for NPZ saving."""
        return {
            't': self.t,
            'X': self.x,
            'q': self.q,
            'delta': self.delta,
            'd_x': self.d_x,
            'ell_x': self.ell_x,
            'ell_x_H': self.ell_x,
            'u_norm': self.u_norm,
            'delta_norm': self.delta_norm,
            'err_norm': self.delta_norm,
            'eta_norm': self.eta_norm,
            'eta_bound': self.eta_bound,
            'lambda2': self.lam2,
            'lambda2_integral': self.lam2_integral,
            'active_edges': self.active_edges,
            's': self.s,
            'rho_H': self.rho_h,
            'R_bar': self.r_bar,
            'd_q': self.d_q,
            'ell_H': self.ell_q,
        }


def simulate_star_maneuver(
    m: float,
    kappa: float,
    t_span: Tuple[float, float] = (0.0, 12.0),
    p0: Optional[np.ndarray] = None,
    delta0: Optional[np.ndarray] = None,
    d_safe: float = 1.0,
    r_c: float = 6.0,
    beta: float = 0.5,
    eps_w: float = 0.05,
    a0: float = 1.70,
    a_inf: float = 0.72,
    omega: float = np.pi / 4.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: float = 0.02,
    method: str = 'DOP853',
    transient_samples: int = 1501,
    remainder_samples: int = 601,
    **kwargs: Any
) -> SimulationResult:
    """High-accuracy adaptive simulation of rotating star formation maneuver.

    Args:
        m: Coupling gain
        kappa: Maneuver scale compression rate
        t_span: Integration interval (t0, tf)
        p0: Template vertices (default: 10-agent star)
        delta0: Initial perturbation (default: random ring r∈[0.05,0.35] around stations)
        d_safe: Minimum safety separation threshold
        r_c: Maximum communication range
        beta: Fractional exponent
        eps_w: Interaction weight regularizer (w̲ = 1/(R_c + eps_w) = 1/6.05)
        a0: Initial formation scale (default 1.70; gives s(0)≈1.44 m > rho_0_max≈1.26 m)
        a_inf: Asymptotic formation scale (default 0.72; ensures a_inf*DQ0=1.033>d_safe)
        omega: Angular velocity of rotation
        rtol, atol, max_step: Solver numerical tolerances
        transient_samples: Sample resolution over transient interval [0, 2.5]
        remainder_samples: Sample resolution over post-transient interval [2.5, tf]

    Returns:
        SimulationResult containing all trajectories, metrics, and certificates.
    """
    t0, tf = t_span
    if p0 is None:
        p0 = build_star_template(n=10)
    n = len(p0)
    dim = p0.shape[1]

    dq0, _, _, _ = template_geometry_metrics(p0)
    # Use backbone spanning tree metrics for certificates (not full-graph diameter)
    ell_h0_backbone, lambda2_lh = backbone_geometry_metrics(p0)
    q0, _ = star_stations(t0, kappa, p0, a0=a0, a_inf=a_inf, omega=omega)

    if delta0 is None:
        delta0, _, _ = generate_random_ring_perturbation(q0, d_safe=d_safe, r_c=r_c)
    delta0_norm = float(np.linalg.norm(delta0))
    eta0 = disagreement_vector(delta0)
    rho0 = np.sqrt(2.0) * float(np.linalg.norm(eta0))
    t_bar_univ = universal_settling_deadline(delta0_norm)

    x0 = q0 + delta0
    y0 = np.concatenate([x0.ravel(), [0.0]])

    def station_fn(t_val: float) -> Tuple[np.ndarray, np.ndarray]:
        return star_stations(t_val, kappa, p0, a0=a0, a_inf=a_inf, omega=omega)

    def rhs(t_val: float, y_val: np.ndarray) -> np.ndarray:
        return closed_loop_ode_rhs(
            t_val, y_val, station_fn, m=m, r_c=r_c, beta=beta, eps_w=eps_w, n=n, dim=dim
        )

    solver_settings = dict(rtol=rtol, atol=atol, max_step=max_step, method='DOP853')
    sol = solve_ivp(
        rhs, (t0, tf), y0, method='DOP853',
        rtol=rtol, atol=atol, max_step=max_step, dense_output=True
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed for m={m}, kappa={kappa}: {sol.message}")

    # Build evaluation mesh: densely sample transient [t0, 2.5]
    t_tr = min(2.5, tf)
    if t_tr < tf:
        t_eval = np.unique(np.concatenate([
            np.linspace(t0, t_tr, transient_samples),
            np.linspace(t_tr, tf, remainder_samples),
        ]))
    else:
        t_eval = np.linspace(t0, tf, transient_samples)

    state_values = sol.sol(t_eval)
    x_values = state_values[:n * dim].T.reshape(-1, n, dim)

    q_values, qdot_values = star_stations_batch(t_eval, kappa, p0, a0=a0, a_inf=a_inf, omega=omega)
    delta_values = x_values - q_values

    # Pairwise distances and backbone edge lengths
    iu = np.triu_indices(n, k=1)
    diffs = x_values[:, :, None, :] - x_values[:, None, :, :]
    dist_values = np.linalg.norm(diffs, axis=-1)
    pair_distances = dist_values[:, iu[0], iu[1]]
    d_x = np.min(pair_distances, axis=1)

    bb_u = [e[0] for e in BACKBONE_EDGES]
    bb_v = [e[1] for e in BACKBONE_EDGES]
    bb_distances = np.linalg.norm(x_values[:, bb_u, :] - x_values[:, bb_v, :], axis=-1)
    ell_x = np.max(bb_distances, axis=1)  # max backbone edge length over time

    # Weights and Laplacians
    diag_mask = np.eye(n, dtype=bool)[None, :, :]
    adjacency = (dist_values < r_c) & (~diag_mask)
    w_values = adjacency / (dist_values + eps_w)
    L_values = -w_values
    diag_idx = np.arange(n)
    L_values[:, diag_idx, diag_idx] = np.sum(w_values, axis=2)

    # Feedback commands and disagreement
    fb_values = -delta_values - sig_beta(delta_values, beta) - m * (L_values @ delta_values)
    u_values = qdot_values + fb_values
    u_norm = np.linalg.norm(u_values.reshape(len(t_eval), -1), axis=1)
    delta_norm = np.linalg.norm(delta_values.reshape(len(t_eval), -1), axis=1)
    eta = delta_values - delta_values.mean(axis=1, keepdims=True)
    eta_norm = np.linalg.norm(eta.reshape(len(t_eval), -1), axis=1)

    # Connectivity integration
    eigenvalues = np.linalg.eigvalsh(L_values)
    lam2 = np.maximum(0.0, eigenvalues[:, 1])
    dt = np.diff(t_eval)
    lam2_int = np.concatenate([[0.0], np.cumsum(0.5 * (lam2[:-1] + lam2[1:]) * dt)])
    eta_bound = eta_norm[0] * np.exp(-t_eval - m * lam2_int)

    # Refine minimum separation
    def _dist_at(tau: float) -> float:
        st = sol.sol(tau)[:n * dim].reshape(n, dim)
        return float(np.min(np.linalg.norm(st[iu[0]] - st[iu[1]], axis=1)))

    i_min = int(np.argmin(d_x))
    lo_min = t_eval[max(0, i_min - 1)]
    hi_min = t_eval[min(len(t_eval) - 1, i_min + 1)]
    if hi_min > lo_min:
        res_min = minimize_scalar(_dist_at, bounds=(lo_min, hi_min), method='bounded', options={'xatol': 1e-8})
        min_dx = min(float(np.min(d_x)), float(res_min.fun))
    else:
        min_dx = float(np.min(d_x))

    max_ell = float(np.max(ell_x))

    # Theoretical certificates — 15-edge backbone algebraic connectivity lambda2(L_H)
    gamma = effective_decay_rate(m, r_c=r_c, eps_w=eps_w, lambda2_lh=lambda2_lh)
    gamma_req = compute_required_rate(
        kappa, rho0, dq0, ell_h0=ell_h0_backbone, d_safe=d_safe, r_c=r_c, a0=a0, a_inf=a_inf, t_bar=t_bar_univ
    )
    rho_t = dynamic_distortion_tube(t_eval, delta0_norm, rho0, gamma)
    s_t = geometric_clearance_margin(t_eval, kappa, dq0, ell_h0=ell_h0_backbone, d_safe=d_safe, r_c=r_c, a0=a0, a_inf=a_inf)

    # Settling times
    settled_4 = delta_norm <= 1e-4
    t_settle_4 = float(t_eval[np.argmax(settled_4)]) if np.any(settled_4) else tf
    settled_6 = delta_norm <= 1e-6
    t_settle_6 = float(t_eval[np.argmax(settled_6)]) if np.any(settled_6) else tf
    settled_8 = delta_norm <= 1e-8
    t_settle_8 = float(t_eval[np.argmax(settled_8)]) if np.any(settled_8) else tf

    t_bar = universal_settling_deadline(delta0_norm)
    certified = bool(np.isfinite(gamma_req) and (gamma > gamma_req) and (min_dx >= d_safe))
    obs_safe = bool(min_dx >= d_safe)
    obs_conn = bool(max_ell <= r_c)
    classification = classify_outcome(certified, obs_safe, obs_conn)
    m_req = compute_required_gain(gamma_req, r_c=r_c, eps_w=eps_w, lambda2_lh=lambda2_lh)

    # Collision window during transient
    mask_tr = t_eval <= 2.5
    tz = t_eval[mask_tr]
    dt_tr = tz[1] - tz[0]
    safe_viol_mask = d_x[mask_tr] < d_safe
    conn_viol_mask = ell_x[mask_tr] > r_c
    viol_t_safe = tz[safe_viol_mask]
    viol_t_conn = tz[conn_viol_mask]
    safe_viol_start = float(viol_t_safe[0]) if viol_t_safe.size else None
    safe_viol_end = float(viol_t_safe[-1]) if viol_t_safe.size else None
    dur_safe = float(np.sum(safe_viol_mask) * dt_tr)
    dur_conn = float(np.sum(conn_viol_mask) * dt_tr)

    metrics = {
        'kappa': kappa,
        'min_separation': min_dx,
        'minimum_clearance': min_dx - d_safe,
        'max_backbone_edge': max_ell,
        'minimum_connectivity_margin': r_c - max_ell,
        'observed_safe': obs_safe,
        'observed_connected': obs_conn,
        'observed_backbone_preserved': obs_conn,
        'observed_graph_connected': bool(np.min(lam2) > 1e-12),
        'settling_times': {'0.0001': t_settle_4, '1e-06': t_settle_6, '1e-08': t_settle_8},
        'T_bar': t_bar,
        'peak_control': float(np.max(u_norm)),
        'peak_individual_control': float(np.max(u_norm)),
        'J_u': float(sol.y[n * dim, -1]),
        'gamma': gamma,
        'gamma_req': float(gamma_req) if np.isfinite(gamma_req) else None,
        'm_required': m_req,
        'certified': certified,
        'classification': classification,
        'safe_viol_start': safe_viol_start,
        'safe_viol_end': safe_viol_end,
        'dur_safe_viol': dur_safe,
        'dur_conn_viol': dur_conn,
        'solver_settings': solver_settings,
    }

    return SimulationResult(
        t=t_eval,
        x=x_values,
        q=q_values,
        delta=delta_values,
        eta=eta,
        d_x=d_x,
        ell_x=ell_x,
        u_norm=u_norm,
        delta_norm=delta_norm,
        eta_norm=eta_norm,
        eta_bound=eta_bound,
        lam2=lam2,
        lam2_integral=lam2_int,
        active_edges=adjacency,
        s=s_t,
        rho_h=rho_t,
        r_bar=universal_tracking_bound(t_eval, delta0_norm, gamma=1.0),
        d_q=star_scale(t_eval, kappa, a0, a_inf) * dq0,
        ell_q=star_scale(t_eval, kappa, a0, a_inf) * ell_h0_backbone,
        sol=sol,
        metrics=metrics,
    )
