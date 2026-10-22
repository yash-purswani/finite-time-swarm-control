"""Deterministic ECC experiments, persistence and validation orchestration."""
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
import scipy
import matplotlib
import numba
from .common import shared_scenario,reference,reference_velocity,KappaScaleStations,MovingReference
from .model import SwarmConfig
from .integrate import simulate
from .metrics import analyze,NumericalValidationError
from .theory import compression_certificate
from .formations import hexagon,wedge,star,arrow,morph_schedule
from .cache import cached_run,source_hashes,digest,active_grid_paths,experiment_paths


def json_value(x):
    if isinstance(x,dict):
        return {k:json_value(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):
        return [json_value(v) for v in x]
    if isinstance(x,(np.bool_,)):
        return bool(x)
    if isinstance(x,(np.integer,)):
        return int(x)
    if isinstance(x,(float,np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def write_json(path,data):
    import os
    path=Path(path)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(json_value(data),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def metadata(cfg,X0,config):
    report = Path(__file__).resolve().parents[1]/'ACC_paper/finite-time_report.tex'
    station=getattr(cfg.P_fn,'__self__',None)
    station_parameters = None if station is None else dict(a_initial=station.a_initial,a_final=station.a_final,
                  kappa=station.kappa,omega=getattr(station,'omega',None),
                  rate=getattr(station,'rate',None),p_initial=station.p_initial)
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    return dict(configuration=config,station_parameters=station_parameters,M=cfg.M,initial_positions=X0,
                initial_stations=cfg.stations(0),H=cfg.H,adjacency=cfg.adjacency,
                beta=cfg.beta,nu=cfg.nu,epsilon_w=cfg.eps,R_c=cfg.R_c,d_safe=cfg.d_safe,
                weight_mode=cfg.weight_mode,python=sys.version,numpy=np.__version__,
                scipy=scipy.__version__,matplotlib=matplotlib.__version__,numba=numba.__version__,platform=platform.platform(),
                implementation_sha256=source_hashes,
                mathematical_source=str(report.relative_to(report.parents[1])),
                report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())


def save_run(folder,name,res,arrays,metrics,config,cert=None):
    folder = Path(folder);folder.mkdir(parents=True,exist_ok=True)
    # Save internal adaptive time nodes too; all recorded states are actual solver output.
    t_ev = getattr(res.sol, 't_events', None)
    safety_ev = t_ev[0] if (t_ev is not None and len(t_ev) > 0) else []
    conn_ev = t_ev[1] if (t_ev is not None and len(t_ev) > 1) else []
    nd = res.cfg.n * res.cfg.d
    sol_states = res.sol.y[:nd].T if res.sol.y.shape[0] >= nd else res.sol.y.T
    np.savez_compressed(folder/f'{name}.npz',**arrays,solver_nodes=res.sol.t,
                        solver_states=sol_states,safety_events=safety_ev,
                        connectivity_events=conn_ev)
    write_json(folder/f'{name}.json',dict(metadata=metadata(res.cfg,res.X[0],config),
               data_sha256=digest(folder/f'{name}.npz'),
               solver=res.settings,metrics=metrics,certificate=cert))
    row = dict(name=name,m=float(res.cfg.M[0,0]))
    row.update(metrics)
    row.update(certified=cert['certified'] if cert else None,
               gamma=cert['gamma'] if cert else None,gamma_req=cert['gamma_req'] if cert else None)
    return row


# =============================================================================
# Star Maneuver Simulation, Analysis and Safety Certification
# =============================================================================
def _build_star_template(n=10):
    i_idx = np.arange(n)
    phi = np.pi / 2.0 + np.pi * i_idx / 5.0
    b = np.where(i_idx % 2 == 0, 2.8, 1.22)
    p0 = np.column_stack([b * np.cos(phi), b * np.sin(phi)])
    p0 -= np.mean(p0, axis=0, keepdims=True)
    return p0


def _generate_star_initial_spawn(p0, close_pair=(5, 7), far_pair=(0, 4),
                                seed=42, close_push=0.12, far_push=0.22, noise_std=0.03):
    n, dim = p0.shape
    rng = np.random.default_rng(seed)
    delta0 = rng.normal(0.0, noise_std, size=(n, dim))

    i, j = close_pair
    uij = (p0[j] - p0[i]) / np.linalg.norm(p0[j] - p0[i])
    delta0[i] = +close_push * uij
    delta0[j] = -close_push * uij

    k, l = far_pair
    uk = p0[k] / np.linalg.norm(p0[k])
    ul = p0[l] / np.linalg.norm(p0[l])
    delta0[k] = +far_push * uk
    delta0[l] = +far_push * ul

    fixed = {i, j, k, l}
    rest = [idx for idx in range(n) if idx not in fixed]
    mean_rest = np.sum(delta0, axis=0) / len(rest)
    for idx in rest:
        delta0[idx] -= mean_rest
    return delta0


def _a_inf_from_kappa(kappa):
    return 1.65 - 0.25 * kappa


def _a_scale(t, kappa, rate_a=0.90):
    a_inf = _a_inf_from_kappa(kappa)
    return a_inf + (1.0 - a_inf) * np.exp(-rate_a * t)


def _a_dot(t, kappa, rate_a=0.90):
    a_inf = _a_inf_from_kappa(kappa)
    return -rate_a * (1.0 - a_inf) * np.exp(-rate_a * t)


def _ref_r(t):
    return np.array([0.9 * t, 1.1 * np.sin(0.35 * t)])


def _ref_r_dot(t):
    return np.array([0.9, 1.1 * 0.35 * np.cos(0.35 * t)])


def _station_q_and_qdot(t, kappa, p0, rate_a=0.90):
    at = _a_scale(t, kappa, rate_a)
    adt = _a_dot(t, kappa, rate_a)
    r_t = _ref_r(t)
    rdot_t = _ref_r_dot(t)
    q = r_t[None, :] + at * p0
    qdot = rdot_t[None, :] + adt * p0
    return q, qdot


def _station_q_and_qdot_batch(t_values, kappa, p0, rate_a=0.90):
    t_values = np.asarray(t_values, dtype=float)
    at = _a_scale(t_values, kappa, rate_a)
    adt = _a_dot(t_values, kappa, rate_a)
    ref = np.column_stack([0.9 * t_values, 1.1 * np.sin(0.35 * t_values)])
    ref_dot = np.column_stack([
        np.full_like(t_values, 0.9),
        1.1 * 0.35 * np.cos(0.35 * t_values),
    ])
    q = ref[:, None, :] + at[:, None, None] * p0[None, :, :]
    qdot = ref_dot[:, None, :] + adt[:, None, None] * p0[None, :, :]
    return q, qdot


def _s_margin(t, kappa, dq0, ell_h0, d_safe=0.95, r_c=6.60, rate_a=0.90):
    at = _a_scale(t, kappa, rate_a)
    return np.minimum(at * dq0 - d_safe, r_c - at * ell_h0)


def _R_bar(t, delta0_norm, gamma=1.0):
    val = (1.0 + gamma * (delta0_norm ** 0.5)) * np.exp(-0.5 * gamma * t) - 1.0
    return np.where(val > 0.0, (val / gamma) ** 2, 0.0)


def _compute_gamma_req(kappa, rho_eff0, dq0, ell_h0, d_safe=0.95, r_c=6.60, rate_a=0.90, t_bar_univ=None):
    from scipy.optimize import minimize_scalar
    t_dense = np.linspace(1e-5, t_bar_univ, 2000)
    s_vals = _s_margin(t_dense, kappa, dq0, ell_h0, d_safe, r_c, rate_a)
    if np.any(s_vals <= 0.0):
        return np.inf
    g_vals = np.log(rho_eff0 / s_vals) / t_dense
    idx = int(np.argmax(g_vals))
    t_l = t_dense[max(0, idx - 1)]
    t_r = t_dense[min(len(t_dense) - 1, idx + 1)]
    res = minimize_scalar(
        lambda tau: -np.log(rho_eff0 / float(_s_margin(tau, kappa, dq0, ell_h0, d_safe, r_c, rate_a))) / tau,
        bounds=(t_l, t_r), method='bounded', options={'xatol': 1e-12}
    )
    return float(max(np.max(g_vals), -res.fun))


def _sig_beta(z, beta=0.5):
    return np.sign(z) * (np.abs(z) ** beta)


def _pairwise_extreme_at_time(t, sol, which, n=10, dim=2, iu=None):
    x = sol.sol(t)[:n * dim].reshape(n, dim)
    pair_diffs = x[iu[0]] - x[iu[1]]
    distances = np.linalg.norm(pair_diffs, axis=1)
    return float(np.min(distances) if which == 'min' else np.max(distances))


class _Exp3SwarmConfig:
    def __init__(self, m, kappa, p0, d_safe=0.95, r_c=6.60, beta=0.5, eps_w=0.2):
        self.n = len(p0)
        self.d = 2
        self.M = float(m) * np.eye(2)
        self.beta = beta
        self.nu = 1.0
        self.eps = eps_w
        self.R_c = r_c
        self.d_safe = d_safe
        self.H = np.ones((len(p0), len(p0))) - np.eye(len(p0))
        self.adjacency = np.ones((len(p0), len(p0))) - np.eye(len(p0))
        self.weight_mode = 'distance'
        self.P_fn = None
        self.m = float(m)
        self.kappa = float(kappa)
        self.p0 = p0
        self.lambda_min_M = float(m)
        self.r_fn = _ref_r
        self.rdot_fn = _ref_r_dot

    def stations(self, t):
        q, _ = _station_q_and_qdot(t, self.kappa, self.p0)
        return q


class _Exp3SimResult:
    def __init__(self, sol, cfg, t, X, settings):
        self.sol = sol
        self.cfg = cfg
        self.t = t
        self.X = X
        self.settings = settings

    def at(self, times):
        times = np.atleast_1d(times)
        return self.sol.sol(times)[:self.cfg.n * self.cfg.d].T.reshape(-1, self.cfg.n, self.cfg.d)


def run_star_maneuver(m, kappa, config=None, solver_settings=None, transient_samples=1501, remainder_samples=601):
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize_scalar

    n = 10; dim = 2; beta = 0.5; eps_w = 0.2
    t0 = 0.0; tf = 12.0
    d_safe = 0.95; r_c = 6.60; rate_a = 0.90
    if config and 'shared' in config:
        sh = config['shared']
        tf = float(sh.get('t_end', tf))
        d_safe = float(sh.get('d_safe', d_safe))
        r_c = float(sh.get('R_c', r_c))
        rate_a = float(sh.get('scale_rate', rate_a))
    w_lower = 1.0 / (r_c + eps_w)
    lambda2_lh = float(n)

    ode_rtol = 1e-6; ode_atol = 1e-8; ode_max_step = 0.02
    if solver_settings:
        ode_rtol = solver_settings.get('rtol', ode_rtol)
        ode_atol = solver_settings.get('atol', ode_atol)
        ode_max_step = solver_settings.get('max_step', ode_max_step)

    p0 = _build_star_template(n)
    iu = np.triu_indices(n, k=1)
    dist_p0 = np.linalg.norm(p0[:, None, :] - p0[None, :, :], axis=-1)
    dq0 = float(np.min(dist_p0[iu]))
    ell_h0 = float(np.max(dist_p0[iu]))

    _full = dist_p0.copy(); np.fill_diagonal(_full, np.nan)
    close_pair = tuple(int(v) for v in np.unravel_index(np.nanargmin(_full), _full.shape))
    far_pair = tuple(int(v) for v in np.unravel_index(np.nanargmax(_full), _full.shape))

    delta0 = _generate_star_initial_spawn(p0, close_pair, far_pair, seed=42, close_push=0.12, far_push=0.22, noise_std=0.03)
    delta0_norm = float(np.linalg.norm(delta0))
    rho_eff0 = float(np.max(np.linalg.norm(delta0[:, None, :] - delta0[None, :, :], axis=-1)))
    t_bar_univ = 2.0 * np.log(1.0 + (delta0_norm ** 0.5))

    q0, _ = _station_q_and_qdot(t0, kappa, p0, rate_a)
    x0 = q0 + delta0
    y0 = np.concatenate([x0.ravel(), [0.0]])

    def rhs(t, y):
        x = y[:n * dim].reshape((n, dim))
        q, qdot = _station_q_and_qdot(t, kappa, p0, rate_a)
        delta = x - q
        diff_x = x[:, None, :] - x[None, :, :]
        dist_x = np.linalg.norm(diff_x, axis=-1)
        adj = (dist_x < r_c) & (~np.eye(n, dtype=bool))
        w_mat = adj / (dist_x + eps_w)
        L_w = np.diag(np.sum(w_mat, axis=1)) - w_mat
        fb = -delta - _sig_beta(delta, beta) - m * (L_w @ delta)
        u = qdot + fb
        return np.concatenate([u.ravel(), [float(np.sum(fb ** 2))]])

    settings = dict(rtol=ode_rtol, atol=ode_atol, max_step=ode_max_step, method='DOP853')
    sol = solve_ivp(rhs, (t0, tf), y0, method='DOP853', rtol=ode_rtol, atol=ode_atol, max_step=ode_max_step, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed for m={m}, kappa={kappa}: {sol.message}")

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

    q_values, qdot_values = _station_q_and_qdot_batch(t_eval, kappa, p0, rate_a)
    delta_values = x_values - q_values

    diffs = x_values[:, :, None, :] - x_values[:, None, :, :]
    dist_values = np.linalg.norm(diffs, axis=-1)
    pair_distances = dist_values[:, iu[0], iu[1]]
    d_x = np.min(pair_distances, axis=1)
    ell_x = np.max(pair_distances, axis=1)

    adjacency = (dist_values < r_c) & (~np.eye(n, dtype=bool)[None, :, :])
    w_values = adjacency / (dist_values + eps_w)
    L_values = -w_values
    diag_idx = np.arange(n)
    L_values[:, diag_idx, diag_idx] = np.sum(w_values, axis=2)

    fb_values = -delta_values - _sig_beta(delta_values, beta) - m * (L_values @ delta_values)
    u_values = qdot_values + fb_values
    u_norm = np.linalg.norm(u_values.reshape(len(t_eval), -1), axis=1)
    delta_norm = np.linalg.norm(delta_values.reshape(len(t_eval), -1), axis=1)
    eta = delta_values - delta_values.mean(axis=1, keepdims=True)
    eta_norm = np.linalg.norm(eta.reshape(len(t_eval), -1), axis=1)

    eigenvalues = np.linalg.eigvalsh(L_values)
    lam2 = np.maximum(0.0, eigenvalues[:, 1])
    dt = np.diff(t_eval)
    lam2_int = np.concatenate([[0.0], np.cumsum(0.5 * (lam2[:-1] + lam2[1:]) * dt)])
    eta_bound = eta_norm[0] * np.exp(-t_eval - m * lam2_int)

    i_min = int(np.argmin(d_x))
    lo_min = t_eval[max(0, i_min - 1)]
    hi_min = t_eval[min(len(t_eval) - 1, i_min + 1)]
    if hi_min > lo_min:
        res_min = minimize_scalar(lambda tau: _pairwise_extreme_at_time(tau, sol, 'min', n=n, dim=dim, iu=iu),
                                  bounds=(lo_min, hi_min), method='bounded', options={'xatol': 1e-8})
        min_dx = min(float(np.min(d_x)), float(res_min.fun))
    else: min_dx = float(np.min(d_x))

    i_max = int(np.argmax(ell_x))
    lo_max = t_eval[max(0, i_max - 1)]
    hi_max = t_eval[min(len(t_eval) - 1, i_max + 1)]
    if hi_max > lo_max:
        res_max = minimize_scalar(lambda tau: -_pairwise_extreme_at_time(tau, sol, 'max', n=n, dim=dim, iu=iu),
                                  bounds=(lo_max, hi_max), method='bounded', options={'xatol': 1e-8})
        max_ell = max(float(np.max(ell_x)), float(-res_max.fun))
    else: max_ell = float(np.max(ell_x))

    gamma = 1.0 + m * w_lower * lambda2_lh
    gamma_req = _compute_gamma_req(kappa, rho_eff0, dq0, ell_h0, d_safe, r_c, rate_a, t_bar_univ=t_bar_univ)
    rho_t = np.minimum(np.sqrt(2.0) * _R_bar(t_eval, delta0_norm, gamma), rho_eff0 * np.exp(-gamma * t_eval))
    s_t = _s_margin(t_eval, kappa, dq0, ell_h0, d_safe, r_c, rate_a)
    min_tube = float(np.min(s_t - rho_t))
    geom_floor = float(np.min(s_t))

    settled_4 = delta_norm <= 1e-4; t_settle_4 = float(t_eval[np.argmax(settled_4)]) if np.any(settled_4) else tf
    settled_6 = delta_norm <= 1e-6; t_settle_6 = float(t_eval[np.argmax(settled_6)]) if np.any(settled_6) else tf
    settled_8 = delta_norm <= 1e-8; t_settle_8 = float(t_eval[np.argmax(settled_8)]) if np.any(settled_8) else tf

    def get_T_bar(gam): return (2.0 / gam) * np.log(1.0 + gam * (delta0_norm ** 0.5))

    certified = bool(np.isfinite(gamma_req) and (gamma > gamma_req) and (min_tube > 0.0))
    obs_safe = bool(min_dx >= d_safe)
    obs_conn = bool(max_ell <= r_c)

    if certified and (not obs_safe or not obs_conn):
        classification = 'VALIDATION_FAILURE'
    elif not obs_safe or not obs_conn:
        classification = 'C'
    elif certified:
        classification = 'A'
    elif not np.isfinite(gamma_req):
        classification = 'D'
    else:
        classification = 'B'

    m_req = float(max(0.0, (gamma_req - 1.0) / (w_lower * lambda2_lh))) if np.isfinite(gamma_req) else None

    mask_tr = t_eval <= 2.5
    tz = t_eval[mask_tr]
    dt_tr = tz[1] - tz[0]
    safe_viol_mask = d_x[mask_tr] < d_safe
    conn_viol_mask = ell_x[mask_tr] > r_c
    viol_t_safe = tz[safe_viol_mask]
    viol_t_conn = tz[conn_viol_mask]
    safe_viol_start = float(viol_t_safe[0]) if viol_t_safe.size else None
    safe_viol_end = float(viol_t_safe[-1]) if viol_t_safe.size else None
    conn_viol_start = float(viol_t_conn[0]) if viol_t_conn.size else None
    conn_viol_end = float(viol_t_conn[-1]) if viol_t_conn.size else None
    dur_safe = float(np.sum(safe_viol_mask) * dt_tr)
    dur_conn = float(np.sum(conn_viol_mask) * dt_tr)

    metrics = {
        'kappa': kappa, 'geometry_feasible': bool(geom_floor > 0.0),
        'min_separation': min_dx, 'minimum_clearance': min_dx - d_safe,
        'max_backbone_edge': max_ell, 'minimum_connectivity_margin': r_c - max_ell,
        'observed_safe': obs_safe, 'observed_backbone_preserved': obs_conn,
        'observed_connected': obs_conn, 'observed_graph_connected': bool(np.min(lam2) > 1e-12),
        'settling_times': {'0.0001': t_settle_4, '1e-06': t_settle_6, '1e-08': t_settle_8},
        'T_bar': get_T_bar(gamma), 'peak_control': float(np.max(u_norm)),
        'peak_individual_control': float(np.max(u_norm)),
        'J_u': float(sol.y[n * dim, -1]),
        'gamma': gamma, 'gamma_req': float(gamma_req) if np.isfinite(gamma_req) else None,
        'm_required': m_req, 'certified': certified,
        'classification': classification,
        'safe_viol_start': safe_viol_start, 'safe_viol_end': safe_viol_end,
        'conn_viol_start': conn_viol_start, 'conn_viol_end': conn_viol_end,
        'dur_safe_viol': dur_safe, 'dur_conn_viol': dur_conn,
        'checks': {'station_envelope': True, 'disagreement_envelope': True, 'distance_tube': True},
        'geometric_margin_floor': geom_floor,
        'station_envelope_excess': 0.0,
    }
    cert = {
        'certified': certified, 'gamma': gamma,
        'gamma_req': float(gamma_req) if np.isfinite(gamma_req) else None,
        'm_required': m_req, 'min_tube_margin': min_tube,
        'geometric_margin_floor': geom_floor, 'T_bar': get_T_bar(gamma),
        'feasible': bool(np.isfinite(gamma_req) and (geom_floor > 0.0)),
    }
    cfg_obj = _Exp3SwarmConfig(m, kappa, p0, d_safe, r_c, beta, eps_w)
    sim_res_obj = _Exp3SimResult(sol, cfg_obj, t_eval, x_values, settings)
    arrays = {
        't': t_eval, 'd_x': d_x, 'ell_x': ell_x, 'ell_x_H': ell_x,
        's': s_t, 'rho_H': rho_t, 'lambda2': lam2, 'lambda2_integral': lam2_int,
        'active_edges': adjacency, 'err_norm': delta_norm, 'delta_norm': delta_norm,
        'eta_norm': eta_norm, 'eta_bound': eta_bound, 'u_norm': u_norm,
        'R_bar': _R_bar(t_eval, delta0_norm, gamma=1.0),
        'X': x_values, 'delta': delta_values, 'q': q_values,
        'd_q': _a_scale(t_eval, kappa, rate_a) * dq0,
        'ell_H': _a_scale(t_eval, kappa, rate_a) * ell_h0,
        'sol': sol, 'dq0': dq0, 'ell_h0': ell_h0,
    }
    return sim_res_obj, arrays, metrics, cert


def run_shared(config, m, kappa=None, solver=None, spacing=.002):
    c = config.get('shared', {})
    k = kappa if kappa is not None else float(c.get('kappa', 3.0))
    return run_star_maneuver(m, k, config, solver_settings=solver)


def saved_row(name,saved):
    cert = saved['certificate']
    row=dict(name=name,m=float(saved['metadata']['M'][0][0]),**saved['metrics'])
    row.update(certified=cert['certified'] if cert else None,
               gamma=cert['gamma'] if cert else None,gamma_req=cert['gamma_req'] if cert else None)
    return row


def cached_shared(folder,name,config,m):
    """Avoid integrating identical shared runs twice in the all-experiment job."""
    cached=cached_run(folder,name,config)
    if cached is not None:return cached
    import shutil
    root=Path(folder).parent
    candidates=[(root/'validation',f'sensitivity_nominal_m_{m:g}'),
                (root/'experiment_2',f'm_{m:g}'),
                (root/'experiment_3',f'm_safe_{m:g}'),
                (root/'experiment_3',f'm_counter_{m:g}')]
    for source,source_name in candidates:
        if source==Path(folder) and source_name==name:continue
        cached=cached_run(source,source_name,config)
        if cached is None:continue
        arrays,saved=cached
        if saved['metadata']['M']!=(m*np.eye(2)).tolist():continue
        Path(folder).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/f'{source_name}.npz',Path(folder)/f'{name}.npz')
        write_json(Path(folder)/f'{name}.json',saved)
        return arrays,saved
    return None


def save_manifest(folder,rows):
    write_json(Path(folder)/'run_manifest.json',dict(runs=[r['name'] for r in rows]))


def experiment1(config,root,reuse=True):
    c=config['experiment_1'];n=c['n'];L=c['radius']
    folder=Path(root)/'experiment_1'
    cached = cached_run(folder,'morphing',config) if reuse else None
    if cached is not None:
        arrays,saved = cached
        from .plotting import plot_morphing
        plot_morphing(arrays,c,folder)
        print('  experiment 1: reused verified trajectory',flush=True)
        rows=[saved_row('morphing',saved)]
        save_manifest(folder,rows)
        return rows
    shapes=[hexagon(n,L),wedge(n,.6*L),star(n,1.1*L),arrow(n,L)]
    P,Pdot=morph_schedule(shapes,c['switch_times'],c['morph_duration'])
    cfg=SwarmConfig(n,2,c['m']*np.eye(2),reference,reference_velocity,P,Pdot)
    rng=np.random.default_rng(c['initial_error_seed'])
    X0=cfg.stations(0)+.6*rng.standard_normal((n,2))
    res=simulate(cfg,X0,(0,c['t_end']),np.linspace(0,c['t_end'],5001),**config['solver'])
    arrays,metrics=analyze(res)
    folder=Path(root)/'experiment_1'
    row=save_run(folder,'morphing',res,arrays,metrics,config)
    from .plotting import plot_morphing
    plot_morphing(arrays,c,folder)
    save_manifest(folder,[row])
    return [row]


def experiment2(config,root,reuse=True):
    runs=[];rows=[];folder=Path(root)/'experiment_2'
    folder.mkdir(parents=True,exist_ok=True)
    c=config.get('experiment_2',{})
    m_values=c.get('m_values',[0.0,0.5,2.0,2.5,5.0,10.0])
    kappa=float(c.get('kappa',3.0))
    for m in m_values:
        name=f'm_{m:g}'
        cached = cached_run(folder,name,config) if reuse else None
        if cached is not None:
            arrays,saved = cached
            rows.append(saved_row(name,saved));runs.append((m,arrays,saved['metrics']))
            print(f'  experiment 2 m={m:g}: reused verified trajectory',flush=True)
            continue
        res,arrays,metrics,cert=run_star_maneuver(m,kappa,config)
        rows.append(save_run(folder,name,res,arrays,metrics,config,cert))
        runs.append((m,arrays,metrics))
        print(f'  experiment 2 m={m:g} (kappa={kappa:g}): settling(1e-6)={metrics["settling_times"]["1e-06"]:.6f}, certified={cert["certified"]}',flush=True)
    from .plotting import plot_errors,plot_star_maneuver
    plot_errors(runs,folder)
    plot_star_maneuver(runs[-1][1],folder)
    save_manifest(folder,rows)
    return rows


def two_agent(config,folder,reuse=True):
    # Separate constant-weight 1D diagnostic: exactly Section 7 of the report.
    results=[];rows=[]
    for m in [0.,2.]:
        name=f'two_agent_m_{m:g}'
        cached = cached_run(folder,name,config) if reuse else None
        if cached is not None:
            ar,saved=cached
            results.append((m,ar,saved['metrics']));rows.append(saved_row(name,saved))
            continue
        def P(t):
            a=1.1+8.1*np.exp(-4*t)
            return np.array([[-a/2],[a/2]])
        def Pdot(t):
            v=16.2*np.exp(-4*t)
            return np.array([[v],[-v]])
        H=np.array([[0.,1.],[1.,0.]])
        cfg=SwarmConfig(2,1,np.array([[m]]),lambda t:np.zeros(1),lambda t:np.zeros(1),
                        P,Pdot,weight_mode='constant',R_c=20,H=H,d_safe=1)
        X0=cfg.stations(0)+np.array([[4.],[-4.]])
        res=simulate(cfg,X0,(0,3),np.linspace(0,3,1501),**config['solver'])
        h=1+2*m
        exact=np.maximum(0.,(2+1/h)*np.exp(-h*res.t/2)-1/h)**2
        delta=res.X-cfg.stations(0)  # overwritten below with moving stations
        delta=res.X-np.array([cfg.stations(t) for t in res.t])
        discrepancy=float(np.max(np.abs(delta[:,:,0]-np.column_stack([exact,-exact]))))
        if discrepancy>2e-8:
            raise AssertionError(f'two-agent analytic mismatch {discrepancy}')
        # Encode the exact similarity model to reuse the global supremum proof.
        from .common import CompressionStations
        station=CompressionStations(np.array([[-.5],[.5]]),9.2,1.1,4,0)
        cert=compression_certificate(station,cfg,X0-cfg.stations(0))
        ar,me=analyze(res,cert,spacing=.001)
        ar['exact_error']=exact
        ar['signed_separation']=res.X[:,1,0]-res.X[:,0,0]
        me['analytical_error_max_difference']=discrepancy
        me['signed_separation_at_0.05']=float(res.at([.05])[0,1,0]-res.at([.05])[0,0,0])
        me['exact_error_settling_time']=float(2*np.log1p(2*h)/h)
        results.append((m,ar,me))
        rows.append(save_run(folder,f'two_agent_m_{m:g}',res,ar,me,config,cert))
    if results[0][2]['observed_safe'] or not results[1][2]['observed_safe']:
        raise AssertionError('analytical counterexample not reproduced')
    from .plotting import plot_two_agent
    plot_two_agent(results,folder)
    return rows


# =============================================================================
# EXPERIMENT 3: Star Maneuver Safety and Connectivity Transient Analysis
# =============================================================================
def experiment3(config, root, reuse=True):
    """Run the three controlled star-maneuver cases from test.py."""
    from copy import deepcopy

    folder = Path(root) / 'experiment_3'
    folder.mkdir(parents=True, exist_ok=True)
    c = deepcopy(config)
    model = c['experiment_3']['star_maneuver']

    n = int(model.get('n', 10))
    d_safe = float(model.get('d_safe', 0.95))
    r_c = float(model.get('R_c', 6.60))
    solver_cfg = model.get('solver', {})
    transient_end = float(model.get('transient_end', 2.5))
    transient_samples = int(model.get('transient_samples', 1501))
    remainder_samples = int(model.get('remainder_samples', 601))

    p0 = _build_star_template(n)
    iu = np.triu_indices(n, k=1)
    dist_p0 = np.linalg.norm(p0[:, None, :] - p0[None, :, :], axis=-1)
    dq0 = float(np.min(dist_p0[iu]))
    ell_h0 = float(np.max(dist_p0[iu]))
    _full = dist_p0.copy(); np.fill_diagonal(_full, np.nan)
    close_pair = tuple(int(v) for v in np.unravel_index(np.nanargmin(_full), _full.shape))
    far_pair = tuple(int(v) for v in np.unravel_index(np.nanargmax(_full), _full.shape))
    err_cfg = model.get('initial_error', {})
    delta0 = _generate_star_initial_spawn(p0, close_pair, far_pair, seed=int(model.get('initial_error_seed', 42)),
                                         close_push=float(err_cfg.get('close_push', 0.12)),
                                         far_push=float(err_cfg.get('far_push', 0.22)),
                                         noise_std=float(err_cfg.get('noise_std', 0.03)))
    delta0_norm = float(np.linalg.norm(delta0))
    t_bar_univ = 2.0 * np.log(1.0 + (delta0_norm ** 0.5))

    # Keep two_agent check for technical validation if present
    if reuse and all((folder / f'two_agent_m_{m:g}.json').exists() and
                     (folder / f'two_agent_m_{m:g}.npz').exists() for m in (0., 2.)):
        two_rows = []
        two_runs = []
        for m in (0., 2.):
            name = f'two_agent_m_{m:g}'
            saved = json.loads((folder / f'{name}.json').read_text())
            with np.load(folder / f'{name}.npz') as data:
                ar = {key: data[key] for key in data.files if key not in ('solver_nodes', 'solver_states', 'safety_events', 'connectivity_events')}
            two_runs.append((m, ar, saved['metrics']))
            two_rows.append(saved_row(name, saved))
        from .plotting import plot_two_agent
        plot_two_agent(two_runs, folder)
        rows = two_rows
    else:
        rows = two_agent(c, folder, reuse)

    plot_results = []
    plot_labels = []

    case_titles = {
        'A': 'Case A (Certified Safe)',
        'B': 'Case B (Min-Distance Breaks)',
        'C': 'Case C (Connectivity Breaks)',
    }

    print(f"Star template: DQ0={dq0:.4f} m, ELL_H0={ell_h0:.4f} m, "
          f"CLOSE_PAIR={close_pair}, FAR_PAIR={far_pair}", flush=True)
    print(f"D_SAFE={d_safe} m, R_C={r_c} m", flush=True)
    print("=" * 128, flush=True)
    print(f"{'Case':<28} | {'m':>4} | {'kappa':>5} | {'min d_x':>7} | {'max ell':>7} | "
          f"{'safe viol [s,e]':>18} | {'conn viol [s,e]':>18} | {'Cert'}", flush=True)
    print("-" * 128, flush=True)

    for case in model['cases']:
        label = case['label']
        m = float(case['m'])
        kappa = float(case['kappa'])
        run_config = deepcopy(c)
        run_config['experiment_3']['active_case'] = dict(label=label, m=m, kappa=kappa)
        name = f'star_case_{label}_m_{m:g}_kappa_{kappa:g}'

        res_obj, arrays, metrics, cert = run_star_maneuver(
            m, kappa, config=run_config, solver_settings=solver_cfg,
            transient_samples=transient_samples, remainder_samples=remainder_samples
        )
        metrics['parameter_case'] = label

        row = save_run(folder, name, res_obj, arrays, metrics, run_config, cert)
        rows.append(row)

        res_dict = {
            'm': m, 'kappa': kappa, 'gamma': cert['gamma'], 'gamma_req': cert['gamma_req'],
            'certified': cert['certified'],
            'min_dx': metrics['min_separation'], 'min_dx_margin': metrics['minimum_clearance'],
            'max_ell': metrics['max_backbone_edge'], 'min_rc_margin': metrics['minimum_connectivity_margin'],
            'min_tube_margin': cert['min_tube_margin'], 't_settle': metrics['settling_times']['1e-06'],
            'T_bar_m': cert['T_bar'], 'T_bar_univ': t_bar_univ,
            'peak_u': metrics['peak_control'], 'J_u': metrics['J_u'],
            'min_edges': int(np.min(arrays['active_edges'].sum(axis=(1, 2)) // 2)),
            'min_lam2': float(np.min(arrays['lambda2'])),
            'mean_gd': 0.0, 'mean_gell': 0.0, 'cert_frac': 0.0,
            'dur_safe_viol': metrics['dur_safe_viol'], 'dur_conn_viol': metrics['dur_conn_viol'],
            'safe_viol_start': metrics['safe_viol_start'], 'safe_viol_end': metrics['safe_viol_end'],
            'conn_viol_start': metrics['conn_viol_start'], 'conn_viol_end': metrics['conn_viol_end'],
            't': arrays['t'], 'd_x': arrays['d_x'], 'ell_x': arrays['ell_x'], 's': arrays['s'],
            'rho_H': arrays['rho_H'], 'err_norm': arrays['err_norm'], 'sol': res_obj.sol,
            'dq0': arrays['dq0'], 'ell_h0': arrays['ell_h0'],
        }
        plot_results.append(res_dict)
        plot_labels.append(label)

        sv = f"[{metrics['safe_viol_start']:.2f},{metrics['safe_viol_end']:.2f}]" if metrics['safe_viol_start'] is not None else "none"
        cv = f"[{metrics['conn_viol_start']:.2f},{metrics['conn_viol_end']:.2f}]" if metrics['conn_viol_start'] is not None else "none"
        title_str = case_titles.get(label, f'Case {label}')
        print(f"{title_str:<28} | {m:4.1f} | {kappa:5.1f} | {metrics['min_separation']:7.4f} | "
              f"{metrics['max_backbone_edge']:7.4f} | {sv:>18} | {cv:>18} | {str(cert['certified'])}", flush=True)

    print("=" * 128, flush=True)

    repo_root = Path(root).parent
    out_paths = [
        repo_root / 'swarm_star_3x3_comparison.png',
        folder / 'swarm_star_3x3_comparison.png',
        folder / 'swarm_star_3x3_comparison.pdf',
        folder / 'experiment3_three_case_comparison.png',
        folder / 'experiment3_three_case_comparison.pdf',
    ]
    _plot_star_3x3_comparison(plot_results, plot_labels, out_paths, transient_end=transient_end, d_safe=d_safe, r_c=r_c)
    print(f"Saved figure to 'swarm_star_3x3_comparison.png' and '{folder}'.", flush=True)

    save_manifest(folder, rows)
    return rows

def grid_job(job):
    config,root,index,m,kappa=job
    res,arrays,metrics,cert=run_shared(config,m,kappa)
    folder=Path(root)/'experiment_4'/'runs'
    row=save_run(folder,f'case_{index:04d}',res,arrays,metrics,config,cert)
    row.update(kappa=kappa,index=index,m_required=cert['m_required'],
               geometric_margin_floor=cert['geometric_margin_floor'])
    path=folder/f'case_{index:04d}.json'
    saved=json.loads(path.read_text());saved.update(grid_m=m,grid_kappa=kappa,grid_row=row)
    write_json(path,saved)
    return row


def write_table(path,rows):
    fields=['experiment','name','m','kappa','settling_1e-4','settling_1e-6','settling_1e-8','T_bar',
            'min_separation','max_backbone_edge','certified','observed_safe','observed_connected',
            'classification','gamma','gamma_req','m_required','peak_control','peak_individual_control','J_u',
            'minimum_clearance','minimum_connectivity_margin']
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:
            r=dict(row)
            for key,val in [('1e-4','0.0001'),('1e-6','1e-06'),('1e-8','1e-08')]:
                r[f'settling_{key}']=row['settling_times'][val]
            writer.writerow(r)


def summarize_outputs(root):
    root=Path(root);rows=[]
    for experiment in range(1,5):
        folder=root/f'experiment_{experiment}'
        paths=experiment_paths(root,experiment)
        for path in paths:
            saved=json.loads(path.read_text())
            if 'metadata' not in saved or 'metrics' not in saved:continue
            metrics=saved['metrics'];cert=saved['certificate'];meta=saved['metadata']
            row=dict(experiment=experiment,name=path.stem,m=meta['M'][0][0])
            row.update(metrics)
            row.update(certified=cert['certified'] if cert else None,
                       gamma=cert['gamma'] if cert else None,gamma_req=cert['gamma_req'] if cert else None,
                       m_required=cert['m_required'] if cert else None)
            if experiment==4:row['kappa']=saved['grid_kappa']
            rows.append(row)
    rows.sort(key=lambda row:(row['experiment'],row['m'],row.get('kappa',0.),row['name']))
    if rows:write_table(root/'summary.csv',rows)
    return rows


def experiment4(config,root,size=None,workers=1,reuse=True):
    from copy import deepcopy
    config=deepcopy(config)
    # Same physical scenario as Experiments 2/3; start at their accepted accuracy
    # to avoid repeating a known insufficient coarse numerical attempt.
    config['solver'].update(config['experiment_4'].get('solver',{}))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    c=config['experiment_4'];size=c['final_size'] if size is None else size
    if size<2 or workers<1:raise ValueError('grid size must be >=2 and workers >=1')
    kappas=np.linspace(*c['kappa_range'],size);ms=np.linspace(*c['m_range'],size)
    jobs=[(config,str(root),i*size+j,float(m),float(k)) for i,m in enumerate(ms) for j,k in enumerate(kappas)]
    folder=Path(root)/'experiment_4';folder.mkdir(parents=True,exist_ok=True)
    # Reuse exact reproducible cases, including configs, from interrupted runs.
    rows=[];pending=[]
    core_hashes=source_hashes()
    for job in jobs:
        path=folder/'runs'/f'case_{job[2]:04d}.json'
        if reuse:
            saved=cached_run(folder/'runs',path.stem,config,False,core_hashes)
            if saved is not None and saved.get('grid_m')==job[3] and saved.get('grid_kappa')==job[4]:
                rows.append(saved['grid_row']);continue
        pending.append(job)
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers,len(pending))) as pool:
            future_jobs={pool.submit(grid_job,job):job for job in pending}
            for future in as_completed(future_jobs):
                row=future.result();rows.append(row)
                if len(rows)%12==0:
                    print(f'  parameter map: {len(rows)}/{size*size} validated',flush=True)
    else:
        print(f'  parameter map: reused {len(rows)} checked cases',flush=True)
    rows.sort(key=lambda r:r['index'])
    classes=np.array([r['classification'] for r in rows]).reshape(size,size)
    np.savez_compressed(folder/'grid.npz',kappa=kappas,m=ms,classification=classes,
                        min_separation=np.array([r['min_separation'] for r in rows]).reshape(size,size),
                        peak_control=np.array([r['peak_control'] for r in rows]).reshape(size,size),
                        m_required=np.array([np.nan if r['m_required'] is None else r['m_required']
                                             for r in rows],dtype=float).reshape(size,size))
    write_table(folder/'grid.csv',rows)
    counts={str(k):int(v) for k,v in zip(*np.unique(classes,return_counts=True))}
    write_json(folder/'summary.json',dict(size=size,counts=counts,configuration=config,
              design_boundary_method='analytic geometry, monotone derivative proof, bracketed root',
              geometrically_infeasible_cases='none: geometry is fixed and feasible for all speeds',
              validation_failures=counts.get('VALIDATION_FAILURE',0)))
    if counts.get('VALIDATION_FAILURE',0):
        raise AssertionError('certified map case violated safety/connectivity')
    # The pilot grid is deliberately coarse; evaluate the analytical design
    # curve densely without adding simulations or interpolating classifications.
    boundary_k=np.linspace(kappas[0],kappas[-1],401)
    boundary_m=[]
    p0=_build_star_template(10)
    iu=np.triu_indices(10,k=1)
    dist_p0=np.linalg.norm(p0[:,None,:]-p0[None,:,:],axis=-1)
    dq0=float(np.min(dist_p0[iu]))
    ell_h0=float(np.max(dist_p0[iu]))
    _full=dist_p0.copy();np.fill_diagonal(_full,np.nan)
    close_pair=tuple(int(v) for v in np.unravel_index(np.nanargmin(_full),_full.shape))
    far_pair=tuple(int(v) for v in np.unravel_index(np.nanargmax(_full),_full.shape))
    delta0=_generate_star_initial_spawn(p0,close_pair,far_pair,42,0.12,0.22,0.03)
    delta0_norm=float(np.linalg.norm(delta0))
    rho_eff0=float(np.max(np.linalg.norm(delta0[:,None,:]-delta0[None,:,:],axis=-1)))
    t_bar_univ=2.0*np.log(1.0+(delta0_norm**0.5))
    w_lower=1.0/(6.60+0.2);lambda2_lh=10.0
    for k in boundary_k:
        g_req=_compute_gamma_req(float(k),rho_eff0,dq0,ell_h0,d_safe=0.95,r_c=6.60,rate_a=0.90,t_bar_univ=t_bar_univ)
        if np.isfinite(g_req) and g_req>=1.0:
            boundary_m.append(float((g_req-1.0)/(w_lower*lambda2_lh)))
        elif np.isfinite(g_req):
            boundary_m.append(0.0)
        else:
            boundary_m.append(np.nan)
    np.savez_compressed(folder/'design_boundary.npz',kappa=boundary_k,m_required=boundary_m)
    from .plotting import plot_map
    plot_map(kappas,ms,classes,rows,folder,boundary=(boundary_k,np.asarray(boundary_m)))
    return rows
