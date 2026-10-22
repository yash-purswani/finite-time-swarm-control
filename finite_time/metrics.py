"""Error signals and measured quantities extracted from a simulation."""

import numpy as np

Vec = np.ndarray


class NumericalValidationError(AssertionError):
    def __init__(self,metrics):
        self.metrics=metrics
        super().__init__(f'Bound validation failed: {metrics}')


def reference(t: Vec, cfg) -> Vec:
    """Stacked reference r(t) of shape (T, d)."""
    return np.array([cfg.r_fn(float(ti)) for ti in np.atleast_1d(t)], dtype=float)


def biases(t: Vec, cfg) -> Vec:
    """Stacked formation biases P(t) of shape (T, n, d)."""
    return np.array([cfg.P_fn(float(ti)) for ti in np.atleast_1d(t)], dtype=float)


def centroid_error(t: Vec, X: Vec, cfg) -> Vec:
    """sigma(t) = c(t) - r(t), Eq. (3). ``X`` is (T, n, d); returns (T, d)."""
    return X.mean(axis=1) - reference(t, cfg)


def formation_errors(t: Vec, X: Vec, cfg) -> Vec:
    """delta_i(t) = x_i - p_i - r, Eq. (5). ``X`` is (T, n, d); returns (T, n, d)."""
    return X - biases(t, cfg) - reference(t, cfg)[:, None, :]


def delta_norm(t: Vec, X: Vec, cfg) -> Vec:
    """Collective ||delta(t)|| = sqrt(sum_i ||delta_i||^2), the quantity Prop. 2 bounds."""
    D = formation_errors(t, X, cfg)
    return np.linalg.norm(D.reshape(D.shape[0], -1), axis=1)


def max_agent_error(t: Vec, X: Vec, cfg) -> Vec:
    """max_i ||delta_i(t)||, the quantity Cor. 1 turns into a docking tolerance."""
    D = formation_errors(t, X, cfg)
    return np.linalg.norm(D, axis=2).max(axis=1)


# =========================================================================
# Derived measurements
# =========================================================================

def empirical_rate(t: Vec, y: Vec, lo: float = None, hi: float = None,
                   floor: float = 1e-9) -> float:
    """Least-squares decay rate of ``y`` from the slope of ln y over [lo, hi].

    Returns ``-d(ln y)/dt``, so a value >= 1 confirms the Prop. 2 guarantee.
    Samples at or below ``floor`` are dropped -- they carry integrator noise, not
    dynamics.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y) & (y > floor)
    if lo is not None:
        m &= t >= lo
    if hi is not None:
        m &= t <= hi
    if m.sum() < 3:
        return np.nan
    slope = np.polyfit(t[m], np.log(y[m]), 1)[0]
    return float(-slope)


def settling_time(t: Vec, y: Vec, tol: float) -> float:
    """First time after which ``y`` stays at or below ``tol`` for the rest of the run.

    Interpolates linearly between the bracketing samples so the answer is not
    quantised to the output grid.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    above = np.where(y > tol)[0]
    if above.size == 0:
        return float(t[0])
    k = above[-1]
    if k + 1 >= t.size:
        return np.inf                       # never settles within the horizon
    y0, y1 = y[k], y[k + 1]
    if y0 == y1:
        return float(t[k + 1])
    frac = (y0 - tol) / (y0 - y1)
    return float(t[k] + frac * (t[k + 1] - t[k]))


def ultimate_value(t: Vec, y: Vec, tail_frac: float = 0.25, stat: str = "max") -> float:
    """Steady-state level of ``y``, taken over the last ``tail_frac`` of the horizon."""
    t = np.asarray(t, dtype=float)
    cut = t[0] + (1.0 - tail_frac) * (t[-1] - t[0])
    tail = np.asarray(y, dtype=float)[t >= cut]
    if tail.size == 0:
        return float(np.asarray(y)[-1])
    return float(tail.max() if stat == "max" else np.mean(tail))


def min_pair_distance(X: Vec) -> Vec:
    """Smallest inter-agent distance at each time step; ``X`` is (T, n, d)."""
    diff = X[:, :, None, :] - X[:, None, :, :]
    dist = np.linalg.norm(diff, axis=3)
    n = X.shape[1]
    iu = np.triu_indices(n, k=1)
    return dist[:, iu[0], iu[1]].min(axis=1)


def max_pair_distance(X: Vec) -> Vec:
    """Largest inter-agent distance at each time step; ``X`` is (T, n, d)."""
    diff = X[:, :, None, :] - X[:, None, :, :]
    return np.linalg.norm(diff, axis=3).reshape(X.shape[0], -1).max(axis=1)


def held_commands(t: Vec, X: Vec) -> Vec:
    """Recover the commanded velocity from a sampled trajectory, shape (T-1, n, d).

    Forward differences, not ``np.gradient``: under a zero-order hold the plant is
    an integrator, so (x_{k+1} - x_k)/dt *is* the command u_k exactly. A central
    difference would average consecutive commands and halve precisely the
    step-to-step switching this is meant to measure.
    """
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)[:, None, None]
    return np.diff(np.asarray(X, dtype=float), axis=0) / dt


def control_effort(t: Vec, X: Vec) -> float:
    """Integral of sum_i ||u_i|| dt over the run."""
    t = np.asarray(t, dtype=float)
    u = held_commands(t, X)
    speed = np.linalg.norm(u, axis=2).sum(axis=1)
    return float(np.sum(speed * np.diff(t)))


def chattering_index(t: Vec, X: Vec, tail_frac: float = 0.5) -> float:
    """Total variation of the *commanded* signal over the tail, per unit time.

    Summed over agents and coordinates rather than over ||u_i||. The discrete-time
    sliding motion is a two-cycle that flips the sign of the command each step
    while leaving its magnitude almost unchanged, so a norm-based measure reports
    zero for a trajectory that is in fact switching at every sample.
    """
    t = np.asarray(t, dtype=float)
    u = held_commands(t, X)
    tu = t[:-1]
    cut = tu[0] + (1.0 - tail_frac) * (tu[-1] - tu[0])
    m = tu >= cut
    if m.sum() < 3:
        return 0.0
    span = tu[m][-1] - tu[m][0]
    tv = np.abs(np.diff(u[m], axis=0)).sum()
    return float(tv / max(span, 1e-12))


def refined_distance_extrema(res, spacing=.002):
    """Observed extrema using dense solver output and pair-specific refinement.

    Scan every pair, including near-coincident encounters, then optimize every
    sampled local extremum on its neighboring interval. These are numerical
    observations, not rigorous continuous-time proofs. Analytic certificates
    are computed separately in theory.py.
    """
    from scipy.optimize import minimize_scalar
    t = np.linspace(res.t[0],res.t[-1],int(np.ceil((res.t[-1]-res.t[0])/spacing))+1)
    X = res.at(t)
    iu = np.triu_indices(res.cfg.n,1)
    distances = np.linalg.norm(X[:,iu[0]]-X[:,iu[1]],axis=2)
    he = np.where(np.triu(res.cfg.H,1)>0) if res.cfg.H is not None else iu
    hmask = {(int(i),int(j)) for i,j in zip(*he)}
    minimum,maximum = float(distances.min()),0.
    tminimum,tmaximum = float(t[np.unravel_index(np.argmin(distances),distances.shape)[0]]),0.
    for k,(i,j) in enumerate(zip(*iu)):
        y = distances[:,k]
        for sign in ([1,-1] if (int(i),int(j)) in hmask else [1]):
            z = sign*y
            candidates = np.where((z[1:-1]<z[:-2])&(z[1:-1]<z[2:])&
                                  (np.maximum(z[:-2]-z[1:-1],z[2:]-z[1:-1])>1e-8))[0]+1
            # Include boundaries; values on the dense grid are retained too.
            best = float(z.min()); when = float(t[np.argmin(z)])
            for index in candidates:
                def objective(tq):
                    pos = res.at([tq])[0]
                    return sign*np.linalg.norm(pos[i]-pos[j])
                opt = minimize_scalar(objective,bounds=(t[index-1],t[index+1]),
                                      method='bounded',options={'xatol':1e-11})
                if opt.fun<best:
                    best,when = float(opt.fun),float(opt.x)
            if sign==1 and best<minimum:
                minimum,tminimum = best,when
            if sign==-1 and -best>maximum:
                maximum,tmaximum = -best,when
    return dict(min_separation=minimum,time_min_separation=tminimum,
                max_backbone_edge=maximum,time_max_backbone_edge=tmaximum,
                minimum_clearance=minimum-res.cfg.d_safe,
                minimum_connectivity_margin=res.cfg.R_c-maximum,
                refinement_spacing=spacing,extrema_method='dense pairwise scan plus local bracket refinement')


def analyze(res,certificate=None,spacing=.002):
    from scipy.integrate import cumulative_trapezoid
    from .model import rhs, rhs_error, graph, lambda2, sig_beta
    from .theory import R_bar,T_bar,disagreement,rho_H,classify
    from .sampling import station_history
    t,X,cfg = res.t,res.X,res.cfg
    q, qdot = station_history(cfg,t)
    delta = X-q
    eta = disagreement(delta)
    dn = np.linalg.norm(delta.reshape(len(t),-1),axis=1)
    en = np.linalg.norm(eta.reshape(len(t),-1),axis=1)
    Ls,edges = zip(*(graph(x,cfg) for x in X))
    Ls,edges = np.array(Ls),np.array(edges)
    # Evaluate the same controller algebra using the already computed graphs.
    correction = -delta-sig_beta(delta,cfg.beta)-np.matmul(Ls,delta)@cfg.M.T
    u = qdot+correction
    un = np.linalg.norm(u.reshape(len(t),-1),axis=1)
    spectra = np.linalg.eigvalsh(Ls)
    lam = np.maximum(0.,spectra[:,1])
    integral = cumulative_trapezoid(lam,t,initial=0.)
    eb = en[0]*np.exp(-(t-t[0])-cfg.lambda_min_M*integral)
    rb = R_bar(t,delta[0],cfg.beta,t[0])
    iu = np.triu_indices(cfg.n,1)
    he = np.where(np.triu(cfg.H,1)>0) if cfg.H is not None else iu
    dx = np.linalg.norm(X[:,iu[0]]-X[:,iu[1]],axis=2)
    dq = np.linalg.norm(q[:,iu[0]]-q[:,iu[1]],axis=2)
    ellx = np.linalg.norm(X[:,he[0]]-X[:,he[1]],axis=2).max(axis=1)
    ellq = np.linalg.norm(q[:,he[0]]-q[:,he[1]],axis=2).max(axis=1)
    s = np.minimum(dq.min(axis=1)-cfg.d_safe,cfg.R_c-ellq)
    # Only a priori certified configurations receive the backbone envelope.
    # For m=0, eta contraction at rate 1 is unconditional even after H loss.
    certified = certificate is not None and certificate['certified']
    if certified or cfg.lambda_min_M==0:
        gamma = certificate['gamma'] if certified else 1.
        rh = rho_H(t,delta[0],gamma,cfg.beta,t[0])
        rho_label = 'certified backbone' if certified else 'graph-independent (mu=0)'
    else:
        rh = np.sqrt(2)*rb
        rho_label = 'graph-independent station tube'
    error_velocity = correction
    dv = np.sum(delta*error_velocity,axis=(1,2))
    dw = np.sum(eta*error_velocity,axis=(1,2))
    lyap_rhs = -dn**2-dn**(1+cfg.beta)
    disagreement_rhs = -(1+cfg.lambda_min_M*lam)*en**2
    centroid = X.mean(axis=1)-np.array([cfg.r_fn(s) for s in t])
    centroid_norm = np.linalg.norm(centroid,axis=1)
    post = t>=T_bar(delta[0],cfg.beta,t[0])
    extrema = refined_distance_extrema(res,spacing)
    # Check settling against both requested samples and adaptive solver nodes,
    # then refine the final threshold crossing on the dense interpolant.
    from scipy.optimize import brentq
    # Adaptive states are already stored at these nodes: do not reevaluate
    # hundreds of thousands of dense interpolants just to recover their values.
    node_states = res.sol.y.T.reshape(-1,cfg.n,cfg.d)
    node_delta = node_states if res.coordinates=='error' else node_states-station_history(cfg,res.sol.t)[0]
    node_norms = np.linalg.norm(node_delta.reshape(len(res.sol.t),-1),axis=1)
    all_times = np.r_[t,res.sol.t]
    order = np.argsort(all_times,kind='stable')
    sorted_times = all_times[order]
    sorted_norms = np.r_[dn,node_norms][order]
    ts, first = np.unique(sorted_times,return_index=True)
    ns = np.maximum.reduceat(sorted_norms,first)
    settles = {}
    for tol in [1e-4,1e-6,1e-8]:
        above = np.where(ns>tol)[0]
        value = float(ts[0])
        if len(above):
            k = above[-1]
            if k+1==len(ts):
                value = np.inf
            else:
                fun = lambda tq:np.linalg.norm(res.at([tq])[0]-cfg.stations(tq))-tol
                value = brentq(fun,ts[k],ts[k+1],xtol=1e-12)
        settles[str(tol)] = float(value)
    metrics = dict(extrema,settling_times=settles,
                   T_bar=T_bar(delta[0],cfg.beta,t[0]),peak_control=float(un.max()),
                   peak_individual_control=float(np.linalg.norm(u,axis=2).max()),
                   J_u=float(np.trapz(np.sum(correction**2,axis=(1,2)),t)),
                   station_envelope_excess=float(np.max(dn-rb)),
                   disagreement_envelope_excess=float(np.max(en-eb)),
                   lyapunov_inequality_excess=float(np.max(dv-lyap_rhs)),
                   disagreement_lyapunov_excess=float(np.max(dw-disagreement_rhs)),
                   centroid_envelope_excess=float(np.max(centroid_norm-rb/np.sqrt(cfg.n))),
                   tube_inequality_excess=float(np.max(np.abs(dx-dq)-rh[:,None])),
                   centered_offset_residual=float(max(np.linalg.norm(cfg.P_fn(s).sum(axis=0)) for s in t)),
                   centroid_error_after_deadline=float(centroid_norm[post].max()) if post.any() else None)
    metrics.update(observed_safe=bool(extrema['minimum_clearance']>0),
                   observed_connected=bool(extrema['minimum_connectivity_margin']>0),
                   rho_label=rho_label,solver_nfev=res.sol.nfev,
                   safety_events=res.sol.t_events[0].tolist(),
                   connectivity_events=res.sol.t_events[1].tolist() if len(res.sol.t_events)>1 else [])
    if certificate is not None:
        metrics['classification'] = classify(certificate,metrics['observed_safe'],metrics['observed_connected'])
    # A 1e-8 absolute norm floor is the explicit numerical residual allowance;
    # actual excess and tighter-solver differences remain in the output.
    checks = dict(laplacian_symmetry=np.max(np.abs(Ls-Ls.swapaxes(1,2)))<1e-12,
                  laplacian_row_sums=np.max(np.abs(Ls.sum(axis=2)))<1e-12,
                  laplacian_psd=np.min(spectra)>=-1e-12,
                  station_envelope=metrics['station_envelope_excess']<1e-8,
                  disagreement_envelope=metrics['disagreement_envelope_excess']<1e-8,
                  lyapunov_inequality=metrics['lyapunov_inequality_excess']<1e-12,
                  disagreement_lyapunov=metrics['disagreement_lyapunov_excess']<1e-12,
                  centroid_envelope=metrics['centroid_envelope_excess']<1e-8,
                  distance_tube=metrics['tube_inequality_excess']<1e-8,
                  centered_offsets=metrics['centered_offset_residual']<1e-11,
                  eta_sum=np.max(np.abs(eta.sum(axis=1)))<1e-12,
                  certified_outcome=not certified or (metrics['observed_safe'] and metrics['observed_connected']))
    metrics['checks'] = {k:bool(v) for k,v in checks.items()}
    if not all(checks.values()):
        raise NumericalValidationError(metrics)
    arrays = dict(t=t,X=X,q=q,qdot=qdot,delta=delta,eta=eta,V=.5*dn**2,W=.5*en**2,
                  delta_norm=dn,eta_norm=en,R_bar=rb,eta_bound=eb,lambda2=lam,
                  lambda2_integral=integral,active_edges=edges,u=u,u_norm=un,
                  correction_norm=np.linalg.norm(correction.reshape(len(t),-1),axis=1),
                  centroid_error=centroid,centroid_error_norm=centroid_norm,
                  V_dot=dv,W_dot=dw,
                  d_x=dx.min(axis=1),d_q=dq.min(axis=1),ell_x_H=ellx,ell_H=ellq,s=s,rho_H=rh)
    return arrays,metrics
