"""Certificates in ACC_paper/finite-time_report.tex (29 September 2026)."""
import numpy as np
from scipy.optimize import brentq
from .model import lambda2,laplacian


def projection(n):
    return np.eye(n)-np.ones((n,n))/n


def disagreement(delta):
    # Translation-invariant centering avoids manufacturing a tiny nonzero eta
    # from rounding in the mean of an exactly identical common translation.
    shifted = delta-delta[..., :1, :]
    return shifted-shifted.mean(axis=-2,keepdims=True)


def R_bar(t,delta0,beta=.5,t0=0.):
    b = 1-beta
    return np.maximum(0.,(1+np.linalg.norm(delta0)**b)*np.exp(-b*(np.asarray(t)-t0))-1)**(1/b)


def T_bar(delta0,beta=.5,t0=0.):
    return float(t0+np.log1p(np.linalg.norm(delta0)**(1-beta))/(1-beta))


def rho_H(t,delta0,gamma,beta=.5,t0=0.):
    rho0 = np.sqrt(2)*np.linalg.norm(disagreement(delta0))
    return np.minimum(np.sqrt(2)*R_bar(t,delta0,beta,t0),rho0*np.exp(-gamma*(np.asarray(t)-t0)))


def affine_exp_sup(C,B,kappa,rho0,deadline):
    """Global sup log(rho0/(C+B exp(-kt)))/t on (0,T).

    C>0 and s(0)>rho0. f' has sign g=-t s'/s-log(rho0/s).
    g'=-t k^2 C B exp(-kt)/s^2. For B>0, g decreases strictly:
    at most one maximum (bracketed root). For B<=0, g increases
    from a positive limit; f increases to the endpoint supremum.
    This exhausts critical points and requires no sampled optimization.
    """
    if C<=0 or C+B<=rho0 or rho0<=0 or deadline<=0:
        raise ValueError('supremum assumptions not satisfied')
    def f(t):
        s = C+B*np.exp(-kappa*t)
        return np.log(rho0/s)/t
    def g(t):
        s = C+B*np.exp(-kappa*t)
        return t*kappa*B*np.exp(-kappa*t)/s-np.log(rho0/s)
    tm = deadline
    if B>0 and kappa>0 and g(deadline)<0:
        tm = brentq(g,0.,deadline,xtol=1e-13)
    return float(f(tm)),float(tm)


def compression_certificate(station,cfg,delta0,slack=1e-8):
    """All-time certificate from exact similarity geometry and global extrema."""
    d0,ell0 = station.geometry(cfg.H)
    ai,af,k = station.a_initial,station.a_final,station.kappa
    if not (ai>=af>0 and k>=0):
        raise ValueError('certificate supports positive monotone compression')
    rho0 = float(np.sqrt(2)*np.linalg.norm(disagreement(delta0)))
    deadline = T_bar(delta0,cfg.beta)
    eligible_floor=float(np.min(cfg.adjacency[cfg.H>0]))
    w_lower = eligible_floor if cfg.weight_mode=='constant' else eligible_floor/(cfg.R_c**cfg.nu+cfg.eps)
    spectral = w_lower*lambda2(laplacian(cfg.H))
    gamma = 1+cfg.lambda_min_M*spectral
    s0 = min(ai*d0-cfg.d_safe,cfg.R_c-ai*ell0)
    geom_floor = min((af if k>0 else ai)*d0-cfg.d_safe,cfg.R_c-ai*ell0)
    info = dict(rho_0=rho0,T_bar=deadline,s_initial=s0,geometric_margin_floor=geom_floor,
                w_lower=w_lower,lambda2_H=lambda2(laplacian(cfg.H)),gamma=gamma,
                gamma_req=None,m_required=None,certified=False,design_satisfied=False,
                feasible=bool(geom_floor>0),initial_margin=bool(rho0<s0),
                certificate_method='analytic endpoints and unique bracketed critical point',
                design_slack=slack)
    if geom_floor<=0 or rho0>=s0:
        return info
    if rho0==0 or deadline==0:
        info.update(certified=True,design_satisfied=True,zero_disagreement=True,m_required=0.)
        return info
    if k==0:
        f1,t1 = affine_exp_sup(ai*d0-cfg.d_safe,0.,k,rho0,deadline)
        f2,t2 = affine_exp_sup(cfg.R_c-ai*ell0,0.,k,rho0,deadline)
    else:
        f1,t1 = affine_exp_sup(af*d0-cfg.d_safe,(ai-af)*d0,k,rho0,deadline)
        f2,t2 = affine_exp_sup(cfg.R_c-af*ell0,-(ai-af)*ell0,k,rho0,deadline)
    req = max(f1,f2)
    info.update(gamma_req=req,gamma_req_time=t1 if f1>=f2 else t2,
                m_required=max(0.,(req-1)/spectral),
                certified=bool(gamma>req+slack),design_satisfied=bool(gamma>req+slack))
    # Independent positive lower bound on s-rho for all time. For t>0,
    # s-rho_exp>=s*(1-exp(-(gamma-req)t)). Near zero use a Lipschitz
    # bound on s and rho_exp<=rho0. After T_bar use geometry alone.
    h = min(deadline/2,(s0-rho0)/(2*max(k*(ai-af)*max(d0,ell0),1e-300)))
    if info['certified']:
        info['continuous_tube_margin_lower_bound'] = min(
            (s0-rho0)/2,geom_floor*(-np.expm1(-(gamma-req)*h)),geom_floor)
    return info


def classify(cert,observed_safe,observed_connected):
    if cert['certified'] and not (observed_safe and observed_connected):
        return 'VALIDATION_FAILURE'
    if not (observed_safe and observed_connected):
        return 'C'
    if not cert.get('feasible', True):
        return 'D'
    return 'A' if cert['certified'] else 'B'
