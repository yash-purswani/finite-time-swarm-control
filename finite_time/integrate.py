"""Adaptive integration of the physical plant; never projects or resets states."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from .model import rhs, rhs_error, solver_jacobian


@dataclass
class SimResult:
    t: np.ndarray
    X: np.ndarray
    cfg: object
    sol: object
    settings: dict
    coordinates: str = 'physical'

    def at(self,times):
        times = np.atleast_1d(times)
        values = self.sol.sol(times).T.reshape(-1,self.cfg.n,self.cfg.d)
        if self.coordinates=='error':
            values = values+np.array([self.cfg.stations(t) for t in times])
        return values


def simulate(cfg,Z0,t_span,t_eval=None,rtol=1e-9,atol=1e-11,
             method='DOP853',max_step=.01,coordinates='physical'):
    if t_eval is None:
        t_eval = np.linspace(*t_span,2001)
    settings = dict(rtol=rtol,atol=atol,method=method,max_step=max_step)
    field = rhs if coordinates=='physical' else rhs_error
    if coordinates not in ('physical','error'):
        raise ValueError('unknown coordinates')
    initial = np.asarray(Z0).reshape(cfg.n,cfg.d)
    if coordinates=='error':
        initial = initial-cfg.stations(t_span[0])
    iu = np.triu_indices(cfg.n,1)
    he = np.where(np.triu(cfg.H,1)>0) if cfg.H is not None else iu
    def position(t,z):
        X = z.reshape(cfg.n,cfg.d)
        return X if coordinates=='physical' else X+cfg.stations(t)
    def separation(t,z):
        X = position(t,z)
        return np.linalg.norm(X[iu[0]]-X[iu[1]],axis=1).min()-cfg.d_safe
    def connectivity(t,z):
        X = position(t,z)
        return cfg.R_c-np.linalg.norm(X[he[0]]-X[he[1]],axis=1).max()
    events = [separation]+([connectivity] if np.isfinite(cfg.R_c) else [])
    evaluate = lambda t,z:field(t,z,cfg)
    from .common import (CompressionStations,compression_field,reference,reference_velocity,
                         reference_parameters,KappaScaleStations,kappa_scale_field)
    station = getattr(cfg.P_fn,'__self__',None)
    ref_parameters = reference_parameters(cfg)
    if (isinstance(station,CompressionStations) and cfg.weight_mode=='distance' and
            ref_parameters is not None and cfg.d==2 and
            getattr(cfg.P_fn,'__func__',None) is CompressionStations.offsets and
            getattr(cfg.Pdot_fn,'__self__',None) is station and
            getattr(cfg.Pdot_fn,'__func__',None) is CompressionStations.velocity):
        evaluate = lambda t,z:compression_field(t,z,station.p_initial,station.a_initial,
            station.a_final,station.kappa,station.omega,cfg.M,cfg.adjacency,cfg.beta,
            cfg.nu,cfg.eps,cfg.R_c,coordinates=='error',ref_parameters)
    elif (isinstance(station,KappaScaleStations) and cfg.weight_mode=='distance' and
            ref_parameters is not None and cfg.d==2 and
            getattr(cfg.P_fn,'__func__',None) is KappaScaleStations.offsets and
            getattr(cfg.Pdot_fn,'__self__',None) is station and
            getattr(cfg.Pdot_fn,'__func__',None) is KappaScaleStations.velocity):
        evaluate = lambda t,z:kappa_scale_field(t,z,station.p_initial,station.kappa,
            station.rate,cfg.M,cfg.adjacency,cfg.beta,cfg.nu,cfg.eps,cfg.R_c,
            coordinates=='error',ref_parameters)
    elif (hasattr(cfg.P_fn,'morph_data') and cfg.r_fn is reference and
            cfg.rdot_fn is reference_velocity and cfg.d==2 and
            cfg.Pdot_fn is getattr(cfg.P_fn,'morph_velocity_fn',None)):
        from .accelerate import morph_field
        shapes,switches,duration = cfg.P_fn.morph_data
        evaluate = lambda t,z:morph_field(t,z,shapes,switches,duration,cfg.M,
            cfg.adjacency,cfg.beta,cfg.nu,cfg.eps,cfg.R_c,
            cfg.weight_mode=='constant',coordinates=='error')
    options = dict(settings)
    if method in ('Radau','BDF','LSODA'):
        options['jac'] = lambda t,z:solver_jacobian(t,z,cfg,coordinates)
    sol = solve_ivp(evaluate,t_span,initial.ravel(),
                    dense_output=True,events=events,**options)
    if not sol.success:
        raise RuntimeError(sol.message)
    res = SimResult(np.asarray(t_eval),None,cfg,sol,settings,coordinates)
    res.X = res.at(t_eval)
    return res
