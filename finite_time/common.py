"""One station model, template and initial state for Experiments 2, 3, 4."""
from dataclasses import dataclass
import numpy as np
from numba import njit
from .model import SwarmConfig,feedback_arrays


@njit(cache=True)
def reference_arrays(t):
    return np.array([.35*t,.18*np.sin(.7*t)]),np.array([.35,.126*np.cos(.7*t)])


def reference(t):
    return reference_arrays(t)[0]


def reference_velocity(t):
    return reference_arrays(t)[1]


@dataclass(frozen=True)
class MovingReference:
    speed: float
    amplitude: float
    frequency: float

    def position(self,t):
        return moving_reference_arrays(t,(self.speed,self.amplitude,self.frequency))[0]

    def velocity(self,t):
        return moving_reference_arrays(t,(self.speed,self.amplitude,self.frequency))[1]


@njit(cache=True)
def moving_reference_arrays(t,parameters):
    v,A,f = parameters
    return np.array([v*t,A*np.sin(f*t)]),np.array([v,A*f*np.cos(f*t)])


def reference_parameters(cfg):
    """Return parameters only when both analytic reference functions match."""
    if cfg.r_fn is reference and cfg.rdot_fn is reference_velocity:
        return (.35,.18,.7)
    obj = getattr(cfg.r_fn,'__self__',None)
    if (isinstance(obj,MovingReference) and
            getattr(cfg.r_fn,'__func__',None) is MovingReference.position and
            getattr(cfg.rdot_fn,'__self__',None) is obj and
            getattr(cfg.rdot_fn,'__func__',None) is MovingReference.velocity):
        return (obj.speed,obj.amplitude,obj.frequency)
    return None


@njit(cache=True)
def compression_arrays(t,P,ai,af,k,omega):
    c,s = np.cos(omega*t),np.sin(omega*t)
    a = af+(ai-af)*np.exp(-k*t)
    adot = -k*(ai-af)*np.exp(-k*t)
    R = np.array([[c,s],[-s,c]])
    Rdot = omega*np.array([[-s,c],[-c,-s]])
    return a*P@R,adot*P@R+a*P@Rdot


@njit(cache=True)
def compression_field(t,z,P,ai,af,k,omega,M,A,beta,nu,eps,R_c,error_coordinates,
                      ref_parameters=(.35,.18,.7)):
    offsets,velocity = compression_arrays(t,P,ai,af,k,omega)
    r,rdot = moving_reference_arrays(t,ref_parameters)
    q = offsets+r
    qdot = velocity+rdot
    state = z.reshape(P.shape)
    delta = state if error_coordinates else state-q
    X = state+q if error_coordinates else state
    feedback = feedback_arrays(X,delta,M,A,beta,nu,eps,R_c,False)
    return feedback.ravel() if error_coordinates else (qdot+feedback).ravel()


@njit(cache=True)
def kappa_scale_field(t,z,P,kappa,rate,M,A,beta,nu,eps,R_c,error_coordinates,
                      ref_parameters=(.35,.18,.7)):
    """Compiled exact physical/error RHS for the Experiment-3 scale family."""
    ainf=1.65-.25*kappa
    decay=np.exp(-rate*t)
    a=ainf+(1.-ainf)*decay
    adot=-rate*(1.-ainf)*decay
    r,rdot=moving_reference_arrays(t,ref_parameters)
    q=a*P+r
    qdot=adot*P+rdot
    state=z.reshape(P.shape)
    delta=state if error_coordinates else state-q
    X=state+q if error_coordinates else state
    feedback=feedback_arrays(X,delta,M,A,beta,nu,eps,R_c,False)
    return feedback.ravel() if error_coordinates else (qdot+feedback).ravel()


@dataclass
class CompressionStations:
    p_initial: np.ndarray
    a_initial: float
    a_final: float
    kappa: float
    omega: float = .08

    def scale(self,t):
        return self.a_final+(self.a_initial-self.a_final)*np.exp(-self.kappa*np.asarray(t))

    def offsets(self,t):
        return compression_arrays(t,self.p_initial,self.a_initial,self.a_final,self.kappa,self.omega)[0]

    def velocity(self,t):
        return compression_arrays(t,self.p_initial,self.a_initial,self.a_final,self.kappa,self.omega)[1]

    def geometry(self,H):
        D = np.linalg.norm(self.p_initial[:,None]-self.p_initial[None,:],axis=-1)
        iu = np.triu_indices(len(D),1)
        return float(D[iu].min()),float(D[np.triu(H,1)>0].max())


@dataclass
class KappaScaleStations:
    """Experiment-3 scale law: a_inf=1.65-0.25*kappa, a=a_inf+(1-a_inf)e^(-rate*t)."""
    p_initial: np.ndarray
    kappa: float
    rate: float = 0.90
    a_initial: float = 1.0

    @property
    def a_final(self):
        return 1.65-0.25*self.kappa

    def scale(self,t):
        return self.a_final+(self.a_initial-self.a_final)*np.exp(-self.rate*np.asarray(t))

    def scale_velocity(self,t):
        return -self.rate*(self.a_initial-self.a_final)*np.exp(-self.rate*np.asarray(t))

    def offsets(self,t):
        return self.scale(t)*self.p_initial

    def velocity(self,t):
        return self.scale_velocity(t)*self.p_initial

    def geometry(self,H):
        D=np.linalg.norm(self.p_initial[:,None]-self.p_initial[None,:],axis=-1)
        iu=np.triu_indices(len(D),1)
        return float(D[iu].min()),float(D[np.triu(H,1)>0].max())


def shared_scenario(config,m,kappa=None):
    c = config['shared']
    P = np.asarray(c['template'],dtype=float)
    P -= P.mean(axis=0)
    k = kappa if kappa is not None else c.get('kappa', 3.0)
    if 'omega' in c:
        station = CompressionStations(P,c['a_initial'],c['a_final'],k,c['omega'])
    else:
        station = KappaScaleStations(P,k,rate=c.get('scale_rate',0.90),a_initial=c.get('a_initial',1.0))
    H = np.ones((len(P),len(P)))-np.eye(len(P))
    ref = MovingReference(**c['reference']) if 'reference' in c else None
    r_fn,rdot_fn = (ref.position,ref.velocity) if ref is not None else (reference,reference_velocity)
    cfg = SwarmConfig(len(P),2,float(m)*np.eye(2),r_fn,rdot_fn,
                      station.offsets,station.velocity,beta=c['beta'],nu=c['nu'],eps=c['epsilon_w'],
                      H=H,R_c=c['R_c'],d_safe=c['d_safe'])
    delta0 = np.asarray(c['delta_initial'],dtype=float)
    return station,cfg,cfg.stations(0)+delta0
