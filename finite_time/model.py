"""Report's first-order local station-error controller (row-major agent stack)."""
from dataclasses import dataclass, replace
from typing import Callable
import numpy as np
from numba import njit


def sig_beta(z, beta):
    return np.sign(z)*np.abs(z)**beta


def validate_M(M, d):
    M = np.asarray(M, dtype=float)
    if M.shape != (d, d) or not np.all(np.isfinite(M)):
        raise ValueError('M must be a finite d by d matrix')
    if not np.allclose(M, M.T, atol=1e-13, rtol=0):
        raise ValueError('M must be symmetric')
    if np.linalg.eigvalsh(M)[0] < -1e-13:
        raise ValueError('M must be positive semidefinite')
    return M


def laplacian(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.allclose(A,A.T,atol=1e-13,rtol=0):
        raise ValueError('adjacency must be square and symmetric')
    if not np.all(np.isfinite(A)) or np.any(A < 0) or np.any(np.diag(A) != 0):
        raise ValueError('adjacency must be nonnegative with zero diagonal')
    return np.diag(A.sum(axis=1))-A


def lambda2(L):
    # Second smallest, not second strictly positive eigenvalue.
    return float(max(0., np.linalg.eigvalsh(L)[1]))


@dataclass
class SwarmConfig:
    n: int
    d: int
    M: np.ndarray
    r_fn: Callable
    rdot_fn: Callable
    P_fn: Callable
    Pdot_fn: Callable
    beta: float = .5
    nu: float = 1.
    eps: float = .2
    adjacency: np.ndarray = None
    weight_mode: str = 'distance'
    R_c: float = np.inf
    H: np.ndarray = None
    d_safe: float = 1.

    def __post_init__(self):
        if self.n<2 or self.d<1 or self.R_c<=0 or np.isnan(self.R_c) or self.d_safe<=0:
            raise ValueError('require n>=2, d>=1, positive R_c and d_safe')
        self.M = validate_M(self.M, self.d)
        if not 0 < self.beta < 1 or not np.isfinite(self.nu) or not np.isfinite(self.eps) or self.nu < 1 or self.eps <= 0:
            raise ValueError('require 0<beta<1, nu>=1, epsilon_w>0')
        if self.adjacency is None:
            self.adjacency = np.ones((self.n,self.n))-np.eye(self.n)
        self.adjacency = np.asarray(self.adjacency, dtype=float)
        laplacian(self.adjacency)
        if self.adjacency.shape != (self.n,self.n):
            raise ValueError('adjacency size differs from n')
        if self.weight_mode not in ('constant','distance'):
            raise ValueError('unknown weight mode')
        if self.H is not None:
            self.H = np.asarray(self.H,dtype=float)
            if self.H.shape != (self.n,self.n) or lambda2(laplacian(self.H)) <= 0:
                raise ValueError('H must be a connected spanning graph')
            if np.any((self.H!=0)&(self.H!=1)):
                raise ValueError('H must be an unweighted backbone adjacency')
            if np.any((self.H>0)&(self.adjacency==0)):
                raise ValueError('backbone links must be eligible')

    def with_(self, **kw):
        return replace(self, **kw)

    @property
    def lambda_min_M(self):
        return float(max(0.,np.linalg.eigvalsh(self.M)[0]))

    def stations(self,t):
        return np.asarray(self.r_fn(t))+np.asarray(self.P_fn(t))

    def station_velocity(self,t):
        return np.asarray(self.rdot_fn(t))+np.asarray(self.Pdot_fn(t))


def graph(X,cfg):
    distances = np.linalg.norm(X[:,None]-X[None,:],axis=-1)
    # Memoryless undirected range rule: eligible H links always available below
    # R_c, with no forced retention above R_c and no hysteresis dropout below it.
    active = cfg.adjacency*(distances<cfg.R_c)
    np.fill_diagonal(active,0.)
    weights = active if cfg.weight_mode=='constant' else active/(distances**cfg.nu+cfg.eps)
    return laplacian(weights), active


def interaction_term(X,delta,cfg):
    L_w,_ = graph(X,cfg)
    return L_w@delta@cfg.M.T


@njit(cache=True)
def feedback_arrays(X,delta,M,A,beta,nu,eps,R_c,constant):
    """Compiled evaluation of exactly the same local controller, no smoothing."""
    n,d = X.shape
    out = -delta-np.sign(delta)*np.abs(delta)**beta
    if np.max(np.abs(M))==0:
        return out
    for i in range(n):
        for j in range(i+1,n):
            squared = 0.
            for coordinate in range(d):
                squared += (X[i,coordinate]-X[j,coordinate])**2
            distance = np.sqrt(squared)
            if A[i,j]==0 or distance>=R_c:
                continue
            weight = A[i,j] if constant else A[i,j]/(distance**nu+eps)
            for coordinate in range(d):
                interaction = 0.
                for other in range(d):
                    interaction += M[coordinate,other]*(delta[i,other]-delta[j,other])
                interaction *= weight
                out[i,coordinate]-=interaction
                out[j,coordinate]+=interaction
    return out


def rhs(t,z,cfg):
    X = np.asarray(z).reshape(cfg.n,cfg.d)
    delta = X-cfg.stations(t)
    return (cfg.station_velocity(t)+feedback_arrays(X,delta,cfg.M,cfg.adjacency,
        cfg.beta,cfg.nu,cfg.eps,cfg.R_c,cfg.weight_mode=='constant')).ravel()


def rhs_error(t,z,cfg):
    delta = np.asarray(z).reshape(cfg.n,cfg.d)
    X = delta+cfg.stations(t)
    return (-delta-sig_beta(delta,cfg.beta)-interaction_term(X,delta,cfg)).ravel()


def solver_jacobian(t,z,cfg,coordinates='physical'):
    """Newton matrix for implicit solvers; does not change the vector field.

    At zero the signed-power derivative is unbounded. Cap ONLY its Jacobian
    magnitude using a 1e-10 evaluation floor. The RHS is still exact sig^beta.
    Include derivatives of distance weights within the current graph mode.
    """
    state = z.reshape(cfg.n,cfg.d)
    delta = state-cfg.stations(t) if coordinates=='physical' else state
    X = state if coordinates=='physical' else state+cfg.stations(t)
    L,active = graph(X,cfg)
    diag = 1+cfg.beta*np.maximum(np.abs(delta),1e-10)**(cfg.beta-1)
    J = -np.diag(diag.ravel())-np.kron(L,cfg.M)
    if cfg.weight_mode=='distance':
        for i in range(cfg.n):
            for j in range(i+1,cfg.n):
                if active[i,j]==0:
                    continue
                v = X[i]-X[j]
                distance = np.linalg.norm(v)
                if distance==0:
                    continue
                grad = -active[i,j]*cfg.nu*distance**(cfg.nu-2)*v/(distance**cfg.nu+cfg.eps)**2
                block = np.outer(cfg.M@(delta[i]-delta[j]),grad)
                ii,jj = slice(i*cfg.d,(i+1)*cfg.d),slice(j*cfg.d,(j+1)*cfg.d)
                J[ii,ii]-=block
                J[ii,jj]+=block
                J[jj,ii]+=block
                J[jj,jj]-=block
    return J
