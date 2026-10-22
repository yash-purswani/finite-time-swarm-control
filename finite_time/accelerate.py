"""Exact compiled physical/error RHS for the shared-reference morph schedule."""
import numpy as np
from numba import njit
from .common import reference_arrays
from .model import feedback_arrays


@njit(cache=True)
def morph_field(t,z,shapes,switches,duration,M,A,beta,nu,eps,R_c,constant,error_coordinates):
    segment = 0
    for switch in switches:
        if t >= switch:
            segment += 1
    offsets = shapes[segment].copy()
    velocity = np.zeros_like(offsets)
    if segment > 0:
        v = (t-switches[segment-1])/duration
        if v < 1.0:
            v = min(max(v,0.0),1.0)
            blend = 10*v**3-15*v**4+6*v**5
            derivative = (30*v**2-60*v**3+30*v**4)/duration
            difference = shapes[segment]-shapes[segment-1]
            offsets = shapes[segment-1]+blend*difference
            velocity = derivative*difference
    r,rdot = reference_arrays(t)
    q = offsets+r
    state = z.reshape(offsets.shape)
    delta = state if error_coordinates else state-q
    X = state+q if error_coordinates else state
    feedback = feedback_arrays(X,delta,M,A,beta,nu,eps,R_c,constant)
    return feedback.ravel() if error_coordinates else (velocity+rdot+feedback).ravel()
