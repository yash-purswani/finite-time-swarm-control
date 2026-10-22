"""Batch station evaluation without changing the prescribed trajectories."""
import numpy as np
from numba import njit
from .common import CompressionStations, compression_arrays, moving_reference_arrays, reference_parameters


@njit(cache=True)
def compression_history(times, P, ai, af, kappa, omega, ref_parameters):
    q = np.empty((len(times), len(P), 2))
    qdot = np.empty_like(q)
    for i in range(len(times)):
        offsets, velocity = compression_arrays(times[i], P, ai, af, kappa, omega)
        r, rdot = moving_reference_arrays(times[i],ref_parameters)
        q[i] = offsets + r
        qdot[i] = velocity + rdot
    return q, qdot


def station_history(cfg, times):
    times = np.asarray(times, dtype=float)
    station = getattr(cfg.P_fn, '__self__', None)
    # The fast path applies only to the shared, planar reference model.
    ref_parameters = reference_parameters(cfg)
    if (isinstance(station, CompressionStations) and cfg.d == 2 and
            ref_parameters is not None and
            getattr(cfg.P_fn,'__func__',None) is CompressionStations.offsets and
            getattr(cfg.Pdot_fn,'__self__',None) is station and
            getattr(cfg.Pdot_fn,'__func__',None) is CompressionStations.velocity):
        return compression_history(times, station.p_initial, station.a_initial,
                                   station.a_final, station.kappa, station.omega, ref_parameters)
    return (np.array([cfg.stations(t) for t in times]),
            np.array([cfg.station_velocity(t) for t in times]))
