r"""Prescribed target station trajectories, template geometries, and morphing schedules.

Implements the rotating-compressing star formation stations:
    q_i(t) = r(t) + a(t) R(\omega t) p_i^0
    \dot{q}_i(t) = \dot{r}(t) + \dot{a}(t) R(\omega t) p_i^0 + a(t) \dot{R}(\omega t) p_i^0
with centroid:
    r(t) = [t, 2*sin(0.5*t)]^T, \dot{r}(t) = [1, cos(0.5*t)]^T
and scale contraction:
    a(t) = a_\infty + (a_0 - a_\infty) * exp(-\kappa * t)
"""
from typing import Tuple, List, Callable
import numpy as np


def reference_centroid_path(t: float) -> np.ndarray:
    """Reference centroid trajectory r(t) = [t, 2*sin(0.5*t)]^T."""
    return np.array([float(t), 2.0 * np.sin(0.5 * float(t))], dtype=float)


def reference_centroid_velocity(t: float) -> np.ndarray:
    r"""Reference centroid velocity \dot{r}(t) = [1, cos(0.5*t)]^T."""
    return np.array([1.0, np.cos(0.5 * float(t))], dtype=float)


def build_star_template(n: int = 10) -> np.ndarray:
    """Construct centered 10-agent five-tip star template p_i^0.

    Radii: 2.8 m for even vertices, 1.22 m for odd vertices.
    Angles: phi_i = pi/2 + pi*i/5.
    Baseline metrics: DQ0 ~ 1.4342 m, ELL_H0 ~ 5.3259 m.
    """
    i_idx = np.arange(n)
    phi = np.pi / 2.0 + np.pi * i_idx / 5.0
    b = np.where(i_idx % 2 == 0, 2.8, 1.22)
    p0 = np.column_stack([b * np.cos(phi), b * np.sin(phi)])
    p0 -= np.mean(p0, axis=0, keepdims=True)
    return p0


def template_geometry_metrics(p0: np.ndarray) -> Tuple[float, float, Tuple[int, int], Tuple[int, int]]:
    """Compute baseline minimum distance, diameter, and closest/farthest pairs.

    Returns:
        dq0: Minimum pairwise distance among template vertices
        ell_h0: Maximum pairwise distance (diameter)
        closest_pair: (i, j) index tuple of closest vertex pair
        farthest_pair: (i, j) index tuple of farthest vertex pair
    """
    n = len(p0)
    iu = np.triu_indices(n, k=1)
    diffs = p0[:, None, :] - p0[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    dq0 = float(np.min(dist[iu]))
    ell_h0 = float(np.max(dist[iu]))

    full_dist = dist.copy()
    np.fill_diagonal(full_dist, np.nan)
    closest_pair = tuple(int(v) for v in np.unravel_index(np.nanargmin(full_dist), full_dist.shape))
    farthest_pair = tuple(int(v) for v in np.unravel_index(np.nanargmax(full_dist), full_dist.shape))
    return dq0, ell_h0, closest_pair, farthest_pair


def star_scale(t: np.ndarray, kappa: float, a0: float = 1.10, a_inf: float = 0.75) -> np.ndarray:
    r"""Scale contraction factor a(t) = a_\infty + (a_0 - a_\infty) * exp(-\kappa * t)."""
    return a_inf + (a0 - a_inf) * np.exp(-kappa * np.asarray(t, dtype=float))


def star_scale_dot(t: np.ndarray, kappa: float, a0: float = 1.10, a_inf: float = 0.75) -> np.ndarray:
    r"""Time derivative of scale factor: \dot{a}(t) = -\kappa * (a_0 - a_\infty) * exp(-\kappa * t)."""
    return -kappa * (a0 - a_inf) * np.exp(-kappa * np.asarray(t, dtype=float))


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """2D rotation matrix R(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotation_matrix_derivative_2d(theta: float, omega: float) -> np.ndarray:
    r"""Time derivative \dot{R}(theta) = omega * [-sin(theta), -cos(theta); cos(theta), -sin(theta)]."""
    c, s = np.cos(theta), np.sin(theta)
    return omega * np.array([[-s, -c], [c, -s]], dtype=float)


def star_stations(
    t: float,
    kappa: float,
    p0: np.ndarray,
    a0: float = 1.10,
    a_inf: float = 0.75,
    omega: float = np.pi / 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Evaluate target station positions q_i(t) and velocities \dot{q}_i(t) at a single time t.

    Returns:
        q: (n, 2) station coordinates
        qdot: (n, 2) analytic station velocities
    """
    at = float(star_scale(t, kappa, a0, a_inf))
    adt = float(star_scale_dot(t, kappa, a0, a_inf))
    theta = omega * t
    r = reference_centroid_path(t)
    rdot = reference_centroid_velocity(t)

    R = rotation_matrix_2d(theta)
    Rdot = rotation_matrix_derivative_2d(theta, omega)

    p_rot = p0 @ R.T
    p_dot = p0 @ Rdot.T

    q = r[None, :] + at * p_rot
    qdot = rdot[None, :] + adt * p_rot + at * p_dot
    return q, qdot


def star_stations_batch(
    t_values: np.ndarray,
    kappa: float,
    p0: np.ndarray,
    a0: float = 1.10,
    a_inf: float = 0.75,
    omega: float = np.pi / 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized evaluation of target stations q_i(t) and velocities across an array of times.

    Returns:
        q: (T, n, 2)
        qdot: (T, n, 2)
    """
    t_values = np.asarray(t_values, dtype=float)
    at = star_scale(t_values, kappa, a0, a_inf)
    adt = star_scale_dot(t_values, kappa, a0, a_inf)
    theta = omega * t_values
    c, s = np.cos(theta), np.sin(theta)

    ref = np.column_stack([t_values, 2.0 * np.sin(0.5 * t_values)])
    ref_dot = np.column_stack([np.ones_like(t_values), np.cos(0.5 * t_values)])

    px, py = p0[:, 0], p0[:, 1]
    rot_x = np.outer(c, px) - np.outer(s, py)
    rot_y = np.outer(s, px) + np.outer(c, py)
    p_rot = np.stack([rot_x, rot_y], axis=-1)

    dot_x = omega * (-np.outer(s, px) - np.outer(c, py))
    dot_y = omega * (np.outer(c, px) - np.outer(s, py))
    p_dot = np.stack([dot_x, dot_y], axis=-1)

    q = ref[:, None, :] + at[:, None, None] * p_rot
    qdot = ref_dot[:, None, :] + adt[:, None, None] * p_rot + at[:, None, None] * p_dot
    return q, qdot


# =============================================================================
# Formation Morphing Schedule (Experiment 1)
# =============================================================================
def build_morphing_schedule(
    shapes: List[np.ndarray],
    switch_times: List[float],
    morph_duration: float
) -> Tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    """Build smooth C^2 quintic polynomial interpolation between discrete shapes.

    Returns:
        P_fn: t -> (n, dim) offsets
        Pdot_fn: t -> (n, dim) velocities
    """
    shapes = [np.asarray(s, dtype=float) - np.mean(s, axis=0, keepdims=True) for s in shapes]
    switches = list(switch_times)

    def P_fn(t: float) -> np.ndarray:
        seg = 0
        for s in switches:
            if t >= s:
                seg += 1
        if seg == 0:
            return shapes[0].copy()
        t_start = switches[seg - 1]
        v = (t - t_start) / morph_duration
        if v >= 1.0:
            return shapes[seg].copy()
        if v <= 0.0:
            return shapes[seg - 1].copy()
        blend = 10.0 * (v ** 3) - 15.0 * (v ** 4) + 6.0 * (v ** 5)
        return shapes[seg - 1] + blend * (shapes[seg] - shapes[seg - 1])

    def Pdot_fn(t: float) -> np.ndarray:
        seg = 0
        for s in switches:
            if t >= s:
                seg += 1
        if seg == 0:
            return np.zeros_like(shapes[0])
        t_start = switches[seg - 1]
        v = (t - t_start) / morph_duration
        if v >= 1.0 or v <= 0.0:
            return np.zeros_like(shapes[0])
        d_blend = (30.0 * (v ** 2) - 60.0 * (v ** 3) + 30.0 * (v ** 4)) / morph_duration
        return d_blend * (shapes[seg] - shapes[seg - 1])

    return P_fn, Pdot_fn
