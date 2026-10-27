r"""Deterministic agent initial state generation and perturbation setup.

Generates zero-mean perturbations \sum_i \delta_i(0) = 0:
Random annular ring spawning around target stations q_i(0)
and verifies that the initial swarm configuration is strictly safe and connected.
"""
from typing import Tuple, Optional
import numpy as np
from .stations import template_geometry_metrics


def verify_initial_conditions(
    x0: np.ndarray,
    d_safe: float = 1.0,
    r_c: float = 6.0
) -> Tuple[float, float, bool]:
    """Verify physical separation d_x(0) > d_safe and diameter ell_x(0) < R_c.

    Returns:
        min_dx: Initial minimum pairwise separation
        max_ell: Initial maximum pairwise separation
        is_valid: True if both initial conditions are strictly satisfied
    """
    n = len(x0)
    iu = np.triu_indices(n, k=1)
    diffs = x0[:, None, :] - x0[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    min_dx = float(np.min(dist[iu]))
    max_ell = float(np.max(dist[iu]))
    is_valid = bool(min_dx > d_safe and max_ell < r_c)
    return min_dx, max_ell, is_valid


def generate_random_ring_perturbation(
    q0: np.ndarray,
    r_min: float = 0.3,
    r_max: float = 0.7,
    d_safe: float = 1.0,
    r_c: float = 6.0,
    seed: int = 123,
    max_attempts: int = 2000
) -> Tuple[np.ndarray, float, float]:
    r"""Spawn agents randomly in an annular ring r in [r_min, r_max] around each target station q_i(0).

    Guarantees:
      1. Zero-mean centroid shift: \sum_i \delta_i(0) = 0.
      2. Initial physical separation d_x(0) > d_safe.
      3. Initial network connectivity \ell_x(0) < R_c.

    Returns:
        delta0: (n, 2) zero-mean perturbation
        min_dx: Verified initial minimum separation
        max_ell: Verified initial maximum diameter
    """
    rng = np.random.default_rng(seed)
    n = len(q0)

    for attempt in range(max_attempts):
        r = rng.uniform(r_min, r_max, size=n)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        delta0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

        # Conserve centroid shift: \sum_i \delta_i = 0
        delta0 -= np.mean(delta0, axis=0, keepdims=True)
        x0 = q0 + delta0

        min_dx, max_ell, is_valid = verify_initial_conditions(x0, d_safe=d_safe, r_c=r_c)
        if is_valid:
            return delta0, min_dx, max_ell

    raise RuntimeError(
        f"Could not find valid initial configuration within {max_attempts} attempts "
        f"for r in [{r_min}, {r_max}], d_safe={d_safe}, R_c={r_c}."
    )


def generate_star_initial_perturbation(
    p0: np.ndarray,
    close_pair: Tuple[int, int] = (5, 7),
    far_pair: Tuple[int, int] = (0, 4),
    seed: int = 42,
    close_push: float = 0.28,
    perp_push: float = 0.25,
    far_push: float = 0.05,
    noise_std: float = 0.02
) -> np.ndarray:
    r"""Generate zero-mean initial station error \delta_0 = x_0 - q_0 (Paper Benchmark).

    Args:
        p0: Centered template vertices (n, 2)
        close_pair: Indices of closest pair in template (default: (5, 7))
        far_pair: Indices of farthest pair in template (default: (0, 4))
        seed: Deterministic RNG seed
        close_push: Inward displacement nudge for closest pair
        perp_push: Perpendicular lateral push for closest pair
        far_push: Outward displacement nudge for farthest pair
        noise_std: Standard deviation of background Gaussian perturbation

    Returns:
        delta0: (n, 2) zero-mean initial perturbation matching exact reference
    """
    n, dim = p0.shape
    rng = np.random.default_rng(seed)
    delta0 = rng.normal(0.0, noise_std, size=(n, dim))

    i, j = close_pair
    uij = (p0[j] - p0[i]) / np.linalg.norm(p0[j] - p0[i])
    perp = np.array([-uij[1], uij[0]])
    delta0[i] += +close_push * uij + perp_push * perp
    delta0[j] += -close_push * uij - perp_push * perp

    k, l = far_pair
    uk = p0[k] / np.linalg.norm(p0[k])
    ul = p0[l] / np.linalg.norm(p0[l])
    delta0[k] += +far_push * uk
    delta0[l] += +far_push * ul

    fixed = {i, j, k, l}
    rest = [idx for idx in range(n) if idx not in fixed]
    mean_rest = np.sum(delta0, axis=0) / len(rest)
    for idx in rest:
        delta0[idx] -= mean_rest
    return delta0
