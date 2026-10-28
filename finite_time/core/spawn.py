r"""Deterministic agent initial state generation and perturbation setup.

Agents are spawned randomly within an annular ring of radius r ∈ [r_min, r_max]
around each target station q_i(0).  No zero-mean centroid constraint is imposed;
the disagreement vector η_i(0) = δ_i(0) − mean(δ(0)) is formed internally by
the simulation and theory modules.

Initial physical safety d_x(0) > d_safe and connectivity ℓ_x(0) < R_c are
verified for every generated sample.
"""
from typing import Tuple, Optional, List
import numpy as np
from .stations import template_geometry_metrics, BACKBONE_EDGES


def verify_initial_conditions(
    x0: np.ndarray,
    d_safe: float = 1.0,
    r_c: float = 6.0,
    backbone_edges: Optional[List[Tuple[int, int]]] = None
) -> Tuple[float, float, bool]:
    """Verify physical separation d_x(0) > d_safe and backbone connectivity ell_H(0) < R_c.

    Returns:
        min_dx: Initial minimum pairwise separation
        max_ell_H: Initial maximum backbone edge length
        is_valid: True if both initial conditions are strictly satisfied
    """
    n = len(x0)
    iu = np.triu_indices(n, k=1)
    diffs = x0[:, None, :] - x0[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    min_dx = float(np.min(dist[iu]))

    if backbone_edges is None:
        backbone_edges = BACKBONE_EDGES

    edge_lens = [float(np.linalg.norm(x0[i] - x0[j])) for i, j in backbone_edges]
    max_ell_H = float(max(edge_lens))

    is_valid = bool(min_dx > d_safe and max_ell_H < r_c)
    return min_dx, max_ell_H, is_valid


def generate_random_ring_perturbation(
    q0: np.ndarray,
    r_min: float = 0.05,
    r_max: float = 0.25,
    d_safe: float = 1.0,
    r_c: float = 6.0,
    seed: int = 123,
    max_attempts: int = 2000,
    backbone_edges: Optional[List[Tuple[int, int]]] = None,
    close_pair: Optional[Tuple[int, int]] = (1, 3),
    close_push: float = 0.55
) -> Tuple[np.ndarray, float, float]:
    r"""Spawn agents randomly in an annular ring r ∈ [r_min, r_max] around each station q_i(0).

    For benchmark transient evaluations, closest station pair `close_pair` has an
    inward push component `close_push`, demonstrating that without inter-agent coupling
    (m = 0), rapid compression causes transient collision, whereas sufficient coupling
    (m >= m_req) provably guarantees safety.

    Guarantees (checked on generated sample):
      1. Initial physical separation  d_x(0) > d_safe.
      2. Initial backbone connectivity  ℓ_{x,H}(0) < R_c.

    Returns:
        delta0:    (n, 2) perturbation vectors (not zero-mean enforced)
        min_dx:    Verified minimum pairwise separation at t=0
        max_ell_H: Verified maximum backbone edge length at t=0
    """
    rng = np.random.default_rng(seed)
    n = len(q0)

    for attempt in range(max_attempts):
        r = rng.uniform(r_min, r_max, size=n)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        delta0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

        if close_pair is not None and close_push > 0.0:
            i, j = close_pair
            diff_ij = q0[j] - q0[i]
            uij = diff_ij / np.linalg.norm(diff_ij)
            delta0[i] += +close_push * uij
            delta0[j] += -close_push * uij

        x0 = q0 + delta0
        min_dx, max_ell_H, is_valid = verify_initial_conditions(
            x0, d_safe=d_safe, r_c=r_c, backbone_edges=backbone_edges
        )
        if is_valid:
            return delta0, min_dx, max_ell_H

    raise RuntimeError(
        f"Could not find valid initial configuration within {max_attempts} attempts "
        f"for r in [{r_min}, {r_max}], d_safe={d_safe}, R_c={r_c}."
    )