r"""Closed-loop kinematic dynamics, interaction weights, and graph Laplacian.

Implements the decentralized control command from Eqs. (1)–(6):
    \dot{x}_i = u_i = \dot{q}_i - \delta_i - sig^beta(\delta_i) - \sum_{j \in N_i(t)} w_{ij} M (\delta_i - \delta_j)
where:
    \delta_i = x_i - q_i
    M = m * I_d
    w_{ij}(x, t) = a_{ij} / (||x_i - x_j|| + eps_w) with a_{ij} = 1 if ||x_i - x_j|| < R_c (i != j)
"""
from typing import Callable, Tuple
import numpy as np


def sig_beta(z: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Componentwise signed power: sign(z) * |z|^beta."""
    return np.sign(z) * (np.abs(z) ** beta)


def pairwise_distance_matrix(x: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix for agents x of shape (n, dim)."""
    diffs = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def proximity_weights(
    dist_matrix: np.ndarray,
    r_c: float = 6.0,
    eps_w: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute adjacency indicator matrix and distance-dependent interaction weights.

    Returns:
        adjacency: Boolean matrix of active edges (i != j and dist < r_c)
        weights: Continuous weight matrix w_{ij} = adjacency / (dist + eps_w)
    """
    n = dist_matrix.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    adjacency = (dist_matrix < r_c) & (~diag_mask)
    weights = adjacency.astype(float) / (dist_matrix + eps_w)
    return adjacency, weights


def laplacian_matrix(weights: np.ndarray) -> np.ndarray:
    """Compute graph Laplacian L_w = diag(sum_j w_ij) - W."""
    degree = np.sum(weights, axis=1)
    return np.diag(degree) - weights


def algebraic_connectivity(L_w: np.ndarray) -> float:
    """Compute algebraic connectivity lambda_2(L_w) (second smallest eigenvalue)."""
    eigenvalues = np.linalg.eigvalsh(L_w)
    return float(max(0.0, eigenvalues[1]))


def evaluate_control_command(
    x: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    m: float,
    r_c: float = 6.0,
    beta: float = 0.5,
    eps_w: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate closed-loop control commands and interaction terms.

    Args:
        x: Agent positions (n, dim)
        q: Target stations (n, dim)
        qdot: Target station velocities (n, dim)
        m: Coupling gain (scalar multiplier for M = m * I_dim)
        r_c: Communication range threshold
        beta: Fractional exponent
        eps_w: Weight denominator regularizer

    Returns:
        u: Control command velocities (n, dim)
        L_w: Weighted Laplacian matrix (n, n)
        delta: Station tracking errors x - q (n, dim)
        lam2: Algebraic connectivity lambda_2(L_w)
    """
    delta = x - q
    dist_mat = pairwise_distance_matrix(x)
    _, w_mat = proximity_weights(dist_mat, r_c=r_c, eps_w=eps_w)
    L_w = laplacian_matrix(w_mat)
    lam2 = algebraic_connectivity(L_w)

    coupling_term = m * (L_w @ delta)
    feedback = -delta - sig_beta(delta, beta) - coupling_term
    u = qdot + feedback
    return u, L_w, delta, lam2


def closed_loop_ode_rhs(
    t: float,
    y: np.ndarray,
    station_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    m: float,
    r_c: float = 6.0,
    beta: float = 0.5,
    eps_w: float = 0.05,
    n: int = 10,
    dim: int = 2
) -> np.ndarray:
    """RHS vector field for ODE solver integrating [x.ravel(), J_u].

    y: Stacked vector of agent coordinates (n * dim,) followed by cumulative control effort J_u.
    Returns:
        dydt: [u.ravel(), sum(||u_i||^2)]
    """
    x = y[:n * dim].reshape((n, dim))
    q, qdot = station_fn(t)
    u, _, _, _ = evaluate_control_command(x, q, qdot, m=m, r_c=r_c, beta=beta, eps_w=eps_w)
    effort_rate = float(np.sum(u ** 2))
    return np.concatenate([u.ravel(), [effort_rate]])
