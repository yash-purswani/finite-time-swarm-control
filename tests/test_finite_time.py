"""Unit and regression tests for finite-time swarm control core theory and backbone graph."""
import numpy as np
import pytest
from finite_time.core.stations import (
    build_star_template,
    template_geometry_metrics,
    backbone_geometry_metrics,
    backbone_laplacian,
    star_stations,
    star_scale,
    BACKBONE_EDGES,
    _BACKBONE_ELL_H0,
    _BACKBONE_LAMBDA2,
)
from finite_time.core.spawn import generate_random_ring_perturbation, verify_initial_conditions
from finite_time.core.theory import (
    effective_decay_rate,
    compute_required_rate,
    compute_required_gain,
    geometric_clearance_margin,
    disagreement_vector,
)
from finite_time.core.simulation import simulate_star_maneuver


def test_backbone_15_edges_and_connectivity():
    """Verify 15-edge backbone graph topology and algebraic connectivity lambda2 = 0.931582."""
    assert len(BACKBONE_EDGES) == 15
    L_H = backbone_laplacian(10)
    eigvals = np.linalg.eigvalsh(L_H)
    lam2 = eigvals[1]
    np.testing.assert_allclose(lam2, _BACKBONE_LAMBDA2, atol=1e-5)
    np.testing.assert_allclose(lam2, 0.931582, atol=1e-5)


def test_template_geometry():
    """Verify baseline metrics: DQ0 = 1.4342 m, ell_H0 = 1.9497 m."""
    p0 = build_star_template(10)
    dq0, ell_max, closest_pair, farthest_pair = template_geometry_metrics(p0)
    ell_h0, lam2 = backbone_geometry_metrics(p0)

    np.testing.assert_allclose(dq0, 1.434196, atol=1e-4)
    np.testing.assert_allclose(ell_h0, 1.949666, atol=1e-4)
    np.testing.assert_allclose(lam2, 0.931582, atol=1e-4)


def test_backbone_contraction_invariance():
    """Verify that during contraction from a0=1.70 to a_inf=0.72, all backbone edges remain < R_c=6.0."""
    t_vals = np.linspace(0.0, 12.0, 100)
    a_vals = star_scale(t_vals, kappa=4.0, a0=1.70, a_inf=0.72)
    max_backbone_edge = a_vals * _BACKBONE_ELL_H0
    assert np.all(max_backbone_edge < 6.0)
    assert max_backbone_edge[0] < 3.32


def test_initial_clearance_guarantee():
    """Verify strict prerequisite rho0 < s(0) = min{s_sep(0), s_conn(0)}."""
    p0 = build_star_template(10)
    q0, _ = star_stations(0.0, kappa=4.0, p0=p0, a0=1.70, a_inf=0.72)
    delta0, min_dx, max_ell_H = generate_random_ring_perturbation(
        q0, r_min=0.05, r_max=0.25, d_safe=1.0, r_c=6.0, seed=123,
        close_pair=(1, 3), close_push=0.55
    )
    eta0 = disagreement_vector(delta0)
    rho0 = float(np.sqrt(2.0) * np.linalg.norm(eta0))

    dq0, _, _, _ = template_geometry_metrics(p0)
    ell_h0, _ = backbone_geometry_metrics(p0)
    s0 = float(geometric_clearance_margin(0.0, 4.0, dq0, ell_h0=ell_h0, d_safe=1.0, r_c=6.0, a0=1.70, a_inf=0.72))

    assert rho0 < s0
    margin = s0 - rho0
    assert margin > 0.10


def test_rates_and_gains():
    """Verify rate calculation gamma(m) = 1 + m * w_lower * lambda2_lh and m_req."""
    w_min = 1.0 / (6.0 + 0.05)
    lam2 = 0.931582
    expected_w_lam2 = w_min * lam2

    # m=18
    gam_18 = effective_decay_rate(18.0)
    np.testing.assert_allclose(gam_18, 1.0 + 18.0 * expected_w_lam2, atol=1e-5)

    # m=0
    gam_0 = effective_decay_rate(0.0)
    np.testing.assert_allclose(gam_0, 1.0, atol=1e-5)

    # Inverse gain recovery
    m_rec = compute_required_gain(gam_18)
    np.testing.assert_allclose(m_rec, 18.0, atol=1e-5)


def test_simulation_certification_outcomes():
    """Verify Experiment 3 safe case (m=18) is Certified Safe, and unsafe case (m=0) is Unsafe."""
    p0 = build_star_template(10)
    q0, _ = star_stations(0.0, kappa=4.0, p0=p0, a0=1.70, a_inf=0.72)
    delta0, _, _ = generate_random_ring_perturbation(
        q0, r_min=0.05, r_max=0.25, d_safe=1.0, r_c=6.0, seed=123,
        close_pair=(1, 3), close_push=0.55
    )

    # Safe run
    res_safe = simulate_star_maneuver(
        m=18.0, kappa=4.0, p0=p0, delta0=delta0, a0=1.70, a_inf=0.72, t_span=(0.0, 3.0)
    )
    assert res_safe.metrics['certified'] is True
    assert res_safe.metrics['observed_safe'] is True
    assert res_safe.metrics['min_separation'] > 1.0
    assert res_safe.metrics['gamma'] > res_safe.metrics['gamma_req']

    # Unsafe run
    res_unsafe = simulate_star_maneuver(
        m=0.0, kappa=4.0, p0=p0, delta0=delta0, a0=1.70, a_inf=0.72, t_span=(0.0, 3.0)
    )
    assert res_unsafe.metrics['certified'] is False
    assert res_unsafe.metrics['observed_safe'] is False
    assert res_unsafe.metrics['min_separation'] < 1.0
    assert res_unsafe.metrics['gamma'] < res_unsafe.metrics['gamma_req']


def test_expansion_simulation_outcomes():
    """Verify Experiment 4 rapid expansion outcomes: m=25 is Certified Safe & Connected, m=0 has connectivity breach."""
    from experiments.exp4_expanding_safety import generate_expansion_initial_perturbation

    p0 = build_star_template(10)
    a0, a_inf, kappa = 1.70, 3.02, 8.0
    q0, _ = star_stations(0.0, kappa, p0, a0=a0, a_inf=a_inf)
    delta0 = generate_expansion_initial_perturbation(q0, seed=123)

    # Safe expansion run (m=25.0)
    res_safe = simulate_star_maneuver(
        m=25.0, kappa=kappa, p0=p0, delta0=delta0, a0=a0, a_inf=a_inf, t_span=(0.0, 2.5)
    )
    assert res_safe.metrics['certified'] is True
    assert res_safe.metrics['observed_safe'] is True
    assert res_safe.metrics['observed_connected'] is True
    assert res_safe.metrics['max_backbone_edge'] <= 6.0

    # Unsafe expansion run (m=0.0): link drops mid-transient
    res_unsafe = simulate_star_maneuver(
        m=0.0, kappa=kappa, p0=p0, delta0=delta0, a0=a0, a_inf=a_inf, t_span=(0.0, 2.5)
    )
    assert res_unsafe.metrics['certified'] is False
    assert res_unsafe.metrics['observed_connected'] is False
    assert res_unsafe.metrics['max_backbone_edge'] > 6.0
