"""Finite-time swarm tracking and transient certification package."""
from .core.dynamics import (
    sig_beta,
    pairwise_distance_matrix,
    proximity_weights,
    laplacian_matrix,
    algebraic_connectivity,
    evaluate_control_command,
    closed_loop_ode_rhs,
)
from .core.stations import (
    reference_centroid_path,
    reference_centroid_velocity,
    build_star_template,
    template_geometry_metrics,
    star_scale,
    star_scale_dot,
    rotation_matrix_2d,
    rotation_matrix_derivative_2d,
    star_stations,
    star_stations_batch,
    build_morphing_schedule,
)
from .core.spawn import (
    generate_random_ring_perturbation,
    verify_initial_conditions,
)
from .core.theory import (
    disagreement_vector,
    disagreement_norm,
    universal_tracking_bound,
    universal_settling_deadline,
    geometric_clearance_margin,
    dynamic_distortion_tube,
    effective_decay_rate,
    compute_required_rate,
    compute_required_gain,
    classify_outcome,
)
from .core.simulation import (
    SimulationResult,
    simulate_star_maneuver,
)
from .plotting.style import use_paper_style, save_figure, COL_W, COL2_W, COLORS
from .plotting.figures import (
    plot_morphing_snapshots,
    plot_gain_evaluation,
    plot_transient_2x3_grid,
    plot_parameter_map_and_control,
)
from .utils.io import (
    digest,
    write_json,
    save_simulation_run,
    load_simulation_run,
)
from .utils.reporting import (
    write_star_gain_table,
    write_star_cases_table,
    write_report_metrics,
)

# Backwards compatibility wrappers
def run_shared(config, m, kappa=None, solver=None, spacing=0.002):
    k = kappa if kappa is not None else float(config.get('shared', {}).get('kappa', 2.0))
    solver_settings = solver if solver is not None else dict(method='DOP853', rtol=1e-6, atol=1e-8, max_step=0.02)
    res = simulate_star_maneuver(m=m, kappa=k, **solver_settings)
    cert = {
        'certified': res.metrics['certified'],
        'gamma': res.metrics['gamma'],
        'gamma_req': res.metrics['gamma_req'],
        'm_required': res.metrics['m_required'],
        'T_bar': res.metrics['T_bar'],
    }
    return res, res.to_dict(), res.metrics, cert

def save_run(folder, name, res, arrays, metrics, config, cert=None):
    return save_simulation_run(folder, name, res, config)
