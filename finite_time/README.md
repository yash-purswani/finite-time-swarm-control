# ECC numerical experiments

The mathematical source is `ACC_paper/finite-time_report.tex`, dated 29 September
2026. The report is not modified. The current controller is local station-error
feedback with station feedforward, componentwise signed power, and unnormalized
Laplacian coupling. All plant integrations use the exact nominal vector field.
No components are latched, smoothed, projected, or reset to manufacture settling.

## Reproduction

**Current revision:** Experiments 2--4 use the approved five-tip star and larger
reference path. Experiment 4 is a small 4 by 4 pilot sweep, replacing the old map.
To reproduce this revision:

```sh
python3 make_shared_formation_gif.py
python3 run_experiments.py --experiment 2
python3 run_experiments.py --experiment 3
python3 run_experiments.py --experiment 4 --grid-size 4 --workers 3
python3 run_experiments.py --experiment validate
```

The GIF shows prescribed stations and a moving star outline. Experiment 2 also
saves `star_maneuver.pdf/png` showing actual positions and station snapshots.
Experiment 3 first retains the analytical two-agent check, then runs three
controlled cases on the shared centered asymmetric star template. It uses
\(a_\infty=1.65-0.25\kappa\),
\(a(t)=a_\infty+(1-a_\infty)e^{-2t}\), and
\(r(t)=(1.25t,1.5\sin(0.4t))\), with analytic station velocities. The cases
\((m,\kappa)=(2.5,3.5),(0.05,6),(0,0.5)\) share the same initial positions,
\(d_{safe}=0.55\), and \(R_c=7\). The 3-by-3 figure compares minimum separation,
longest complete-backbone edge, and geometric margin against the
graph-independent distortion envelope. Raw trajectories include active-edge and
weighted-\(\lambda_2\) histories. Case C loses complete-backbone links while the
range-limited active graph remains connected; the two notions are reported
separately. The three cases do not satisfy the initial distortion-margin
condition, so none is presented as certified. The untouched root-level
`test.py` is not used by the experiment runner.

From the repository root, use Python 3 (the available executable is `python3`):

```sh
python3 -m pip install -r requirements.txt
python3 run_experiments.py --experiment all --workers 3
```

The complete command executes automatic validation and all four experiments,
including the current 4 by 4 pilot parameter map. This is a deterministic parameter
sweep, not a Monte Carlo experiment. Running individual experiments:

```sh
python3 run_experiments.py --experiment validate
python3 run_experiments.py --experiment 1
python3 run_experiments.py --experiment 2
python3 run_experiments.py --experiment 3
python3 run_experiments.py --experiment 4 --grid-size 4 --workers 3
python3 -m unittest finite_time.verify.MathematicalTests -v
python3 -m unittest finite_time.test_workflow -v
```

Configuration is in
`configs/ecc.json`; use `--config PATH` and `--output PATH` for separate revisions.
Completed simulations and validation are now reused automatically when their
configuration, scientific source hashes, library versions and data checksums
match. This applies to all four experiments and the adaptive-node audit, not
only the parameter map. Identical nominal shared runs are also reused across
validation and Experiment 2. Figure generation remains separate and runs on
the selected data. To force fresh computation:

```sh
python3 run_experiments.py --experiment all --rerun
python3 validate_saved_results.py --rerun
```

Changing scientific code or parameters invalidates old caches. The first run
after an implementation change still performs fresh integration. Completed
checks are not reused if their output files are missing or modified. Run
manifests select the active gains and grid cases, so leftovers from earlier
larger grids and standalone animation files do not enter ECC summaries.

BLAS uses one thread per process to prevent oversubscription. The default is at
most three workers because full-horizon dense-output solves retain substantial
adaptive state histories. Choose fewer workers on smaller machines. The nonsmooth controller makes high-accuracy adaptive
integration expensive near zero, even without large coupling. All raw states are
retained; successful grid cases can be resumed using the same command and config.

Plots can be regenerated from saved arrays without integrating again:

```sh
python3 plot_results.py --output results
```

## Structure

- `model.py`: PSD gain validation, graph construction, exact local controller,
  independent closed-loop error RHS, and optional implicit-solver Newton matrix.
- `integrate.py`: configurable `solve_ivp`, dense output, nonterminal separation
  and range events. The main experiments integrate physical positions.
- `formations.py`: reused centered shapes, Hungarian assignment and analytic
  quintic transitions.
- `common.py`: shared similarity station model and configurable analytic
  reference; Experiments 2--4 use the same centered star and initial positions.
- `theory.py`: station envelope/deadline, projection, disagreement, backbone tube,
  continuous-time gain certificate and explicit A/B/C/D/failure classification.
- `metrics.py`: full physical command, feedback energy, bounds, graph histories,
  numerical settling, dense pairwise extremum refinement and automatic checks.
- `experiments.py`: experiment orchestration, parameter sweep and raw persistence.
- `plotting.py` / `style.py`: reusable PDF figures and PNG previews.
- `verify.py`: deterministic algebra tests, physical/error comparison, translation
  and disagreement diagnostics, analytical two-agent checks (in experiment 3),
  Radau comparison and tighter-solver sensitivity runs.
- `accelerate.py` / `sampling.py`: exact compiled morph controller and batch
  station evaluation; generic/custom references keep their original RHS.
- `cache.py`: checksummed run reuse and selection of active experiment cases.
- `test_workflow.py`: regression checks for acceleration, cache integrity,
  custom references and mixed/stale output folders.

The accelerated Numba kernel evaluates the same nominal controller. It is checked
against the independent matrix/Kronecker formulation. Its optimization changes
neither the graph nor feedback. The implicit solver uses a capped derivative only
in its Newton matrix at zero, where the exact derivative is unbounded; its RHS is
never regularized.

## Certificates and observations

All coordinates and time use the report's normalized units. No hardware scale,
velocity cap, disturbances, second-order dynamics, or centroid feedback is added.

Experiments 2 and 4 retain their centered rotating-star model and common station
configuration. Experiment 3 now uses the same centered asymmetric star template,
with the parameterized station motion
`q_i=r(t)+a(t)p_i^0`, `a_inf=1.65-0.25*kappa`,
`a(t)=a_inf+(1-a_inf)*exp(-2*t)`, and
`r(t)=(1.25*t,1.5*sin(0.4*t))`. Analytic station velocities are used. All three
cases use the same seeded zero-mean initial error, beta=0.5, `d_safe=0.55`,
`R_c=7`, and complete eligible graph with memoryless range cutoff; only `m` and
`kappa` vary. The cases `(m,kappa)=(2.5,3.5),(0.05,6),(0,0.5)` respectively
show an observed safe/connected maneuver, a separation violation under a
geometrically infeasible compression, and an expansion that loses complete-
backbone links while leaving the sampled range-limited graph connected. The
initial distortion exceeds `s(0)`, so `gamma_req` is not evaluated and no
certificate is claimed. The report and raw metrics keep backbone persistence
separate from connectivity of the active graph.
Experiment 4 varies only the physical parameters m and kappa: m={0,1,2,3},
kappa={0.18,1.62,3.06,4.5}; all runs keep the common 12-unit horizon. Numerical
integration starts at rtol=1e-10, atol=1e-12, max_step=0.005, matching the accepted
accuracy of Experiment 2. Experiment 3 uses its explicit settings in
`experiment_3.star_maneuver.solver`.
This coarse grid is a pilot: classification colors apply only at simulated
parameter points. Its continuous-time analytical gain boundary is evaluated
separately at 401 speeds and saved as `results/experiment_4/design_boundary.npz`.
The earlier asymmetric map is archived in `results/experiment_4_before_star/`
with its configuration in `configs/ecc_experiment4_legacy.json`.
The fixed H in both scenarios is a
**complete ten-node graph**, explicitly chosen to give a useful spectral floor.
It is a connected spanning backbone, not a sparse tree. Eligible edges are active
exactly while their physical lengths are below the configured R_c; no link is retained
outside range. The actual communication graph is recomputed at every RHS call.

For these positive monotone compressions, minimum planned separation decreases
to `a_final*d_initial`, and maximum planned H length decreases from
`a_initial*ell_initial`. Checking their endpoint and limiting margins proves
planned feasibility for all time, including after the simulation horizon.

The gain supremum is global: each margin branch is `C+B*exp(-kappa*t)` with C>0.
For `f=log(rho0/s)/t`, its derivative has the sign of
`g=-t*s'/s-log(rho0/s)`, and `g'=-t*kappa**2*C*B*exp(-kappa*t)/s**2`.
The decreasing separation branch therefore has at most one maximum of f, found
by a bracketed root or the deadline limit. The increasing connectivity branch has
its supremum at the deadline limit. Taking the larger branch value exhausts all
critical points and endpoints. A strict `1e-8` rate slack guards the floating-point
root calculation. This is an analytically justified global-extremum calculation,
not interval arithmetic or a computer-assisted formal proof. A separate positive
lower bound on `s-rho_H` is saved for each certified case. The first-exit argument
from the report then establishes backbone preservation, rather than assuming it.

In this family, a graph-independent upper tube also certifies connectivity for
all gains: the maximum initial prescribed H length plus `sqrt(2)*||delta0||` is
less than R_c, and both terms decrease during compression. Thus the map principally
studies **separation**. No backbone-loss example is claimed. Geometry remains
feasible for every speed, so category D is implemented and tested but can be
absent from the plotted grid. Gain certification is sufficient and conservative;
B means failure of this selected design condition, not proof that no other
certificate exists. VALIDATION_FAILURE is an error, never an expected category.

Observed safety extrema use dense output on a 0.002 time grid, nonterminal
crossing events, and individual pairwise local-extremum optimization. Numerical
oscillations below 1e-8 are not refined as geometric extrema; the full dense scan
is still retained in the min/max calculation. Sensitivity uses a 0.001 grid,
half the maximum integration step and tenfold tighter solver tolerances. These
observations are not proofs for all continuous time. Experiment 1 interpolations
are qualitative station-tracking examples and have no safety certificate.

Sensitivity refines the **actual accepted** solver settings (including any
automatic accuracy retry), not merely the configured initial settings. For the
slow star, the refined interval is [0,3], including the full settling transient.
This shortening is allowed only when exact post-deadline geometry proves that
no later distance extremum can improve the already observed extrema on [3,12].
The computed post-window separation lower bound and backbone upper bound are
saved with each sensitivity comparison. Every adaptive node over the full
12-unit nominal trajectory is independently audited; the faster diagnostic is
refined over its entire 3-unit horizon.

Experiment 4 separately audits every full-horizon adaptive node. Its nearest
observed B/C cases are refined with tenfold tighter actual tolerances and half
the maximum step through the full settling transient. After the universal
deadline, exact station geometry supplies the remaining distance extrema on
[3,12]. The comparison records this analytic extension separately from the raw
refined trajectory; no states are reset or appended to solver data.

Settling is numerical, at 1e-4, 1e-6 and 1e-8, over the finite evaluation horizon.
Both adaptive nodes and output samples are checked and the final crossing is
refined on the dense interpolant. This does not establish exact zero or exclude
arbitrarily short excursions between those samples. Finite-time exact arrival is
the theorem, not a floating-point measurement. Near-zero solver residuals are
reported in every JSON. The absolute norm budget for envelope checks is 1e-8 and
is audited by tighter runs, rather than hidden by clamping or smoothing. Physical
rtol scales with absolute positions; error-coordinate rtol scales with errors.
Their comparison therefore uses a separately tightened short diagnostic.

## Outputs and interpretation

Each experiment directory contains PDF/PNG figures, NPZ trajectories and JSON
configuration/metrics/certificates. NPZ includes X, q, qdot, delta, eta, V, W,
Laplacian lambda2, integrated spectrum, active edges, commands, tubes and adaptive
solver nodes/states. JSON includes solver settings, library versions, full initial
positions, M, H, adjacency and the SHA-256 of the unchanged report. Null in a
settling field means the threshold was not reached over the evaluation horizon.

The map additionally stores each run, a full CSV, a grid NPZ and classification
counts. `results/summary.csv` consolidates the available experiments, and
`results/TECHNICAL_RESULTS.txt` discusses final numerical outcomes and limitations.

The old root `exp1_centroid.py` through `exp7_baselines.py` and old standalone
validation/figure scripts target an earlier mathematical draft. They are retained
as historical research sources and are not current reproduction commands. The
current modules intentionally reject those obsolete controller options rather
than silently simulating the earlier centroid-based model.

The sensitivity summary flags the nominal m=25 1e-8 settling measurement as
solver-sensitive (about 0.4000 versus 0.3033 with tighter tolerances). Use 1e-6
for comparisons; the tighter raw trajectory is retained. If a slow maneuver
exceeds the fixed 1e-8 near-zero residual budget, the runner retries at tenfold
tighter tolerances and half the step and saves the failed attempt metrics.
The validation allowance is never increased.

The complete runner additionally audits every saved adaptive integration node
and reruns the closest observed B/C classification-boundary cases at tighter
accuracy. This audit can be repeated independently:

```sh
python3 validate_saved_results.py --output results --workers 4
```

`results/validation/adaptive_node_audit.json` records node-bound residuals, and
`map_boundary_summary.json` plus `map_boundary/` retain the boundary comparisons.

Grid resumption checks the configuration, actual m/kappa, hashes of the
scientific modules, theory report, library versions and raw-data checksum.
Changing the controller, integrator, station model, theory or metric
implementation invalidates cached runs. Adaptive settling checks read the
stored solver-node states directly; they retain the same dense crossing
refinement, envelope residual budget, tolerances and safety-event checks.

`ACC_paper/main.tex` still describes the older shared centroid corrector and
`M/n` coupling (model near lines 175–186). These experiments follow the specified
`finite-time_report.tex`. The conference manuscript needs alignment with that
model before using the new figures; neither LaTeX source was edited.
