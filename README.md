# Finite-time swarm formation control

This repository contains numerical experiments for first-order agents tracking
time-varying formations. The controller combines station feedforward, local
finite-time station-error feedback, and distance-weighted interactions over a
range-limited communication graph. 

## Setup

From the repository root, create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Python 3.11 or newer. FFmpeg must also be installed and available on `PATH`
to export MP4 videos. GIF export uses Matplotlib's Pillow writer.

## Run the experiments

The main runner executes the supported experiments (1, 2, 3, 4, and 6):

```bash
python3 run_experiments.py --experiment all
```

Select an individual experiment with `--experiment 1`, `2`, `3`, `4`, or `6`.
Results go to `results/experiment_N/` by default. Use `--output DIR` to choose a
different results root. The runner uses each experiment's Python defaults; use
the dedicated scripts below when changing experiment-specific parameters.

### Experiment 1: moving, reconfiguring formations

Agents track a moving formation that transitions through circle, wedge, star,
and arrow templates with smooth interpolation.

```bash
python3 run_experiments.py --experiment 1
# Or choose the coupling gain:
python3 experiments/exp1_morphing.py --m 2.0
```

The snapshots and summary are saved as
`results/experiment_1/formation_snapshots.pdf`, `.png`, and `morphing.json`.

### Experiment 2: convergence and coupling effort

Compare station tracking error, disagreement, and control effort for different
coupling strengths while tracking a rotating star.

```bash
python3 run_experiments.py --experiment 2
# Or supply gains, maneuver rate, and simulation horizon:
python3 experiments/exp2_gain_evaluation.py \
  --m-values 0 0.5 2 5 10 --kappa 2 --t-end 12
```

The default gains are `0, 0.5, 2, 5, 10`. The three-panel figure is saved as
`results/experiment_2/error_disagreement_control.pdf` and `.png`. Per-gain
trajectories and metrics are saved as NPZ and JSON files; the table is
`results/experiment_2/star_gain_table.tex`.

### Experiment 3: star compression and separation

Compare two coupling gains during rapid rotating-star compression. The default
script uses an inward perturbation of agents 1 and 3, using zero-based indices.

```bash
python3 run_experiments.py --experiment 3
# Reproduce the m=70 versus m=1, kappa=15 comparison:
python3 experiments/exp3_transient_safety.py \
  --m-safe 70 --m-unsafe 1 --kappa 15 --transient-end 2.5
```

Figures `swarm_star_2x3_comparison` and `swarm_star_3x2_comparison`, in PDF and
PNG, the case table, and trajectory NPZ/JSON files are written under
`results/experiment_3/`. The gain labels identify the two cases; inspect the
saved metrics to determine whether a particular choice is certified or violates
a threshold.

### Experiment 4: star expansion and backbone preservation

Compare two coupling gains during rapid expansion. This experiment uses an
outward perturbation of agents 0 and 1 and a fixed 15-edge backbone.

```bash
python3 run_experiments.py --experiment 4
# Reproduce the m=70 versus m=1, kappa=15 comparison:
python3 experiments/exp4_expanding_safety.py \
  --m-safe 70 --m-unsafe 1 --kappa 15 --transient-end 2.5
```

The dedicated script also accepts `--a0` and `--a-inf` to change initial and
final formation scales. The comparison figures, case table, and trajectories
are written under `results/experiment_4/`. A backbone-link loss does not by
itself imply that the entire active communication graph is disconnected.

### Experiment 6: corridor traversal

The agents track a ring-to-chevron-to-ring maneuver through a narrow corridor.

```bash
python3 run_experiments.py --experiment 6
# Change the gain and force a new simulation:
python3 experiments/exp6_corridor.py --m 10 --rerun
```

The maneuver and certificate figures and the summary JSON are written under
`results/experiment_6/`. Experiment 6 otherwise reuses an existing summary;
use `--rerun` when changing its parameters or regenerating figures. The
standalone script currently accepts `--animate`, but it does not generate a
new animation.

For the available options on any script, run it with `--help`.

## Animate compression and expansion

`animate_swarm.py` uses the same station trajectories, initial-perturbation
recipes, and simulator as Experiments 3 and 4. It shows agents, desired station
outlines, active communication links, lost backbone links, and threshold
warnings. Render the four comparisons used above with:

```bash
python3 animate_swarm.py --mode both --m-safe 70 --m-unsafe 1 \
  --kappa 15 --t-end 2.5 --playback-speed 0.25 \
  --fps 20 --dpi 100 --output results/animations
```

`--mode compression` or `--mode expansion` selects one maneuver; `--m 70`
renders a single gain. The output is an MP4 by default; pass an `.mp4` or `.gif`
path to `--output` for a single video, or a directory for multiple videos.
`--playback-speed 0.25` plays one simulation second over four video seconds.
The saved videos for the command above are in `results/animations/`.

Add `--use-saved` to render the matching existing Experiment 3/4 trajectory files
instead of solving again. The script checks the requested parameters, station
motion, and initial state against those saved files. Remove `--use-saved` when
running new cases or when the corresponding trajectory files do not exist.

## Repository layout

```text
animate_swarm.py             Compression and expansion video generation
run_experiments.py           Main experiment runner
experiments/                 Individual experiment scripts
finite_time/core/            Dynamics, stations, simulation, and theory
finite_time/plotting/        Figure generation
finite_time/utils/           Output and reporting helpers
configs/ecc.json             Configuration data used by output metadata
ECC_paper/                   Paper and mathematical report
results/                     Figures, metrics, trajectories, and animations
```

The individual experiment scripts currently define their run parameters in
Python defaults and command-line arguments. `configs/ecc.json` is not a global
configuration switch for every experiment. Existing results may correspond to
earlier parameter choices, so check each JSON record and rerun the relevant
experiment when preparing figures.
