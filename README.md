# TMS — vision-inspired adaptive traffic signal control on a synthetic city

See `PROJECT_CONTEXT.md` for the spec and `CLAUDE.md` for working conventions.

## Setup

```bash
pip install -r requirements.txt
```

`eclipse-sumo` bundles the SUMO binaries (including `sumo-gui`) plus `sumolib`/`traci`;
`src/sumo_env.py` locates them at runtime (honouring `SUMO_HOME` if you have a separate
SUMO install). To make `sumolib`/`traci` importable everywhere — and resolvable by
Pylance/IDEs — add a `.pth` once:

```bash
python -c "import site,sumo,os;open(os.path.join(site.getsitepackages()[-1],'sumo_tools.pth'),'w').write(os.path.join(sumo.SUMO_HOME,'tools'))"
```

## The synthetic city (`src/citygen`)

A small grid inspired by Barcelona's Eixample: uniform two-way streets, every
crossing a 4-way signalized junction, with a wider avenue through the middle in
each direction. Rows/cols are configurable; `configs/network_single.yaml` is the
1x1 case (one intersection), `configs/network_eixample.yaml` the 3x3 grid.
Geometry choices and their calibration are documented in the config comments.

Demand is a diurnal curve (night floor, daytime plateau, rush-hour peaks at
08:45 and 18:00) sampled as a non-homogeneous Poisson process per perimeter
entry. Four profiles ship in `configs/demand_*.yaml`: `light`, `balanced`,
`imbalanced` (same total as balanced, skewed to the west side), `heavy`.

Build a scenario:

```bash
python -m src.citygen --network configs/network_eixample.yaml --demand configs/demand_balanced.yaml
```

Plot the demand curve against the sampled departures (do this before trusting a new profile):

```bash
python scripts/plot_demand.py --network configs/network_eixample.yaml --demand configs/demand_balanced.yaml
```

Watch it in the GUI (fixed 20 s per-approach timing — the baseline the controller is compared against):

```bash
sumo-gui -c sumo_scenarios/eixample_3x3__balanced.sumocfg
```

Generated files in `sumo_scenarios/` are git-ignored; regenerate them with the command above.

## The control loop (`src/sim`)

Every step, for each signalized junction independently: ground-truth halting
vehicles per approach (traci) → exponential smoothing (`src/smoothing`) → at the
start of each cycle, queue-proportional green split with a floor (`src/controller`)
→ phase durations pushed back to SUMO. No coordination between junctions.

```bash
python -m src.sim --scenario sumo_scenarios/eixample_3x3__balanced.sumocfg --control adaptive
python -m src.sim --scenario sumo_scenarios/eixample_3x3__balanced.sumocfg --control fixed      # baseline
python -m src.sim --scenario sumo_scenarios/eixample_3x3__balanced.sumocfg --gui --begin 25200 --end 39600
```

Run the whole validation matrix (4 demand profiles × fixed / adaptive / adaptive-without-smoothing):

```bash
python scripts/run_experiments.py
```

Synthetic detector noise (occlusion misses, duplicates, frame drops, phantom bursts —
`configs/noise_detector.yaml`) can be injected into the raw counts to test the
smoothing stage; `--aggregate instant` makes the controller decide on a single
frame's count instead of the previous cycle's peak:

```bash
python scripts/run_experiments.py --demands imbalanced --modes adaptive adaptive_nosmooth --noise configs/noise_detector.yaml --aggregate instant --begin 25200 --end 39600
```

## Results so far (3x3 grid, fixed 96 s cycle)

06:00–13:00 window, ground-truth counts, default `cycle_max` aggregation:

| demand | fixed: mean wait | adaptive: mean wait | fixed teleports | adaptive teleports |
|---|---|---|---|---|
| light | 114 s | 102 s | 0 | 0 |
| balanced | 173 s | 121 s | 1 | 0 |
| imbalanced | 200 s | 118 s | 0 | 0 |
| heavy | 352 s | 128 s | 148 | 0 |

Smoothing experiment, imbalanced, 07:00–11:00 (fixed baseline: 215 s wait):

| aggregation | counts | EMA | mean wait | green oscillation |
|---|---|---|---|---|
| instant | clean | off | 195 s | 14.7 s/cycle |
| instant | noisy | off | 209 s | 16.4 s/cycle |
| instant | noisy | **on** | **149 s** | **10.9 s/cycle** |
| cycle_max | clean | off | 120 s | 6.1 s/cycle |
| cycle_max | noisy | off | 124 s | 4.5 s/cycle |
| cycle_max | noisy | on | 125 s | 5.7 s/cycle |

Deciding on a single frame's count thrashes and forfeits most of the gain; on a noisy
signal the EMA cuts oscillation by a third and wait by 29 %. Aggregating over the whole
previous cycle is a stronger filter still, and makes the EMA redundant on this signal.

## Tests

```bash
pytest tests/
```
