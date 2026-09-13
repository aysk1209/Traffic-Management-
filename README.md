# TMS — vision-inspired adaptive traffic signal control on a synthetic city

See `PROJECT_CONTEXT.md` for the spec and `CLAUDE.md` for working conventions.

## Setup

```bash
pip install -r requirements.txt
```

`eclipse-sumo` bundles the SUMO binaries plus `sumolib`/`traci`; `src/sumo_env.py`
locates them (honouring `SUMO_HOME` if you have a separate SUMO install).

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

## Tests

```bash
pytest tests/
```
