# Project Context — Real-Time Traffic Automation System

## Background
Originally a team project (Sep 2024) built under time pressure; the codebase was lost and no artifacts survived. This is a from-scratch rebuild, solo, using the original scope as the target spec. Nothing from the "previous" implementation should be assumed — this document is the source of truth going forward.

## Problem statement
Fixed-interval traffic signals allocate green time on a timer, regardless of actual queue length per approach. This wastes green time on light approaches while congested approaches keep waiting. The goal is a vision-based, demand-responsive controller: measure real vehicle queue density per lane/approach and allocate signal timing proportionally, so congested lanes clear faster and lightly loaded ones don't hog a full cycle.

## No real camera / real city data
There is no camera feed and no plan to obtain one. Instead of recreating a specific real city, the project builds a **fully synthetic city**: a made-up road network of our own choosing, with a traffic demand generator that mimics real-world diurnal patterns (rush-hour peaks around 9am and 6pm, low overnight, moderate midday). This gives full control over test scenarios (light/heavy/imbalanced) without depending on any external data source. Detection (YOLOv6) is retained only as a proof-of-concept layer to demonstrate the vision component is implementable — it runs on rendered frames from the simulation itself, not on any external footage, and the main control loop does not depend on it succeeding.

## Scope
- **In scope:** a synthetic road network + time-varying demand generator that mimics real diurnal traffic patterns; ground-truth per-approach queue counts read from SUMO via traci; temporal smoothing; queue-proportional green-time allocation; full-loop validation across varying/imbalanced traffic volumes; a standalone YOLOv6 proof-of-concept that detects vehicles in rendered simulation frames and is spot-checked against ground truth.
- **Out of scope:** live hardware deployment at a real intersection, any real camera feed, recreating a specific real-world city or road network, multi-intersection network coordination (green-wave / corridor-level optimization), pedestrian signal logic.

## Pipeline (five stages)

### 0. City & demand generation
- A synthetic road network built directly in SUMO (via netedit/netconvert or hand-written `.net.xml`) — start with a single 4-way signalized intersection; expand to a small grid (e.g. 2x2) only if the single-intersection loop is solid and time allows.
- A demand generator produces SUMO route/flow definitions with a **time-varying vehicle arrival rate** shaped to mimic a real diurnal traffic curve: low overnight, peak ~9am, moderate midday dip, peak ~6pm. Randomize arrivals (e.g. Poisson process per approach) around that curve rather than using a deterministic rate, and support deliberately imbalanced per-approach demand (e.g. one approach much busier than others) as a test scenario.
- This stage's output (`.rou.xml` + `.sumocfg`) is what stages 1–4 run against. Treat the demand curve's shape and parameters as a real design choice worth documenting, not an arbitrary default.

### 1. Detection (proof-of-concept, decoupled from the main loop)
- YOLOv6 run on rendered frames captured from a live SUMO-GUI simulation (screenshot capture) or from a recorded video of a simulation run — never on external/real footage, since none exists.
- Output: vehicle count per lane per frame, using predefined polygon/ROI per lane matching the synthetic network's geometry.
- Spot-check detection output against SUMO's own ground-truth vehicle counts (available via traci) at matching timesteps, as an accuracy sanity check — this is the "proof we could implement it" evidence, not a requirement that detection replace ground truth in the working pipeline.
- Known failure modes worth demonstrating resilience to: occlusion between vehicles, momentary misdetections/flicker — these must **not** propagate directly into the controller (see stage 3), whether the counts come from detection or (for noise-testing purposes) an artificially perturbed ground-truth signal.

### 2. Smoothing
- Raw per-frame per-lane counts are noisy. Before reaching the controller, they must be converted into a stable **density estimate** per lane.
- Purpose: prevent transient occlusion or a single missed/duplicate detection from causing the controller to oscillate between signal states.
- Method: TBD — options include moving average, exponential smoothing, or a Kalman filter per lane. This is a real design decision (accuracy/responsiveness tradeoff) — decide deliberately, don't default silently. Moving average is the simplest starting point; revisit if oscillation persists in testing.

### 3. Controller
- Input: smoothed density estimate per approach/lane.
- Output: green time allocated per approach for the next cycle.
- Core rule: green time per approach is proportional to that approach's measured queue density, so higher-demand approaches get more time within a cycle.
- Must be a pure, simulator-agnostic function/module — no traci or I/O dependencies — so it can be unit-tested with synthetic inputs and reused/swapped independently of the SUMO integration.
- Open design question: whether to enforce a **minimum green-time floor** per approach (so a near-empty approach still isn't starved indefinitely) — decide and document once reached.

### 4. Simulation / validation (primary loop)
- The primary end-to-end loop is: SUMO ground-truth per-approach counts (via traci) → smoothing → controller → apply green times back to SUMO → repeat next cycle. This loop must work fully on ground truth alone, with no dependency on the detection stage.
- Runs against the demand generated in stage 0, across multiple scenarios: light/moderate/heavy overall volume, and a deliberately imbalanced-across-approaches scenario, plus a full simulated day (or compressed version of one) to show the controller responding correctly to the 9am/6pm peaks.
- Validation target: controller should hold up across these varying volumes without manual retuning per scenario.
- The detection module (stage 1) can optionally be substituted in place of ground-truth counts as a stretch goal, once the ground-truth-driven loop is proven — but it is not required for the project's core validation claim.

## Tech stack
- **City & demand:** SUMO (netedit/netconvert for the network, a Python demand generator for time-varying `.rou.xml` flows)
- **Detection (proof-of-concept):** Python, YOLOv6, OpenCV (frame capture from sim, ROI/lane masking, preprocessing)
- **Control logic:** pure Python
- **Simulation loop:** SUMO + traci/sumolib
- **Testing:** pytest

## Success criteria
- Demand generator produces a plausible 24-hour (or compressed) demand curve with visible peaks around 9am and 6pm — plot it and eyeball it before trusting it downstream.
- End-to-end ground-truth→smoothing→controller→SUMO loop runs against at least 3 distinct scenarios (light, heavy, imbalanced-across-approaches) without the controller oscillating or starving any approach.
- Smoothing stage demonstrably reduces oscillation vs. feeding raw/noisy counts directly into the controller (have both a "with smoothing" and "without smoothing" comparison — good for a report/writeup).
- Controller module has unit tests covering: balanced demand, heavily skewed demand, and near-zero demand on one approach.
- Detection proof-of-concept produces per-lane counts on rendered sim frames and reports rough agreement with ground truth on at least one recorded run — this alone satisfies the "we can implement detection" requirement; it does not need to be wired into the live loop.

## Open decisions to make early (flagged, not yet settled)
1. Road network topology: single 4-way intersection vs. small grid (start single; expand only if time allows).
2. Demand curve shape/parameters: peak height and width around 9am/6pm, midday floor, overnight floor, degree of per-approach imbalance to test.
3. Smoothing method: moving average vs. exponential smoothing vs. Kalman filter.
4. Green-time formula: strict proportional split vs. proportional with a minimum floor per approach.
5. Cycle length: fixed total cycle time with proportional split of that fixed budget, vs. variable cycle length driven by total demand.
6. How detection frames are captured: SUMO-GUI screenshot via traci at intervals, vs. a recorded screen-capture video processed afterward.
7. Whether lane ROIs (for both the SUMO network and detection stage) are hardcoded per scenario or made configurable via `configs/`.

## Non-goals / things to explicitly resist scope-creeping into
- Multi-intersection coordination
- Pedestrian/cyclist signal phases
- Real hardware deployment, camera procurement, or recreating a specific real city's actual road network/data
- Any ML beyond the detection proof-of-concept (e.g. no traffic *prediction* models — this is reactive control based on current measured density, not forecasting)
