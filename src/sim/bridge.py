"""traci bridge: run a scenario in SUMO with the smoothing -> controller loop applied
to every signalized junction independently.

Each step, for every junction:
  ground-truth halting vehicles per approach (sum over that approach's lanes)
    -> ExponentialSmoother -> (at the start of each cycle) allocate_green
    -> phase durations pushed back to SUMO via traci.trafficlight.setPhaseDuration.

Approach lanes are discovered from the ``per_approach`` program itself (the links
shown 'G' in ``<A>_green``), so nothing here depends on citygen's naming beyond the
phase names. ``control="fixed"`` runs the same loop without pushing durations, which
gives the baseline with identical logging.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.controller import ControllerConfig
from src.sim.junction import CycleRecord, JunctionController, approach_of_phase
from src.sim.noise import CountNoise, NoiseConfig
from src.smoothing import SmoothingConfig
from src.sumo_env import ensure_sumo_tools, sumo_binary

PROGRAM_ID = "per_approach"
CONTROL_MODES = ("fixed", "adaptive")


@dataclass(frozen=True)
class RunConfig:
    sumocfg: Path
    control: str = "adaptive"          # "fixed" (baseline) or "adaptive"
    gui: bool = False
    begin_s: float | None = None
    end_s: float | None = None
    out_dir: Path = Path("sumo_scenarios/output")
    label: str | None = None           # output file stem; default derived from scenario + mode
    seed: int | None = None
    aggregate: str = "cycle_max"       # see src/sim/junction.py AGGREGATES
    noise: NoiseConfig = NoiseConfig.none()   # synthetic detector noise on the raw counts

    @property
    def run_name(self) -> str:
        if self.label:
            return self.label
        return f"{self.sumocfg.stem}__{self.control}"


@dataclass(frozen=True)
class RunStats:
    run_name: str
    control: str
    vehicles_arrived: int
    mean_travel_time_s: float
    mean_waiting_time_s: float
    mean_time_loss_s: float
    teleports: int
    max_halting: int
    cycles_logged: int
    green_oscillation_s: float = 0.0   # mean |green_t - green_{t-1}| per approach across cycles

    def as_dict(self) -> dict:
        return asdict(self)


def discover_approach_lanes(tls_id: str, phases: list) -> dict[str, list[str]]:
    """Map approach -> incoming lane ids, from which links are green in ``<A>_green``."""
    import traci

    links = traci.trafficlight.getControlledLinks(tls_id)  # index -> [(in, out, via), ...]
    lanes: dict[str, list[str]] = {}
    for ph in phases:
        approach = approach_of_phase(ph.name)
        if approach is None:
            continue
        found: list[str] = []
        for idx, ch in enumerate(ph.state):
            if ch in "Gg":
                for in_lane, _out, _via in links[idx]:
                    if in_lane not in found:
                        found.append(in_lane)
        lanes[approach] = found
    return lanes


def build_junctions(controller_cfg: ControllerConfig, smoothing_cfg: SmoothingConfig,
                    aggregate: str = "cycle_max") -> dict[str, JunctionController]:
    import traci

    junctions: dict[str, JunctionController] = {}
    for tls_id in traci.trafficlight.getIDList():
        if traci.trafficlight.getProgram(tls_id) != PROGRAM_ID:
            raise RuntimeError(f"{tls_id}: active program is {traci.trafficlight.getProgram(tls_id)!r}, expected {PROGRAM_ID!r}")
        logic = next(l for l in traci.trafficlight.getAllProgramLogics(tls_id) if l.programID == PROGRAM_ID)
        phase_names = [ph.name for ph in logic.phases]
        junctions[tls_id] = JunctionController(
            tls_id=tls_id,
            phase_names=phase_names,
            approach_lanes=discover_approach_lanes(tls_id, logic.phases),
            controller_cfg=controller_cfg,
            smoothing_cfg=smoothing_cfg,
            aggregate=aggregate,
        )
    return junctions


def _sumo_command(cfg: RunConfig, tripinfo: Path, summary: Path) -> list[str]:
    cmd = [
        sumo_binary("sumo-gui" if cfg.gui else "sumo"),
        "-c", str(cfg.sumocfg),
        "--tripinfo-output", str(tripinfo),
        "--summary", str(summary),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--duration-log.statistics", "false",
    ]
    if cfg.begin_s is not None:
        cmd += ["--begin", str(cfg.begin_s)]
    if cfg.end_s is not None:
        cmd += ["--end", str(cfg.end_s)]
    if cfg.seed is not None:
        cmd += ["--seed", str(cfg.seed)]
    if cfg.gui:
        cmd += ["--start", "true"]
    return cmd


def run(cfg: RunConfig, controller_cfg: ControllerConfig, smoothing_cfg: SmoothingConfig,
        progress_every_s: float | None = 3600.0) -> RunStats:
    """Run one scenario to completion and return summary statistics.

    Writes ``<run_name>_cycles.csv`` (one row per junction per cycle: raw, smoothed,
    green per approach), ``<run_name>_tripinfo.xml``, ``<run_name>_summary.xml`` and
    ``<run_name>_stats.json`` into ``cfg.out_dir``.
    """
    if cfg.control not in CONTROL_MODES:
        raise ValueError(f"control must be one of {CONTROL_MODES}")
    ensure_sumo_tools()
    import traci
    import traci.constants as tc

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    tripinfo = cfg.out_dir / f"{cfg.run_name}_tripinfo.xml"
    summary = cfg.out_dir / f"{cfg.run_name}_summary.xml"
    cycles_path = cfg.out_dir / f"{cfg.run_name}_cycles.csv"

    traci.start(_sumo_command(cfg, tripinfo, summary))
    n_cycles = 0
    noise = CountNoise(cfg.noise)
    osc = _Oscillation()
    try:
        junctions = build_junctions(controller_cfg, smoothing_cfg, cfg.aggregate)
        for j in junctions.values():
            traci.trafficlight.subscribe(j.tls_id, [tc.TL_CURRENT_PHASE])
            for lane in _all_lanes(j):
                traci.lane.subscribe(lane, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER])

        with open(cycles_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s", "tls", "approach", "raw", "smoothed", "green_s"])
            period = progress_every_s or float("inf")
            next_progress = (traci.simulation.getTime() // period + 1) * period if progress_every_s else float("inf")
            end_time = traci.simulation.getEndTime()  # SUMO does not stop a traci client by itself
            while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < end_time:
                traci.simulationStep()
                t = traci.simulation.getTime()
                lane_res = traci.lane.getAllSubscriptionResults()
                tls_res = traci.trafficlight.getAllSubscriptionResults()
                for j in junctions.values():
                    counts = {
                        a: float(sum(lane_res[l][tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for l in lanes))
                        for a, lanes in j.approach_lanes.items()
                    }
                    phase = tls_res[j.tls_id][tc.TL_CURRENT_PHASE]
                    action = j.step(t, phase, noise.apply(counts))
                    if action.new_cycle is not None:
                        n_cycles += 1
                        osc.add(action.new_cycle)
                        _write_cycle(writer, action.new_cycle)
                    if cfg.control == "adaptive" and action.set_phase_duration is not None:
                        traci.trafficlight.setPhaseDuration(j.tls_id, action.set_phase_duration)
                if t >= next_progress:
                    print(f"  t={t/3600:5.2f} h  running={traci.vehicle.getIDCount():5d}  "
                          f"arrived={traci.simulation.getArrivedNumber()}", flush=True)
                    next_progress += period
    except traci.exceptions.FatalTraCIError as exc:
        # SUMO ended the connection (GUI window closed, or quit at end time): finish with what we have
        print(f"  SUMO closed the connection ({exc}); finishing run", flush=True)
    finally:
        try:
            traci.close()
        except Exception:
            pass

    stats = summarise(cfg.run_name, cfg.control, tripinfo, summary, n_cycles, osc.mean())
    (cfg.out_dir / f"{cfg.run_name}_stats.json").write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    return stats


def _all_lanes(j: JunctionController) -> Iterable[str]:
    seen: set[str] = set()
    for lanes in j.approach_lanes.values():
        for lane in lanes:
            if lane not in seen:
                seen.add(lane)
                yield lane


class _Oscillation:
    """Running mean of |green change| between consecutive cycles, per junction/approach."""

    def __init__(self) -> None:
        self._last: dict[str, dict[str, float]] = {}
        self._sum = 0.0
        self._n = 0

    def add(self, rec: CycleRecord) -> None:
        prev = self._last.get(rec.tls_id)
        if prev is not None:
            for a, g in rec.green.items():
                self._sum += abs(g - prev[a])
                self._n += 1
        self._last[rec.tls_id] = dict(rec.green)

    def mean(self) -> float:
        return self._sum / self._n if self._n else 0.0


def _write_cycle(writer, rec: CycleRecord) -> None:
    for a in rec.green:
        writer.writerow([f"{rec.time_s:.0f}", rec.tls_id, a, f"{rec.raw[a]:.0f}",
                         f"{rec.smoothed[a]:.3f}", f"{rec.green[a]:.2f}"])


def summarise(run_name: str, control: str, tripinfo: Path, summary: Path, n_cycles: int,
              green_oscillation_s: float = 0.0) -> RunStats:
    """Aggregate SUMO's tripinfo and summary outputs into :class:`RunStats`."""
    n = 0
    dur = wait = loss = 0.0
    for _event, el in ET.iterparse(tripinfo, events=("end",)):
        if el.tag == "tripinfo":
            n += 1
            dur += float(el.get("duration"))
            wait += float(el.get("waitingTime"))
            loss += float(el.get("timeLoss"))
            el.clear()
    teleports = 0
    max_halting = 0
    for _event, el in ET.iterparse(summary, events=("end",)):
        if el.tag == "step":
            teleports = int(el.get("teleports"))
            max_halting = max(max_halting, int(el.get("halting")))
            el.clear()
    return RunStats(
        run_name=run_name, control=control, vehicles_arrived=n,
        mean_travel_time_s=dur / n if n else 0.0,
        mean_waiting_time_s=wait / n if n else 0.0,
        mean_time_loss_s=loss / n if n else 0.0,
        teleports=teleports, max_halting=max_halting, cycles_logged=n_cycles,
        green_oscillation_s=green_oscillation_s,
    )
