"""Per-junction control logic, independent of traci.

:class:`JunctionController` owns one smoother and applies the pure controller once per
signal cycle. The traci bridge (``src/sim/bridge.py``) feeds it the current phase index
and raw per-approach counts every step and acts on what it returns. Keeping this
class free of simulator calls makes the cycle logic unit-testable with plain data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.controller import ControllerConfig, allocate_green
from src.smoothing import ExponentialSmoother, SmoothingConfig

# How the per-step smoothed estimates are condensed into the one number per approach
# the controller sees at the start of a cycle:
#   instant    - the smoothed value at that moment. Biased: the approach served last
#                has just discharged and always looks empty at decision time.
#   cycle_max  - peak of the smoothed estimate over the previous cycle, i.e. the queue
#                each approach built up during its red. Fair across phase order.
#   cycle_mean - mean of the smoothed estimate over the previous cycle.
AGGREGATES = ("instant", "cycle_max", "cycle_mean")


def approach_of_phase(phase_name: str) -> str | None:
    """'W_green' -> 'W'; anything that is not a green phase -> None."""
    if phase_name.endswith("_green"):
        return phase_name[: -len("_green")]
    return None


@dataclass(frozen=True)
class CycleRecord:
    """What the controller decided at the start of one cycle."""

    time_s: float
    tls_id: str
    raw: dict[str, float]
    smoothed: dict[str, float]
    green: dict[str, float]


@dataclass(frozen=True)
class StepAction:
    """Instruction for the bridge after one step: optionally set the duration of the
    phase that just started, and optionally a new cycle record to log."""

    set_phase_duration: float | None = None
    new_cycle: CycleRecord | None = None


@dataclass
class JunctionController:
    tls_id: str
    phase_names: list[str]
    approach_lanes: dict[str, list[str]]
    controller_cfg: ControllerConfig
    smoothing_cfg: SmoothingConfig
    aggregate: str = "cycle_max"
    smoother: ExponentialSmoother = field(init=False)
    plan: dict[str, float] = field(default_factory=dict, init=False)
    last_phase: int = field(default=-1, init=False)
    _acc: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _acc_n: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}")
        self.smoother = ExponentialSmoother(self.smoothing_cfg)
        greens = [approach_of_phase(n) for n in self.phase_names]
        seen = [a for a in greens if a is not None]
        expected = list(self.controller_cfg.approaches)
        if seen != expected:
            raise ValueError(
                f"{self.tls_id}: program green phases {seen} do not match controller approaches {expected}"
            )
        self.first_green_index = greens.index(expected[0])

    def step(self, time_s: float, phase_index: int, raw_counts: Mapping[str, float]) -> StepAction:
        """Fold this step's counts into the estimate; on a phase change, hand back the
        duration to apply. A new plan is computed whenever the first green phase begins."""
        smoothed = self.smoother.update(raw_counts)
        self._accumulate(smoothed)
        if phase_index == self.last_phase:
            return StepAction()
        self.last_phase = phase_index

        record = None
        if phase_index == self.first_green_index or not self.plan:
            estimate = self._cycle_estimate(smoothed)
            self.plan = allocate_green(estimate, self.controller_cfg)
            record = CycleRecord(time_s, self.tls_id, dict(raw_counts), estimate, dict(self.plan))
            self._acc.clear()
            self._acc_n = 0

        approach = approach_of_phase(self.phase_names[phase_index])
        duration = self.plan[approach] if approach is not None else None
        return StepAction(set_phase_duration=duration, new_cycle=record)

    def _accumulate(self, smoothed: Mapping[str, float]) -> None:
        self._acc_n += 1
        for a, v in smoothed.items():
            if self.aggregate == "cycle_max":
                self._acc[a] = max(self._acc.get(a, 0.0), v)
            else:
                self._acc[a] = self._acc.get(a, 0.0) + v

    def _cycle_estimate(self, smoothed: Mapping[str, float]) -> dict[str, float]:
        if self.aggregate == "instant" or self._acc_n == 0:
            return dict(smoothed)
        if self.aggregate == "cycle_max":
            return dict(self._acc)
        return {a: v / self._acc_n for a, v in self._acc.items()}
