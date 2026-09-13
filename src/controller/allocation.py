"""Queue-proportional green-time allocation.

Pure functions only: density estimates in, green time per approach out. No I/O, no
SUMO/traci imports — this module must stay simulator-agnostic and testable with
plain dicts.

Design (decided 2026-09-13, see PROJECT_CONTEXT.md "open decisions" 4 and 5):

* **Fixed cycle.** The total cycle length is a constant (default 96 s = 4 x 20 s green
  + 4 x (3 s yellow + 1 s all-red)), the same budget as the fixed-timing baseline
  program written by ``src/citygen``. The controller only changes how that budget is
  *split*, so any improvement over the baseline is attributable to reallocation alone.
* **Proportional with a floor.** Every approach is guaranteed ``min_green_s``; the
  remaining green budget is divided in proportion to each approach's measured queue
  density. A near-empty approach is therefore never starved, and with all densities
  zero the budget is split equally. An optional ``max_green_s`` caps any single
  approach, with the excess redistributed proportionally among the uncapped ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

DEFAULT_APPROACHES: tuple[str, ...] = ("N", "E", "S", "W")


@dataclass(frozen=True)
class ControllerConfig:
    """Timing constants for :func:`allocate_green`. All times in seconds."""

    cycle_s: float = 96.0
    yellow_s: float = 3.0
    all_red_s: float = 1.0
    min_green_s: float = 5.0
    max_green_s: float | None = None
    approaches: tuple[str, ...] = DEFAULT_APPROACHES

    @property
    def n(self) -> int:
        return len(self.approaches)

    @property
    def lost_time_s(self) -> float:
        """Cycle time not available as green: one yellow + all-red interval per approach."""
        return self.n * (self.yellow_s + self.all_red_s)

    @property
    def green_budget_s(self) -> float:
        """Total green time to distribute across approaches each cycle."""
        return self.cycle_s - self.lost_time_s

    def validate(self) -> None:
        if self.n == 0:
            raise ValueError("at least one approach is required")
        if len(set(self.approaches)) != self.n:
            raise ValueError("approach names must be unique")
        if min(self.cycle_s, self.yellow_s, self.all_red_s, self.min_green_s) < 0:
            raise ValueError("timings must be non-negative")
        if self.green_budget_s < self.n * self.min_green_s:
            raise ValueError(
                f"cycle {self.cycle_s}s cannot fit {self.n} x min_green {self.min_green_s}s "
                f"plus {self.lost_time_s}s lost time"
            )
        if self.max_green_s is not None:
            if self.max_green_s < self.min_green_s:
                raise ValueError("max_green_s must be >= min_green_s")
            if self.n * self.max_green_s < self.green_budget_s:
                raise ValueError(
                    f"{self.n} x max_green {self.max_green_s}s cannot absorb the "
                    f"{self.green_budget_s}s green budget"
                )


def allocate_green(densities: Mapping[str, float], cfg: ControllerConfig) -> dict[str, float]:
    """Split the cycle's green budget across approaches in proportion to queue density.

    ``densities`` maps approach name -> non-negative queue/density estimate (any unit;
    only ratios matter). Approaches missing from the mapping count as zero. Returns a
    mapping approach -> green seconds, in ``cfg.approaches`` order, that always sums
    to ``cfg.green_budget_s`` and honours ``min_green_s``/``max_green_s``.
    """
    cfg.validate()
    unknown = set(densities) - set(cfg.approaches)
    if unknown:
        raise KeyError(f"densities for unknown approaches: {sorted(unknown)}")

    raw = {a: float(densities.get(a, 0.0)) for a in cfg.approaches}
    if any(math.isnan(v) for v in raw.values()):
        raise ValueError("density estimates must not be NaN")
    d = {a: max(0.0, v) for a, v in raw.items()}  # negative estimates count as empty

    floor = cfg.min_green_s
    spare = cfg.green_budget_s - cfg.n * floor
    total = sum(d.values())
    if total <= 0.0:
        weights = {a: 1.0 / cfg.n for a in cfg.approaches}
    else:
        weights = {a: v / total for a, v in d.items()}

    if cfg.max_green_s is None:
        return {a: floor + spare * weights[a] for a in cfg.approaches}
    return _allocate_with_cap(weights, cfg)


def _allocate_with_cap(weights: Mapping[str, float], cfg: ControllerConfig) -> dict[str, float]:
    """Water-filling: approaches whose proportional share exceeds ``max_green_s`` are
    pinned to the cap, and the rest of the budget is re-split (floor + proportional)
    among the remaining approaches; repeat until nothing overflows."""
    cap = cfg.max_green_s
    assert cap is not None
    capped: set[str] = set()
    while True:
        free = [a for a in cfg.approaches if a not in capped]
        if not free:
            return {a: cap for a in cfg.approaches}
        spare = cfg.green_budget_s - cap * len(capped) - cfg.min_green_s * len(free)
        w_free = sum(weights[a] for a in free)
        green = {a: cap for a in capped}
        for a in free:
            share = weights[a] / w_free if w_free > 0 else 1.0 / len(free)
            green[a] = cfg.min_green_s + spare * share
        over = [a for a in free if green[a] > cap + 1e-9]
        if not over:
            return {a: green[a] for a in cfg.approaches}
        capped.update(over)


def phase_plan(green: Mapping[str, float], cfg: ControllerConfig) -> list[tuple[str, float]]:
    """Ordered (phase name, duration) list for one cycle, matching the ``per_approach``
    program layout produced by ``src/citygen``: ``<A>_green``, ``<A>_yellow``,
    ``<A>_allred`` for each approach in order. All-red phases are omitted when zero."""
    plan: list[tuple[str, float]] = []
    for a in cfg.approaches:
        plan.append((f"{a}_green", float(green[a])))
        plan.append((f"{a}_yellow", cfg.yellow_s))
        if cfg.all_red_s > 0:
            plan.append((f"{a}_allred", cfg.all_red_s))
    return plan
