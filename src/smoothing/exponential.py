"""Exponential smoothing of per-approach vehicle counts into stable density estimates.

Design (decided 2026-09-13, PROJECT_CONTEXT.md open decision 3): exponential moving
average (EMA), one state value per approach::

    estimate_t = alpha * count_t + (1 - alpha) * estimate_{t-1}

* One knob. ``alpha`` is expressed as a half-life in seconds (the time for the
  influence of an old sample to halve) so the setting is independent of the sample
  rate: ``alpha = 1 - exp(-ln 2 * sample_interval / half_life)``.
* O(1) memory, no window to fill: the first observation seeds the state directly, so
  there is no warm-up bias at simulation start.
* A one-sample spike of height ``h`` moves the estimate by only ``alpha * h`` and decays
  geometrically — which is exactly the flicker/occlusion resilience the controller
  needs. ``alpha = 1`` disables smoothing (pass-through), used for the
  "without smoothing" comparison runs.

Pure Python, no simulator imports: feed it any ``Mapping[str, float]`` of counts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping


def alpha_from_half_life(half_life_s: float, sample_interval_s: float) -> float:
    """EMA weight giving a sample's influence a half-life of ``half_life_s``."""
    if half_life_s <= 0 or sample_interval_s <= 0:
        raise ValueError("half_life_s and sample_interval_s must be > 0")
    return 1.0 - math.exp(-math.log(2.0) * sample_interval_s / half_life_s)


@dataclass(frozen=True)
class SmoothingConfig:
    alpha: float

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

    @classmethod
    def from_half_life(cls, half_life_s: float, sample_interval_s: float = 1.0) -> "SmoothingConfig":
        return cls(alpha_from_half_life(half_life_s, sample_interval_s))

    @classmethod
    def passthrough(cls) -> "SmoothingConfig":
        """No smoothing at all — the estimate is the raw count."""
        return cls(alpha=1.0)


@dataclass
class ExponentialSmoother:
    """Stateful EMA over a dict of named series (one per approach or lane)."""

    config: SmoothingConfig
    _state: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def update(self, counts: Mapping[str, float]) -> dict[str, float]:
        """Fold one sample per key into the estimates and return a copy of all estimates.

        Keys absent from ``counts`` keep their previous estimate; keys seen for the
        first time are initialised to their raw value.
        """
        a = self.config.alpha
        for key, raw in counts.items():
            x = float(raw)
            if math.isnan(x):
                raise ValueError(f"count for {key!r} is NaN")
            prev = self._state.get(key)
            self._state[key] = x if prev is None else a * x + (1.0 - a) * prev
        return dict(self._state)

    @property
    def estimates(self) -> dict[str, float]:
        return dict(self._state)

    def reset(self) -> None:
        self._state.clear()


def ema_series(values: Iterable[float], alpha: float) -> list[float]:
    """Offline EMA of a single sequence (same recurrence as :class:`ExponentialSmoother`).
    Handy for plots and tests."""
    cfg = SmoothingConfig(alpha)
    out: list[float] = []
    prev: float | None = None
    for v in values:
        prev = float(v) if prev is None else cfg.alpha * float(v) + (1.0 - cfg.alpha) * prev
        out.append(prev)
    return out
