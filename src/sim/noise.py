"""Synthetic detector noise applied to ground-truth counts before smoothing.

Ground-truth traci counts are exact, so on their own they cannot show what the
smoothing stage is for. This module perturbs them the way a per-frame vision
counter would fail (PROJECT_CONTEXT.md, stage 1 "known failure modes"):

* ``miss_p``      - each queued vehicle is independently missed with this probability
                    (occlusion / partial visibility). Binomial thinning of the count.
* ``dup_p``       - each vehicle is independently double-counted with this probability
                    (split boxes, ghost detections).
* ``dropout_p``   - with this probability per step the whole approach reads 0
                    (frame drop / detector flicker).
* ``spike_p``     - with this probability per step a spurious burst of ``spike_max``
                    extra vehicles is reported (false positives on background clutter).

Pure and seeded; ``NoiseConfig.none()`` is the identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class NoiseConfig:
    miss_p: float = 0.0
    dup_p: float = 0.0
    dropout_p: float = 0.0
    spike_p: float = 0.0
    spike_max: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("miss_p", "dup_p", "dropout_p", "spike_p"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.spike_max < 0:
            raise ValueError("spike_max must be >= 0")

    @classmethod
    def none(cls) -> "NoiseConfig":
        return cls()

    @property
    def is_identity(self) -> bool:
        return self.miss_p == 0 and self.dup_p == 0 and self.dropout_p == 0 and self.spike_p == 0


def noise_config_from_dict(raw: dict[str, Any]) -> NoiseConfig:
    return NoiseConfig(
        miss_p=float(raw.get("miss_p", 0.0)),
        dup_p=float(raw.get("dup_p", 0.0)),
        dropout_p=float(raw.get("dropout_p", 0.0)),
        spike_p=float(raw.get("spike_p", 0.0)),
        spike_max=int(raw.get("spike_max", 0)),
        seed=int(raw.get("seed", 0)),
    )


def load_noise_config(path: Path | str) -> NoiseConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return noise_config_from_dict(raw)


class CountNoise:
    """Stateful (seeded) perturbation of a ``{approach: count}`` mapping."""

    def __init__(self, cfg: NoiseConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def apply(self, counts: Mapping[str, float]) -> dict[str, float]:
        cfg = self.cfg
        if cfg.is_identity:
            return dict(counts)
        rng = self._rng
        out: dict[str, float] = {}
        for key, raw in counts.items():
            n = int(round(max(0.0, float(raw))))
            if cfg.dropout_p > 0 and rng.random() < cfg.dropout_p:
                out[key] = 0.0
                continue
            seen = n - rng.binomial(n, cfg.miss_p) if cfg.miss_p > 0 else n
            seen += rng.binomial(seen, cfg.dup_p) if cfg.dup_p > 0 else 0
            if cfg.spike_p > 0 and rng.random() < cfg.spike_p:
                seen += int(rng.integers(1, cfg.spike_max + 1)) if cfg.spike_max > 0 else 0
            out[key] = float(seen)
        return out
