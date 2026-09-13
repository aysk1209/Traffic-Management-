"""Tests for src/sim/noise.py: synthetic detector noise on counts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.sim.noise import CountNoise, NoiseConfig, load_noise_config, noise_config_from_dict

ROOT = Path(__file__).resolve().parents[1]


def test_identity_when_all_zero() -> None:
    n = CountNoise(NoiseConfig.none())
    assert NoiseConfig.none().is_identity
    assert n.apply({"N": 3, "E": 0.0}) == {"N": 3, "E": 0.0}


@pytest.mark.parametrize("bad", [dict(miss_p=1.5), dict(dup_p=-0.1), dict(spike_max=-1)])
def test_invalid_config(bad: dict) -> None:
    with pytest.raises(ValueError):
        noise_config_from_dict(bad)


def test_shipped_config_loads() -> None:
    cfg = load_noise_config(ROOT / "configs" / "noise_detector.yaml")
    assert not cfg.is_identity and 0 < cfg.miss_p < 1


def test_seeded_and_deterministic() -> None:
    cfg = NoiseConfig(miss_p=0.3, dup_p=0.1, dropout_p=0.1, spike_p=0.1, spike_max=5, seed=11)
    a = [CountNoise(cfg).apply({"N": 20}) for _ in range(1)][0]
    b = CountNoise(cfg).apply({"N": 20})
    assert a == b


def test_miss_rate_thins_counts_on_average() -> None:
    n = CountNoise(NoiseConfig(miss_p=0.25, seed=1))
    vals = np.array([n.apply({"N": 40})["N"] for _ in range(2000)])
    assert vals.mean() == pytest.approx(30, abs=1.0)
    assert vals.min() >= 0


def test_dropout_produces_zeros_at_configured_rate() -> None:
    n = CountNoise(NoiseConfig(dropout_p=0.2, seed=2))
    zeros = sum(n.apply({"N": 10})["N"] == 0 for _ in range(5000))
    assert zeros / 5000 == pytest.approx(0.2, abs=0.03)


def test_spikes_add_bounded_bursts() -> None:
    n = CountNoise(NoiseConfig(spike_p=1.0, spike_max=8, seed=3))
    vals = [n.apply({"N": 0})["N"] for _ in range(500)]
    assert min(vals) >= 1 and max(vals) <= 8


def test_dup_rate_inflates_counts_on_average() -> None:
    n = CountNoise(NoiseConfig(dup_p=0.5, seed=4))
    vals = np.array([n.apply({"N": 40})["N"] for _ in range(2000)])
    assert vals.mean() == pytest.approx(60, abs=1.5)


def test_keys_independent_and_negative_clamped() -> None:
    n = CountNoise(NoiseConfig(miss_p=0.5, seed=5))
    out = n.apply({"N": -3, "E": 0})
    assert out == {"N": 0.0, "E": 0.0}
