"""Tests for src/smoothing: exponential moving average on synthetic noisy count sequences."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.smoothing import (
    ExponentialSmoother,
    SmoothingConfig,
    alpha_from_half_life,
    ema_series,
    load_smoothing_config,
)
from src.smoothing.config import smoothing_config_from_dict

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- config

def test_half_life_conversion() -> None:
    a = alpha_from_half_life(10.0, 1.0)
    # after one half-life the weight of an old sample has halved
    assert (1 - a) ** 10 == pytest.approx(0.5)
    assert alpha_from_half_life(10.0, 10.0) == pytest.approx(0.5)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_alpha_bounds(alpha: float) -> None:
    with pytest.raises(ValueError):
        SmoothingConfig(alpha)


def test_shipped_config_loads() -> None:
    cfg = load_smoothing_config(ROOT / "configs" / "smoothing.yaml")
    assert 0.0 < cfg.alpha < 0.2


def test_alpha_overrides_half_life_and_unknown_method_rejected() -> None:
    assert smoothing_config_from_dict({"half_life_s": 10, "alpha": 0.3}).alpha == 0.3
    with pytest.raises(ValueError):
        smoothing_config_from_dict({"method": "kalman", "half_life_s": 10})


# ----------------------------------------------------------------------------- behaviour

def test_first_sample_seeds_state_no_warmup_bias() -> None:
    s = ExponentialSmoother(SmoothingConfig(0.1))
    assert s.update({"N": 12.0}) == {"N": 12.0}


def test_constant_input_is_fixed_point() -> None:
    s = ExponentialSmoother(SmoothingConfig(0.1))
    for _ in range(50):
        out = s.update({"N": 7.0})
    assert out["N"] == pytest.approx(7.0)


def test_step_response_follows_closed_form() -> None:
    alpha = 0.2
    series = ema_series([0.0] * 5 + [10.0] * 20, alpha)
    for n in range(1, 21):
        assert series[4 + n] == pytest.approx(10.0 * (1 - (1 - alpha) ** n))


def test_single_spike_attenuated_by_alpha_and_decays() -> None:
    alpha = 0.1
    series = ema_series([0, 0, 0, 20, 0, 0, 0, 0], alpha)
    assert series[3] == pytest.approx(alpha * 20)
    assert all(series[i + 1] < series[i] for i in range(3, 7))
    assert series[-1] == pytest.approx(alpha * 20 * (1 - alpha) ** 4)


def test_noise_variance_reduced_but_mean_preserved() -> None:
    rng = np.random.default_rng(0)
    true_level = 15.0
    raw = true_level + rng.normal(0, 4.0, size=5000)          # jittery counts
    raw += rng.choice([0, 0, 0, 0, 0, 8, -8], size=5000)         # occasional flicker
    sm = np.array(ema_series(raw, alpha=0.067))[500:]
    assert sm.mean() == pytest.approx(raw[500:].mean(), abs=0.3)
    assert sm.std() < 0.3 * raw[500:].std()


def test_smoothing_tracks_queue_ramp_within_a_cycle() -> None:
    """A real queue build-up (0 -> 20 over 40 s) must be substantially reflected
    within a 96 s cycle at the shipped half-life."""
    cfg = load_smoothing_config(ROOT / "configs" / "smoothing.yaml")
    ramp = list(np.linspace(0, 20, 40)) + [20.0] * 56
    sm = ema_series(ramp, cfg.alpha)
    assert sm[39] > 0.5 * 20     # halfway through the cycle
    assert sm[95] > 0.9 * 20     # by the end of the cycle


def test_passthrough_equals_raw() -> None:
    s = ExponentialSmoother(SmoothingConfig.passthrough())
    for x in [3, 0, 9, 9, 1]:
        assert s.update({"N": x})["N"] == x


def test_keys_are_independent_and_missing_keys_hold_value() -> None:
    s = ExponentialSmoother(SmoothingConfig(0.5))
    s.update({"N": 10, "E": 2})
    out = s.update({"N": 0})
    assert out["E"] == 2 and out["N"] == 5
    assert s.estimates == out
    s.reset()
    assert s.estimates == {}


def test_nan_rejected() -> None:
    with pytest.raises(ValueError):
        ExponentialSmoother(SmoothingConfig(0.5)).update({"N": math.nan})
