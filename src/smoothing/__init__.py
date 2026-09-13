"""Temporal smoothing: raw counts -> stable density estimates (exponential moving average)."""
from src.smoothing.config import load_smoothing_config
from src.smoothing.exponential import ExponentialSmoother, SmoothingConfig, alpha_from_half_life, ema_series

__all__ = ["ExponentialSmoother", "SmoothingConfig", "alpha_from_half_life", "ema_series", "load_smoothing_config"]
