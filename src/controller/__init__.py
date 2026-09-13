"""Queue-proportional green-time controller (pure, simulator-agnostic)."""
from src.controller.allocation import ControllerConfig, allocate_green, phase_plan
from src.controller.config import load_controller_config

__all__ = ["ControllerConfig", "allocate_green", "phase_plan", "load_controller_config"]
