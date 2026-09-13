"""Locate SUMO and make its Python tools (sumolib, traci) importable.

Resolution order for SUMO_HOME:
  1. the SUMO_HOME environment variable, if set
  2. the ``eclipse-sumo`` pip package (exposes ``sumo.SUMO_HOME``)

Nothing here hardcodes a path. Call :func:`ensure_sumo_tools` once before importing
``sumolib`` or ``traci``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def find_sumo_home() -> Path:
    """Return the SUMO installation root, or raise if none can be found."""
    env = os.environ.get("SUMO_HOME")
    if env:
        home = Path(env)
        if home.is_dir():
            return home
        raise RuntimeError(f"SUMO_HOME is set to {env!r} but that directory does not exist")
    try:
        import sumo  # type: ignore  # provided by the eclipse-sumo pip package
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "SUMO not found: set SUMO_HOME or `pip install eclipse-sumo`"
        ) from exc
    return Path(sumo.SUMO_HOME)


def ensure_sumo_tools() -> Path:
    """Put ``$SUMO_HOME/tools`` on ``sys.path`` (idempotent) and return SUMO_HOME."""
    home = find_sumo_home()
    tools = str(home / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return home


def sumo_binary(name: str) -> str:
    """Absolute path to a SUMO executable such as ``netconvert`` or ``sumo``."""
    home = find_sumo_home()
    for candidate in (home / "bin" / f"{name}.exe", home / "bin" / name):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"{name} not found under {home / 'bin'}")
