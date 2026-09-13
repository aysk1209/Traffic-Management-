"""One-time helper: make sumolib/traci importable without sys.path tricks (and
resolvable by IDEs) by writing a .pth file that points at eclipse-sumo's tools dir.

    python scripts/setup_sumo_path.py
"""
from __future__ import annotations

import os
import site
import sys


def main() -> int:
    try:
        import sumo  # eclipse-sumo pip package
    except ImportError:
        print("eclipse-sumo is not installed: pip install -r requirements.txt", file=sys.stderr)
        return 1
    tools = os.path.join(sumo.SUMO_HOME, "tools")
    target = os.path.join(site.getsitepackages()[-1], "sumo_tools.pth")
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(tools + "
")
    print(f"wrote {target} -> {tools}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
