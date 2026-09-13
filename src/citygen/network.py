"""Build the synthetic Eixample-style grid as a SUMO network.

Pipeline: :class:`NetworkConfig` -> plain-XML node/edge files -> ``netconvert`` ->
``<name>.net.xml`` -> per-approach traffic-light programs in ``<name>.tls.add.xml``.

Node naming (row 0 = south, col 0 = west):
    n_{row}_{col}     signalized grid intersection
    ext_N_{col} / ext_S_{col} / ext_E_{row} / ext_W_{row}   perimeter dead-ends
Edge id: ``{from_node}-{to_node}``. Every road is two-way.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import quoteattr

from src.citygen.config import NetworkConfig
from src.sumo_env import ensure_sumo_tools, sumo_binary

# Compass order used for approach phases; the controller sees approaches in this order.
APPROACH_ORDER = ("N", "E", "S", "W")
PROGRAM_ID = "per_approach"


@dataclass(frozen=True)
class Node:
    id: str
    x: float
    y: float
    kind: str  # "traffic_light" or "dead_end"


@dataclass(frozen=True)
class Edge:
    id: str
    from_node: str
    to_node: str
    lanes: int
    speed_ms: float


@dataclass
class GridPlan:
    """Pure geometry of the grid, before netconvert."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    @property
    def tls_ids(self) -> list[str]:
        return [n.id for n in self.nodes if n.kind == "traffic_light"]

    @property
    def entry_edges(self) -> list[Edge]:
        """Edges leading from a perimeter dead-end into the grid (where vehicles depart)."""
        return [e for e in self.edges if e.from_node.startswith("ext_")]

    @property
    def exit_edges(self) -> list[Edge]:
        """Edges leading out to a perimeter dead-end (where vehicles arrive)."""
        return [e for e in self.edges if e.to_node.startswith("ext_")]


def side_of_ext_node(node_id: str) -> str:
    """Map a perimeter node id such as ``ext_W_1`` to its side name (``west``)."""
    letter = node_id.split("_")[1]
    return {"N": "north", "E": "east", "S": "south", "W": "west"}[letter]


def plan_grid(cfg: NetworkConfig) -> GridPlan:
    plan = GridPlan()
    b, a = cfg.block_length_m, cfg.approach_length_m

    def grid_id(r: int, c: int) -> str:
        return f"n_{r}_{c}"

    for r in range(cfg.rows):
        for c in range(cfg.cols):
            plan.nodes.append(Node(grid_id(r, c), c * b, r * b, "traffic_light"))
    for c in range(cfg.cols):
        plan.nodes.append(Node(f"ext_S_{c}", c * b, -a, "dead_end"))
        plan.nodes.append(Node(f"ext_N_{c}", c * b, (cfg.rows - 1) * b + a, "dead_end"))
    for r in range(cfg.rows):
        plan.nodes.append(Node(f"ext_W_{r}", -a, r * b, "dead_end"))
        plan.nodes.append(Node(f"ext_E_{r}", (cfg.cols - 1) * b + a, r * b, "dead_end"))

    def two_way(u: str, v: str, lanes: int) -> None:
        plan.edges.append(Edge(f"{u}-{v}", u, v, lanes, cfg.speed_ms))
        plan.edges.append(Edge(f"{v}-{u}", v, u, lanes, cfg.speed_ms))

    # N-S roads (one per column): south stub -> grid nodes -> north stub
    for c in range(cfg.cols):
        lanes = cfg.lanes_for_col(c)
        chain = [f"ext_S_{c}", *[grid_id(r, c) for r in range(cfg.rows)], f"ext_N_{c}"]
        for u, v in zip(chain, chain[1:]):
            two_way(u, v, lanes)
    # E-W roads (one per row): west stub -> grid nodes -> east stub
    for r in range(cfg.rows):
        lanes = cfg.lanes_for_row(r)
        chain = [f"ext_W_{r}", *[grid_id(r, c) for c in range(cfg.cols)], f"ext_E_{r}"]
        for u, v in zip(chain, chain[1:]):
            two_way(u, v, lanes)
    return plan


def nodes_xml(plan: GridPlan) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<nodes>"]
    for n in plan.nodes:
        lines.append(
            f'    <node id={quoteattr(n.id)} x="{n.x:.2f}" y="{n.y:.2f}" type="{n.kind}"/>'
        )
    lines.append("</nodes>")
    return "\n".join(lines) + "\n"


def edges_xml(plan: GridPlan) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<edges>"]
    for e in plan.edges:
        lines.append(
            f'    <edge id={quoteattr(e.id)} from={quoteattr(e.from_node)} '
            f'to={quoteattr(e.to_node)} numLanes="{e.lanes}" speed="{e.speed_ms:.2f}"/>'
        )
    lines.append("</edges>")
    return "\n".join(lines) + "\n"


def run_netconvert(cfg: NetworkConfig, out_dir: Path) -> Path:
    """Write plain XML for ``cfg`` into ``out_dir`` and compile it to ``<name>.net.xml``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_grid(cfg)
    nod = out_dir / f"{cfg.name}.nod.xml"
    edg = out_dir / f"{cfg.name}.edg.xml"
    net = out_dir / f"{cfg.name}.net.xml"
    nod.write_text(nodes_xml(plan), encoding="utf-8")
    edg.write_text(edges_xml(plan), encoding="utf-8")
    cmd = [
        sumo_binary("netconvert"),
        "--node-files", str(nod),
        "--edge-files", str(edg),
        "--output-file", str(net),
        "--tls.default-type", "static",
        "--tls.yellow.time", str(int(cfg.yellow_s)),
        "--no-turnarounds", "true" if cfg.no_turnarounds else "false",
        "--junctions.corner-detail", "5",
        "--no-warnings", "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"netconvert failed:\n{result.stdout}\n{result.stderr}")
    return net


# --------------------------------------------------------------------------- TLS programs


@dataclass(frozen=True)
class ApproachLinks:
    approach: str  # one of APPROACH_ORDER
    in_edge: str
    link_indices: tuple[int, ...]


def approach_of_incoming_edge(edge) -> str:  # edge: sumolib.net.edge.Edge
    """Which compass side traffic on ``edge`` arrives *from* at its destination junction."""
    fx, fy = edge.getFromNode().getCoord()
    tx, ty = edge.getToNode().getCoord()
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy):
        return "W" if dx > 0 else "E"  # heading east means it came from the west
    return "S" if dy > 0 else "N"


def tls_approaches(net, tls_id: str) -> list[ApproachLinks]:
    """Group a junction's controlled links by the approach they come from, in compass order."""
    tls = net.getTLS(tls_id)
    by_edge: dict[str, list[int]] = {}
    approach_of: dict[str, str] = {}
    for in_lane, _out_lane, link_idx in tls.getConnections():
        edge = in_lane.getEdge()
        by_edge.setdefault(edge.getID(), []).append(link_idx)
        approach_of[edge.getID()] = approach_of_incoming_edge(edge)
    groups = [
        ApproachLinks(approach_of[eid], eid, tuple(sorted(set(idx))))
        for eid, idx in by_edge.items()
    ]
    return sorted(groups, key=lambda g: APPROACH_ORDER.index(g.approach))


def per_approach_phases(
    groups: Iterable[ApproachLinks], n_links: int, cfg: NetworkConfig
) -> list[tuple[float, str, str]]:
    """(duration, state, name) tuples: per approach a green, a yellow and an optional all-red."""
    phases: list[tuple[float, str, str]] = []
    for g in groups:
        green = ["r"] * n_links
        yellow = ["r"] * n_links
        for i in g.link_indices:
            green[i], yellow[i] = "G", "y"
        phases.append((cfg.green_s, "".join(green), f"{g.approach}_green"))
        phases.append((cfg.yellow_s, "".join(yellow), f"{g.approach}_yellow"))
        if cfg.all_red_s > 0:
            phases.append((cfg.all_red_s, "r" * n_links, f"{g.approach}_allred"))
    return phases


def write_tls_programs(
    net_path: Path, cfg: NetworkConfig, out_path: Path
) -> dict[str, list[ApproachLinks]]:
    """Write a ``tlLogic`` per junction with one green phase per approach; return the link map."""
    ensure_sumo_tools()
    import sumolib  # noqa: E402  (needs SUMO tools on sys.path)

    net = sumolib.net.readNet(str(net_path), withPrograms=True)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<additional>"]
    link_map: dict[str, list[ApproachLinks]] = {}
    for tls in net.getTrafficLights():
        tls_id = tls.getID()
        groups = tls_approaches(net, tls_id)
        n_links = max(idx for _, _, idx in tls.getConnections()) + 1
        link_map[tls_id] = groups
        lines.append(
            f'    <tlLogic id={quoteattr(tls_id)} type="static" programID="{PROGRAM_ID}" offset="0">'
        )
        for dur, state, name in per_approach_phases(groups, n_links, cfg):
            lines.append(f'        <phase duration="{dur:g}" state="{state}" name="{name}"/>')
        lines.append("    </tlLogic>")
    lines.append("</additional>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return link_map


def build_network(cfg: NetworkConfig, out_dir: Path) -> tuple[Path, Path, GridPlan]:
    """Full network build: returns (net.xml path, tls.add.xml path, grid plan)."""
    net = run_netconvert(cfg, out_dir)
    tls = out_dir / f"{cfg.name}.tls.add.xml"
    write_tls_programs(net, cfg, tls)
    return net, tls, plan_grid(cfg)
