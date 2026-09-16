#!/usr/bin/env python3
"""
gen_cases.py - deterministic random test-case generator for the
Amazon Robotics Hackathon 2026 drive-unit routing game.

    python3 dev/gen_cases.py --out dev/cases --seed 1 --per-level 40 --stress 12 [--validate]

Writes  <out>/level1/gen_l1_NNN.json, <out>/level2/gen_l2_NNN.json,
        <out>/level3/gen_l3_NNN.json and <out>/stress/gen_s_NNN.json.

Standard library only.  Every case is drawn from its own random.Random
instance seeded from (--seed, level, index), so a given seed always
reproduces the same files.  --validate additionally loads every file
through ar_hackathon's loader and plays it with the example basic driver
through the real GameEngine.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import sys
import time
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_LETTERS = "ABCDEFGH"

LEVELS = ("level1", "level2", "level3", "stress")
LEVEL_NO = {"level1": 1, "level2": 2, "level3": 3, "stress": 4}
FILE_PREFIX = {"level1": "gen_l1", "level2": "gen_l2", "level3": "gen_l3", "stress": "gen_s"}
# Upper bound on max_time_steps so a bad driver never runs forever.
MTS_CAP = {"level1": 400, "level2": 400, "level3": 400, "stress": 700}


# --------------------------------------------------------------------------
# Case builder
# --------------------------------------------------------------------------
class CaseBuilder:
    """Accumulates nodes/edges/units/pods and offers graph queries."""

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.units = []
        self.pods = []
        self._pairs = set()
        self._counts = Counter()

    # ---- construction -----------------------------------------------------
    def add_node(self, ntype="travel", name=None, capacity=None):
        nid = len(self.nodes)
        if name is None:
            self._counts[ntype] += 1
            k = self._counts[ntype]
            if ntype == "storage":
                name = f"Storage-{STORAGE_LETTERS[(k - 1) % len(STORAGE_LETTERS)]}"
            elif ntype == "station":
                name = f"Station-{k}"
            else:
                name = f"Aisle-{k}"
        node = {"id": nid, "name": name, "type": ntype}
        if capacity is not None:
            node["capacity"] = int(capacity)
        self.nodes.append(node)
        return nid

    def add_edge(self, a, b, weight, capacity=None, bidirectional=True):
        if a == b:
            return False
        key = frozenset((a, b))
        if key in self._pairs:
            return False
        self._pairs.add(key)
        edge = {"from_node": a, "to_node": b, "weight": int(weight)}
        if capacity is not None:
            edge["capacity"] = int(capacity)
        if not bidirectional:
            edge["bidirectional"] = False
        self.edges.append(edge)
        return True

    def has_edge(self, a, b):
        return frozenset((a, b)) in self._pairs

    def add_unit(self, start, capacity=1):
        unit = {"id": len(self.units), "start_node": start}
        if capacity != 1:
            unit["capacity"] = int(capacity)
        self.units.append(unit)

    def add_pod(self, src, dst, entry_time):
        self.pods.append({"source_node": src, "destination_station": dst,
                          "entry_time": int(entry_time)})

    # ---- queries ------------------------------------------------------------
    def ids_of(self, ntype):
        return [n["id"] for n in self.nodes if n["type"] == ntype]

    @property
    def storages(self):
        return self.ids_of("storage")

    @property
    def stations(self):
        return self.ids_of("station")

    @property
    def travels(self):
        return self.ids_of("travel")

    def node(self, nid):
        return self.nodes[nid]

    def node_capacity(self, nid):
        return self.nodes[nid].get("capacity")

    def adjacency(self):
        adj = {n["id"]: [] for n in self.nodes}
        for e in self.edges:
            adj[e["from_node"]].append((e["to_node"], e["weight"]))
            if e.get("bidirectional", True):
                adj[e["to_node"]].append((e["from_node"], e["weight"]))
        return adj

    def undirected_neighbors(self, nid):
        out = set()
        for e in self.edges:
            if e["from_node"] == nid:
                out.add(e["to_node"])
            elif e["to_node"] == nid:
                out.add(e["from_node"])
        return out

    def dijkstra(self, src, adj=None):
        adj = adj or self.adjacency()
        return dijkstra(adj, src)

    def important_nodes(self):
        imp = set(self.storages) | set(self.stations)
        imp |= {u["start_node"] for u in self.units}
        return imp

    def important_connected(self):
        adj = self.adjacency()
        imp = self.important_nodes()
        return all(imp <= reachable(adj, u) for u in imp)

    # ---- output ---------------------------------------------------------------
    def finalize_pods(self):
        indexed = list(enumerate(self.pods))
        indexed.sort(key=lambda t: (t[1]["entry_time"], t[0]))
        out = []
        for k, (_, p) in enumerate(indexed, start=1):
            out.append({"id": f"P{k}", "source_node": p["source_node"],
                        "destination_station": p["destination_station"],
                        "entry_time": p["entry_time"]})
        self.pods = out

    def to_data(self, max_time_steps, description):
        return {
            "metadata": {"max_time_steps": int(max_time_steps), "description": description},
            "nodes": self.nodes,
            "edges": self.edges,
            "drive_units": self.units,
            "pods": self.pods,
        }


def dijkstra(adj, src):
    dist = {src: 0}
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def reachable(adj, src):
    seen = {src}
    stack = [src]
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def split_int(rng, total, parts, lo, hi):
    """Random composition of `total` into `parts` integers in [lo, hi]."""
    assert parts * lo <= total <= parts * hi
    vals = [lo] * parts
    remaining = total - parts * lo
    while remaining > 0:
        idx = [i for i in range(parts) if vals[i] < hi]
        vals[rng.choice(idx)] += 1
        remaining -= 1
    return vals


def staggered_times(rng, n, first_batch, step_choices, cap):
    """`first_batch` pods at t=0, the rest at cumulative random increments (<= cap)."""
    times, t = [], 0
    for i in range(n):
        if i >= first_batch:
            t = min(cap, t + rng.choice(step_choices))
        times.append(t)
    return times


def random_connected(c, rng, ids, n_extra, wmin, wmax, cap_fn=lambda: None, locality=3):
    """Random spanning tree (attached locally, so paths are long-ish) plus extra edges."""
    order = list(ids)
    rng.shuffle(order)
    for i in range(1, len(order)):
        j = rng.randrange(max(0, i - locality), i)
        c.add_edge(order[i], order[j], rng.randint(wmin, wmax), cap_fn())
    added = tries = 0
    while added < n_extra and tries < 60 * (n_extra + 1):
        tries += 1
        a, b = rng.sample(ids, 2)
        if c.add_edge(a, b, rng.randint(wmin, wmax), cap_fn()):
            added += 1


def build_grid(c, rng, rows, cols, weight_fn, cap_fn, special):
    """Row-major grid. `special` maps (r, c) -> (type, capacity)."""
    ids = {}
    for r in range(rows):
        for col in range(cols):
            ntype, cap = special.get((r, col), ("travel", None))
            name = f"Aisle-r{r}c{col}" if ntype == "travel" else None
            ids[(r, col)] = c.add_node(ntype, name=name, capacity=cap)
    for r in range(rows):
        for col in range(cols):
            if col + 1 < cols:
                c.add_edge(ids[(r, col)], ids[(r, col + 1)], weight_fn(), cap_fn())
            if r + 1 < rows:
                c.add_edge(ids[(r, col)], ids[(r + 1, col)], weight_fn(), cap_fn())
    return ids


def pick_unit_starts(c, rng, n_units, preferred, p_odd=0.15):
    """Start nodes: mostly spread over `preferred`, occasionally a station/travel node.
    Respects node capacities (never more starters than a node can hold)."""
    occupancy = Counter()

    def room(nid):
        cap = c.node_capacity(nid)
        return cap is None or occupancy[nid] < cap

    pref = list(dict.fromkeys(preferred))
    rng.shuffle(pref)
    pref_set = set(pref)
    others = [n["id"] for n in c.nodes if n["id"] not in pref_set]
    starts = []
    for i in range(n_units):
        cand = None
        if others and rng.random() < p_odd:
            pool = [x for x in others if room(x)]
            if pool:
                cand = rng.choice(pool)
        if cand is None:
            pool = [x for x in pref if room(x)]
            if pool:
                cand = pool[i % len(pool)]
        if cand is None:
            pool = [n["id"] for n in c.nodes if room(n["id"])]
            cand = rng.choice(pool)
        occupancy[cand] += 1
        starts.append(cand)
    return starts


def unit_capacities(rng, n_units, mode):
    if mode == "all1":
        return [1] * n_units
    if mode == "all2":
        return [2] * n_units
    if mode == "some3":
        caps = [rng.choice([1, 2, 2]) for _ in range(n_units)]
        caps[rng.randrange(n_units)] = 3
        return caps
    # mixed: at least one unit with capacity 2
    caps = [rng.choice([1, 2]) for _ in range(n_units)]
    if 2 not in caps:
        caps[rng.randrange(n_units)] = 2
    return caps


# --------------------------------------------------------------------------
# Level 1: one drive unit, weighted shortest path
# --------------------------------------------------------------------------
def gen_level1(rng):
    kind = rng.choice(["chain_shortcut", "chain_shortcut", "two_station_loop",
                       "random_sparse", "random_sparse"])
    c = CaseBuilder()
    notes = []

    if kind == "chain_shortcut":
        s = c.add_node("storage")
        k = rng.randint(2, 5)
        chain = [c.add_node("travel") for _ in range(k)]
        d = c.add_node("station")
        path = [s] + chain + [d]
        m = len(path) - 1
        w = [1] * m
        for _ in range(rng.randint(0, 9 - m)):
            w[rng.randrange(m)] += 1
        for i in range(m):
            c.add_edge(path[i], path[i + 1], w[i])
        direct = rng.randint(sum(w) + 1, 10)
        c.add_edge(s, d, direct)
        notes.append(f"direct aisle weight {direct} vs {m}-hop path of length {sum(w)}")
        if rng.random() < 0.5:
            x = c.add_node("travel", name="Side-Aisle")
            c.add_edge(s, x, 1)
            c.add_edge(x, d, rng.randint(max(7, sum(w)), 10))
            notes.append("cheapest first hop leads onto a long aisle")
        if rng.random() < 0.35:
            d2 = c.add_node("station")
            c.add_edge(rng.choice(chain), d2, rng.randint(2, 6))
        storages, stations = [s], c.stations
        n_pods = rng.randint(1, 3)
        label = "chain + long shortcut"

    elif kind == "two_station_loop":
        s = c.add_node("storage")
        c.add_node("station")
        c.add_node("station")
        travels = [c.add_node("travel") for _ in range(rng.randint(3, 6))]
        ids = [s] + c.stations + travels
        random_connected(c, rng, ids, n_extra=rng.randint(2, 4), wmin=1, wmax=6)
        storages, stations = [s], c.stations
        n_pods = rng.randint(2, 4)
        label = "two stations, looping aisles"

    else:  # random_sparse
        n_storage = rng.choice([1, 1, 2])
        n_station = rng.choice([1, 2])
        n_total = rng.randint(max(5, n_storage + n_station + 2), 10)
        storages = [c.add_node("storage") for _ in range(n_storage)]
        stations = [c.add_node("station") for _ in range(n_station)]
        travels = [c.add_node("travel") for _ in range(n_total - n_storage - n_station)]
        random_connected(c, rng, storages + stations + travels,
                         n_extra=rng.randint(1, 4), wmin=1, wmax=10)
        n_pods = rng.randint(1, 4)
        label = "random sparse graph"

    t = 0
    for i in range(n_pods):
        if i > 0:
            t += rng.randint(6, 20)
        c.add_pod(rng.choice(storages), rng.choice(stations), t)

    if kind != "chain_shortcut" and rng.random() < 0.6:
        pod = rng.choice(c.pods)
        s, d = pod["source_node"], pod["destination_station"]
        dist = c.dijkstra(s).get(d)
        if dist is not None and 2 <= dist <= 9 and not c.has_edge(s, d):
            direct = rng.randint(dist + 1, 10)
            c.add_edge(s, d, direct)
            notes.append(f"long direct aisle (weight {direct}) vs multi-hop path of length {dist}")

    if rng.random() < 0.3:
        e = rng.choice(c.edges)
        e["capacity"] = rng.choice([1, 2])
        notes.append("one aisle has a capacity, irrelevant for a single unit")

    start = pick_unit_starts(c, rng, 1, storages, p_odd=0.15)[0]
    c.add_unit(start, 1)
    return c, label, notes


# --------------------------------------------------------------------------
# Level 2: several units, narrow aisles
# --------------------------------------------------------------------------
def gen_level2(rng):
    kind = rng.choice(["grid", "grid", "bridge_detour", "bridge_detour", "random_narrow"])
    c = CaseBuilder()
    notes = []
    n_units = rng.choice([2, 2, 3, 3])

    if kind == "grid":
        rows, cols = rng.choice([(2, 3), (3, 3), (3, 3), (3, 4)])
        p_cap = 1.0 if rng.random() < 0.7 else rng.uniform(0.6, 0.9)
        w0 = rng.choice([1, 2, 2, 3])
        vary = rng.random() < 0.4
        n_storage = rng.randint(1, min(3, rows))
        n_station = rng.randint(1, min(2, rows))
        left = [(r, 0) for r in range(rows)]
        right = [(r, cols - 1) for r in range(rows)]
        special = {}
        for pos in rng.sample(left, n_storage):
            special[pos] = ("storage", None)
        for pos in rng.sample(right, n_station):
            special[pos] = ("station", None)
        ids = build_grid(c, rng, rows, cols,
                         weight_fn=lambda: w0 + (rng.choice([0, 0, 1]) if vary else 0),
                         cap_fn=lambda: 1 if rng.random() < p_cap else None,
                         special=special)
        preferred = c.storages + [ids[p] for p in left if p not in special]
        label = f"grid {rows}x{cols}"
        notes.append("all aisles capacity 1" if p_cap >= 1.0 else "most aisles capacity 1")

    elif kind == "bridge_detour":
        n_storage = rng.randint(1, 3)
        n_station = rng.choice([1, 1, 2])
        storages = [c.add_node("storage") for _ in range(n_storage)]
        a = c.add_node("travel", name="Junction-A")
        b = c.add_node("travel", name="Junction-B")
        stations = [c.add_node("station") for _ in range(n_station)]
        n_det = rng.randint(1, 2)
        det = [c.add_node("travel", name=f"Detour-{i + 1}") for i in range(n_det)]
        extra = []
        while len(c.nodes) < 6 or (len(extra) < 2 and rng.random() < 0.3):
            extra.append(c.add_node("travel", name=f"Parking-{len(extra) + 1}"))
        p_side = rng.choice([0.0, 0.0, 0.5, 1.0])
        for s in storages:
            c.add_edge(s, a, rng.randint(1, 3), 1 if rng.random() < p_side else None)
        wb = rng.randint(4, 7)
        c.add_edge(a, b, wb, 1)
        total = wb + rng.randint(1, 3)
        parts = split_int(rng, total, n_det + 1, 1, 10)
        chain = [a] + det + [b]
        det_cap = 1 if rng.random() < 0.4 else None
        for i in range(len(chain) - 1):
            c.add_edge(chain[i], chain[i + 1], parts[i], det_cap)
        for d in stations:
            c.add_edge(b, d, rng.randint(1, 3), 1 if rng.random() < p_side else None)
        for x in extra:
            c.add_edge(x, rng.choice([a, b] + storages), rng.randint(1, 3))
        preferred = storages + extra
        label = "bridge + detour"
        notes.append(f"narrow bridge weight {wb} (capacity 1) vs detour of length {total}")

    else:  # random_narrow
        n_storage = rng.randint(1, 3)
        n_station = rng.choice([1, 2])
        n_total = rng.randint(max(6, n_storage + n_station + 2), 12)
        storages = [c.add_node("storage") for _ in range(n_storage)]
        stations = [c.add_node("station") for _ in range(n_station)]
        travels = [c.add_node("travel") for _ in range(n_total - n_storage - n_station)]
        p_cap = rng.choice([1.0, 1.0, 0.8, 0.7])
        random_connected(c, rng, storages + stations + travels,
                         n_extra=rng.randint(2, 5), wmin=1, wmax=5,
                         cap_fn=lambda: 1 if rng.random() < p_cap else None)
        preferred = storages
        label = "random narrow aisles"
        notes.append("all aisles capacity 1" if p_cap >= 1.0 else "most aisles capacity 1")

    for st in pick_unit_starts(c, rng, n_units, preferred, p_odd=0.12):
        c.add_unit(st, 1)

    n_pods = rng.randint(2, 8)
    first_batch = rng.randint(1, min(3, n_pods))
    for t in staggered_times(rng, n_pods, first_batch, [0, 2, 4, 4, 6, 8], 20):
        c.add_pod(rng.choice(c.storages), rng.choice(c.stations), t)
    return c, label, notes


# --------------------------------------------------------------------------
# Level 3: docks and multi-pod units
# --------------------------------------------------------------------------
def gen_level3(rng):
    kind = rng.choice(["single_dock", "single_dock", "two_stations_docks",
                       "two_stations_docks", "grid_docks", "random_docks"])
    c = CaseBuilder()
    notes = []
    n_units = rng.choice([2, 2, 3, 3])
    cap_mode = rng.choice(["all2", "all2", "mixed", "mixed", "all1", "some3"])

    def station_caps(n):
        caps = [rng.choice([1, 2]) for _ in range(n)]
        if 1 not in caps:
            caps[rng.randrange(n)] = 1
        return caps

    if kind == "single_dock":
        n_storage = rng.choice([1, 1, 2])
        storages = [c.add_node("storage") for _ in range(n_storage)]
        parking = [c.add_node("travel", name=f"Parking-{i + 1}") for i in range(n_units)]
        approaches = [c.add_node("travel", name="Approach")]
        if rng.random() < 0.4:
            approaches.append(c.add_node("travel", name="Approach-2"))
        while len(c.nodes) < 5:
            parking.append(c.add_node("travel", name=f"Parking-{len(parking) + 1}"))
        station = c.add_node("station", capacity=1)
        for i, p in enumerate(parking):
            c.add_edge(p, storages[i % n_storage], rng.randint(1, 3))
        for i, s in enumerate(storages):
            c.add_edge(s, approaches[i % len(approaches)], rng.randint(1, 3))
        if n_storage == 2 and rng.random() < 0.5:
            c.add_edge(storages[0], storages[1], rng.randint(2, 5))
        for ap in approaches:
            c.add_edge(ap, station, rng.randint(2, 4))
        if len(approaches) == 2:
            c.add_edge(approaches[0], approaches[1], rng.randint(1, 3))
        preferred = parking
        n_pods = rng.randint(4, 8)
        label = "single dock (station capacity 1)"
        notes.append("one station with capacity 1 shared by everyone")

    elif kind == "two_stations_docks":
        n_storage = rng.randint(2, 3)
        n_station = rng.randint(2, 3)
        storages = [c.add_node("storage") for _ in range(n_storage)]
        junctions = [c.add_node("travel", name=f"Junction-{i + 1}") for i in range(n_storage)]
        approaches = [c.add_node("travel", name=f"Approach-{i + 1}") for i in range(n_station)]
        stations = [c.add_node("station", capacity=cp) for cp in station_caps(n_station)]
        for s, j in zip(storages, junctions):
            c.add_edge(s, j, rng.randint(1, 3))
        for i in range(len(junctions) - 1):
            c.add_edge(junctions[i], junctions[i + 1], rng.randint(3, 5))
        if len(junctions) == 3 and rng.random() < 0.5:
            c.add_edge(junctions[0], junctions[2], rng.randint(3, 6))
        for i, j in enumerate(junctions):
            c.add_edge(j, approaches[i % n_station], rng.randint(2, 4))
        for i in range(len(approaches) - 1):
            c.add_edge(approaches[i], approaches[i + 1], rng.randint(1, 3))
        for ap, st in zip(approaches, stations):
            c.add_edge(ap, st, rng.randint(1, 3))
        if rng.random() < 0.5:
            c.add_edge(rng.choice(junctions), rng.choice(approaches), rng.randint(3, 6))
        preferred = junctions if rng.random() < 0.6 else storages
        n_pods = rng.randint(4, 10)
        label = "storage junctions + station docks"

    elif kind == "grid_docks":
        rows, cols = rng.choice([(3, 3), (3, 4), (2, 4), (2, 5)])
        n_storage = rng.randint(2, 3)
        n_station = rng.randint(1, 3)
        left = [(r, 0) for r in range(rows)] + [(r, 1) for r in range(rows)]
        right = [(r, cols - 1) for r in range(rows)] + [(r, cols - 2) for r in range(rows)]
        left = [p for p in left if p not in right]
        special = {}
        for pos in rng.sample(left, n_storage):
            special[pos] = ("storage", None)
        for pos, cp in zip(rng.sample(right, n_station), station_caps(n_station)):
            special[pos] = ("station", cp)
        edge_mode = rng.choice(["open", "open", "narrow1", "narrow2"])
        p_cap = 0.0 if edge_mode == "open" else rng.uniform(0.3, 0.7)
        ecap = 1 if edge_mode == "narrow1" else 2
        ids = build_grid(c, rng, rows, cols,
                         weight_fn=lambda: rng.randint(1, 3),
                         cap_fn=lambda: ecap if rng.random() < p_cap else None,
                         special=special)
        preferred = c.storages + [ids[p] for p in left if p not in special]
        n_pods = rng.randint(4, 10)
        label = f"grid {rows}x{cols} with docks"

    else:  # random_docks
        n_storage = rng.randint(2, 3)
        n_station = rng.randint(1, 3)
        n_total = rng.randint(max(7, n_storage + n_station + 2), 12)
        storages = [c.add_node("storage") for _ in range(n_storage)]
        stations = [c.add_node("station", capacity=cp) for cp in station_caps(n_station)]
        travels = [c.add_node("travel") for _ in range(n_total - n_storage - n_station)]
        p_cap = rng.choice([0.0, 0.3, 0.6])
        random_connected(c, rng, storages + stations + travels,
                         n_extra=rng.randint(2, 5), wmin=1, wmax=5,
                         cap_fn=lambda: rng.choice([1, 2]) if rng.random() < p_cap else None)
        preferred = storages
        n_pods = rng.randint(4, 10)
        label = "random graph with docks"

    caps = unit_capacities(rng, n_units, cap_mode)
    for st, cp in zip(pick_unit_starts(c, rng, n_units, preferred, p_odd=0.12), caps):
        c.add_unit(st, cp)

    first_batch = rng.randint(2, min(4, n_pods))
    for t in staggered_times(rng, n_pods, first_batch, [0, 0, 5, 10, 10], 30):
        c.add_pod(rng.choice(c.storages), rng.choice(c.stations), t)
    return c, label, notes


# --------------------------------------------------------------------------
# Stress: big grids, many units and pods
# --------------------------------------------------------------------------
def gen_stress(rng):
    c = CaseBuilder()
    notes = []
    rows, cols = rng.randint(4, 6), rng.randint(4, 6)
    n_storage = rng.randint(2, 4)
    n_station = rng.randint(2, 4)
    left = [(r, col) for r in range(rows) for col in (0, 1)]
    right = [(r, col) for r in range(rows) for col in (cols - 2, cols - 1)]
    special = {}
    for pos in rng.sample(left, n_storage):
        special[pos] = ("storage", None)
    for pos in rng.sample(right, n_station):
        special[pos] = ("station", rng.choice([1, 2, 3, None]))

    def cap_fn():
        r = rng.random()
        return None if r < 0.4 else (1 if r < 0.75 else 2)

    ids = build_grid(c, rng, rows, cols, weight_fn=lambda: rng.randint(1, 4),
                     cap_fn=cap_fn, special=special)
    for _ in range(rng.randint(1, 4)):
        r, col = rng.randrange(rows - 1), rng.randrange(cols - 1)
        a, b = ((r, col), (r + 1, col + 1)) if rng.random() < 0.5 else ((r + 1, col), (r, col + 1))
        c.add_edge(ids[a], ids[b], rng.randint(3, 6), cap_fn())

    n_units = rng.randint(3, 6)
    starts = pick_unit_starts(c, rng, n_units, c.storages, p_odd=0.3)
    for st in starts:
        c.add_unit(st, rng.choice([1, 1, 2, 2, 3]))

    n_pods = rng.randint(10, 30)
    times = [0] * rng.randint(2, 5) + [rng.randint(0, 60) for _ in range(n_pods)]
    for t in sorted(times[:n_pods]):
        c.add_pod(rng.choice(c.storages), rng.choice(c.stations), t)
    label = f"grid {rows}x{cols}"
    notes.append("mixed aisle capacities")
    return c, label, notes


# --------------------------------------------------------------------------
# Decorations shared by levels 2/3/stress
# --------------------------------------------------------------------------
def make_one_way(c, rng, k):
    """Turn up to k bidirectional edges into one-way aisles, keeping every
    storage/station/start node mutually reachable."""
    cand = [e for e in c.edges if e.get("bidirectional", True)]
    rng.shuffle(cand)
    made = 0
    for e in cand:
        if made >= k:
            break
        flips = [False, True]
        rng.shuffle(flips)
        for flip in flips:
            if flip:
                e["from_node"], e["to_node"] = e["to_node"], e["from_node"]
            e["bidirectional"] = False
            if c.important_connected():
                made += 1
                break
            del e["bidirectional"]
    return made


def add_node_capacities(c, rng, p_storage, p_travel):
    starts = Counter(u["start_node"] for u in c.units)
    notes = []

    def near_capped(nid):
        capped = {n["id"] for n in c.nodes if n.get("capacity") is not None}
        return nid in capped or bool(c.undirected_neighbors(nid) & capped)

    if rng.random() < p_storage:
        pool = [s for s in c.storages if not near_capped(s)]
        if pool:
            s = rng.choice(pool)
            cap = max(rng.choice([1, 2, 2, 3]), starts[s], 1)
            c.nodes[s]["capacity"] = cap
            notes.append(f"storage node {s} has capacity {cap}")
    if rng.random() < p_travel:
        pool = [t for t in c.travels
                if len(c.undirected_neighbors(t)) >= 2 and not near_capped(t)]
        if pool:
            t = rng.choice(pool)
            cap = max(rng.choice([1, 1, 2]), starts[t], 1)
            c.nodes[t]["capacity"] = cap
            notes.append(f"travel node {t} has capacity {cap}")
    return notes


def choose_max_time_steps(c, rng, cap):
    adj = c.adjacency()
    dist_cache = {}
    total = 0
    for p in c.pods:
        s = p["source_node"]
        if s not in dist_cache:
            dist_cache[s] = dijkstra(adj, s)
        total += dist_cache[s].get(p["destination_station"], 0)
    last_entry = max(p["entry_time"] for p in c.pods)
    est = total / max(1, len(c.units)) + last_entry + 10
    mts = int(round(est * rng.uniform(3.0, 5.0)))
    mts = max(60, mts, last_entry + 40)
    return min(mts, max(cap, last_entry + 40))


# --------------------------------------------------------------------------
# Validation (pure structural checks on the JSON dict)
# --------------------------------------------------------------------------
def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate_data(data):
    problems = []
    meta = data.get("metadata", {})
    mts = meta.get("max_time_steps")
    if not _is_int(mts) or mts < 60:
        problems.append("metadata.max_time_steps missing or < 60")
        mts = 60
    if not isinstance(meta.get("description"), str) or not meta["description"]:
        problems.append("metadata.description missing")

    nodes = data.get("nodes", [])
    ids = [n.get("id") for n in nodes]
    if ids != list(range(len(nodes))):
        problems.append("node ids are not 0..N-1 in order")
    types = {}
    for n in nodes:
        t = n.get("type", "travel")
        if t not in ("storage", "station", "travel"):
            problems.append(f"node {n.get('id')} bad type {t}")
        if "capacity" in n and (not _is_int(n["capacity"]) or n["capacity"] < 1):
            problems.append(f"node {n.get('id')} bad capacity")
        types[n.get("id")] = t

    pairs = set()
    adj = {i: [] for i in types}
    for e in data.get("edges", []):
        a, b = e.get("from_node"), e.get("to_node")
        if a not in types or b not in types:
            problems.append(f"edge {a}-{b} references unknown node")
            continue
        if a == b:
            problems.append(f"self-loop at {a}")
        key = frozenset((a, b))
        if key in pairs:
            problems.append(f"duplicate edge {a}-{b}")
        pairs.add(key)
        if not _is_int(e.get("weight")) or e["weight"] < 1:
            problems.append(f"edge {a}-{b} bad weight")
        if "capacity" in e and (not _is_int(e["capacity"]) or e["capacity"] < 1):
            problems.append(f"edge {a}-{b} bad capacity")
        bidi = e.get("bidirectional", True)
        if not isinstance(bidi, bool):
            problems.append(f"edge {a}-{b} bad bidirectional flag")
        adj[a].append((b, 1))
        if bidi:
            adj[b].append((a, 1))

    units = data.get("drive_units", [])
    if not units:
        problems.append("no drive units")
    uids = [u.get("id") for u in units]
    if len(set(uids)) != len(uids):
        problems.append("duplicate drive unit ids")
    starts = Counter()
    for u in units:
        if u.get("start_node") not in types:
            problems.append(f"unit {u.get('id')} starts at unknown node")
            continue
        starts[u["start_node"]] += 1
        if "capacity" in u and (not _is_int(u["capacity"]) or u["capacity"] < 1):
            problems.append(f"unit {u.get('id')} bad capacity")
    for n in nodes:
        if n.get("capacity") is not None and starts[n["id"]] > n["capacity"]:
            problems.append(f"node {n['id']} holds more starting units than its capacity")

    pods = data.get("pods", [])
    if not pods:
        problems.append("no pods")
    pids = [p.get("id") for p in pods]
    if len(set(pids)) != len(pids) or not all(isinstance(x, str) and x for x in pids):
        problems.append("pod ids not unique non-empty strings")
    for p in pods:
        if types.get(p.get("source_node")) != "storage":
            problems.append(f"pod {p.get('id')} source is not a storage node")
        if types.get(p.get("destination_station")) != "station":
            problems.append(f"pod {p.get('id')} destination is not a station")
        t = p.get("entry_time")
        if not _is_int(t) or t < 0 or t >= mts - 20:
            problems.append(f"pod {p.get('id')} bad entry_time {t}")

    important = {i for i, t in types.items() if t in ("storage", "station")}
    important |= {u["start_node"] for u in units if u.get("start_node") in types}
    for u in important:
        missing = important - reachable(adj, u)
        if missing:
            problems.append(f"nodes {sorted(missing)} not reachable from {u}")
    return problems


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
GENERATORS = {"level1": gen_level1, "level2": gen_level2, "level3": gen_level3,
              "stress": gen_stress}


def describe(level, label, c, notes, seed, idx):
    caps = sorted({u.get("capacity", 1) for u in c.units})
    cap_txt = "/".join(str(x) for x in caps)
    st_caps = [n.get("capacity") for n in c.nodes if n["type"] == "station"]
    dock_txt = ""
    if any(x is not None for x in st_caps):
        dock_txt = " with capacities " + "/".join("inf" if x is None else str(x) for x in st_caps)
    head = "Stress" if level == "stress" else f"Level {LEVEL_NO[level]}"
    txt = (f"{head} ({label}): {len(c.units)} drive unit(s) (capacity {cap_txt}), "
           f"{len(c.pods)} pod(s) from {len(c.storages)} storage node(s) to "
           f"{len(c.stations)} station(s){dock_txt}")
    if notes:
        txt += "; " + "; ".join(notes)
    txt += f". Generated by dev/gen_cases.py, seed {seed}, case {idx}."
    return txt


def generate_case(level, seed, idx):
    rng = random.Random(seed * 1_000_003 + LEVEL_NO[level] * 100_003 + idx)
    for _attempt in range(50):
        c, label, notes = GENERATORS[level](rng)
        if level in ("level2", "level3", "stress"):
            p_oneway = 0.30 if level == "stress" else 0.12
            if rng.random() < p_oneway:
                k = rng.randint(2, 5) if level == "stress" else rng.randint(1, 3)
                made = make_one_way(c, rng, k)
                if made:
                    notes.append(f"{made} one-way aisle(s)")
            p_cap = 0.3 if level == "stress" else 0.12
            notes += add_node_capacities(c, rng, p_cap, p_cap)
        c.finalize_pods()
        mts = choose_max_time_steps(c, rng, MTS_CAP[level])
        data = c.to_data(mts, describe(level, label, c, notes, seed, idx))
        problems = validate_data(data)
        if not problems:
            return data
    raise RuntimeError(f"could not build a valid {level} case #{idx}: {problems}")


def dump_case(data):
    lines = ["{"]
    meta = json.dumps(data["metadata"], indent=2).replace("\n", "\n  ")
    lines.append(f'  "metadata": {meta},')
    keys = ["nodes", "edges", "drive_units", "pods"]
    for key in keys:
        items = data[key]
        lines.append(f'  "{key}": [')
        for i, item in enumerate(items):
            lines.append("    " + json.dumps(item) + ("," if i < len(items) - 1 else ""))
        lines.append("  ]" + ("," if key != keys[-1] else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def summarize(written):
    def stats(vals):
        return f"{min(vals):>3}-{max(vals):<3} avg {sum(vals) / len(vals):5.1f}"

    print("\nDistribution summary (min-max avg):")
    print(f"{'level':<8}{'n':>4}  {'nodes':<18}{'edges':<18}{'units':<18}{'pods':<18}"
          f"{'max_time_steps':<18}{'one-way':>8}{'node-cap':>9}")
    for level in LEVELS:
        cases = [d for lv, _, d in written if lv == level]
        if not cases:
            continue
        one_way = sum(any(not e.get("bidirectional", True) for e in d["edges"]) for d in cases)
        node_cap = sum(any(n.get("capacity") is not None for n in d["nodes"]
                           if n["type"] != "station") for d in cases)
        print(f"{level:<8}{len(cases):>4}  "
              f"{stats([len(d['nodes']) for d in cases]):<18}"
              f"{stats([len(d['edges']) for d in cases]):<18}"
              f"{stats([len(d['drive_units']) for d in cases]):<18}"
              f"{stats([len(d['pods']) for d in cases]):<18}"
              f"{stats([d['metadata']['max_time_steps'] for d in cases]):<18}"
              f"{one_way:>8}{node_cap:>9}")


def run_validation(written):
    """Load every file through the real loader and play it with the example
    basic driver through the real GameEngine."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import logging
    from ar_hackathon.utils.json_loader import load_test_case
    from ar_hackathon.engine.game_engine import GameEngine
    from ar_hackathon.examples import basic_driver as bd
    logging.getLogger("ar_hackathon.engine.game_engine").setLevel(logging.ERROR)

    t0 = time.time()
    failures = []
    per_level = {lv: {"files": 0, "delivered": 0, "pods": 0, "steps": 0} for lv in LEVELS}
    for level, path, data in written:
        errors = []

        def driver(unit_id, state):
            try:
                return bd.basic_driver(unit_id, state)
            except Exception as exc:  # recorded, then re-raised for the engine
                errors.append(exc)
                raise

        try:
            tc = load_test_case(path)
            if tc.total_pods() != len(data["pods"]):
                raise RuntimeError("loader pod count mismatch")
            if len(tc.nodes) != len(data["nodes"]) or len(tc.edges) != len(data["edges"]):
                raise RuntimeError("loader node/edge count mismatch")
            bd.visited.clear()
            engine = GameEngine(path, driver)
            stats = engine.run_until_finished()
            if errors:
                raise RuntimeError(f"basic driver raised: {errors[0]!r}")
            if stats["total_time_steps"] > data["metadata"]["max_time_steps"]:
                raise RuntimeError("engine ran past max_time_steps")
            pl = per_level[level]
            pl["files"] += 1
            pl["delivered"] += stats["delivered_pods"]
            pl["pods"] += stats["total_pods"]
            pl["steps"] += stats["total_time_steps"]
        except Exception as exc:
            failures.append((path, repr(exc)))
        finally:
            bd.visited.clear()
    elapsed = time.time() - t0

    print("\nValidation (loader + GameEngine with basic_driver):")
    for lv in LEVELS:
        pl = per_level[lv]
        if pl["files"]:
            print(f"  {lv:<8}{pl['files']:>4} files ok, basic driver delivered "
                  f"{pl['delivered']}/{pl['pods']} pods, {pl['steps']} total steps")
    for path, err in failures:
        print(f"  FAIL {path}: {err}")
    print(f"  {len(written) - len(failures)}/{len(written)} files passed in {elapsed:.1f}s")
    return not failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--out", default="dev/cases", help="output directory (default dev/cases)")
    ap.add_argument("--seed", type=int, default=1, help="master seed (default 1)")
    ap.add_argument("--per-level", type=int, default=40, help="cases per level 1-3 (default 40)")
    ap.add_argument("--stress", type=int, default=12, help="number of stress cases (default 12)")
    ap.add_argument("--validate", action="store_true",
                    help="after writing, load each file and play it with the basic driver")
    args = ap.parse_args(argv)

    counts = {"level1": args.per_level, "level2": args.per_level,
              "level3": args.per_level, "stress": args.stress}
    written = []
    for level in LEVELS:
        outdir = os.path.join(args.out, level)
        os.makedirs(outdir, exist_ok=True)
        for idx in range(counts[level]):
            data = generate_case(level, args.seed, idx)
            path = os.path.join(outdir, f"{FILE_PREFIX[level]}_{idx:03d}.json")
            with open(path, "w") as fh:
                fh.write(dump_case(data))
            written.append((level, path, data))
        print(f"{level}: wrote {counts[level]} cases to {outdir}")

    summarize(written)
    if args.validate:
        ok = run_validation(written)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
