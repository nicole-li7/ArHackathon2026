"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name: CookieLovers
Email address: jadenbergernorth@gmail.com
*******************

==============================================================================
DESIGN OVERVIEW
==============================================================================
The engine is deterministic and we are handed the full state, so the router
is built as a *planner* rather than a reactive heuristic:

  1. Graph          - node/edge tables, all-pairs shortest distances.
  2. Snapshot       - compact copy of the current floor state (units, pods).
  3. Reservations   - a space-time table of edge and node occupancy that
                      mirrors the engine's capacity rules exactly, including
                      the "lower unit id commits first" tie-break.
  4. plan_path      - space-time A* for one unit that respects reservations
                      (it can wait, detour, or take a narrow aisle later).
  5. Plan           - for every unit an ordered list of stops
                      ('P', pod) = go pick up pod, ('D', pod) = deliver it,
                      plus a priority order in which units get routed.
  6. simulate       - an exact replica of the engine's step loop that executes
                      the planned routes and returns the real score the
                      engine would give for the pods we currently know about.
  7. Local search   - greedy insertion of new pods, then relocate / swap /
                      reorder / priority moves, each candidate scored by (6).
                      Runs as an anytime algorithm inside a strict time budget
                      and continues across time steps (warm start).
  8. Controller     - per-step caching, episode reset, validity checks, and a
                      simple Dijkstra fallback so we never return garbage.

Idle units are parked on storage nodes (where pods spawn) and never left on
a capacity-limited station dock.
==============================================================================
"""

import heapq
import itertools
import math
import random
import time
from typing import Dict, List, Optional, Tuple

from ar_hackathon.models.graph_state import GraphState

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
BUDGET_FIRST = 0.25       # s, first plan of an episode
BUDGET_EVENT = 0.18       # s, new pods appeared or plan diverged
BUDGET_STEADY = 0.10      # s, routine improvement
BUDGET_CONVERGED = 0.05   # s, search converged, re-check cheaply
TOTAL_BUDGET = 45.0       # s, planning time per episode (test case)
HORIZON_MAX = 300         # steps simulated / searched ahead
ASTAR_MAX_EXP = 5000      # expansions per space-time search
UNLIMITED = 1 << 30
INF = float("inf")
DEBUG = False
# Dev-only hook: the benchmark harness may set this to a list of
# (pod_id, source_node_id, destination_station_id, entry_time) tuples so the
# planner can be evaluated with perfect knowledge of future pods.  It is
# always None in the real game.
ORACLE_PODS = None
# Dev-only hook: when True the controller plans once (first call) and then
# blindly executes that plan so the harness can compare the internal replica's
# predictions with what the real engine does.  Always False in the real game.
FREEZE = False


# ===========================================================================
# 1. Graph
# ===========================================================================
class Graph(object):
    """Static floor description with the engine's edge-lookup semantics."""

    def __init__(self, state):
        nodes = state.nodes
        self.n = len(nodes)
        self.ids = [nd.id for nd in nodes]
        self.idx = {}
        for i, nd in enumerate(nodes):
            if nd.id not in self.idx:          # engine's get_node returns the first
                self.idx[nd.id] = i
        self.ncap = [nd.capacity if nd.capacity is not None else UNLIMITED
                     for nd in nodes]
        self.ntype = [nd.node_type for nd in nodes]
        self.storage = [i for i in range(self.n) if self.ntype[i] == "storage"]

        # Edges: mirror GraphState.get_edge (first edge in list order that
        # connects the ordered pair).
        self.pair_edge = {}        # (a, b) -> edge index used for that move
        self.edge_pairs = []       # k -> ordered pairs edge k "connects"
        self.ecap = []
        self.ew = []
        for e in state.edges:
            a = self.idx.get(e.from_node)
            b = self.idx.get(e.to_node)
            k = len(self.ew)
            if a is None or b is None or a == b:
                pairs = []
            else:
                pairs = [(a, b)] + ([(b, a)] if e.bidirectional else [])
            w = int(math.ceil(e.weight)) if e.weight is not None else 1
            if w < 1:
                w = 1
            self.edge_pairs.append(pairs)
            self.ecap.append(e.capacity if e.capacity is not None else UNLIMITED)
            self.ew.append(w)
            for p in pairs:
                if p not in self.pair_edge:
                    self.pair_edge[p] = k
        used = sorted(set(self.pair_edge.values()))
        # A unit moving on pair p counts toward the occupancy of every used
        # edge whose connects() accepts p (normally exactly one edge).
        self.pair_affects = {}
        for p in self.pair_edge:
            self.pair_affects[p] = [k for k in used if p in self.edge_pairs[k]]
        self.out = [[] for _ in range(self.n)]
        for (a, b), k in self.pair_edge.items():
            self.out[a].append((b, self.ew[k], k))
        for lst in self.out:
            lst.sort()
        self.spawn_freq = {}       # storage node -> pods seen spawning there
        self.has_caps = any(c < UNLIMITED for c in self.ncap) or \
            any(self.ecap[k] < UNLIMITED for k in used)
        # All-pairs shortest distances in time steps (capacities ignored).
        self.dist = [self._dijkstra(i) for i in range(self.n)]

    def _dijkstra(self, src):
        d = [INF] * self.n
        d[src] = 0
        heap = [(0, src)]
        while heap:
            dd, u = heapq.heappop(heap)
            if dd > d[u]:
                continue
            for v, w, _ in self.out[u]:
                nd = dd + w
                if nd < d[v]:
                    d[v] = nd
                    heapq.heappush(heap, (nd, v))
        return d


# ===========================================================================
# 2. Snapshot of the current state
# ===========================================================================
class UnitS(object):
    __slots__ = ("id", "node", "cap", "carrying", "transit", "dest", "rem",
                 "pair", "polled")


class PodS(object):
    __slots__ = ("id", "node", "dest", "entry", "carrier", "src")


class Snapshot(object):
    """Compact, index-based copy of the GraphState at planning time."""

    def __init__(self, G, state, t0, first_uid):
        self.t0 = t0
        units = sorted(state.drive_units, key=lambda u: u.id)
        self.units = []
        self.uidx = {}
        for u in units:
            if u.id in self.uidx:
                continue
            s = UnitS()
            s.id = u.id
            s.node = G.idx.get(u.current_node)
            s.cap = u.capacity if u.capacity is not None else 1
            s.carrying = []
            s.transit = bool(u.in_transit)
            s.dest = G.idx.get(u.transit_destination) if s.transit else None
            s.rem = int(math.ceil(u.transit_remaining_time)) if s.transit else 0
            if s.transit and s.rem < 1:
                s.rem = 1
            if s.transit and (s.dest is None or s.node is None):
                s.transit = False
                s.dest = None
                s.rem = 0
            s.pair = (s.node, s.dest) if s.transit else None
            s.polled = (not s.transit) and u.id < first_uid
            self.uidx[u.id] = len(self.units)
            self.units.append(s)
        self.pods = []
        self.pidx = {}
        for p in state.active_pods:
            if p.delivery_time is not None or p.id in self.pidx:
                continue
            dest = G.idx.get(p.destination_station)
            if dest is None:
                continue
            ps = PodS()
            ps.id = p.id
            ps.dest = dest
            ps.entry = p.entry_time
            ps.carrier = None
            ps.node = None
            ps.src = None
            if p.carried_by is not None and p.carried_by in self.uidx:
                ps.carrier = self.uidx[p.carried_by]
            elif p.current_node is not None:
                ps.node = G.idx.get(p.current_node)
                if ps.node is None:
                    continue
            else:
                continue
            self.pidx[p.id] = len(self.pods)
            self.pods.append(ps)
        if ORACLE_PODS:
            delivered = set(p.id for p in state.delivered_pods)
            for pid, src_id, dest_id, entry in ORACLE_PODS:
                if entry <= t0 or pid in self.pidx or pid in delivered:
                    continue
                src = G.idx.get(src_id)
                dest = G.idx.get(dest_id)
                if src is None or dest is None:
                    continue
                ps = PodS()
                ps.id = pid
                ps.dest = dest
                ps.entry = entry
                ps.carrier = None
                ps.node = None
                ps.src = src
                self.pidx[pid] = len(self.pods)
                self.pods.append(ps)
        # carrying lists in the engine's own order
        for u in units:
            if u.id in self.uidx:
                s = self.units[self.uidx[u.id]]
                s.carrying = [self.pidx[pid] for pid in u.carrying if pid in self.pidx]


# ===========================================================================
# 3. Reservation table (space-time occupancy with engine tie-break rules)
# ===========================================================================
class Reservations(object):
    """
    Node intervals (a, d, uid): unit is inbound/standing at the node from the
    step it departs toward it (a; -1 if already there) until the step it
    departs (d; INF if it stays).  Edge intervals (s0, s1, uid): unit is on
    the edge during routing phases s0..s1 inclusive.

    Presence of unit i "just before unit k is polled at step s":
      node: (a < s or (a == s and i < k)) and (s < d or (s == d and i > k))
      edge: (s0 < s or (s0 == s and i < k)) and s <= s1
    """

    def __init__(self, G):
        self.G = G
        self.node_iv = [[] for _ in range(G.n)]
        self.node_ent = [[] for _ in range(G.n)]
        self.edge_iv = [[] for _ in range(len(G.ew))]
        self.edge_ent = [[] for _ in range(len(G.ew))]

    def remove_unit(self, j):
        for lst in (self.node_iv, self.node_ent, self.edge_iv, self.edge_ent):
            for i in range(len(lst)):
                if lst[i]:
                    lst[i] = [x for x in lst[i] if x[-1] != j]

    def add_node(self, j, n, a, d):
        self.node_iv[n].append((a, d, j))
        if a >= 0:
            self.node_ent[n].append((a, j))

    def add_move(self, j, pair, s, w):
        for k in self.G.pair_affects[pair]:
            self.edge_iv[k].append((s, s + w - 1, j))
            self.edge_ent[k].append((s, j))

    def add_transit(self, j, pair, s1):
        for k in self.G.pair_affects.get(pair, ()):
            self.edge_iv[k].append((-1, s1, j))

    def node_occ(self, n, s, k, j):
        c = 0
        for a, d, i in self.node_iv[n]:
            if i == j:
                continue
            if (a < s or (a == s and i < k)) and (s < d or (s == d and i > k)):
                c += 1
        return c

    def edge_occ(self, e, s, k, j):
        c = 0
        for s0, s1, i in self.edge_iv[e]:
            if i == j:
                continue
            if (s0 < s or (s0 == s and i < k)) and s <= s1:
                c += 1
        return c

    def can_enter_node(self, j, n, s_dep, s_arr):
        cap = self.G.ncap[n]
        if cap >= UNLIMITED:
            return True
        if self.node_occ(n, s_dep, j, j) >= cap:
            return False
        for sp, i in self.node_ent[n]:
            if i == j or sp < s_dep or sp > s_arr:
                continue
            if sp == s_dep and i < j:
                continue        # i committed before our reservation existed
            if sp == s_arr and i > j:
                continue        # depends on our action at s_arr (deferred)
            if self.node_occ(n, sp, i, j) >= cap - 1:
                return False
        return True

    def can_wait_node(self, j, n, s):
        cap = self.G.ncap[n]
        if cap >= UNLIMITED:
            return True
        for sp, i in self.node_ent[n]:
            if i == j:
                continue
            if (sp == s and i > j) or (sp == s + 1 and i < j):
                if self.node_occ(n, sp, i, j) >= cap - 1:
                    return False
        return True

    def can_enter_edge(self, j, pair, s, w):
        G = self.G
        own = G.pair_edge[pair]
        for k in G.pair_affects[pair]:
            cap = G.ecap[k]
            if cap >= UNLIMITED:
                continue
            if k == own and self.edge_occ(k, s, j, j) >= cap:
                return False
            last = s + w - 1
            for sp, i in self.edge_ent[k]:
                if i == j or sp < s or sp > last:
                    continue
                if sp == s and i < j:
                    continue
                if self.edge_occ(k, sp, i, j) >= cap - 1:
                    return False
        return True


# ===========================================================================
# 4. Space-time A* for one unit
# ===========================================================================
def plan_path(G, R, j, start, t0, goal, horizon):
    """Earliest-arrival path from (start, t0) to goal honouring reservations.

    Returns (moves, t_arrival) with moves = [(depart_step, from, to, w)], or
    None if unreachable within the horizon / expansion cap.
    """
    dist = G.dist
    if dist[start][goal] >= INF:
        return None
    if start == goal:
        return [], t0
    if not G.has_caps:
        moves = []
        node, t = start, t0
        while node != goal:
            best = None
            for nb, w, _ in G.out[node]:
                if w + dist[nb][goal] == dist[node][goal]:
                    best = (nb, w)
                    break
            if best is None:
                return None
            moves.append((t, node, best[0], best[1]))
            t += best[1]
            node = best[0]
        return moves, t

    parent = {(start, t0): None}
    heap = [(t0 + dist[start][goal], t0, start)]
    closed = set()
    expansions = 0
    while heap:
        f, t, node = heapq.heappop(heap)
        key = (node, t)
        if key in closed:
            continue
        closed.add(key)
        if node == goal:
            keys = []
            k = key
            while k is not None:
                keys.append(k)
                k = parent[k]
            keys.reverse()
            moves = []
            for (n1, t1), (n2, t2) in zip(keys, keys[1:]):
                if n1 != n2:
                    moves.append((t1, n1, n2, t2 - t1))
            return moves, t
        expansions += 1
        if expansions > ASTAR_MAX_EXP:
            return None
        if t + 1 <= horizon:
            k2 = (node, t + 1)
            if k2 not in parent and R.can_wait_node(j, node, t):
                parent[k2] = key
                heapq.heappush(heap, (t + 1 + dist[node][goal], t + 1, node))
        for nb, w, _ in G.out[node]:
            if dist[nb][goal] >= INF:
                continue
            t2 = t + w
            if t2 > horizon:
                continue
            k2 = (nb, t2)
            if k2 in parent:
                continue
            if R.can_enter_edge(j, (node, nb), t, w) and \
                    R.can_enter_node(j, nb, t, t2):
                parent[k2] = key
                heapq.heappush(heap, (t2 + dist[nb][goal], t2, nb))
    return None


# ===========================================================================
# 5. Plans (stop sequences + priority order) and route construction
# ===========================================================================
class Plan(object):
    __slots__ = ("order", "stops")

    def __init__(self, order, stops):
        self.order = order      # list of unit indices, routing priority
        self.stops = stops      # per unit index: list of (kind, pod index)

    def copy(self):
        return Plan(list(self.order), [list(s) for s in self.stops])


def stops_feasible(snap, u, stops):
    """Capacity check: never plan more pickups than the unit can hold."""
    unit = snap.units[u]
    load = set(unit.carrying)
    for kind, p in stops:
        if kind == "P":
            if p in load:
                continue
            if len(load) >= unit.cap:
                return False
            load.add(p)
        else:
            load.discard(p)
    return True


def build_routes(G, snap, plan, horizon):
    """Turn a plan into concrete space-time routes for every unit.

    Units are routed in priority order; each unit's delivery stops are
    followed immediately by its parking move so that later units see when a
    dock or aisle becomes free.  Returns (routes, finish, final, infeasible).
    """
    R = Reservations(G)
    t0 = snap.t0
    U = len(snap.units)
    # Provisional reservations: everybody is where they are until planned.
    for j, u in enumerate(snap.units):
        if u.transit:
            R.add_transit(j, u.pair, t0 + u.rem - 1)
            R.add_node(j, u.dest, -1, t0 + u.rem)
        elif u.node is not None:
            R.add_node(j, u.node, -1, t0 + (1 if u.polled else 0))

    routes = [[] for _ in range(U)]
    finish = [t0] * U
    final = [None] * U
    infeasible = 0
    parked = {}

    for j in plan.order:
        u = snap.units[j]
        if u.node is None:
            continue
        R.remove_unit(j)
        if u.transit:
            R.add_transit(j, u.pair, t0 + u.rem - 1)
            node, t = u.dest, t0 + u.rem
        else:
            node, t = u.node, t0 + (1 if u.polled else 0)
        cur_a = -1
        ok = True
        for kind, p in plan.stops[j]:
            pod = snap.pods[p]
            if kind == "P":
                goal = pod.node if pod.node is not None else pod.src
            else:
                goal = pod.dest
            if goal is None:
                continue
            if goal != node:
                res = plan_path(G, R, j, node, t, goal, horizon)
                if res is None:
                    ok = False
                    break
                for s, a, b, w in res[0]:
                    R.add_node(j, a, cur_a, s)
                    R.add_move(j, (a, b), s, w)
                    routes[j].append((s, a, b, w))
                    cur_a = s
                node, t = goal, res[1]
            if kind == "P" and pod.entry > t:
                t = pod.entry          # wait for the pod to spawn
        if not ok:
            infeasible += 1
        finish[j] = t
        final[j] = node
        # Parking: leave docks, head for a storage node with room.
        target = choose_parking(G, node, parked, U)
        if target is not None and target != node:
            res = plan_path(G, R, j, node, t, target, horizon)
            if res is not None:
                for s, a, b, w in res[0]:
                    R.add_node(j, a, cur_a, s)
                    R.add_move(j, (a, b), s, w)
                    routes[j].append((s, a, b, w))
                    cur_a = s
                node = target
        R.add_node(j, node, cur_a, INF)
        final[j] = node
        parked[node] = parked.get(node, 0) + 1
    return routes, finish, final, infeasible


def parking_room(G, n, parked, U):
    """Can one more idle unit stand on node n without choking it?"""
    cap = G.ncap[n]
    if cap >= UNLIMITED:
        return True
    here = parked.get(n, 0)
    return here + 1 < cap or (here + 1 <= cap and cap >= U)


def choose_parking(G, node, parked, U):
    """Storage node with room, busiest spawn points first, spreading idle
    units over distinct nodes; never a dock."""
    best = None
    freq = G.spawn_freq
    for s in G.storage:
        if not parking_room(G, s, parked, U):
            continue
        d = G.dist[node][s]
        if d >= INF:
            continue
        key = (parked.get(s, 0), -freq.get(s, 0), d, s)
        if best is None or key < best[0]:
            best = (key, s)
    if best is not None:
        return best[1]
    if G.ntype[node] != "station" and parking_room(G, node, parked, U):
        return node
    for i in sorted(range(G.n), key=lambda i: (G.dist[node][i], i)):
        if G.ntype[i] != "station" and parking_room(G, i, parked, U) and G.dist[node][i] < INF:
            return i
    return node


# ===========================================================================
# 6. Exact simulation of the engine on the known pods
# ===========================================================================
def simulate(G, snap, routes, horizon):
    """Replay the engine's step loop executing the given routes.

    Returns (score, first_moves, mismatches, delivered_times).
    first_moves: {unit index: next node index} for the current step.
    """
    t0 = snap.t0
    U = len(snap.units)
    P = len(snap.pods)
    node = [u.node for u in snap.units]
    transit = [u.transit for u in snap.units]
    dest = [u.dest for u in snap.units]
    rem = [u.rem for u in snap.units]
    pair = [u.pair for u in snap.units]
    cap = [u.cap for u in snap.units]
    polled = [u.polled for u in snap.units]
    carrying = [list(u.carrying) for u in snap.units]
    pnode = [p.node for p in snap.pods]
    pdest = [p.dest for p in snap.pods]
    pentry = [p.entry for p in snap.pods]
    pid = [p.id for p in snap.pods]
    psrc = [p.src for p in snap.pods]
    spawn = {}
    for p in range(P):
        if pnode[p] is None and psrc[p] is not None and pentry[p] > t0:
            spawn.setdefault(pentry[p], []).append(p)
    pdeliv = [None] * P
    standing = [0] * G.n
    inbound = [0] * G.n
    onpair = {}
    for u in range(U):
        if transit[u]:
            inbound[dest[u]] += 1
            onpair[pair[u]] = onpair.get(pair[u], 0) + 1
        elif node[u] is not None:
            standing[node[u]] += 1
    waiting = [[] for _ in range(G.n)]
    for p in range(P):
        if pnode[p] is not None:
            waiting[pnode[p]].append(p)
    for lst in waiting:
        if len(lst) > 1:
            lst.sort(key=lambda p: (pentry[p], pid[p]))
    undelivered = P
    ptr = [0] * U
    first_moves = {}
    mism = 0

    def resolve(s):
        cnt = 0
        for u in range(U):
            if transit[u] or node[u] is None:
                continue
            nd = node[u]
            if carrying[u]:
                keep = []
                for p in carrying[u]:
                    if pdest[p] == nd:
                        pdeliv[p] = s
                        cnt += 1
                    else:
                        keep.append(p)
                carrying[u] = keep
            lst = waiting[nd]
            if lst and len(carrying[u]) < cap[u]:
                while lst and len(carrying[u]) < cap[u]:
                    p = lst.pop(0)
                    pnode[p] = None
                    carrying[u].append(p)
        return cnt

    s = t0
    while s < horizon:
        if s > t0:
            if s in spawn:
                for p in spawn[s]:
                    pnode[p] = psrc[p]
                    lst = waiting[psrc[p]]
                    lst.append(p)
                    if len(lst) > 1:
                        lst.sort(key=lambda q: (pentry[q], pid[q]))
            undelivered -= resolve(s)
        for u in range(U):
            if transit[u] or node[u] is None:
                continue
            if s == t0 and polled[u]:
                continue
            r = routes[u]
            k = ptr[u]
            if k >= len(r):
                continue
            sd, a, b, w = r[k]
            if sd > s:
                continue
            if a != node[u]:
                mism += 1
                ptr[u] = len(r)
                continue
            e = G.pair_edge.get((a, b))
            ok = e is not None
            if ok and G.ecap[e] < UNLIMITED:
                occ = 0
                for pp in G.edge_pairs[e]:
                    occ += onpair.get(pp, 0)
                if occ >= G.ecap[e]:
                    ok = False
            if ok and G.ncap[b] < UNLIMITED and standing[b] + inbound[b] >= G.ncap[b]:
                ok = False
            if ok:
                transit[u] = True
                dest[u] = b
                rem[u] = w
                pair[u] = (a, b)
                standing[a] -= 1
                inbound[b] += 1
                onpair[(a, b)] = onpair.get((a, b), 0) + 1
                ptr[u] += 1
                if s == t0:
                    first_moves[u] = b
            else:
                mism += 1
        moving = False
        for u in range(U):
            if transit[u]:
                rem[u] -= 1
                if rem[u] <= 0:
                    inbound[dest[u]] -= 1
                    onpair[pair[u]] -= 1
                    node[u] = dest[u]
                    standing[node[u]] += 1
                    transit[u] = False
                    pair[u] = None
                    dest[u] = None
                else:
                    moving = True
        undelivered -= resolve(s)
        s += 1
        if undelivered <= 0:
            break
        if not moving and all(ptr[u] >= len(routes[u]) for u in range(U)) and \
                not any(k > s for k in spawn):
            break
    score = 0.0
    for p in range(P):
        if pdeliv[p] is not None:
            score += 100.0 * math.exp(-(pdeliv[p] - pentry[p]) / 50.0)
    return score, first_moves, mism, pdeliv


# ===========================================================================
# 7. Plan evaluation and local search
# ===========================================================================
class Evaluation(object):
    __slots__ = ("value", "score", "first_moves", "mism", "finish", "final",
                 "routes", "pdeliv")


def evaluate(G, snap, plan, horizon):
    routes, finish, final, infeasible = build_routes(G, snap, plan, horizon)
    score, first_moves, mism, pdeliv = simulate(G, snap, routes, horizon)
    ev = Evaluation()
    ev.score = score
    ev.first_moves = first_moves
    ev.mism = mism
    ev.finish = finish
    ev.final = final
    ev.routes = routes
    ev.pdeliv = pdeliv
    t0 = snap.t0
    tail = 0.0
    park = 0.0
    for j in range(len(snap.units)):
        if plan.stops[j]:
            tail += max(0, finish[j] - t0)
        if final[j] is not None and G.storage:
            best = INF
            for s in G.storage:
                d = G.dist[final[j]][s]
                if d < best:
                    best = d
            if best < INF:
                park += best
    ev.value = score - 0.002 * tail - 0.0005 * park - 0.05 * mism - 5.0 * infeasible
    return ev


def route_len_estimate(G, snap, u, stops):
    """Cheap distance-only length of a stop sequence (for screening)."""
    unit = snap.units[u]
    node = unit.dest if unit.transit else unit.node
    total = unit.rem if unit.transit else 0
    for kind, p in stops:
        pod = snap.pods[p]
        if kind == "P":
            goal = pod.node if pod.node is not None else pod.src
        else:
            goal = pod.dest
        if goal is None or goal == node:
            continue
        d = G.dist[node][goal]
        if d >= INF:
            return INF
        total += d
        node = goal
    return total


def insertion_candidates(G, snap, plan, p):
    """All (unit, stops) ways of inserting pod p, with a cheap estimate."""
    pod = snap.pods[p]
    cands = []
    for u in range(len(snap.units)):
        unit = snap.units[u]
        if unit.node is None:
            continue
        base = [st for st in plan.stops[u] if st[1] != p]
        L = len(base)
        if pod.carrier is not None:
            if pod.carrier != u:
                continue
            for k in range(L + 1):
                st = base[:k] + [("D", p)] + base[k:]
                if stops_feasible(snap, u, st):
                    cands.append((route_len_estimate(G, snap, u, st), u, st))
            continue
        if pod.node is None and pod.src is None:
            continue
        for i in range(L + 1):
            for k in range(i + 1, L + 2):
                st = base[:i] + [("P", p)] + base[i:]
                st = st[:k] + [("D", p)] + st[k:]
                if stops_feasible(snap, u, st):
                    cands.append((route_len_estimate(G, snap, u, st), u, st))
    cands.sort(key=lambda c: (c[0], c[1]))
    return cands


def owner_of(plan, p):
    for u, st in enumerate(plan.stops):
        for kind, q in st:
            if q == p:
                return u
    return None


class Planner(object):
    """Anytime local search over plans for one time step."""

    def __init__(self, G, snap, horizon, rng):
        self.G = G
        self.snap = snap
        self.horizon = horizon
        self.rng = rng
        self.best = None
        self.best_ev = None
        self.evals = 0
        self.deadline = None

    def time_left(self):
        return time.perf_counter() < self.deadline

    def consider(self, plan):
        if not all(stops_feasible(self.snap, u, plan.stops[u])
                   for u in range(len(self.snap.units))):
            return False
        ev = evaluate(self.G, self.snap, plan, self.horizon)
        self.evals += 1
        if self.best_ev is None or ev.value > self.best_ev.value + 1e-9:
            self.best = plan
            self.best_ev = ev
            return True
        return False

    def insert_pod(self, p, max_exact=8):
        cands = insertion_candidates(self.G, self.snap, self.best, p)
        if not cands:
            return
        base_plan = self.best
        base_ev = self.best_ev
        self.best_ev = None
        tried = 0
        for est, u, st in cands:
            if est >= INF:
                break
            if tried >= max_exact and (tried >= 2 * max_exact or not self.time_left()):
                break
            plan = base_plan.copy()
            plan.stops[u] = st
            self.consider(plan)
            tried += 1
        if self.best_ev is None:          # nothing evaluated: keep base
            self.best = base_plan
            self.best_ev = base_ev

    def exhaustive(self, max_evals=900):
        """Enumerate the whole plan space when it is tiny: every assignment
        of waiting pods to units, every capacity-feasible stop order and
        every priority order.  Returns True if it ran to completion."""
        snap = self.snap
        U = len(snap.units)
        pods = list(range(len(snap.pods)))
        if not pods or U == 0:
            return True
        units = [u for u in range(U) if snap.units[u].node is not None]
        waiting = [p for p in pods if snap.pods[p].carrier is None]
        carried = [p for p in pods if snap.pods[p].carrier is not None]
        # rough size check before enumerating
        size = (len(units) ** len(waiting)) * math.factorial(min(U, 4))
        for p in waiting:
            size *= 2
        if size > max_evals or len(waiting) > 4:
            return False
        orders = [list(o) for o in itertools.permutations(range(U))]
        base = [[] for _ in range(U)]
        for p in carried:
            base[snap.pods[p].carrier].append(("D", p))
        count = 0
        for assign in itertools.product(units, repeat=len(waiting)):
            per_unit = [list(base[u]) for u in range(U)]
            for p, u in zip(waiting, assign):
                per_unit[u].append(("P", p))
                per_unit[u].append(("D", p))
            # all capacity-feasible orderings per unit (small lists only)
            options = []
            for u in range(U):
                st = per_unit[u]
                if len(st) <= 1:
                    options.append([st])
                    continue
                seen = set()
                opts = []
                for perm in itertools.permutations(st):
                    ok = True
                    pos = {}
                    for i, x in enumerate(perm):
                        pos[x] = i
                    for kind, p in perm:
                        if kind == "D" and ("P", p) in pos and pos[("P", p)] > pos[("D", p)]:
                            ok = False
                            break
                    if not ok or perm in seen:
                        continue
                    seen.add(perm)
                    if stops_feasible(snap, u, list(perm)):
                        opts.append(list(perm))
                    if len(opts) > 24:
                        break
                options.append(opts or [st])
            for combo in itertools.product(*options):
                for o in orders:
                    if not self.time_left():
                        return False
                    count += 1
                    if count > max_evals:
                        return False
                    self.consider(Plan(o, [list(x) for x in combo]))
        return True

    def improve(self):
        """Local search until the deadline.

        Phase 1: systematic sweeps (relocate / priority / reorder / swap)
        until no single move improves.  Phase 2: large-neighbourhood search -
        rip out 1-3 pods and re-insert them by exact evaluation - with a
        late-acceptance random walk so plateaus can be crossed.
        Returns (improved_any, converged).
        """
        snap = self.snap
        U = len(snap.units)
        pods = list(range(len(snap.pods)))
        improved_any = False
        converged = False
        while self.time_left():
            if self.sweep(pods, U):
                improved_any = True
            else:
                converged = True
                break
        if not pods or not self.time_left():
            return improved_any, converged
        # Phase 2: LNS from the best plan.
        current = self.best
        current_val = self.best_ev.value
        rounds = 0
        while self.time_left():
            rounds += 1
            k = 1 + self.rng.randrange(min(3, len(pods)))
            removed = self.rng.sample(pods, k)
            plan = current.copy()
            for u in range(U):
                plan.stops[u] = [x for x in plan.stops[u] if x[1] not in removed]
            self.rng.shuffle(removed)
            ev = None
            for p in removed:
                plan, ev = self.best_insertion(plan, p)
                if plan is None:
                    break
            if plan is None or ev is None:
                continue
            if ev.value > self.best_ev.value + 1e-9:
                self.best, self.best_ev = plan, ev
                current, current_val = plan, ev.value
                improved_any = True
                converged = False
            elif ev.value >= current_val - 0.25:
                current, current_val = plan, ev.value
            if rounds % 25 == 0:
                current, current_val = self.best, self.best_ev.value
        return improved_any, converged

    def best_insertion(self, plan, p, limit=12):
        """Insert pod p into plan at the best exactly-evaluated position."""
        cands = insertion_candidates(self.G, self.snap, plan, p)
        best = None
        best_ev = None
        tried = 0
        for est, u, st in cands:
            if est >= INF or tried >= limit:
                break
            if tried >= 3 and not self.time_left():
                break
            cand = plan.copy()
            cand.stops[u] = st
            if not stops_feasible(self.snap, u, st):
                continue
            ev = evaluate(self.G, self.snap, cand, self.horizon)
            self.evals += 1
            tried += 1
            if best_ev is None or ev.value > best_ev.value + 1e-9:
                best, best_ev = cand, ev
        return best, best_ev

    def sweep(self, pods, U):
        """One pass over the small neighbourhoods; True if anything improved."""
        snap = self.snap
        progress = False
        # 1. relocate one pod (both stops) to the best position anywhere
        self.rng.shuffle(pods)
        for p in pods:
            if not self.time_left():
                return progress
            cands = insertion_candidates(self.G, snap, self.best, p)
            tried = 0
            for est, u, st in cands:
                if est >= INF or tried >= 10 or not self.time_left():
                    break
                if st == self.best.stops[u]:
                    continue
                plan = self.best.copy()
                for v in range(U):
                    if v != u:
                        plan.stops[v] = [x for x in plan.stops[v] if x[1] != p]
                plan.stops[u] = st
                tried += 1
                if self.consider(plan):
                    progress = True
        # 2. priority orders: all permutations for small fleets
        if U > 1:
            if U <= 4:
                orders = [list(o) for o in itertools.permutations(range(U))]
            else:
                orders = []
                for i in range(U - 1):
                    o = list(self.best.order)
                    o[i], o[i + 1] = o[i + 1], o[i]
                    orders.append(o)
                for _ in range(4):
                    o = list(self.best.order)
                    self.rng.shuffle(o)
                    orders.append(o)
            for o in orders:
                if not self.time_left():
                    return progress
                if o == self.best.order:
                    continue
                plan = self.best.copy()
                plan.order = o
                if self.consider(plan):
                    progress = True
        # 3. intra-unit reorders (move one stop elsewhere)
        for u in range(U):
            st = self.best.stops[u]
            if len(st) < 3:
                continue
            for i in range(len(st)):
                for k in range(len(st)):
                    if i == k:
                        continue
                    if not self.time_left():
                        return progress
                    lst = list(st)
                    x = lst.pop(i)
                    lst.insert(k, x)
                    plan = self.best.copy()
                    plan.stops[u] = lst
                    if self.consider(plan):
                        progress = True
        # 4. swap two pods between units
        if len(pods) > 1 and U > 1:
            for _ in range(min(20, len(pods) * len(pods))):
                if not self.time_left():
                    return progress
                a, b = self.rng.sample(pods, 2)
                ua = owner_of(self.best, a)
                ub = owner_of(self.best, b)
                if ua is None or ub is None or ua == ub:
                    continue
                plan = self.best.copy()
                plan.stops[ua] = [(k, b if q == a else q) for k, q in plan.stops[ua]]
                plan.stops[ub] = [(k, a if q == b else q) for k, q in plan.stops[ub]]
                if self.consider(plan):
                    progress = True
        return progress


# ===========================================================================
# 8. Controller
# ===========================================================================
class Controller(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.G = None
        self.sig = None
        self.last_t = -1
        self.plan = None           # Plan in snapshot indices of the last step
        self.plan_units = []       # unit ids by index for self.plan
        self.plan_pods = []        # pod ids by index for self.plan
        self.cache_t = None
        self.cache_moves = {}
        self.spent = 0.0
        self.rng = random.Random(20260915)
        self.converged = False
        self.known_pods = set()
        self.calls = 0
        self.frozen = None         # dev-only (FREEZE): unit id -> route list
        self.frozen_ptr = {}
        self.predicted = {}        # dev-only: pod id -> predicted delivery
        self.divergences = 0

    # -- episode handling -------------------------------------------------
    @staticmethod
    def signature(state):
        return (tuple((nd.id, nd.node_type, nd.capacity) for nd in state.nodes),
                tuple((e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional)
                      for e in state.edges),
                tuple((u.id, u.capacity) for u in state.drive_units))

    def ensure_episode(self, state, t):
        sig = self.signature(state)
        if self.G is None or sig != self.sig or t < self.last_t:
            self.reset()
            self.sig = sig
            self.G = Graph(state)
        self.last_t = t

    # -- main entry ---------------------------------------------------------
    def next_move(self, uid, state):
        t = state.current_time_step
        self.ensure_episode(state, t)
        self.calls += 1
        unit = state.get_drive_unit(uid)
        if unit is None or unit.in_transit:
            return None
        if FREEZE and self.frozen is not None:
            return self.frozen_move(uid, t, state, unit)
        if self.cache_t != t:
            start = time.perf_counter()
            try:
                self.replan(state, t, uid)
            finally:
                self.spent += time.perf_counter() - start
            self.cache_t = t
        if FREEZE and self.frozen is not None:
            return self.frozen_move(uid, t, state, unit)
        mv = self.cache_moves.get(uid)
        if mv is None:
            return None
        if valid_move_now(self.G, state, unit, mv):
            return mv
        # Divergence from the plan: force a fresh plan next step.
        self.cache_t = None
        self.converged = False
        return fallback_move(self.G, state, unit)

    def frozen_move(self, uid, t, state, unit):
        route = self.frozen.get(uid, [])
        k = self.frozen_ptr.get(uid, 0)
        if k >= len(route):
            return None
        sd, a, b, w = route[k]
        if sd > t:
            return None
        mv = self.G.ids[b]
        if a != self.G.idx.get(unit.current_node) or \
                not valid_move_now(self.G, state, unit, mv):
            self.divergences += 1
            return None
        self.frozen_ptr[uid] = k + 1
        return mv

    # -- planning -----------------------------------------------------------
    def budget_for(self, new_pods, first):
        if self.spent >= TOTAL_BUDGET:
            return 0.01
        if first:
            b = BUDGET_FIRST
        elif new_pods:
            b = BUDGET_EVENT
        elif not self.converged:
            b = BUDGET_STEADY
        else:
            b = BUDGET_CONVERGED
        remaining = TOTAL_BUDGET - self.spent
        return min(b, max(0.01, remaining / 20.0))

    def replan(self, state, t, first_uid):
        G = self.G
        snap = Snapshot(G, state, t, first_uid)
        horizon = t + HORIZON_MAX
        pod_ids = set(p.id for p in snap.pods)
        new_pods = set(pid for pid in pod_ids if pid not in self.known_pods)
        first = self.plan is None
        self.known_pods |= pod_ids
        for p in snap.pods:
            if p.id in new_pods and p.node is not None:
                G.spawn_freq[p.node] = G.spawn_freq.get(p.node, 0) + 1
        budget = self.budget_for(bool(new_pods), first)
        planner = Planner(G, snap, horizon, self.rng)
        planner.deadline = time.perf_counter() + budget

        # Warm start: previous plan translated to the current snapshot.
        plan = self.translate_plan(snap)
        planner.consider(plan)
        if planner.best is None:
            planner.best = plan
            planner.best_ev = evaluate(G, snap, plan, horizon)

        # Insert pods not yet in the plan (new or dropped), earliest first.
        assigned = set()
        for st in plan.stops:
            for kind, p in st:
                assigned.add(p)
        todo = [p for p in range(len(snap.pods)) if p not in assigned]
        todo.sort(key=lambda p: (snap.pods[p].entry, snap.pods[p].id))
        for p in todo:
            planner.insert_pod(p)

        planner.exhaustive()
        improved, converged = planner.improve()
        if new_pods or improved:
            self.converged = False
        elif converged:
            self.converged = True

        best = planner.best
        self.plan = best
        self.plan_units = [u.id for u in snap.units]
        self.plan_pods = [p.id for p in snap.pods]
        moves = {}
        for u, nb in planner.best_ev.first_moves.items():
            moves[snap.units[u].id] = G.ids[nb]
        self.cache_moves = moves
        self.predicted = {}
        for p, d in enumerate(planner.best_ev.pdeliv):
            if d is not None:
                self.predicted[snap.pods[p].id] = d
        if FREEZE:
            self.frozen = {}
            for j, u in enumerate(snap.units):
                self.frozen[u.id] = list(planner.best_ev.routes[j])
            self.frozen_ptr = {}
        if DEBUG:
            print("[t=%d] evals=%d score=%.2f value=%.3f mism=%d budget=%.2f moves=%s stops=%s order=%s" % (
                t, planner.evals, planner.best_ev.score, planner.best_ev.value,
                planner.best_ev.mism, budget, moves,
                [[(k, snap.pods[p].id) for k, p in st] for st in best.stops], best.order))

    def translate_plan(self, snap):
        """Map the previous plan (unit ids / pod ids) onto the new snapshot."""
        U = len(snap.units)
        stops = [[] for _ in range(U)]
        order = list(range(U))
        if self.plan is not None:
            new_order = []
            for u in self.plan.order:
                if u < len(self.plan_units):
                    uid = self.plan_units[u]
                    if uid in snap.uidx and snap.uidx[uid] not in new_order:
                        new_order.append(snap.uidx[uid])
            for u in range(U):
                if u not in new_order:
                    new_order.append(u)
            order = new_order
            for u, st in enumerate(self.plan.stops):
                if u >= len(self.plan_units):
                    continue
                uid = self.plan_units[u]
                if uid not in snap.uidx:
                    continue
                nu = snap.uidx[uid]
                for kind, p in st:
                    if p < len(self.plan_pods):
                        pid = self.plan_pods[p]
                        if pid in snap.pidx:
                            stops[nu].append((kind, snap.pidx[pid]))
        # Normalise against reality.
        for p, pod in enumerate(snap.pods):
            if pod.carrier is not None:
                c = pod.carrier
                for u in range(U):
                    stops[u] = [x for x in stops[u]
                                if x[1] != p or (u == c and x[0] == "D")]
                if ("D", p) not in stops[c]:
                    stops[c].append(("D", p))
            else:
                owner = None
                for u in range(U):
                    if any(x[1] == p for x in stops[u]):
                        owner = u
                        break
                if owner is None:
                    continue
                st = stops[owner]
                pi = next((i for i, x in enumerate(st) if x == ("P", p)), None)
                di = next((i for i, x in enumerate(st) if x == ("D", p)), None)
                rest = [x for x in st if x[1] != p]
                if pi is None:
                    pi = di if di is not None else 0
                pi_rel = sum(1 for x in st[:pi] if x[1] != p)
                di_rel = sum(1 for x in st[:di] if x[1] != p) if di is not None else pi_rel
                if di_rel < pi_rel:
                    di_rel = pi_rel
                rest.insert(pi_rel, ("P", p))
                rest.insert(di_rel + 1, ("D", p))
                stops[owner] = rest
                for u in range(U):
                    if u != owner:
                        stops[u] = [x for x in stops[u] if x[1] != p]
        for u in range(U):
            if not stops_feasible(snap, u, stops[u]):
                stops[u] = repair_stops(snap, u, stops[u])
        return Plan(order, stops)


def repair_stops(snap, u, stops):
    """Greedy fix of capacity violations: deliver before picking up more."""
    unit = snap.units[u]
    load = set(unit.carrying)
    out = []
    pending = []
    for kind, p in stops:
        if kind == "P":
            if p in load:
                continue
            if len(load) >= unit.cap:
                pending.append(p)
                continue
            load.add(p)
            out.append((kind, p))
        else:
            if p in load:
                load.discard(p)
                out.append((kind, p))
    for p in pending:
        out.append(("P", p))
        out.append(("D", p))
    return out


# ===========================================================================
# Validity check and fallback (mirrors ar_hackathon.utils.routing_utils)
# ===========================================================================
def valid_move_now(G, state, unit, next_id):
    if unit.in_transit or next_id == unit.current_node:
        return False
    a = G.idx.get(unit.current_node)
    b = G.idx.get(next_id)
    if a is None or b is None:
        return False
    k = G.pair_edge.get((a, b))
    if k is None:
        return False
    if G.ecap[k] < UNLIMITED:
        occ = 0
        pairs = G.edge_pairs[k]
        for v in state.drive_units:
            if v.in_transit:
                va = G.idx.get(v.current_node)
                vb = G.idx.get(v.transit_destination)
                if (va, vb) in pairs:
                    occ += 1
        if occ >= G.ecap[k]:
            return False
    if G.ncap[b] < UNLIMITED:
        occ = 0
        for v in state.drive_units:
            if v.in_transit:
                if G.idx.get(v.transit_destination) == b:
                    occ += 1
            elif G.idx.get(v.current_node) == b:
                occ += 1
        if occ >= G.ncap[b]:
            return False
    return True


def fallback_move(G, state, unit):
    """Greedy shortest-path step toward the most sensible target."""
    a = G.idx.get(unit.current_node)
    if a is None:
        return None
    target = None
    if unit.carrying:
        pod = state.get_pod(unit.carrying[0])
        if pod is not None:
            target = G.idx.get(pod.destination_station)
    else:
        best = INF
        for pod in state.active_pods:
            if pod.carried_by is None and pod.current_node is not None:
                n = G.idx.get(pod.current_node)
                if n is not None and G.dist[a][n] < best:
                    best = G.dist[a][n]
                    target = n
    if target is None or target == a:
        return None
    for nb, w, _ in G.out[a]:
        if w + G.dist[nb][target] == G.dist[a][target]:
            if valid_move_now(G, state, unit, G.ids[nb]):
                return G.ids[nb]
    return None


CONTROLLER = Controller()


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    This is the function that students will implement. The game engine will
    call this function for each idle drive unit at each time step to
    determine where it should go next.

    Pickups and deliveries are automatic: a drive unit with free capacity
    that stops at (or passes through) a node with a waiting pod picks it up,
    and a drive unit that reaches a carried pod's destination station drops
    it off.

    Args:
        drive_unit_id: ID of the drive unit being routed
        state: GraphState object containing the current state of the floor

    Returns:
        next_node_id: ID of an adjacent node to move to, or None to wait
                      at the current node
    """
    try:
        return CONTROLLER.next_move(drive_unit_id, state)
    except Exception:
        if DEBUG:
            raise
        try:
            unit = state.get_drive_unit(drive_unit_id)
            if unit is None or unit.in_transit or CONTROLLER.G is None:
                return None
            return fallback_move(CONTROLLER.G, state, unit)
        except Exception:
            return None
