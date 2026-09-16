#!/usr/bin/env python3
"""In-process benchmark / grading loop for the routing solution. See dev/README.md."""
import argparse
import functools
import glob
import heapq
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from ar_hackathon.utils.json_loader import load_test_case

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ar_hackathon.engine.game_engine import GameEngine  # noqa: E402

LEVELS = ("level1", "level2", "level3", "stress")
SLOW_CALL, SLOW_CASE, TIMEOUT = 0.5, 60.0, 1.0  # seconds
REGRESSION = 0.5  # points


# ----------------------------------------------------------------------------- cases
def collect_cases(args):
    """Expand files / directories / globs into sorted case paths relative to ROOT."""
    if not args:
        args = [os.path.join(ROOT, "test_cases", "level*", "*.json")]
        if os.path.isdir(os.path.join(ROOT, "dev", "cases")):
            args.append(os.path.join(ROOT, "dev", "cases", "**", "*.json"))
    found = set()
    for a in args:
        hits = glob.glob(a, recursive=True) or [a]
        for p in hits:
            if os.path.isdir(p):
                found.update(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
            elif os.path.isfile(p):
                found.add(p)
            else:
                print(f"warning: nothing matches {a!r}", file=sys.stderr)
    paths = {os.path.relpath(os.path.abspath(p), ROOT) for p in found
             if os.path.basename(p) != "schema.json"}
    return sorted(paths)


def level_of(path):
    parent = os.path.basename(os.path.dirname(path))
    return next((lv for lv in LEVELS if parent.startswith(lv)), "other")


# ----------------------------------------------------------------------------- bounds
def dijkstra(adj, src):
    dist, heap = {src: 0}, [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj.get(u, ()):
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return dist


def upper_bounds(tc):
    """Optimistic scores no solution can beat: (bound, bound2).

    lb  = D - 1, D = shortest travel time source -> station (a unit departing at
          step t on an edge of weight w arrives and delivers during step t+w-1).
    lb2 = lb + max(0, min_unit dist(unit_start, source) - entry_time): the
          nearest unit may pre-position during the pod's entry delay.
    A pod that cannot be delivered before max_time_steps contributes 0.
    """
    adj = {}
    for e in tc.edges:
        w = math.ceil(e.weight)  # engine decrements by 1 per step, arrives at <= 0
        adj.setdefault(e.from_node, []).append((e.to_node, w))
        if e.bidirectional:
            adj.setdefault(e.to_node, []).append((e.from_node, w))
    fetch = {}
    for u in tc.drive_units:
        for n, d in dijkstra(adj, u.current_node).items():
            fetch[n] = min(fetch.get(n, math.inf), d)
    from_src, pts, pts2, n, last = {}, 0.0, 0.0, 0, tc.max_time_steps - 1
    for pods in tc.pods_by_time.values():
        for pod in pods:
            n += 1
            if pod.current_node not in from_src:
                from_src[pod.current_node] = dijkstra(adj, pod.current_node)
            lb = max(0, from_src[pod.current_node].get(pod.destination_station, math.inf) - 1)
            lb2 = lb + max(0, fetch.get(pod.current_node, math.inf) - pod.entry_time)
            pts += 100 * math.exp(-lb / 50) if pod.entry_time + lb <= last else 0
            pts2 += 100 * math.exp(-lb2 / 50) if pod.entry_time + lb2 <= last else 0
    return (pts / n, pts2 / n) if n else (0.0, 0.0)


# ----------------------------------------------------------------------------- running
class CallStats:
    def __init__(self):
        self.calls = self.exceptions = self.timeouts = 0
        self.total = self.max = 0.0
        self.first_tb = None

    def wrap(self, fn):
        @functools.wraps(fn)  # safe_execute logs func.__name__
        def wrapped(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            except Exception:
                self.exceptions += 1
                self.first_tb = self.first_tb or traceback.format_exc()
                raise  # engine's safe_execute still sees the exception
            finally:
                dt = time.perf_counter() - t0
                self.calls += 1
                self.total += dt
                self.max = max(self.max, dt)
                self.timeouts += dt >= TIMEOUT
        return wrapped


def load_driver(name):
    """Return (driver, reset_fn). The default driver is never reset: we want to see leaks."""
    if name == "basic":
        from ar_hackathon.examples import basic_driver as mod
        return mod.basic_driver, mod.visited.clear
    if name.endswith(".py"):
        # any routing-like module by path (dev copies of routing.py)
        import importlib.util
        spec = importlib.util.spec_from_file_location("routing_candidate", os.path.join(ROOT, name))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.drive_unit_next_move, None
    from ar_hackathon.api.routing import drive_unit_next_move
    return drive_unit_next_move, None


ORACLE = False


def set_oracle(path):
    """Dev hook: let the planner see every pod of the case in advance."""
    import ar_hackathon.api.routing as routing
    if not ORACLE:
        routing.ORACLE_PODS = None
        return
    tc = load_test_case(os.path.join(ROOT, path))
    routing.ORACLE_PODS = [(p.id, p.current_node, p.destination_station, p.entry_time)
                           for pods in tc.pods_by_time.values() for p in pods]
    routing.BUDGET_FIRST, routing.BUDGET_EVENT = 0.6, 0.4
    routing.BUDGET_STEADY, routing.BUDGET_CONVERGED = 0.25, 0.1
    routing.TOTAL_BUDGET = 200.0


def run_case(path, driver):
    set_oracle(path)
    st, rec, t0 = CallStats(), {"path": path, "level": level_of(path)}, time.perf_counter()
    try:
        engine = GameEngine(os.path.join(ROOT, path), st.wrap(driver))
        s = engine.run_until_finished()
        b1, b2 = upper_bounds(engine.test_case)
        rec.update(score=s["score"], delivered=s["delivered_pods"], total=s["total_pods"],
                   steps=s["total_time_steps"], bound=b1, bound2=b2, gap=b2 - s["score"])
    except Exception:
        rec.update(score=0.0, delivered=0, total=0, steps=0, bound=0.0, bound2=0.0, gap=0.0,
                   error=traceback.format_exc())
    rec.update(wall=time.perf_counter() - t0, calls=st.calls, max_call=st.max,
               mean_call=st.total / st.calls if st.calls else 0.0, exceptions=st.exceptions,
               timeouts=st.timeouts, first_traceback=st.first_tb)
    return rec


def run_cases(paths, driver_name, quiet):
    """Run cases sequentially in this process with the driver module imported once."""
    driver, reset = load_driver(driver_name)
    if quiet:
        logging.getLogger("ar_hackathon.engine.game_engine").setLevel(logging.ERROR)
    out = []
    for p in paths:
        if reset:
            reset()
        rec = run_case(p, driver)
        if not quiet:
            print(f"  {rec['score']:7.2f} / {rec['bound2']:6.2f}  {rec['wall']:7.2f}s  {p}", flush=True)
        out.append(rec)
    return out


def _worker(args):
    global ORACLE
    paths, driver_name, quiet, oracle = args
    ORACLE = oracle
    return run_cases(paths, driver_name, quiet)


def run_all(paths, driver_name, jobs, quiet):
    if jobs <= 1 or len(paths) <= 1:
        return run_cases(paths, driver_name, quiet)
    chunks = [paths[i::jobs] for i in range(jobs) if paths[i::jobs]]
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        parts = pool.map(_worker, [(c, driver_name, quiet, ORACLE) for c in chunks])
    return sorted((r for part in parts for r in part), key=lambda r: r["path"])


# ----------------------------------------------------------------------------- reporting
def f2(x):
    return "-" if x is None else f"{x:.2f}"


def table(rows, cols):
    cells = [[h for h, _ in cols]] + [[f(r) for _, f in cols] for r in rows]
    w = [max(len(c[i]) for c in cells) for i in range(len(cols))]
    return "\n".join("  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i])
                               for i, c in enumerate(row)) for row in cells)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(recs):
    per_level = {}
    for lv in [*LEVELS, "other"]:
        rs = [r for r in recs if r["level"] == lv]
        if rs:
            per_level[lv] = dict(n=len(rs), score=mean(r["score"] for r in rs),
                                 bound=mean(r["bound2"] for r in rs), gap=mean(r["gap"] for r in rs),
                                 delivered=100 * sum(r["delivered"] for r in rs) / max(1, sum(r["total"] for r in rs)),
                                 max_call=max(r["max_call"] for r in rs), wall=mean(r["wall"] for r in rs))
    return dict(
        n=len(recs), mean_score=mean(r["score"] for r in recs), sum_score=sum(r["score"] for r in recs),
        mean_gap=mean(r["gap"] for r in recs), per_level=per_level,
        worst_gap=[r["path"] for r in sorted(recs, key=lambda r: -r["gap"])[:5]],
        worst_call=[r["path"] for r in sorted(recs, key=lambda r: -r["max_call"])[:5]],
        slow_calls=[r["path"] for r in recs if r["max_call"] > SLOW_CALL],
        slow_cases=[r["path"] for r in recs if r["wall"] > SLOW_CASE],
        exceptions=sum(r["exceptions"] for r in recs), timeouts=sum(r["timeouts"] for r in recs),
        errors=[r["path"] for r in recs if r.get("error")])


def report(recs, agg, baseline):
    by = {r["path"]: r for r in recs}
    cols = [("case", lambda r: r["path"]), ("score", lambda r: f2(r["score"])),
            ("bound", lambda r: f2(r["bound2"])), ("bound1", lambda r: f2(r["bound"])),
            ("gap", lambda r: f2(r["gap"])), ("deliv", lambda r: f"{r['delivered']}/{r['total']}"),
            ("steps", lambda r: str(r["steps"])), ("calls", lambda r: str(r["calls"])),
            ("max_ms", lambda r: f2(1e3 * r["max_call"])), ("mean_ms", lambda r: f2(1e3 * r["mean_call"])),
            ("wall_s", lambda r: f2(r["wall"])),
            ("exc/to", lambda r: f"{r['exceptions']}/{r['timeouts']}" if r["exceptions"] or r["timeouts"] else "-")]
    if baseline is not None:
        cols.append(("delta", lambda r: "new" if r.get("delta") is None else f"{r['delta']:+.2f}"))
    print("\n== Per case (bound = fetch-aware upper bound, bound1 = travel-only, gap = bound - score)")
    print(table(recs, cols))
    lv_cols = [("level", lambda r: r[0]), ("n", lambda r: str(r[1]["n"])), ("score", lambda r: f2(r[1]["score"])),
               ("bound", lambda r: f2(r[1]["bound"])), ("gap", lambda r: f2(r[1]["gap"])),
               ("deliv%", lambda r: f2(r[1]["delivered"])), ("max_ms", lambda r: f2(1e3 * r[1]["max_call"])),
               ("wall_s", lambda r: f2(r[1]["wall"]))]
    print("\n== Per level (means; max_ms = max)")
    print(table(list(agg["per_level"].items()), lv_cols))
    print(f"\n== Overall: {agg['n']} cases  mean score {f2(agg['mean_score'])}  sum score {f2(agg['sum_score'])}"
          f"  mean gap {f2(agg['mean_gap'])}  exceptions {agg['exceptions']}  timeouts {agg['timeouts']}")
    print("worst by gap:      " + ", ".join(f"{p} ({f2(by[p]['gap'])})" for p in agg["worst_gap"]))
    print("worst by max call: " + ", ".join(f"{p} ({f2(1e3 * by[p]['max_call'])} ms)" for p in agg["worst_call"]))
    for p in agg["slow_calls"]:
        print(f"!!! SLOW CALL  {f2(1e3 * by[p]['max_call'])} ms (> {int(SLOW_CALL * 1e3)} ms): {p}")
    for p in agg["slow_cases"]:
        print(f"!!! SLOW CASE  {f2(by[p]['wall'])} s (> {int(SLOW_CASE)} s): {p}")
    for p in agg["errors"]:
        print(f"!!! HARNESS/ENGINE ERROR in {p}:\n{by[p]['error']}")
    for r in recs:
        if r["first_traceback"]:
            print(f"!!! driver raised {r['exceptions']}x in {r['path']}; first:\n{r['first_traceback']}")
    if baseline is not None:
        matched = [r for r in recs if r.get("delta") is not None]
        net = sum(r["delta"] for r in matched)
        regressed = [r for r in matched if r["delta"] < -REGRESSION]
        missing = sorted(set(baseline) - set(by))
        print(f"\n== Baseline: net {net:+.2f} over {len(matched)} matched cases "
              f"({len(recs) - len(matched)} new, {len(missing)} missing from this run)")
        for r in regressed:
            print(f"!!! REGRESSED {r['delta']:+.2f} ({f2(r['score'])} vs {f2(baseline[r['path']])}): {r['path']}")
        if not regressed:
            print("no regressions > %.1f points" % REGRESSION)
        agg["baseline"] = dict(net=net, matched=len(matched), regressed=[r["path"] for r in regressed])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Benchmark the routing driver over test cases.")
    ap.add_argument("paths", nargs="*", help="case files, directories or globs (default: test_cases + dev/cases)")
    ap.add_argument("--driver", default="default", help="default | basic | path/to/module.py")
    ap.add_argument("--json", metavar="OUT", help="write per-case records + aggregate to this file")
    ap.add_argument("--baseline", metavar="PREV", help="previous --json output to diff against")
    ap.add_argument("--seq", action="store_true", help="force sequential single-process run")
    ap.add_argument("--oracle", action="store_true", help="planner sees all future pods (reference only)")
    ap.add_argument("--jobs", type=int, default=1, metavar="N", help="worker processes (each runs its share sequentially)")
    ap.add_argument("--quiet", action="store_true", help="no per-case progress lines or engine warnings")
    args = ap.parse_args()

    paths = collect_cases(args.paths)
    if not paths:
        sys.exit("bench: no test cases found")
    jobs = 1 if args.seq else max(1, args.jobs)
    global ORACLE
    ORACLE = args.oracle
    print(f"bench: {len(paths)} cases, driver={args.driver}, jobs={jobs}, oracle={ORACLE}")
    t0 = time.perf_counter()
    recs = run_all(paths, args.driver, jobs, args.quiet)
    total_wall = time.perf_counter() - t0

    baseline = None
    if args.baseline:
        with open(args.baseline) as f:
            baseline = {c["path"]: c["score"] for c in json.load(f)["cases"]}
        for r in recs:
            r["delta"] = r["score"] - baseline[r["path"]] if r["path"] in baseline else None
    agg = summarize(recs)
    report(recs, agg, baseline)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(driver=args.driver, when=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           total_wall=total_wall, cases=recs, aggregate=agg), f, indent=1)
        print(f"wrote {args.json}")
    print(f"total wall time {total_wall:.2f}s")


if __name__ == "__main__":
    main()
