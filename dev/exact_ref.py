#!/usr/bin/env python3
"""Dev-only reference: exhaustive plan-space search instead of the local search.

Monkeypatches ar_hackathon.api.routing so that, whenever the current instance is
small (<= PMAX known pods, <= UMAX units), Planner.improve enumerates EVERY plan
in the planner's own plan space - every assignment of waiting pods to units,
every capacity-feasible ordering of each unit's stops (a 'P' anywhere before
its 'D'; carried pods keep a single 'D' on their carrier) and every priority
order - scores each one with routing.evaluate and keeps the best value.  The
greedy/warm-start plan is kept as a candidate, so the step is never worse than
what the planner had before improve().  Bigger instances fall back to the
original local search.  The engine's 1-second timeout is replaced with a
passthrough and the planner's total budget is disabled.

Usable in two ways (see dev/README.md):
  * CLI:    python3 dev/exact_ref.py [paths...] [--jobs N] [--pmax 6] [--cap 120000]
                                     [--json out.json] [--baseline dev/run_v4.json]
  * driver: python3 dev/bench.py --driver dev/exact_ref.py   (env EXACT_PMAX / EXACT_CAP / EXACT_COMPARE_LS)
--compare-ls additionally runs the original local search at every enumerated step from
the same greedy start (its result is discarded) and counts the steps where it stops
short of the exhaustive optimum - a per-step measure that is free of trajectory noise.
The patches are applied at import time.  Nothing under ar_hackathon/ is modified.
"""
import argparse
import functools
import itertools
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "dev")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ar_hackathon.api.routing as routing  # noqa: E402
import ar_hackathon.engine.game_engine as game_engine  # noqa: E402

PMAX = int(os.environ.get("EXACT_PMAX", 6))          # enumerate when <= PMAX known pods
UMAX = int(os.environ.get("EXACT_UMAX", 3))          # ... and <= UMAX units
EVAL_CAP = int(os.environ.get("EXACT_CAP", 120000))  # evaluations per step
COMPARE_LS = bool(int(os.environ.get("EXACT_COMPARE_LS", 0)))  # also run the local search per step
EPS = 1e-9
MAX_EVENTS = 40


# ----------------------------------------------------------------------------- stats
def new_stats():
    return dict(steps=0, enum_steps=0, empty_steps=0, big_steps=0, improved_steps=0,
                evals=0, max_evals=0, cap_hits=0, max_space=0, max_pods=0, max_pods_big=0,
                gain_sum=0.0, gain_max=0.0, wall=0.0, events=[], errors=[], infeasible=0,
                ls_steps=0, ls_stuck=0, ls_gap_value=0.0, ls_gap_score=0.0, ls_gap_max=0.0,
                ls_better=0, ls_wall=0.0, ls_events=[])


STATS = new_stats()


# ----------------------------------------------------------------------------- plan space
@functools.lru_cache(maxsize=None)
def n_orderings(cap, onboard, pending):
    """Number of capacity-feasible stop sequences for one unit.

    onboard = pods on the unit that still need a 'D' (carried + already picked
    in the sequence), pending = pods still needing 'P' then 'D'.
    """
    if not onboard and not pending:
        return 1
    n = 0
    if onboard:
        n += onboard * n_orderings(cap, onboard - 1, pending)
    if pending and onboard < cap:
        n += pending * n_orderings(cap, onboard + 1, pending - 1)
    return n


def orderings(cap, carried, new):
    """Yield every capacity-feasible stop list: 'D' for each carried pod, 'P'
    then 'D' for each new pod, never more than cap pods on board."""
    seq = []

    def rec(onboard, pending):
        if not onboard and not pending:
            yield list(seq)
            return
        for i, p in enumerate(onboard):
            seq.append(("D", p))
            yield from rec(onboard[:i] + onboard[i + 1:], pending)
            seq.pop()
        if len(onboard) < cap:
            for i, p in enumerate(pending):
                seq.append(("P", p))
                yield from rec(onboard + (p,), pending[:i] + pending[i + 1:])
                seq.pop()

    yield from rec(tuple(carried), tuple(new))


def exhaustive(self):
    """Replacement for Planner.improve on small instances."""
    G, snap, horizon = self.G, self.snap, self.horizon
    st = STATS
    U, P = len(snap.units), len(snap.pods)
    active = [u for u in range(U) if snap.units[u].node is not None]
    inactive = [u for u in range(U) if snap.units[u].node is None]
    carried = [[] for _ in range(U)]
    free = []
    for p, pod in enumerate(snap.pods):
        if pod.carrier is not None:
            carried[pod.carrier].append(p)
        elif pod.node is not None or pod.src is not None:
            free.append(p)
    if not active:
        return False, True
    orders = [list(o) + inactive for o in itertools.permutations(active)]
    assigns = list(itertools.product(active, repeat=len(free)))
    space = 0
    for a in assigns:
        k = [0] * U
        for u in a:
            k[u] += 1
        prod = 1
        for u in active:
            prod *= n_orderings(snap.units[u].cap, len(carried[u]), k[u])
        space += prod
    space *= len(orders)
    random.Random(snap.t0 * 1000003 + 7919 * P + len(free)).shuffle(assigns)

    prev_ev = self.best_ev
    best_plan, best_ev = self.best, self.best_ev
    stops = [None] * U
    for u in inactive:                      # never routed; keep their D stops for the record
        stops[u] = [("D", p) for p in carried[u]]
    state = dict(evals=0, capped=False)

    def leaf():
        nonlocal best_plan, best_ev
        if not all(routing.stops_feasible(snap, u, stops[u]) for u in active):
            st["infeasible"] += 1
            return
        for order in orders:
            if state["evals"] >= EVAL_CAP:
                state["capped"] = True
                return
            plan = routing.Plan(list(order), [list(s) for s in stops])
            ev = routing.evaluate(G, snap, plan, horizon)
            state["evals"] += 1
            if ev.value > best_ev.value + EPS:
                best_plan, best_ev = plan, ev

    def rec(i, per):
        if i == len(active):
            leaf()
            return
        u = active[i]
        for seq in orderings(snap.units[u].cap, carried[u], per[u]):
            stops[u] = seq
            rec(i + 1, per)
            if state["capped"]:
                return

    for a in assigns:
        per = [[] for _ in range(U)]
        for p, u in zip(free, a):
            per[u].append(p)
        rec(0, per)
        if state["capped"]:
            break

    evals = state["evals"]
    self.evals += evals
    st["evals"] += evals
    st["max_evals"] = max(st["max_evals"], evals)
    st["max_space"] = max(st["max_space"], space)
    if state["capped"]:
        st["cap_hits"] += 1
    improved = best_ev is not prev_ev and best_ev.value > prev_ev.value + EPS
    if improved:
        self.best, self.best_ev = best_plan, best_ev
        gain = best_ev.value - prev_ev.value
        st["improved_steps"] += 1
        st["gain_sum"] += gain
        st["gain_max"] = max(st["gain_max"], gain)
        if len(st["events"]) < MAX_EVENTS:
            st["events"].append(dict(t=snap.t0, pods=P, free=len(free), evals=evals, space=space,
                                     capped=state["capped"], prev=round(prev_ev.value, 4),
                                     new=round(best_ev.value, 4),
                                     prev_score=round(prev_ev.score, 4), new_score=round(best_ev.score, 4)))
    return improved, True


def improve(self):
    """Patched Planner.improve: exhaustive on small instances, original otherwise."""
    snap = self.snap
    st = STATS
    st["steps"] += 1
    P, U = len(snap.pods), len(snap.units)
    if P > PMAX or U > UMAX:
        st["big_steps"] += 1
        st["max_pods_big"] = max(st["max_pods_big"], P)
        return _orig_improve(self)
    if P == 0:
        st["empty_steps"] += 1
    else:
        st["enum_steps"] += 1
        st["max_pods"] = max(st["max_pods"], P)
    t0 = time.perf_counter()
    try:
        if COMPARE_LS and P > 0:
            # Same greedy start, the planner's own budget: what does the local search reach?
            g_plan, g_ev = self.best, self.best_ev
            _orig_improve(self)
            ls_ev = self.best_ev
            self.best, self.best_ev = g_plan, g_ev
            st["ls_steps"] += 1
            st["ls_wall"] += time.perf_counter() - t0
            r = exhaustive(self)
            ex_ev = self.best_ev
            if ex_ev.value > ls_ev.value + 1e-6:
                st["ls_stuck"] += 1
                st["ls_gap_value"] += ex_ev.value - ls_ev.value
                st["ls_gap_score"] += ex_ev.score - ls_ev.score
                st["ls_gap_max"] = max(st["ls_gap_max"], ex_ev.score - ls_ev.score)
                if len(st["ls_events"]) < MAX_EVENTS:
                    st["ls_events"].append(dict(t=snap.t0, pods=P, ls=round(ls_ev.value, 4), ex=round(ex_ev.value, 4),
                                                ls_score=round(ls_ev.score, 4), ex_score=round(ex_ev.score, 4)))
            elif ls_ev.value > ex_ev.value + 1e-6:
                st["ls_better"] += 1          # a plan outside the enumerated space
            return r
        return exhaustive(self)
    except Exception:
        # routing.drive_unit_next_move swallows exceptions; keep them visible.
        st["errors"].append(traceback.format_exc())
        raise
    finally:
        st["wall"] += time.perf_counter() - t0


# ----------------------------------------------------------------------------- patches
def _passthrough(func, *args, timeout_seconds=10, default_return_value=None, **kwargs):
    return func(*args, **kwargs)


_orig_improve = routing.Planner.improve
_orig_safe_execute = game_engine.safe_execute


def install():
    routing.Planner.improve = improve
    routing.TOTAL_BUDGET = 1e9
    game_engine.safe_execute = _passthrough


install()
drive_unit_next_move = routing.drive_unit_next_move


# ----------------------------------------------------------------------------- running
def _task(args):
    path, pmax, cap, compare_ls = args
    global PMAX, EVAL_CAP, COMPARE_LS
    PMAX, EVAL_CAP, COMPARE_LS = pmax, cap, compare_ls
    import bench
    logging.getLogger("ar_hackathon.engine.game_engine").setLevel(logging.ERROR)
    STATS.clear()
    STATS.update(new_stats())
    rec = bench.run_case(path, drive_unit_next_move)
    rec["exact"] = json.loads(json.dumps(STATS))
    return rec


def run_all(paths, jobs, pmax, cap, baseline, compare_ls=False):
    tasks = [(p, pmax, cap, compare_ls) for p in paths]
    recs = []

    def progress(r):
        x = r["exact"]
        base = baseline.get(r["path"]) if baseline else None
        d = "   new " if base is None else f"{r['score'] - base:+7.2f}"
        flag = " CAP" if x["cap_hits"] else ""
        err = "  ERROR" if r.get("error") or x["errors"] else ""
        print(f"  {('-' if base is None else f'{base:6.2f}'):>6} -> {r['score']:6.2f} {d}  enum {x['enum_steps']:3d}"
              f"  impr {x['improved_steps']:3d}  maxev {x['max_evals']:6d}  space {x['max_space']:7d}{flag}"
              f"  {r['wall']:6.1f}s  {r['path']}{err}", flush=True)

    if jobs <= 1 or len(paths) <= 1:
        for t in tasks:
            r = _task(t)
            progress(r)
            recs.append(r)
    else:
        with mp.get_context("spawn").Pool(jobs) as pool:
            for r in pool.imap_unordered(_task, tasks, chunksize=1):
                progress(r)
                recs.append(r)
    return sorted(recs, key=lambda r: r["path"])


# ----------------------------------------------------------------------------- reporting
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def summarize(recs, baseline):
    import bench
    levels = [lv for lv in [*bench.LEVELS, "other"] if any(r["level"] == lv for r in recs)]
    per_level = {}
    for lv in levels:
        rs = [r for r in recs if r["level"] == lv]
        m = [r for r in rs if r.get("online") is not None]
        per_level[lv] = dict(
            n=len(rs), online=mean(r["online"] for r in m) if m else None,
            exact=mean(r["score"] for r in rs), delta=mean(r["delta"] for r in m) if m else None,
            bound=mean(r["bound2"] for r in rs),
            gained=sum(1 for r in m if r["delta"] > 1e-6), lost=sum(1 for r in m if r["delta"] < -1e-6),
            enum_steps=sum(r["exact"]["enum_steps"] for r in rs),
            improved_steps=sum(r["exact"]["improved_steps"] for r in rs),
            ls_steps=sum(r["exact"]["ls_steps"] for r in rs), ls_stuck=sum(r["exact"]["ls_stuck"] for r in rs),
            ls_gap_score=sum(r["exact"]["ls_gap_score"] for r in rs),
            ls_better=sum(r["exact"]["ls_better"] for r in rs),
            cap_hits=sum(r["exact"]["cap_hits"] for r in rs),
            max_evals=max(r["exact"]["max_evals"] for r in rs),
            wall=sum(r["wall"] for r in rs))
    m = [r for r in recs if r.get("online") is not None]
    return dict(
        n=len(recs), matched=len(m), online=mean(r["online"] for r in m) if m else None,
        exact=mean(r["score"] for r in recs), exact_matched=mean(r["score"] for r in m) if m else None,
        delta=mean(r["delta"] for r in m) if m else None, net=sum(r["delta"] for r in m),
        gained=sum(1 for r in m if r["delta"] > 1e-6), lost=sum(1 for r in m if r["delta"] < -1e-6),
        per_level=per_level,
        top_gains=[r["path"] for r in sorted(m, key=lambda r: -r["delta"]) if r["delta"] > 1e-6][:10],
        top_losses=[r["path"] for r in sorted(m, key=lambda r: r["delta"]) if r["delta"] < -1e-6][:10],
        cap_hit_cases=[r["path"] for r in recs if r["exact"]["cap_hits"]],
        error_cases=[r["path"] for r in recs if r.get("error") or r["exact"]["errors"]],
        enum_steps=sum(r["exact"]["enum_steps"] for r in recs),
        big_steps=sum(r["exact"]["big_steps"] for r in recs),
        ls_steps=sum(r["exact"]["ls_steps"] for r in recs), ls_stuck=sum(r["exact"]["ls_stuck"] for r in recs),
        ls_gap_score=sum(r["exact"]["ls_gap_score"] for r in recs),
        ls_better=sum(r["exact"]["ls_better"] for r in recs),
        ls_stuck_cases=[r["path"] for r in sorted(recs, key=lambda r: -r["exact"]["ls_gap_score"])
                        if r["exact"]["ls_stuck"]],
        evals=sum(r["exact"]["evals"] for r in recs),
        max_evals=max(r["exact"]["max_evals"] for r in recs))


def report(recs, agg, total_wall, pmax, cap):
    import bench
    by = {r["path"]: r for r in recs}
    cols = [("case", lambda r: r["path"]),
            ("online", lambda r: f2(r.get("online"))), ("exact", lambda r: f2(r["score"])),
            ("delta", lambda r: "new" if r.get("delta") is None else f"{r['delta']:+.2f}"),
            ("bound", lambda r: f2(r["bound2"])),
            ("deliv", lambda r: f"{r['delivered']}/{r['total']}"),
            ("enum", lambda r: str(r["exact"]["enum_steps"])), ("big", lambda r: str(r["exact"]["big_steps"])),
            ("impr", lambda r: str(r["exact"]["improved_steps"])),
            ("gain_v", lambda r: f2(r["exact"]["gain_sum"])),
            ("max_ev", lambda r: str(r["exact"]["max_evals"])), ("max_space", lambda r: str(r["exact"]["max_space"])),
            ("cap", lambda r: "HIT" if r["exact"]["cap_hits"] else "-"),
            ("wall_s", lambda r: f2(r["wall"]))]
    print(f"\n== Per case (online = baseline score, exact = exhaustive plan space with pmax={pmax}, cap={cap};"
          " enum/big = steps enumerated / handed to the local search; impr = steps where enumeration beat the"
          " greedy plan; gain_v = summed replica value gain)")
    print(bench.table(recs, cols))
    lv_cols = [("level", lambda r: r[0]), ("n", lambda r: str(r[1]["n"])),
               ("online", lambda r: f2(r[1]["online"])), ("exact", lambda r: f2(r[1]["exact"])),
               ("delta", lambda r: "-" if r[1]["delta"] is None else f"{r[1]['delta']:+.3f}"),
               ("bound", lambda r: f2(r[1]["bound"])),
               ("gained", lambda r: str(r[1]["gained"])), ("lost", lambda r: str(r[1]["lost"])),
               ("enum", lambda r: str(r[1]["enum_steps"])), ("impr", lambda r: str(r[1]["improved_steps"])),
               ("cap_hits", lambda r: str(r[1]["cap_hits"])), ("max_ev", lambda r: str(r[1]["max_evals"])),
               ("wall_s", lambda r: f2(r[1]["wall"]))]
    if agg["ls_steps"]:
        lv_cols[9:9] = [("ls_stuck", lambda r: f"{r[1]['ls_stuck']}/{r[1]['ls_steps']}"),
                        ("ls_gap", lambda r: f2(r[1]["ls_gap_score"]))]
    print("\n== Per level (means; gained/lost = cases with delta > / < 0; enum/impr/cap_hits = step sums)")
    print(bench.table(list(agg["per_level"].items()), lv_cols))
    delta = "-" if agg["delta"] is None else f"{agg['delta']:+.3f}"
    print(f"\n== Overall: {agg['n']} cases ({agg['matched']} with baseline)  mean online {f2(agg['online'])}"
          f"  mean exact {f2(agg['exact_matched'])}  mean delta {delta}"
          f"  net {agg['net']:+.2f}  gained {agg['gained']}  lost {agg['lost']}")
    print(f"   enumerated steps {agg['enum_steps']}, steps handed to local search {agg['big_steps']},"
          f" evaluations {agg['evals']}, max evaluations in a step {agg['max_evals']},"
          f" cap hit in {len(agg['cap_hit_cases'])} cases")
    if agg["ls_steps"]:
        print(f"   local search vs exhaustive at the same step: stuck in a local optimum on {agg['ls_stuck']} of"
              f" {agg['ls_steps']} steps (summed replica-score gap {agg['ls_gap_score']:.2f}),"
              f" local search strictly better (plan outside the enumerated space) on {agg['ls_better']} steps")
        for p in agg["ls_stuck_cases"][:10]:
            x = by[p]["exact"]
            print(f"     {x['ls_stuck']:2d} stuck steps, gap {x['ls_gap_score']:7.2f} (max {x['ls_gap_max']:.2f})  {p}: " +
                  "; ".join(f"t={e['t']} P={e['pods']} {e['ls_score']:.2f}->{e['ex_score']:.2f}" for e in x["ls_events"][:4]))
    print("\n== Largest gains")
    for p in agg["top_gains"]:
        r = by[p]
        print(f"  {r['delta']:+6.2f}  ({f2(r['online'])} -> {f2(r['score'])}, bound {f2(r['bound2'])})  {p}")
    if not agg["top_gains"]:
        print("  none")
    print("\n== Largest losses")
    for p in agg["top_losses"]:
        r = by[p]
        print(f"  {r['delta']:+6.2f}  ({f2(r['online'])} -> {f2(r['score'])}, bound {f2(r['bound2'])})  {p}")
        for e in r["exact"]["events"][:6]:
            print(f"      t={e['t']:3d} pods={e['pods']} free={e['free']} evals={e['evals']:6d}/{e['space']:6d}"
                  f" value {e['prev']:.3f} -> {e['new']:.3f} (score {e['prev_score']:.3f} -> {e['new_score']:.3f})")
    if not agg["top_losses"]:
        print("  none")
    for p in agg["cap_hit_cases"]:
        x = by[p]["exact"]
        print(f"!!! CAP HIT in {x['cap_hits']} step(s), max space {x['max_space']}: {p}")
    for p in agg["error_cases"]:
        r = by[p]
        print(f"!!! ERROR in {p}:\n{r.get('error') or ''}{''.join(r['exact']['errors'][:1])}")
    print(f"\ntotal wall time {total_wall:.1f}s")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Exhaustive plan-space reference for small instances.")
    ap.add_argument("paths", nargs="*", help="case files, directories or globs (default: test_cases + dev/cases levels, no stress)")
    ap.add_argument("--jobs", type=int, default=1, metavar="N")
    ap.add_argument("--pmax", type=int, default=6, help="enumerate when at most this many pods are known")
    ap.add_argument("--cap", type=int, default=120000, help="max evaluations per step")
    ap.add_argument("--compare-ls", action="store_true",
                    help="at every enumerated step also run the original local search from the same greedy start"
                         " (with its normal budget) and record when it falls short of the exhaustive optimum")
    ap.add_argument("--json", metavar="OUT", help="write per-case records + aggregate")
    ap.add_argument("--baseline", default=os.path.join("dev", "run_v4.json"), metavar="PREV",
                    help="bench --json output with the online planner's scores")
    args = ap.parse_args()
    import bench
    paths = bench.collect_cases(args.paths or [os.path.join(ROOT, "test_cases", "level*", "*.json"),
                                               os.path.join(ROOT, "dev", "cases", "level*", "*.json")])
    if not paths:
        sys.exit("exact_ref: no test cases found")
    baseline = {}
    bpath = os.path.join(ROOT, args.baseline) if not os.path.isabs(args.baseline) else args.baseline
    if os.path.isfile(bpath):
        with open(bpath) as f:
            baseline = {c["path"]: c["score"] for c in json.load(f)["cases"]}
    else:
        print(f"warning: baseline {args.baseline!r} not found; no deltas", file=sys.stderr)
    print(f"exact_ref: {len(paths)} cases, jobs={args.jobs}, pmax={args.pmax}, umax={UMAX}, cap={args.cap},"
          f" baseline={args.baseline if baseline else None}")
    t0 = time.perf_counter()
    recs = run_all(paths, max(1, args.jobs), args.pmax, args.cap, baseline, args.compare_ls)
    total_wall = time.perf_counter() - t0
    for r in recs:
        r["online"] = baseline.get(r["path"])
        r["delta"] = None if r["online"] is None else r["score"] - r["online"]
    agg = summarize(recs, baseline)
    report(recs, agg, total_wall, args.pmax, args.cap)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(driver="dev/exact_ref.py", pmax=args.pmax, umax=UMAX, cap=args.cap, compare_ls=args.compare_ls,
                           baseline=args.baseline if baseline else None,
                           when=time.strftime("%Y-%m-%dT%H:%M:%S"), total_wall=total_wall,
                           cases=recs, aggregate=agg), f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
