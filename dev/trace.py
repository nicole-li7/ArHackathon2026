#!/usr/bin/env python3
"""Run one case with the current routing.py and print a compact timeline."""
import sys, math, importlib
from ar_hackathon.engine.game_engine import GameEngine
import ar_hackathon.api.routing as routing

def main():
    path = sys.argv[1]
    debug = "--debug" in sys.argv
    routing.DEBUG = debug
    eng = GameEngine(path, routing.drive_unit_next_move)
    tc = eng.test_case
    prev_carry = {}
    while not eng.is_finished:
        gs, fin = eng.step()
        t = gs.current_time_step - 1
        parts = []
        for u in sorted(gs.drive_units, key=lambda u: u.id):
            if u.in_transit:
                parts.append("U%d:%d->%d(%d)%s" % (u.id, u.current_node, u.transit_destination, u.transit_remaining_time, "".join("[%s]" % c for c in u.carrying)))
            else:
                parts.append("U%d:@%d%s" % (u.id, u.current_node, "".join("[%s]" % c for c in u.carrying)))
        waiting = ["%s@%d" % (p.id, p.current_node) for p in gs.active_pods if p.carried_by is None]
        print("t=%3d  %s   waiting: %s" % (t, "  ".join(parts), " ".join(waiting)))
    st = eng.stats
    print("score %.2f  delivered %d/%d  steps %d" % (st["score"], st["delivered_pods"], st["total_pods"], st["total_time_steps"]))
    for p in sorted(eng.graph_state.delivered_pods, key=lambda p: p.entry_time):
        d = p.delivery_time - p.entry_time
        print("  %s entry %3d delivered %3d duration %3d  -> %.1f pts" % (p.id, p.entry_time, p.delivery_time, d, 100 * math.exp(-d / 50)))

if __name__ == "__main__":
    main()
