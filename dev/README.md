# dev/ - developer tooling (not part of the submission)

## Test-case generator (`dev/gen_cases.py`)

- Generate a benchmark suite: `python3 dev/gen_cases.py --out dev/cases --seed 1 --per-level 40 --stress 12`
- Output: `dev/cases/level{1,2,3}/gen_l{1,2,3}_NNN.json` plus `dev/cases/stress/gen_s_NNN.json`, same JSON schema as `test_cases/` and sized/flavoured like the provided cases per level (stress = bigger grids for time-limit robustness).
- Deterministic: the same `--seed` always reproduces identical files (each case uses its own `random.Random` seeded from seed, level and index).
- Add `--validate` to load every generated file through `ar_hackathon.utils.json_loader` and play it with the example `basic_driver` through the real `GameEngine`; the run exits non-zero if any file fails.
- Run a generated case exactly like a provided one: `python3 scripts/run_game.py dev/cases/level2/gen_l2_000.json`.

## Exhaustive plan-space reference (`dev/exact_ref.py`)

- Answers "how much would the planner gain if its local search were perfect?" for small instances. It monkeypatches `routing.Planner.improve` so that whenever at most `--pmax` (default 6) pods are known and at most 3 units exist, the step enumerates the *entire* plan space of `routing.py` - every assignment of waiting pods to units, every capacity-feasible ordering of each unit's stops (`P` anywhere before its `D`, carried pods keep a single `D` on their carrier) and every priority order - scoring each plan with `routing.evaluate`. The greedy/warm-start plan stays a candidate, so a step is never worse than what the planner had before `improve()`. Bigger instances fall back to the real local search. Enumeration is capped at `--cap` (default 120000) evaluations per step and cap hits are reported.
- The engine's 1-second `safe_execute` timeout is replaced by a passthrough and `routing.TOTAL_BUDGET` is disabled, so this is a reference, not a submission.
- Run: `python3 dev/exact_ref.py --jobs 6 --json dev/run_exact.json` (default set: `test_cases/level*` + `dev/cases/level*`, no stress). Deltas are against `--baseline dev/run_v4.json` (a `bench --json` file). The report lists per-case online/exact/delta, enumerated steps, steps where enumeration beat the greedy plan, max evaluations and cap hits, then per-level means, the largest gains and losses (a loss means a plan with a better replica value led to a worse real outcome later, e.g. because of pods that were not yet known) and total runtime.
- `--compare-ls` additionally runs the planner's own local search at every enumerated step from the same greedy start (result discarded) and reports on how many steps it stopped short of the exhaustive optimum and by how much replica score - a per-step measure that is free of trajectory noise.
- It also works as a bench driver: `python3 dev/bench.py --driver dev/exact_ref.py` (configure with env `EXACT_PMAX`, `EXACT_CAP`, `EXACT_COMPARE_LS`).
- Caveat when reading deltas: the online planner is wall-clock budgeted, so its score on a given case can differ between runs (e.g. under `--jobs 6` load vs. solo); compare against a baseline produced under the same load, or check the `impr` column - a case with `impr = 0` never changed a plan, so its delta is pure timing noise.
