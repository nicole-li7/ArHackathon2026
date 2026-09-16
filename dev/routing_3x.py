"""Dev wrapper: the submitted planner with 3x time budgets (search convergence test)."""
import ar_hackathon.api.routing as r
r.BUDGET_FIRST, r.BUDGET_EVENT, r.BUDGET_STEADY, r.BUDGET_CONVERGED = 0.75, 0.54, 0.30, 0.15
r.TOTAL_BUDGET = 150.0
drive_unit_next_move = r.drive_unit_next_move
