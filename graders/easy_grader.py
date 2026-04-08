"""
Easy Grader – Relieve single line overload.

Per-step rewards:
  +0.15  each step L1-2 loading < 90%
  +0.05  each step frequency in 49.5–50.5 Hz
  -0.20  if blackout occurs
  -0.10  if critical load shed > 0
  -0.05  do_nothing while L1-2 still overloaded (wasted step)

Final score (0–1):
  0.50  × (steps L1-2 < 90%) / max_steps
  0.30  × (1 – min(critical_shed, 30) / 30)
  0.20  × (1 if no blackout else 0)
"""

from __future__ import annotations


def step_reward(action, state, task_def, apply_msg=""):
    from environment import Reward

    breakdown: dict[str, float] = {}
    value = 0.0

    # Blackout = immediate heavy penalty
    if state.blackout_occurred:
        return Reward(value=-1.0,
                      breakdown={"blackout": -1.0},
                      message="💥 BLACKOUT – episode failed.")

    # Find L1-2
    l12 = next((l for l in state.lines if l.id == "L1-2"), None)

    if l12 and l12.loading_pct < 90.0:
        breakdown["line_safe"] = 0.15
        value += 0.15
    elif l12 and l12.loading_pct < 100.0:
        breakdown["line_warning"] = 0.05
        value += 0.05
    else:
        breakdown["line_overloaded"] = -0.05
        value -= 0.05

    # Frequency
    f = state.grid_frequency_hz
    if 49.5 <= f <= 50.5:
        breakdown["freq_ok"] = 0.05
        value += 0.05

    # Critical load shed penalty
    if state.critical_load_shed_mw > 0:
        breakdown["critical_shed"] = -0.10
        value -= 0.10

    # Do-nothing while overloaded
    if action.action_type == "do_nothing" and l12 and l12.loading_pct >= 90.0:
        breakdown["wasted_step"] = -0.05
        value -= 0.05

    return Reward(value=round(value, 4), breakdown=breakdown,
                  message=f"L1-2={l12.loading_pct:.1f}% f={f:.2f}Hz | {apply_msg}")


def final_score(state, task_def) -> float:
    """
    Final score components (weights sum to 1.0):
      0.50  line_score    – fraction of steps where L1-2 stayed below 90%
                           (estimated from cumulative reward; each safe step = +0.15)
      0.30  shed_score    – penalises critical load shed (linear, capped at 30 MW)
      0.20  no_blackout   – 1.0 if episode ended without blackout, 0.0 otherwise
    """
    if state.blackout_occurred:
        return 0.001

    max_steps = task_def["max_steps"]

    # Each step where L1-2 < 90% contributed +0.15 to cumulative_reward.
    # Divide by 0.15 to estimate how many steps were "safe", clamped to [0, max_steps].
    # We use max(0, ...) to guard against negative cumulative reward edge cases.
    safe_steps = min(state.step, max(0, round(state.cumulative_reward / 0.15)))

    line_score  = min(1.0, safe_steps / max_steps)
    shed_score  = max(0.0, 1.0 - state.critical_load_shed_mw / 30.0)
    no_blackout = 1.0  # guaranteed true if we reached this branch

    return round(max(0.001, min(0.999, 0.50 * line_score + 0.30 * shed_score + 0.20 * no_blackout)), 4)
