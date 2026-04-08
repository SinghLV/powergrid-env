"""
Hard Grader – N-2 cascading failure, island risk, multi-objective.

Per-step:
  +0.12  frequency 49.5–50.5 Hz
  +0.10  all voltages 0.92–1.08 pu
  +0.08  no line overloaded
  -0.20  blackout
  -0.15  cascade trip triggered (overloaded line > 2 steps)
  -0.12  island created (wrong pair of lines opened)
  -0.10  critical bus load shed
  -0.08  frequency < 49.0 Hz (pre-blackout zone)

Final score:
  0.30  survival (no blackout)
  0.25  frequency restored to 49.5–50.5 Hz in last 5 steps
  0.20  voltage stability
  0.15  no cascade / island
  0.10  critical load protection
"""

from __future__ import annotations


def step_reward(action, state, task_def, apply_msg=""):
    from environment import Reward

    if state.blackout_occurred:
        return Reward(value=-1.0, breakdown={"blackout": -1.0},
                      message="💥 BLACKOUT – cascading failure complete.")

    breakdown: dict[str, float] = {}
    value = 0.0

    f = state.grid_frequency_hz

    # Frequency
    if 49.5 <= f <= 50.5:
        breakdown["freq_ok"] = 0.12
        value += 0.12
    elif f < 49.0:
        breakdown["freq_critical"] = -0.08
        value -= 0.08

    # Voltage
    v_bad = [b for b in state.buses if not (0.92 <= b.voltage_pu <= 1.08)]
    if not v_bad:
        breakdown["voltage_ok"] = 0.10
        value += 0.10
    else:
        breakdown["voltage_bad"] = round(-0.04 * len(v_bad), 4)
        value += breakdown["voltage_bad"]

    # Line overloads
    overloaded = [l for l in state.lines if l.status == "closed" and l.loading_pct > 100]
    if not overloaded:
        breakdown["lines_ok"] = 0.08
        value += 0.08
    else:
        breakdown["lines_over"] = round(-0.06 * len(overloaded), 4)
        value += breakdown["lines_over"]

    # Island detection
    open_lines = {l.id for l in state.lines if l.status == "open"}
    for pair in task_def.get("island_risk_pairs", []):
        if set(pair).issubset(open_lines):
            breakdown["island_risk"] = -0.12
            value -= 0.12
            break

    # Critical load shed
    if state.critical_load_shed_mw > 0:
        breakdown["critical_shed"] = -0.10
        value -= 0.10

    return Reward(value=round(value, 4), breakdown=breakdown,
                  message=f"f={f:.2f}Hz V_bad={len(v_bad)} OL={len(overloaded)} | {apply_msg}")


def final_score(state, task_def) -> float:
    if state.blackout_occurred:
        return 0.05   # partial credit for lasting a few steps

    steps = max(state.step, 1)
    avg   = state.cumulative_reward / steps

    # Survival
    survival = 1.0

    # Frequency restored estimate
    f = state.grid_frequency_hz
    freq_score = 1.0 if 49.5 <= f <= 50.5 else max(0.0, 1 - abs(f - 50.0) / 1.5)

    # Voltage
    v_ok = all(0.92 <= b.voltage_pu <= 1.08 for b in state.buses)
    voltage_score = 1.0 if v_ok else 0.4

    # Cascade / island
    open_lines = {l.id for l in state.lines if l.status == "open"}
    island = any(set(p).issubset(open_lines) for p in task_def.get("island_risk_pairs", []))
    cascade_score = 0.0 if island else 1.0

    # Critical load
    crit_score = max(0.0, 1.0 - state.critical_load_shed_mw / 50.0)

    return round(
        max(0.001, min(0.999,
            0.30 * survival +
            0.25 * freq_score +
            0.20 * voltage_score +
            0.15 * cascade_score +
            0.10 * crit_score
        )),
        4
    )
