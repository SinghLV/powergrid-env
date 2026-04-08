"""
Medium Grader – Multi-objective: voltage + frequency + line loading.

Per-step rewards:
  +0.10  all voltages 0.95–1.05 pu
  +0.08  frequency 49.8–50.2 Hz
  +0.07  all lines < 95% loading
  -0.15  any voltage outside 0.90–1.10 (emergency zone)
  -0.10  frequency outside 49.5–50.5 Hz
  -0.20  blackout
  -0.12  critical load shed (per occurrence)

Final score weights:
  0.35 voltage stability (steps in range / total)
  0.30 frequency stability
  0.25 line safety
  0.10 no critical load shed
"""

from __future__ import annotations


def step_reward(action, state, task_def, apply_msg=""):
    from environment import Reward

    if state.blackout_occurred:
        return Reward(value=-1.0, breakdown={"blackout": -1.0},
                      message="💥 BLACKOUT")

    breakdown: dict[str, float] = {}
    value = 0.0

    # Voltage
    v_bad = [b for b in state.buses if not (0.95 <= b.voltage_pu <= 1.05)]
    v_emergency = [b for b in state.buses if not (0.90 <= b.voltage_pu <= 1.10)]
    if not v_bad:
        breakdown["voltage_ok"] = 0.10
        value += 0.10
    elif v_emergency:
        breakdown["voltage_emergency"] = -0.15
        value -= 0.15

    # Frequency
    f = state.grid_frequency_hz
    if 49.8 <= f <= 50.2:
        breakdown["freq_ok"] = 0.08
        value += 0.08
    elif not (49.5 <= f <= 50.5):
        breakdown["freq_bad"] = -0.10
        value -= 0.10

    # Lines
    overloaded = [l for l in state.lines if l.status == "closed" and l.loading_pct > 95]
    if not overloaded:
        breakdown["lines_ok"] = 0.07
        value += 0.07
    else:
        breakdown["lines_overloaded"] = -0.05 * len(overloaded)
        value -= 0.05 * len(overloaded)

    # Critical shed
    if state.critical_load_shed_mw > 0:
        breakdown["critical_shed"] = -0.12
        value -= 0.12

    buses_str = " ".join(f"{b.id}:{b.voltage_pu:.2f}" for b in state.buses)
    return Reward(value=round(value, 4), breakdown=breakdown,
                  message=f"f={f:.2f}Hz {buses_str} | {apply_msg}")


def final_score(state, task_def) -> float:
    """
    Final score components (weights sum to 1.0):
      0.35  voltage_score  – fraction of buses in safe range (0.95–1.05 pu) at episode end
      0.30  freq_score     – how close frequency is to nominal (50.0 Hz), scaled 0→1
      0.25  line_score     – fraction of closed lines below 95% loading at episode end
      0.10  shed_score     – penalises critical load shed (linear, capped at 100 MW)

    Unlike the per-step reward (which uses a single average), each sub-score here
    is derived directly from the terminal state so the three objectives are distinguishable.
    """
    if state.blackout_occurred:
        return 0.001

    # ── Voltage score: fraction of buses within 0.95–1.05 pu ─────────────────
    buses_ok = sum(1 for b in state.buses if 0.95 <= b.voltage_pu <= 1.05)
    voltage_score = buses_ok / max(len(state.buses), 1)

    # ── Frequency score: proximity to 50.0 Hz, full credit within ±0.2 Hz ────
    f = state.grid_frequency_hz
    freq_score = max(0.0, 1.0 - abs(f - 50.0) / 0.5)   # 0→1 over [49.5, 50.5]

    # ── Line score: fraction of closed lines below 95% loading ────────────────
    closed_lines = [l for l in state.lines if l.status == "closed"]
    lines_ok = sum(1 for l in closed_lines if l.loading_pct < 95.0)
    line_score = lines_ok / max(len(closed_lines), 1)

    # ── Critical shed penalty ─────────────────────────────────────────────────
    shed_score = 1.0 if state.critical_load_shed_mw == 0 else max(
        0.0, 1.0 - state.critical_load_shed_mw / 100.0
    )

    return round(
        max(0.001, min(0.999,
            0.35 * voltage_score +
            0.30 * freq_score +
            0.25 * line_score +
            0.10 * shed_score
        )),
        4,
    )
