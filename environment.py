"""
PowerGridEnv – Real-world power grid failure prevention environment.

An AI grid operator must monitor bus voltages, line loads, and generator
outputs across a simulated distribution network, then issue control
actions (load shedding, generator redispatch, line switching, capacitor
switching) to prevent cascading failures before they propagate.

OpenEnv interface: reset() → step() → state()
Typed models: Observation, Action, Reward, State  (Pydantic v2)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Domain primitives
# ──────────────────────────────────────────────────────────────────────────────

class Bus(BaseModel):
    id: str
    voltage_pu: float          # per-unit voltage (nominal = 1.0)
    load_mw: float             # active load in MW
    load_mvar: float           # reactive load in MVAr
    is_critical: bool = False  # hospital, water plant, etc.


class Line(BaseModel):
    id: str
    from_bus: str
    to_bus: str
    flow_mw: float             # current flow
    capacity_mw: float         # thermal limit
    status: str = "closed"     # "closed" | "open"

    @property
    def loading_pct(self) -> float:
        return round(abs(self.flow_mw) / self.capacity_mw * 100, 2)

    @property
    def is_overloaded(self) -> bool:
        return self.loading_pct > 100.0


class Generator(BaseModel):
    id: str
    bus_id: str
    output_mw: float
    min_mw: float
    max_mw: float
    gen_type: str              # "coal" | "gas" | "solar" | "wind" | "hydro"
    ramp_rate_mw_per_step: float  # max change per timestep


class Alert(BaseModel):
    severity: str              # "warning" | "critical" | "emergency"
    component: str
    message: str


# ──────────────────────────────────────────────────────────────────────────────
# OpenEnv typed models
# ──────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    step: int
    max_steps: int
    task_description: str
    buses: list[Bus]
    lines: list[Line]
    generators: list[Generator]
    alerts: list[Alert]
    grid_frequency_hz: float   # nominal 50.0 Hz
    previous_actions: list[dict] = Field(default_factory=list)


class Action(BaseModel):
    """
    action_type choices:
      redispatch      – change a generator's output_mw
      shed_load       – reduce load on a bus by shed_mw
      switch_line     – open or close a transmission line
      switch_capacitor– switch a reactive compensation bank (raises voltage)
      do_nothing      – explicit no-op
    """
    action_type: str
    target_id: str                   # generator id / bus id / line id
    value: float | None = None       # new MW setpoint or shed_mw or None
    switch_to: str | None = None     # "open" | "closed" for line actions
    reasoning: str | None = None


class Reward(BaseModel):
    value: float
    breakdown: dict[str, float]
    message: str


class State(BaseModel):
    task_id: str
    step: int
    done: bool
    buses: list[Bus]
    lines: list[Line]
    generators: list[Generator]
    grid_frequency_hz: float
    blackout_occurred: bool
    critical_load_shed_mw: float
    total_load_shed_mw: float
    actions_taken: list[dict]
    cumulative_reward: float


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

from tasks.easy   import EASY_TASK
from tasks.medium import MEDIUM_TASK
from tasks.hard   import HARD_TASK

TASKS = {"easy": EASY_TASK, "medium": MEDIUM_TASK, "hard": HARD_TASK}


class PowerGridEnv:
    """
    OpenEnv-compliant power grid failure prevention environment.

    Usage
    -----
    env = PowerGridEnv(task_id="easy")
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    score = env.final_score()
    """

    def __init__(self, task_id: str = "easy") -> None:
        if task_id not in TASKS:
            raise ValueError(f"task_id must be one of {list(TASKS)}")
        self.task_id = task_id
        self._task_def = TASKS[task_id]
        self._state: State | None = None

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(self) -> Observation:
        td = self._task_def
        self._state = State(
            task_id=self.task_id,
            step=0,
            done=False,
            buses=[Bus(**b) for b in td["initial_buses"]],
            lines=[Line(**l) for l in td["initial_lines"]],
            generators=[Generator(**g) for g in td["initial_generators"]],
            grid_frequency_hz=td.get("initial_frequency_hz", 50.0),
            blackout_occurred=False,
            critical_load_shed_mw=0.0,
            total_load_shed_mw=0.0,
            actions_taken=[],
            cumulative_reward=0.0,
        )
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode finished. Call reset().")

        self._state.step += 1

        # Apply action to grid state
        apply_msg = self._apply_action(action)

        # Advance grid physics one timestep
        self._advance_physics()

        # Compute reward
        grader = self._task_def["grader"]
        reward = grader.step_reward(action=action, state=self._state,
                                    task_def=self._task_def,
                                    apply_msg=apply_msg)
        self._state.cumulative_reward += reward.value
        self._state.actions_taken.append(action.model_dump())

        # Check terminal conditions
        done = self._check_done()
        self._state.done = done

        obs = self._build_observation()
        info = {
            "step": self._state.step,
            "cumulative_reward": self._state.cumulative_reward,
            "blackout": self._state.blackout_occurred,
            "frequency_hz": self._state.grid_frequency_hz,
        }
        return obs, reward, done, info

    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return deepcopy(self._state)

    def final_score(self) -> float:
        grader = self._task_def["grader"]
        return grader.final_score(state=self._state, task_def=self._task_def)

    # ── Grid physics ───────────────────────────────────────────────────────

    def _apply_action(self, action: Action) -> str:
        s = self._state

        if action.action_type == "redispatch":
            for g in s.generators:
                if g.id == action.target_id:
                    if action.value is None:
                        return "redispatch requires a value"
                    new_out = max(g.min_mw, min(g.max_mw,
                                  g.output_mw + max(-g.ramp_rate_mw_per_step,
                                  min(g.ramp_rate_mw_per_step,
                                  action.value - g.output_mw))))
                    g.output_mw = round(new_out, 2)
                    return f"Redispatched {g.id} to {g.output_mw} MW"
            return f"Generator {action.target_id} not found"

        elif action.action_type == "shed_load":
            for b in s.buses:
                if b.id == action.target_id:
                    shed = min(action.value or 0.0, b.load_mw)
                    b.load_mw = round(max(0.0, b.load_mw - shed), 2)
                    s.total_load_shed_mw += shed
                    if b.is_critical:
                        s.critical_load_shed_mw += shed
                    return f"Shed {shed:.1f} MW on bus {b.id}"
            return f"Bus {action.target_id} not found"

        elif action.action_type == "switch_line":
            for ln in s.lines:
                if ln.id == action.target_id:
                    ln.status = action.switch_to or "open"
                    if ln.status == "open":
                        ln.flow_mw = 0.0
                    return f"Line {ln.id} switched to {ln.status}"
            return f"Line {action.target_id} not found"

        elif action.action_type == "switch_capacitor":
            # Raises voltage on the target bus by ~0.02 pu
            for b in s.buses:
                if b.id == action.target_id:
                    b.voltage_pu = round(min(1.10, b.voltage_pu + 0.02), 3)
                    return f"Capacitor switched on bus {b.id}, V={b.voltage_pu} pu"
            return f"Bus {action.target_id} not found"

        elif action.action_type == "do_nothing":
            return "No action taken"

        return f"Unknown action_type: {action.action_type}"

    def _advance_physics(self) -> None:
        """
        Simplified physics tick:
        - Frequency deviates from 50 Hz based on generation-load imbalance
        - Line flows shift slightly (random walk ±2%) to simulate disturbances
        - Voltage sags on overloaded lines
        - Blackout triggers if frequency < 48.5 Hz or > 51.5 Hz
        """
        import random
        s = self._state
        td = self._task_def

        # Generation vs load balance
        total_gen  = sum(g.output_mw for g in s.generators)
        total_load = sum(b.load_mw for b in s.buses)
        imbalance  = total_gen - total_load          # +ve → over-generation

        # Frequency deviation (simplified droop: 0.1 Hz per 10 MW imbalance)
        delta_f = imbalance * 0.01
        s.grid_frequency_hz = round(
            max(48.0, min(52.0, s.grid_frequency_hz + delta_f * 0.3)), 3
        )

        # Evolve line flows (disturbance scenario from task_def)
        scenario_step = td.get("disturbance_schedule", {}).get(s.step, {})
        for ln in s.lines:
            if ln.status == "open":
                continue
            delta = scenario_step.get(ln.id, random.uniform(-0.02, 0.02) * ln.flow_mw)
            ln.flow_mw = round(ln.flow_mw + delta, 2)

        # Voltage sag on overloaded buses
        for b in s.buses:
            connected_overloads = sum(
                1 for ln in s.lines
                if (ln.from_bus == b.id or ln.to_bus == b.id)
                and ln.is_overloaded and ln.status == "closed"
            )
            if connected_overloads:
                b.voltage_pu = round(max(0.85, b.voltage_pu - 0.01 * connected_overloads), 3)

        # Blackout condition
        freq_ok    = 48.5 <= s.grid_frequency_hz <= 51.5
        voltage_ok = all(0.90 <= b.voltage_pu <= 1.10 for b in s.buses)
        if not freq_ok or not voltage_ok:
            s.blackout_occurred = True

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        s = self._state
        alerts = self._generate_alerts()
        return Observation(
            step=s.step,
            max_steps=self._task_def["max_steps"],
            task_description=self._task_def["description"],
            buses=deepcopy(s.buses),
            lines=deepcopy(s.lines),
            generators=deepcopy(s.generators),
            alerts=alerts,
            grid_frequency_hz=s.grid_frequency_hz,
            previous_actions=list(s.actions_taken),
        )

    def _generate_alerts(self) -> list[Alert]:
        s = self._state
        alerts = []
        for ln in s.lines:
            if ln.loading_pct > 100:
                alerts.append(Alert(severity="emergency", component=ln.id,
                    message=f"Line {ln.id} OVERLOADED at {ln.loading_pct:.1f}%"))
            elif ln.loading_pct > 85:
                alerts.append(Alert(severity="critical", component=ln.id,
                    message=f"Line {ln.id} near limit: {ln.loading_pct:.1f}%"))
            elif ln.loading_pct > 70:
                alerts.append(Alert(severity="warning", component=ln.id,
                    message=f"Line {ln.id} at {ln.loading_pct:.1f}% loading"))
        for b in s.buses:
            if b.voltage_pu < 0.90 or b.voltage_pu > 1.10:
                alerts.append(Alert(severity="emergency", component=b.id,
                    message=f"Bus {b.id} voltage CRITICAL: {b.voltage_pu:.3f} pu"))
            elif b.voltage_pu < 0.95 or b.voltage_pu > 1.05:
                alerts.append(Alert(severity="warning", component=b.id,
                    message=f"Bus {b.id} voltage: {b.voltage_pu:.3f} pu"))
        f = s.grid_frequency_hz
        if f < 49.0 or f > 51.0:
            alerts.append(Alert(severity="critical", component="grid",
                message=f"Frequency deviation: {f:.3f} Hz (nominal 50.0)"))
        return alerts

    def _check_done(self) -> bool:
        return (
            self._state.blackout_occurred
            or self._state.step >= self._task_def["max_steps"]
        )
