"""
Medium Task – Voltage Collapse Prevention + Frequency Regulation.

Scenario: A summer heatwave has pushed regional demand 22% above forecast.
Three buses show under-voltage (<0.95 pu). Grid frequency is slipping
(49.4 Hz). A renewable curtailment event has just cut 80 MW of solar.

The agent must simultaneously:
  1. Restore voltages to 0.95–1.05 pu range using capacitor banks + redispatch
  2. Restore frequency to 49.8–50.2 Hz range
  3. Prevent any single line from exceeding 95% loading
  4. Minimise critical load shedding

The disturbance worsens each step — the agent has 15 steps.
"""

MEDIUM_TASK = {
    "task_id": "medium",
    "description": (
        "HEATWAVE EMERGENCY – Solar curtailment + demand spike.\n"
        "Grid frequency: 49.4 Hz (sagging). Three buses under-voltage.\n"
        "Lines L2-4 and L3-5 are approaching thermal limits.\n\n"
        "Available actions:\n"
        "  redispatch      – ramp a generator up or down\n"
        "  shed_load       – drop non-critical load on a bus\n"
        "  switch_capacitor– energise reactive compensation on a bus\n"
        "  switch_line     – open/close a tie line\n"
        "  do_nothing      – wait\n\n"
        "Goals (all must hold for final 3 steps):\n"
        "  • Frequency: 49.8 – 50.2 Hz\n"
        "  • All bus voltages: 0.95 – 1.05 pu\n"
        "  • No line > 95% loading\n"
        "  • Zero critical load shed"
    ),
    "max_steps": 15,
    "initial_frequency_hz": 49.4,

    "initial_buses": [
        {"id": "B1", "voltage_pu": 1.00, "load_mw": 100.0, "load_mvar": 25.0, "is_critical": False},
        {"id": "B2", "voltage_pu": 0.93, "load_mw": 180.0, "load_mvar": 45.0, "is_critical": True},   # hospital district
        {"id": "B3", "voltage_pu": 0.94, "load_mw": 150.0, "load_mvar": 38.0, "is_critical": False},
        {"id": "B4", "voltage_pu": 0.92, "load_mw": 200.0, "load_mvar": 50.0, "is_critical": True},   # water treatment
        {"id": "B5", "voltage_pu": 0.96, "load_mw": 120.0, "load_mvar": 30.0, "is_critical": False},
    ],

    "initial_lines": [
        {"id": "L1-2", "from_bus": "B1", "to_bus": "B2", "flow_mw": 85.0,  "capacity_mw": 120.0},
        {"id": "L1-3", "from_bus": "B1", "to_bus": "B3", "flow_mw": 75.0,  "capacity_mw": 100.0},
        {"id": "L2-4", "from_bus": "B2", "to_bus": "B4", "flow_mw": 109.0, "capacity_mw": 120.0},  # 91%
        {"id": "L3-5", "from_bus": "B3", "to_bus": "B5", "flow_mw": 91.0,  "capacity_mw": 100.0},  # 91%
        {"id": "L4-5", "from_bus": "B4", "to_bus": "B5", "flow_mw": 60.0,  "capacity_mw": 80.0},
        {"id": "L1-4", "from_bus": "B1", "to_bus": "B4", "flow_mw": 0.0,   "capacity_mw": 100.0, "status": "open"},  # tie line
    ],

    "initial_generators": [
        {"id": "G1", "bus_id": "B1", "output_mw": 150.0, "min_mw": 50.0,
         "max_mw": 200.0, "gen_type": "coal",  "ramp_rate_mw_per_step": 20.0},
        {"id": "G2", "bus_id": "B3", "output_mw": 80.0,  "min_mw": 0.0,
         "max_mw": 150.0, "gen_type": "gas",   "ramp_rate_mw_per_step": 40.0},
        {"id": "G3", "bus_id": "B5", "output_mw": 60.0,  "min_mw": 0.0,
         "max_mw": 100.0, "gen_type": "wind",  "ramp_rate_mw_per_step": 10.0},
        {"id": "G4", "bus_id": "B2", "output_mw": 30.0,  "min_mw": 0.0,
         "max_mw": 80.0,  "gen_type": "solar", "ramp_rate_mw_per_step": 5.0},  # curtailed
    ],

    # Demand keeps rising +5 MW/bus/step for first 5 steps (heatwave)
    "disturbance_schedule": {
        1: {"L2-4": 4.0, "L3-5": 3.0},
        2: {"L2-4": 4.0, "L3-5": 3.0},
        3: {"L2-4": 3.0, "L3-5": 2.0},
        4: {"L2-4": 2.0, "L3-5": 2.0},
        5: {"L2-4": 1.0, "L3-5": 1.0},
    },

    # Stability window: last 3 steps all conditions must hold
    "stability_window": 3,

    "grader": None,
}

from graders import medium_grader
MEDIUM_TASK["grader"] = medium_grader
