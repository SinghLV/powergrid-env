"""
Hard Task – N-2 Cascading Failure Prevention.

Scenario: A severe storm has caused simultaneous faults on two
transmission lines (N-2 contingency). The remaining network is
heavily stressed. A gas plant is ramping to compensate but its
ramp rate is insufficient. Frequency is at 49.1 Hz and dropping.

The agent faces conflicting objectives:
  - Shedding load on B3 would fix frequency but B3 is a hospital
  - Opening line L2-5 would relieve overload but risks islanding B5
  - Ramping G4 (diesel peaker) is fast but very expensive
  - B6 voltage is critically low and approaching collapse

The agent must navigate these tradeoffs across 7 buses, 9 lines,
and 5 generators within 20 steps, without blackout.

Hard because:
  • N-2 contingency (two simultaneous failures)
  • Conflicting actions (what helps frequency hurts voltage)
  • Island risk if wrong line is opened
  • Time pressure: frequency crosses blackout at step ~12 if unaddressed
"""

HARD_TASK = {
    "task_id": "hard",
    "description": (
        "STORM EMERGENCY – N-2 contingency. Lines L1-3 and L4-6 have faulted.\n"
        "Frequency: 49.1 Hz and falling. Bus B6 voltage: 0.88 pu (critical).\n"
        "Lines L2-5 and L3-5 are overloaded. B3 is a hospital — protect it.\n\n"
        "Available actions:\n"
        "  redispatch       – ramp a generator (check ramp limits!)\n"
        "  shed_load        – shed MW from a bus (avoid B3, B7)\n"
        "  switch_line      – open/close a line (watch for islanding)\n"
        "  switch_capacitor – raise voltage on a bus\n"
        "  do_nothing       – wait (frequency keeps falling)\n\n"
        "Goals:\n"
        "  • Prevent blackout (f < 48.5 Hz or V < 0.90 pu)\n"
        "  • Restore frequency to 49.5–50.5 Hz\n"
        "  • Restore all voltages to 0.92–1.08 pu\n"
        "  • Minimise load shed (especially critical buses B3, B7)\n"
        "  • No line > 100% loading for more than 2 consecutive steps"
    ),
    "max_steps": 20,
    "initial_frequency_hz": 49.1,

    "initial_buses": [
        {"id": "B1", "voltage_pu": 0.99, "load_mw": 120.0, "load_mvar": 30.0, "is_critical": False},
        {"id": "B2", "voltage_pu": 0.97, "load_mw": 100.0, "load_mvar": 25.0, "is_critical": False},
        {"id": "B3", "voltage_pu": 0.95, "load_mw": 180.0, "load_mvar": 45.0, "is_critical": True},  # hospital
        {"id": "B4", "voltage_pu": 0.96, "load_mw": 90.0,  "load_mvar": 22.0, "is_critical": False},
        {"id": "B5", "voltage_pu": 0.93, "load_mw": 140.0, "load_mvar": 35.0, "is_critical": False},
        {"id": "B6", "voltage_pu": 0.88, "load_mw": 160.0, "load_mvar": 40.0, "is_critical": False},  # near collapse
        {"id": "B7", "voltage_pu": 0.96, "load_mw": 110.0, "load_mvar": 28.0, "is_critical": True},  # water plant
    ],

    "initial_lines": [
        # L1-3 and L4-6 are FAULTED (open)
        {"id": "L1-3", "from_bus": "B1", "to_bus": "B3", "flow_mw": 0.0,   "capacity_mw": 100.0, "status": "open"},
        {"id": "L4-6", "from_bus": "B4", "to_bus": "B6", "flow_mw": 0.0,   "capacity_mw": 80.0,  "status": "open"},
        # Healthy lines — some overloaded
        {"id": "L1-2", "from_bus": "B1", "to_bus": "B2", "flow_mw": 95.0,  "capacity_mw": 120.0},
        {"id": "L2-3", "from_bus": "B2", "to_bus": "B3", "flow_mw": 88.0,  "capacity_mw": 100.0},
        {"id": "L2-5", "from_bus": "B2", "to_bus": "B5", "flow_mw": 112.0, "capacity_mw": 100.0},  # 112% OVERLOAD
        {"id": "L3-5", "from_bus": "B3", "to_bus": "B5", "flow_mw": 105.0, "capacity_mw": 100.0},  # 105% OVERLOAD
        {"id": "L5-6", "from_bus": "B5", "to_bus": "B6", "flow_mw": 78.0,  "capacity_mw": 80.0},   # 97.5%
        {"id": "L5-7", "from_bus": "B5", "to_bus": "B7", "flow_mw": 60.0,  "capacity_mw": 90.0},
        {"id": "L6-7", "from_bus": "B6", "to_bus": "B7", "flow_mw": 45.0,  "capacity_mw": 70.0},
    ],

    "initial_generators": [
        {"id": "G1", "bus_id": "B1", "output_mw": 200.0, "min_mw": 100.0,
         "max_mw": 250.0, "gen_type": "coal",   "ramp_rate_mw_per_step": 15.0},
        {"id": "G2", "bus_id": "B2", "output_mw": 80.0,  "min_mw": 0.0,
         "max_mw": 160.0, "gen_type": "gas",    "ramp_rate_mw_per_step": 35.0},
        {"id": "G3", "bus_id": "B4", "output_mw": 120.0, "min_mw": 50.0,
         "max_mw": 180.0, "gen_type": "hydro",  "ramp_rate_mw_per_step": 25.0},
        {"id": "G4", "bus_id": "B6", "output_mw": 20.0,  "min_mw": 0.0,
         "max_mw": 100.0, "gen_type": "diesel", "ramp_rate_mw_per_step": 50.0},  # fast peaker
        {"id": "G5", "bus_id": "B7", "output_mw": 40.0,  "min_mw": 0.0,
         "max_mw": 60.0,  "gen_type": "wind",   "ramp_rate_mw_per_step": 8.0},
    ],

    # Storm disturbance: overloaded lines get worse for first 8 steps
    "disturbance_schedule": {
        1: {"L2-5": 5.0, "L3-5": 4.0, "L5-6": 2.0},
        2: {"L2-5": 4.0, "L3-5": 3.0, "L5-6": 2.0},
        3: {"L2-5": 3.0, "L3-5": 3.0},
        4: {"L2-5": 2.0, "L3-5": 2.0},
        5: {"L2-5": 2.0, "L3-5": 1.0},
        6: {"L2-5": 1.0},
        7: {"L2-5": 1.0},
        8: {},
    },

    # Island risk: if L2-5 opened AND L3-5 opened, B5/B6/B7 island
    "island_risk_pairs": [("L2-5", "L3-5")],

    # Lines that if overloaded >2 consecutive steps trigger cascade trip
    "cascade_trigger_lines": {"L2-5", "L3-5", "L5-6"},
    "consecutive_overload_limit": 2,

    "grader": None,
}

from graders import hard_grader
HARD_TASK["grader"] = hard_grader
