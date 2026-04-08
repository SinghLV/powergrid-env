"""
Easy Task – Single Transmission Line Overload.

Scenario: A coal generator trips offline during evening peak demand.
Line L1-2 is now carrying 118% of its thermal limit and will fail in
~8 timesteps unless the agent redispatches the standby gas turbine.

Objective: Relieve the overload on L1-2 within 10 steps without
           triggering a blackout or shedding critical load.
"""

EASY_TASK = {
    "task_id": "easy",
    "description": (
        "GRID EMERGENCY – Evening peak load, Unit G1 (coal) has tripped.\n"
        "Line L1-2 is overloaded at 118% capacity and will thermally fail "
        "in ~8 timesteps, causing a cascading blackout.\n\n"
        "Your available actions:\n"
        "  redispatch  – increase standby gas turbine G2's output\n"
        "  shed_load   – reduce load on a non-critical bus\n"
        "  do_nothing  – wait (dangerous)\n\n"
        "Goal: Reduce line L1-2 loading below 90% without blacking out. "
        "Avoid shedding load on critical buses."
    ),
    "max_steps": 10,
    "initial_frequency_hz": 49.7,   # already sagging due to lost generator

    "initial_buses": [
        {"id": "B1", "voltage_pu": 0.98, "load_mw": 80.0,  "load_mvar": 20.0, "is_critical": True},
        {"id": "B2", "voltage_pu": 0.96, "load_mw": 120.0, "load_mvar": 30.0, "is_critical": False},
        {"id": "B3", "voltage_pu": 0.97, "load_mw": 60.0,  "load_mvar": 15.0, "is_critical": True},
    ],

    "initial_lines": [
        {"id": "L1-2", "from_bus": "B1", "to_bus": "B2", "flow_mw": 118.0, "capacity_mw": 100.0},
        {"id": "L2-3", "from_bus": "B2", "to_bus": "B3", "flow_mw": 55.0,  "capacity_mw": 80.0},
        {"id": "L1-3", "from_bus": "B1", "to_bus": "B3", "flow_mw": 38.0,  "capacity_mw": 60.0},
    ],

    "initial_generators": [
        # G1 tripped — not in list
        {"id": "G2", "bus_id": "B1", "output_mw": 40.0,  "min_mw": 0.0,
         "max_mw": 120.0, "gen_type": "gas",   "ramp_rate_mw_per_step": 30.0},
        {"id": "G3", "bus_id": "B3", "output_mw": 80.0,  "min_mw": 10.0,
         "max_mw": 90.0,  "gen_type": "hydro", "ramp_rate_mw_per_step": 15.0},
    ],

    # Disturbance: L1-2 flow increases 3 MW/step if not addressed
    "disturbance_schedule": {
        i: {"L1-2": 3.0} for i in range(1, 11)
    },

    "grader": None,
    "success_threshold_pct": 90.0,  # L1-2 must be below this
}

from graders import easy_grader
EASY_TASK["grader"] = easy_grader
