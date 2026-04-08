#!/usr/bin/env python3
"""
inference.py – Baseline inference for PowerGridEnv.

Connects to a Hugging Face Inference Endpoint via the OpenAI-compatible
client. API key is read from HF_TOKEN environment variable.

Usage
-----
# Full LLM-based evaluation (requires HF token):
HF_TOKEN=hf_xxx python inference.py [--task easy|medium|hard|all] [--verbose]

# Local smoke-test without a token (uses heuristic do-nothing agent):
python inference.py --dry-run [--task easy|medium|hard|all] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# openai is imported lazily inside main() so that --dry-run works
# without having the openai package installed.
from environment import Action, PowerGridEnv

HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

SYSTEM_PROMPT = """You are an AI power grid operator. Your job is to prevent
blackouts by issuing control actions in real time.

At each step you receive the current grid state (buses, lines, generators,
alerts) and must output EXACTLY ONE JSON object — no prose, no markdown:

{
  "action_type": "<redispatch|shed_load|switch_line|switch_capacitor|do_nothing>",
  "target_id":   "<generator_id | bus_id | line_id>",
  "value":       <float or null>,
  "switch_to":   "<open|closed or null>",
  "reasoning":   "<one sentence explaining your decision>"
}

Rules:
- redispatch: set 'value' to the desired new MW output for that generator
- shed_load:  set 'value' to MW to shed from that bus
- switch_line: set 'switch_to' to "open" or "closed"
- switch_capacitor: set target_id to bus id, value/switch_to can be null
- Respect generator ramp limits (ramp_rate_mw_per_step)
- Do NOT shed critical buses unless absolutely necessary
- Always include a 'reasoning' field
"""


def build_prompt(obs) -> str:
    buses = "\n".join(
        f"  {b.id}: V={b.voltage_pu:.3f}pu  load={b.load_mw}MW"
        f"  critical={b.is_critical}"
        for b in obs.buses
    )
    lines = "\n".join(
        f"  {l.id}: {l.from_bus}→{l.to_bus}  flow={l.flow_mw}MW"
        f"  cap={l.capacity_mw}MW  loading={l.loading_pct:.1f}%  status={l.status}"
        for l in obs.lines
    )
    gens = "\n".join(
        f"  {g.id}@{g.bus_id}: {g.output_mw}MW  [{g.min_mw}–{g.max_mw}]"
        f"  ramp±{g.ramp_rate_mw_per_step}  type={g.gen_type}"
        for g in obs.generators
    )
    alerts = "\n".join(
        f"  [{a.severity.upper()}] {a.component}: {a.message}"
        for a in obs.alerts
    ) or "  None"

    return f"""TASK: {obs.task_description}

STEP {obs.step}/{obs.max_steps}
Frequency: {obs.grid_frequency_hz:.3f} Hz

BUSES:
{buses}

LINES:
{lines}

GENERATORS:
{gens}

ALERTS:
{alerts}

Previous actions: {json.dumps([a.get('action_type') for a in obs.previous_actions[-3:]])}

Output ONE JSON action object now:"""


def query_model(client, prompt: str) -> dict:  # client: openai.OpenAI
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences if the model wraps its JSON
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()
    return json.loads(raw)


def run_task(client, task_id: str, verbose: bool = False) -> float:  # client: openai.OpenAI
    env = PowerGridEnv(task_id=task_id)
    obs = env.reset()

    print(f"\n{'='*65}")
    print(f"START  |  PowerGridEnv Task: {task_id.upper()}")
    print(f"  Buses: {len(obs.buses)}  Lines: {len(obs.lines)}  Generators: {len(obs.generators)}")
    print(f"{'='*65}")

    total_rew = 0.0
    for step_num in range(env._task_def["max_steps"]):
        prompt = build_prompt(obs)
        try:
            action_dict = query_model(client, prompt)
            action = Action(**action_dict)
        except Exception as e:
            print(f"  [step {step_num+1}] Parse error: {e} – doing nothing")
            action = Action(action_type="do_nothing", target_id="grid",
                            reasoning="parse error fallback")

        obs, reward, done, info = env.step(action)
        total_rew += reward.value

        if verbose:
            print(f"STEP {info['step']:02d}: {action.action_type:16s} "
                  f"target={action.target_id:8s} "
                  f"f={info['frequency_hz']:.2f}Hz "
                  f"rew={reward.value:+.3f}  {reward.message[:60]}")
        if done:
            if info["blackout"]:
                print(f"  💥 BLACKOUT at step {info['step']}")
            break

    score = env.final_score()
    print(f"\nEND  |  Steps: {info['step']}  |  Total reward: {total_rew:+.3f}  |  Score: {score:.4f}")
    return score


def run_task_dry(task_id: str, verbose: bool = False) -> float:
    """
    Heuristic smoke-test agent: issues do_nothing every step.
    No HF token required. Used to validate the environment runs without errors.
    """
    env = PowerGridEnv(task_id=task_id)
    obs = env.reset()

    print(f"\n{'='*65}")
    print(f"START  |  PowerGridEnv [DRY-RUN] Task: {task_id.upper()}")
    print(f"  Buses: {len(obs.buses)}  Lines: {len(obs.lines)}  Generators: {len(obs.generators)}")
    print(f"{'='*65}")

    total_rew = 0.0
    info = {"step": 0, "frequency_hz": obs.grid_frequency_hz, "blackout": False}
    for step_num in range(env._task_def["max_steps"]):
        # Heuristic: always do-nothing (worst-case baseline)
        action = Action(action_type="do_nothing", target_id="grid",
                        reasoning="dry-run smoke test")
        obs, reward, done, info = env.step(action)
        total_rew += reward.value
        if verbose:
            print(f"STEP {info['step']:02d}: do_nothing  "
                  f"f={info['frequency_hz']:.2f}Hz  "
                  f"rew={reward.value:+.3f}  {reward.message[:60]}")
        if done:
            if info["blackout"]:
                print(f"  ⚠  BLACKOUT at step {info['step']} (expected in dry-run)")
            break

    score = env.final_score()
    print(f"\nEND  |  Steps: {info['step']}  |  Total reward: {total_rew:+.3f}  |  Score: {score:.4f}")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PowerGridEnv inference baseline. Use --dry-run to validate without HF_TOKEN."
    )
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a heuristic do-nothing agent (no HF_TOKEN required). Useful for smoke-testing.",
    )
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    scores: dict[str, float] = {}

    if args.dry_run:
        # ── Dry-run mode: validate environment without an LLM ──────────────
        # openai package is NOT required in this path.
        print("[DRY-RUN] Running heuristic do-nothing agent for smoke-test...")
        for tid in tasks:
            scores[tid] = run_task_dry(tid, verbose=args.verbose)
    else:
        # ── LLM mode: requires OPENAI_API_KEY + openai package ───────────────────
        if not HF_TOKEN:
            print("ERROR: HF_TOKEN (or OPENAI_API_KEY) environment variable is not set.", file=sys.stderr)
            print("       Run with --dry-run to validate without a token.", file=sys.stderr)
            sys.exit(1)
        try:
            from openai import OpenAI  # lazy import — not needed for dry-run
        except ImportError:
            print("ERROR: 'openai' package not installed. Run: pip install openai", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        for tid in tasks:
            scores[tid] = run_task(client, tid, verbose=args.verbose)

    print(f"\n{'='*65}")
    print("  SCORES")
    print(f"{'='*65}")
    for tid, sc in scores.items():
        bar = "█" * int(sc * 30) + "░" * (30 - int(sc * 30))
        print(f"  {tid:<8}  {bar}  {sc:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"{'='*65}")
    print(f"  AVERAGE   {avg:.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
