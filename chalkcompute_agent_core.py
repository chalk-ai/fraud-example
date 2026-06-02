#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute>=1.5.17", "openai", "chalkpy", "python-dotenv"]
# ///
"""Refund-abuse investigation agent — the agentic loop, as a Chalk Compute function.

This module is deliberately free of any web-framework imports. When the function
is deployed, Chalk Compute ships THIS file as `user_module.py` into the image and
imports it, so it must import cleanly with only the image's deps (openai, chalkpy).
The FastAPI UI (chalkcompute_agent_demo_ui.py) imports `investigate_stream` from
here and forwards its streamed events to the browser as SSE.

The whole agent runs on Chalk Compute:
  - The LLM tool-use loop, the plan, and every chalk.query tool call execute
    inside the deployed function, so the run shows up in Chalk's function tracing.
  - The function is a *generator* (-> Iterator[str]): each event is yielded and
    streamed back to the caller live, so the UI still updates step by step.

Deploy (once, before the demo, so the first call is warm):
  ./chalkcompute_agent_core.py deploy

Run the UI against it:
  ./chalkcompute_agent_demo_ui.py            # AGENT_REMOTE=1 by default
  AGENT_REMOTE=0 ./chalkcompute_agent_demo_ui.py   # run the loop in-process instead
"""

import json
import os
import re
from typing import Iterator

import chalkcompute

# .env load is only meaningful for local execution / deploy-time; inside the
# deployed container, secrets are injected by Chalk (see `secrets=` below).
try:
    from dotenv import load_dotenv

    _dir = os.getcwd()
    for _ in range(5):
        _candidate = os.path.join(_dir, ".env")
        if os.path.exists(_candidate):
            load_dotenv(_candidate, override=True)
            break
        _dir = os.path.dirname(_dir)
except Exception:
    pass

from openai import OpenAI
from chalk.client import ChalkClient

_model = "gpt-4o"
_llm: OpenAI | None = None
_chalk: ChalkClient | None = None


def _clients() -> tuple[OpenAI, ChalkClient]:
    global _llm, _chalk
    if _llm is None:
        _llm = OpenAI()
    if _chalk is None:
        _chalk = ChalkClient()
    return _llm, _chalk


SYSTEM_PROMPT = (
    "You investigate refund claims for potential fraud. "
    "Before gathering any evidence, your FIRST action must be to call submit_plan "
    "with your ordered investigation steps — author a crisp 4-7 word imperative label "
    "for each step in your own words. After the plan is accepted, follow this sequence — "
    "do not skip steps: "
    "1. Call get_fraud_prediction to check the user's individual fraud signals. "
    "2. Call check_refund_volume_trend to check for a broader refund anomaly — do this for every claim regardless of individual signals. "
    "3. If a volume spike is detected, call investigate_spike_pattern to characterise the suspicious cohort. "
    "4. Call check_cohort_match to determine how closely this user fits the cohort. "
    "Issue a verdict only after all four steps. "
    "If the user is individually clean but matches a suspicious broader pattern (2+ cohort factors), ESCALATE — do not APPROVE in isolation. "
    "Format your final response exactly as (no markdown, no bullet symbols, no --- separators):\n"
    "APPROVE|DENY|ESCALATE\n"
    "<one sentence of reasoning>"
)

# ── Investigation plan ────────────────────────────────────────────────────────
# The agent drafts the plan as its FIRST action each investigation, by calling the
# submit_plan tool. INVESTIGATION_PLAN below is the canonical fallback + the source
# of default labels and the canonical tool order.

INVESTIGATION_PLAN = [
    {"id": "h1", "label": "Check user fraud baseline",          "tool": "get_fraud_prediction"},
    {"id": "h2", "label": "Check broader refund context",       "tool": "check_refund_volume_trend"},
    {"id": "h3", "label": "Drill into spike pattern",           "tool": "investigate_spike_pattern"},
    {"id": "h4", "label": "Assess cohort match for this user",  "tool": "check_cohort_match"},
]

_TOOL_TO_HYP = {step["tool"]: step["id"] for step in INVESTIGATION_PLAN}

_CANON_TOOLS    = [step["tool"] for step in INVESTIGATION_PLAN]
_DEFAULT_LABELS = {step["tool"]: step["label"] for step in INVESTIGATION_PLAN}

# PLAN_MODE controls how much freedom the agent has over the plan:
#   "labels" — agent authors the step labels, but the steps/order are normalized to the
#              canonical four tools. Reliable for demos. (current)
#   "free"   — agent chooses which tools to use and in what order; the tree renders
#              whatever it returns.
PLAN_MODE = "labels"


def _normalize_plan(plan_steps: list) -> list[dict]:
    """Normalize the agent's submitted plan into renderable tree steps."""
    authored: dict[str, str] = {}
    order: list[str] = []
    for step in plan_steps or []:
        if not isinstance(step, dict):
            continue
        tool  = step.get("tool")
        label = (step.get("label") or "").strip()
        if tool in _CANON_TOOLS and tool not in authored:
            authored[tool] = label or _DEFAULT_LABELS[tool]
            order.append(tool)

    tools = order if (PLAN_MODE == "free" and order) else _CANON_TOOLS
    return [
        {"id": f"h{i}", "label": authored.get(tool, _DEFAULT_LABELS[tool]), "tool": tool}
        for i, tool in enumerate(tools, 1)
    ]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": (
                "Submit your ordered investigation plan. Call this FIRST, before any other "
                "tool. Each step pairs a short imperative label (your own words) with the tool "
                "you will use for that step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Crisp 4-7 word imperative step label."},
                                "tool":  {"type": "string", "enum": _CANON_TOOLS},
                            },
                            "required": ["label", "tool"],
                        },
                    },
                },
                "required": ["plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fraud_prediction",
            "description": "Fetch real-time fraud prediction for a user from the Chalk fraud_model named query. Returns is_fraud and name_email_match_score.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "integer"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_refund_volume_trend",
            "description": "Check whether refund volume over the last 24 hours is anomalous compared to the 30-day baseline.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "investigate_spike_pattern",
            "description": "Characterise the suspicious refund cohort driving the current spike — account age, email age, filing time, merchant category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_window": {"type": "string", "description": "e.g. '24h'"},
                },
                "required": ["time_window"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_cohort_match",
            "description": "Check how many characteristics of the suspicious refund cohort this specific user matches.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "integer"}},
                "required": ["user_id"],
            },
        },
    },
]

# ── Mock aggregate data ───────────────────────────────────────────────────────

_COHORT_MATCH: dict[int, dict] = {
    1: {"matches": 3, "total": 4,
        "matched": ["account_age <45d", "email_age <21d", "filed 01:00–04:30 UTC"],
        "missed":  ["merchant category"]},
    2: {"matches": 2, "total": 4,
        "matched": ["email_age <21d", "filed 01:00–04:30 UTC"],
        "missed":  ["account_age", "merchant category"]},
    3: {"matches": 4, "total": 4,
        "matched": ["account_age <45d", "email_age <21d", "filed 01:00–04:30 UTC", "merchant category"],
        "missed":  []},
}
_DEFAULT_COHORT = {"matches": 2, "total": 4,
                   "matched": ["account_age <45d", "filed 01:00–04:30 UTC"],
                   "missed":  ["email_age", "merchant category"]}


def _run_tool(name: str, inp: dict) -> str:
    _, chalk = _clients()
    try:
        if name == "get_fraud_prediction":
            uid = int(inp["user_id"])
            ctx = chalk.query(
                input={"user.id": uid},
                output=["user.is_fraud", "user.name_email_match_score"],
                query_name="fraud_model",
                query_name_version="1.0.0",
            )
            is_fraud = ctx.get_feature_value("user.is_fraud")
            score    = ctx.get_feature_value("user.name_email_match_score")
            score_str = str(round(score, 1)) if score is not None else "unknown"
            return f"is_fraud={is_fraud}, name_email_match_score={score_str}"

        if name == "check_refund_volume_trend":
            return (
                "Yesterday: 52 refunds (6.4× above 30-day avg of 8.1/day). "
                "Spike began ~01:00 UTC. STATUS: ANOMALY"
            )

        if name == "investigate_spike_pattern":
            return (
                "44/52 claims in the cohort share: account age <45 days, "
                "email age <21 days, filed between 01:00–04:30 UTC, "
                "merchant category: consumer electronics."
            )

        if name == "check_cohort_match":
            uid   = int(inp["user_id"])
            match = _COHORT_MATCH.get(uid, _DEFAULT_COHORT)
            missed = ", ".join(match["missed"]) if match["missed"] else "none"
            return (
                f"{match['matches']}/{match['total']} cohort factors match. "
                f"Matched: {', '.join(match['matched'])}. "
                f"Not matched: {missed}."
            )

        return "error: unknown tool"

    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def _hyp_status(tool_name: str, result: str) -> str:
    """Derive a hypothesis status from the tool result."""
    if tool_name in ("check_refund_volume_trend", "investigate_spike_pattern"):
        return "alert"
    if tool_name == "check_cohort_match":
        m = re.match(r"(\d+)/(\d+)", result)
        if m and int(m.group(1)) >= 2:
            return "alert"
        return "done"
    return "done"


def _investigate_impl(messages_json: str, followup: bool = False) -> Iterator[str]:
    """The agent loop as a generator of JSON-encoded UI events.

    Each yielded item is a JSON string (one SSE event). The final item is always a
    `_state` event carrying the full updated message history, which the caller stores
    so follow-up turns can continue the same conversation. Streamed back to the UI live.
    """
    messages = json.loads(messages_json)
    try:
        llm, _ = _clients()
        tool_to_hyp = dict(_TOOL_TO_HYP)
        plan_emitted = followup  # follow-ups never plan

        if not followup:
            yield json.dumps({"type": "planning"})

        while True:
            response = llm.chat.completions.create(
                model=_model, max_tokens=1024,
                tools=TOOLS if not followup else None,
                messages=messages,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                text = (msg.content or "").strip()
                verdict_match = re.search(r"\b(APPROVE|DENY|ESCALATE)\b", text)
                messages.append({"role": "assistant", "content": text})
                if verdict_match:
                    verdict = verdict_match.group(1)
                    reasoning = re.sub(r"^.*?\b(?:APPROVE|DENY|ESCALATE)\b[^\n]*\n?", "", text, flags=re.DOTALL).strip()
                    body = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", reasoning).strip()
                    yield json.dumps({"type": "decision", "verdict": verdict, "text": body})
                else:
                    yield json.dumps({"type": "question", "text": text})
                break

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                inp = json.loads(tc.function.arguments)

                # The agent's first action: it authors the plan, which builds the tree.
                if tc.function.name == "submit_plan":
                    plan = _normalize_plan(inp.get("plan", []))
                    tool_to_hyp = {step["tool"]: step["id"] for step in plan}
                    plan_emitted = True
                    yield json.dumps({"type": "plan", "steps": plan})
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": "Plan accepted. Proceed with the investigation."})
                    continue

                # Fallback: if the agent jumped straight to a tool, synthesize the plan.
                if not plan_emitted:
                    plan = _normalize_plan([])
                    tool_to_hyp = {step["tool"]: step["id"] for step in plan}
                    plan_emitted = True
                    yield json.dumps({"type": "plan", "steps": plan})

                hyp = tool_to_hyp.get(tc.function.name)

                yield json.dumps({"type": "tool_call", "id": tc.id, "name": tc.function.name, "args": inp})
                if hyp:
                    yield json.dumps({"type": "hypothesis_update", "id": hyp, "status": "running"})

                result = _run_tool(tc.function.name, inp)
                status = _hyp_status(tc.function.name, result)

                yield json.dumps({"type": "tool_result", "id": tc.id, "result": result})
                if hyp:
                    yield json.dumps({"type": "hypothesis_update", "id": hyp,
                                      "status": status, "summary": result[:90]})

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)})
    finally:
        # Always hand back the updated history so the caller can continue the session.
        yield json.dumps({"type": "_state", "messages": messages})


@chalkcompute.function(
    image=chalkcompute.Image.debian_slim().pip_install(["openai", "chalkpy"]),
    secrets=[chalkcompute.Secret.from_local_env_file(".env")],
    min_replicas=1,
    max_replicas=10,
)
def investigate(messages_json: str, followup: bool) -> str:
    """Run the whole refund investigation on Chalk Compute; return all UI events.

    NOTE on streaming: chalkcompute 2.0.0's generator-streaming call path
    (defer().stream()) does not deliver chunks back from the deployed runtime — even
    a trivial `yield "x"` function hangs on the client poll, while ordinary
    request/response calls work fine. So this function runs the loop to completion and
    returns the full event list as a JSON string; the UI replays it with light pacing
    to keep the step-by-step feel. The whole agent still runs here on Chalk Compute, so
    the run is traced in the Chalk console (server-side tracing is on by default).
    """
    return json.dumps(list(_investigate_impl(messages_json, followup)))


if __name__ == "__main__":
    import sys

    if "deploy" in sys.argv[1:]:
        print("Deploying investigate to Chalk Compute …")
        investigate.deploy()
        investigate.wait_ready()
        print("Deployed and ready (min_replicas=1, kept warm).")
    else:
        print(__doc__)
