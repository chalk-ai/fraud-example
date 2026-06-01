#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute>=1.5.17", "openai", "pandas"]
# ///
"""Refund-abuse investigation agent — Snowflake Summit demo.

Four things to notice in this file:

  1. SPRINKLE CHALK ON YOUR AGENT — one decorator, one entry point.
  2. SECRETS — the agent never holds the LLM key. Chalk injects it.
  3. CONTEXT ENGINE — tool calls pull real-time features from your store,
     inside your VPC, no data leaves.
  4. BRING YOUR OWN MODEL — swap Option A (hosted API) for Option B
     (self-hosted on Chalk Compute, see chalkcompute_vllm_server.py).
     Nothing — prompt, response, or weights — ever leaves your cloud.

Setup (once):
  - Deploy the features:   chalk apply
  - Local .env:            CHALK_API_SERVER, CHALK_CLIENT_ID,
                           CHALK_CLIENT_SECRET, CHALK_ENVIRONMENT(_ID),
                           VLLM_URL (from chalkcompute_vllm_server.py)

Run:
  ./chalkcompute_agent_demo.py            # investigate one order
  ./chalkcompute_agent_demo.py fanout     # fan out across 50 orders
"""

import chalkcompute


SYSTEM_PROMPT = (
    "You investigate refund claims for potential abuse. "
    "You have access to real-time signals from the Chalk feature store — "
    "relevant tools include the fraud prediction and account risk signals. "
    "Use your tools to gather the evidence you need — look up the fraud prediction first, "
    "then decide whether you need more context before ruling. "
    "Reply with APPROVE, DENY, or ESCALATE on the first line, "
    "then one sentence of reasoning."
)


@chalkcompute.function(
    image=chalkcompute.Image.debian_slim().pip_install(
        ["openai", "chalkpy"],
    ),
    secrets=[chalkcompute.Secret.from_local_env_file(".env")],
    min_replicas=1,
    max_replicas=10,
)
def investigate_refund(user_id: int, reason: str) -> str:
    import json
    import os
    from openai import OpenAI
    from chalk.client import ChalkClient

    chalk = ChalkClient()

    # ── LLM Option A: self-hosted Qwen2.5-7B on Chalk Compute ────────────────────
    # client = OpenAI(base_url=os.environ["VLLM_URL"] + "/v1", api_key="EMPTY")
    # model = "Qwen/Qwen2.5-7B-Instruct"

    # ── LLM Option B: hosted Claude API (active) ──────────────────────────────────
    client = OpenAI(
        base_url="https://api.anthropic.com/v1",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        default_headers={"anthropic-version": "2023-06-01"},
    )
    model = "claude-sonnet-4-6"

    # ── Tools: the agent decides which to call and in what order ──────────────
    # TODO: Replace this with a call to the MCP Gateway
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_fraud_prediction",
                "description": (
                    "Fetch the real-time fraud prediction for a user from the Chalk "
                    "fraud_model named query. Returns is_fraud (bool) and "
                    "name_email_match_score (0–100, higher = better match)."
                ),
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
                "name": "get_account_risk",
                "description": (
                    "Look up account risk signals for a user: whether they appear on a "
                    "denylist, and how old their email address is in days."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "integer"}},
                    "required": ["user_id"],
                },
            },
        },
    ]

    # ── Context engine: one line each, runs in your VPC, no data leaves ───────
    def run_tool(name: str, inp: dict) -> str:
        import traceback as _tb
        try:
            uid = int(inp["user_id"])
            if name == "get_fraud_prediction":
                ctx = chalk.query(
                    input={"user.id": uid},
                    output=["user.is_fraud", "user.name_email_match_score"],
                    query_name="fraud_model",
                    query_name_version="1.0.0",
                )
                is_fraud = ctx.get_feature_value("user.is_fraud")
                score = ctx.get_feature_value("user.name_email_match_score")
                score_str = str(round(score, 1)) if score is not None else "unknown"
                return f"is_fraud={is_fraud}, name_email_match_score={score_str}"
            if name == "get_account_risk":
                ctx = chalk.query(
                    input={"user.id": uid},
                    output=["user.denylisted", "user.email_age_days"],
                    query_name="fraud_model",
                    query_name_version="1.0.0",
                )
                denylisted = ctx.get_feature_value("user.denylisted")
                email_age = ctx.get_feature_value("user.email_age_days")
                return f"denylisted={denylisted}, email_age_days={email_age}"
            return "error: unknown tool"
        except Exception as e:
            return f"error: {type(e).__name__}: {e}\n{_tb.format_exc()}"

    # ── Agentic tool-use loop ─────────────────────────────────────────────────
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User {user_id}. Refund reason: {reason!r}."},
    ]
    steps: list[str] = []

    while True:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            decision = msg.content or ""
            trace = "\n".join(steps)
            return f"{trace}\n\n{decision}".lstrip() if steps else decision

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            inp = json.loads(tc.function.arguments)
            result = run_tool(tc.function.name, inp)
            args = ", ".join(f"{k}={v!r}" for k, v in inp.items())
            steps.append(f"  {tc.function.name}({args}) → {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


_DEMO_USERS = {
    "1": (1, "high risk — flagged account"),
    "2": (2, "low risk — clean account"),
    "3": (3, "medium risk — new email"),
}


def run_one() -> None:
    """Scene — investigate a single refund."""
    print("\nSelect a user:")
    for key, (user_id, description) in _DEMO_USERS.items():
        print(f"  {key}. user_id={user_id}  ({description})")
    choice = input("\nUser [1]: ").strip() or "1"
    user_id, _ = _DEMO_USERS.get(choice, _DEMO_USERS["1"])

    reason = input("Refund reason: ").strip() or "Item arrived damaged"

    print()
    try:
        print(investigate_refund(user_id, reason))
    except RuntimeError as e:
        if "503" in str(e):
            print("Error: vLLM server unavailable (503). Re-run ./chalkcompute_vllm_server.py to restart it.")
        else:
            raise


def run_fanout(n: int = 50) -> None:
    """Scene — fan out across N historical orders, concurrently."""
    import re
    import time
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    @pd.api.extensions.register_dataframe_accessor("chalk")
    class _ChalkAccessor:
        def __init__(self, df: pd.DataFrame):
            self._df = df

        def apply(self, fn) -> list:
            with ThreadPoolExecutor(max_workers=len(self._df)) as ex:
                return list(ex.map(lambda r: fn(*r[1:]), self._df.itertuples()))

    known = [1, 2, 3]
    reasons = [
        "Item arrived damaged",
        "Wrong item received",
        "Item not as described",
        "Quality issue",
        "Item never delivered",
    ]
    orders = pd.DataFrame({
        "user_id": [known[i % len(known)] if i < 30 else (100 + i) for i in range(n)],
        "reason": [reasons[i % len(reasons)] for i in range(n)],
    })
    print(f"\nFan-out: {len(orders)} users\n")

    # ── ONE LINE: fan out N concurrent agent invocations ──
    t0 = time.time()
    try:
        decisions = orders.chalk.apply(investigate_refund)
    except RuntimeError as e:
        if "503" in str(e):
            print("Error: vLLM server unavailable (503). Re-run ./chalkcompute_vllm_server.py to restart it.")
            return
        raise
    elapsed = time.time() - t0

    def _verdict(text: str) -> str:
        m = re.search(r"\b(APPROVE|DENY|ESCALATE)\b", text)
        return m.group(0) if m else text.split("\n")[0].strip()

    orders["decision"] = [_verdict(d) for d in decisions]
    approve = orders["decision"].eq("APPROVE").sum()
    deny = orders["decision"].eq("DENY").sum()
    escalate = orders["decision"].eq("ESCALATE").sum()
    print(orders[["user_id", "reason", "decision"]].to_string(index=False))
    print(
        f"\n{len(orders)} agents finished in {elapsed:.1f}s "
        f"(~{elapsed / len(orders) * 1000:.0f}ms/agent average) — "
        f"{approve} APPROVE, {deny} DENY, {escalate} ESCALATE"
    )


if __name__ == "__main__":
    import sys

    if "fanout" in sys.argv[1:]:
        run_fanout()
    else:
        run_one()
