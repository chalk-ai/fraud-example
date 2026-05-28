#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute>=1.5.17", "pandas"]
# ///
"""Refund-abuse investigation agent — Snowflake Summit demo.

Four things to notice in this file:

  1. SPRINKLE CHALK ON YOUR AGENT — one decorator, one entry point.
  2. SECRETS — the agent never holds the LLM key. Chalk injects it.
  3. CONTEXT ENGINE — one line pulls real-time features from your store,
     inside your VPC, no data leaves.
  4. BRING YOUR OWN MODEL — swap Option A (hosted API) for Option B
     (self-hosted on Chalk Compute, see chalkcompute_vllm_server.py).
     Nothing — prompt, response, or weights — ever leaves your cloud.

Setup (once):
  - Deploy the features:   chalk apply
  - Local .env:            CHALK_API_SERVER, CHALK_CLIENT_ID,
                           CHALK_CLIENT_SECRET, CHALK_ENVIRONMENT(_ID),
                           ANTHROPIC_API_KEY

Run:
  ./chalkcompute_agent_demo.py            # investigate one order
  ./chalkcompute_agent_demo.py fanout     # fan out across 50 orders
"""

import chalkcompute


SYSTEM_PROMPT = (
    "You investigate refund claims. Reply with APPROVE or DENY on the "
    "first line, then one sentence of reasoning."
)


@chalkcompute.function(
    image=chalkcompute.Image.debian_slim().pip_install(
        ["pydantic-ai-slim[anthropic,openai]", "chalkpy"],
    ),
    secrets=[chalkcompute.Secret.from_local_env_file(".env")],  # ← uploaded once, injected at runtime
    min_replicas=1,
    max_replicas=10,  # lets Chalk fan out across pods for the 50-order run
)
def investigate_refund(order_id: str, reason: str) -> str:
    import os  # noqa: F401  — used by Option B
    from chalk.client import ChalkClient
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel  # noqa: F401
    from pydantic_ai.providers.openai import OpenAIProvider  # noqa: F401

    # ── Chalk context engine: one line, runs in your VPC, no data leaves ──
    ctx = ChalkClient().query(
        input={"order.id": order_id},
        output=["order.refund_risk_score", "order.customer_prior_refunds"],
    )
    risk = ctx.get_feature_value("order.refund_risk_score")
    prior = ctx.get_feature_value("order.customer_prior_refunds")

    # ── Pick your LLM. Same agent code; just point at a different model. ──

    # OPTION A — third-party API (prompt + response leave your cloud):
    agent = Agent("anthropic:claude-sonnet-4-6", system_prompt=SYSTEM_PROMPT)

    # OPTION B — self-hosted on Chalk Compute (nothing leaves your cloud):
    # agent = Agent(
    #     OpenAIChatModel(
    #         "Qwen/Qwen2.5-7B-Instruct",
    #         provider=OpenAIProvider(
    #             base_url=f"{os.environ['VLLM_URL']}/v1",
    #             api_key="not-required",
    #         ),
    #     ),
    #     system_prompt=SYSTEM_PROMPT,
    # )

    prompt = (
        f"Order {order_id}. Refund reason: {reason!r}. "
        f"Risk score: {risk:.2f}. Prior refund claims: {prior}."
    )
    return agent.run_sync(prompt).output


_DEMO_ORDERS = {
    "1": ("ORD-8823", "high risk — 4 prior claims"),
    "2": ("ORD-1001", "low risk — no prior claims"),
    "3": ("ORD-4242", "medium risk — 2 prior claims"),
}


def run_one() -> None:
    """Scene 4 — investigate a single refund."""
    print("\nSelect an order:")
    for key, (order_id, description) in _DEMO_ORDERS.items():
        print(f"  {key}. {order_id}  ({description})")
    choice = input("\nOrder [1]: ").strip() or "1"
    order_id, _ = _DEMO_ORDERS.get(choice, _DEMO_ORDERS["1"])

    reason = input("Refund reason: ").strip() or "Item arrived damaged"

    print()
    print(investigate_refund(order_id, reason))


def run_fanout(n: int = 50) -> None:
    """Scene 5 — fan out across N historical orders, concurrently."""
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

    known = ["ORD-8823", "ORD-1001", "ORD-4242"]
    reasons = [
        "Item arrived damaged",
        "Wrong item received",
        "Item not as described",
        "Quality issue",
        "Item never delivered",
    ]
    orders = pd.DataFrame({
        "order_id": [known[i % len(known)] if i < 30 else f"ORD-{9000+i}" for i in range(n)],
        "reason": [reasons[i % len(reasons)] for i in range(n)],
    })
    print(f"\nFan-out validation: {len(orders)} orders\n")
    print(orders.head(), "\n")

    # ── ONE LINE: fan out N concurrent agent invocations ──
    t0 = time.time()
    decisions = orders.chalk.apply(investigate_refund)
    elapsed = time.time() - t0

    orders["decision"] = [d.split("\n")[0].strip() for d in decisions]
    approve = orders["decision"].str.startswith("APPROVE").sum()
    deny = orders["decision"].str.startswith("DENY").sum()
    print(orders[["order_id", "reason", "decision"]].to_string(index=False))
    print(
        f"\n{len(orders)} agents finished in {elapsed:.1f}s "
        f"(~{elapsed / len(orders) * 1000:.0f}ms/agent average) — "
        f"{approve} APPROVE, {deny} DENY"
    )


if __name__ == "__main__":
    import sys

    if "fanout" in sys.argv[1:]:
        run_fanout()
    else:
        run_one()
