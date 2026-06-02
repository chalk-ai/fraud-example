"""The entire Chalk Compute integration for the refund-abuse demo.

The whole story is the top of this file: resolve the deployed agent by name, call
it, and read the text it returns. The agent (`investigate_refund`) runs its whole
agentic loop *inside Chalk Compute* — discovering and querying features on its
own, secrets injected by Chalk, data never leaving the VPC. From here it's one
function call. Everything below `investigate()` is just parsing that text.
"""

import os
import re
from urllib.parse import urlencode

from chalkcompute import RemoteFunction
from dotenv import load_dotenv

load_dotenv()  # CHALK_* creds for the from_name lookup below


# resolve deployed agent by name — no URL, no client wiring
investigate_refund = RemoteFunction.from_name("investigate_refund")


def investigate(user_id: int, reason: str):
    # call the agent — runs server-side in Chalk Compute, we get text back
    agent_response = "".join(investigate_refund.remote(user_id, reason))   # -> "{trace}\n\n{verdict}"

    # we drive the UI from that text
    verdict, reasoning = split_verdict(agent_response)   # APPROVE / DENY / ESCALATE + one line
    steps              = parse_steps(agent_response)      # the agent's feature queries / tool calls

    return agent_response, verdict, reasoning, steps


# ── Reading the agent's text response ────────────────────────────────────────

# The verdict keyword. Feature values in the trace never contain these words, so
# the first match marks the boundary between the trace and the decision.
_VERDICT_RE = re.compile(r"\b(APPROVE|DENY|ESCALATE)\b")

# A trace line is `  name(args) → result`, where result may span multiple lines.
# Leading spaces are 0–2: the agent indents every step two spaces but lstrip()s
# the whole blob, so the FIRST line loses its indent. Call lines are identified by
# the `name(...) →` shape, so flush-left result lines (`user.total_spend: …`) never
# match. Match each call's result up to the next call line or end-of-trace.
_STEP_RE = re.compile(r"^ {0,2}(\w+)\((.*?)\)\s*→\s*(.*?)(?=\n {0,2}\w+\(.*?\)\s*→|\Z)",
                      re.DOTALL | re.MULTILINE)
# key=value pairs in an args string; values may be quoted/bracketed (and contain
# commas), so match those before the bare-token case.
_ARG_RE = re.compile(r"(\w+)=('[^']*'|\"[^\"]*\"|\[[^\]]*\]|[^,]+)")


def trace_block(raw: str) -> str:
    """The leading tool-call trace, i.e. everything before the verdict."""
    m = _VERDICT_RE.search(raw)
    return raw[:m.start()] if m else raw


def split_verdict(raw: str) -> tuple[str | None, str]:
    """Pull (APPROVE|DENY|ESCALATE, reasoning) out of the response.

    Keyed off the first verdict keyword rather than block splitting, because the
    model may put a blank line between the verdict and its reasoning.
    """
    m = _VERDICT_RE.search(raw)
    if not m:
        return None, raw.strip()
    reasoning = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", raw[m.end():]).strip(" :\n-")
    return m.group(1), reasoning or raw[m.start():].strip()


def parse_steps(raw: str) -> list[dict]:
    """Reconstruct the ordered tool calls from the trace.

    Each step carries its char offset (`end`) within trace_block(raw) so the UI
    can type the transcript out in step-sized slices.
    """
    block = trace_block(raw)
    steps = []
    for i, m in enumerate(_STEP_RE.finditer(block)):
        name, args, result = m.group(1), m.group(2), m.group(3)
        steps.append({
            "id": f"s{i}",
            "tool": name,
            "label": _step_label(name, args),
            "args": _parse_args(args),
            "result": result.strip(),
            "end": m.end(),
        })
    return steps


def _parse_args(args: str) -> dict:
    """Turn an args string like `user_id=1, features='a,b'` into a dict."""
    out: dict = {}
    for k, v in _ARG_RE.findall(args):
        v = v.strip().strip("'\"")
        out[k] = int(v) if v.lstrip("-").isdigit() else v
    return out


def _step_label(name: str, args: str) -> str:
    """Short node title — feature short-names for get_chalk_features."""
    if name == "get_chalk_features":
        feats = re.findall(r"\b\w+\.(\w+)", args)
        if feats:
            shown = ", ".join(feats[:3])
            return f"{shown} +{len(feats) - 3}" if len(feats) > 3 else shown
    return name.replace("_", " ")


# ── Console trace link (every call is traced server-side) ────────────────────
# We can't mint the per-span deep link client-side (operator/span ids are
# server-assigned), but we can deep-link to the scaling group's flame-graph view
# scoped to the call's time window — the user's run sits right at the top.

CONSOLE_BASE = os.environ.get("CHALK_CONSOLE_BASE", "https://chalk.ai").rstrip("/")
CONSOLE_PROJECT = os.environ.get("CHALK_CONSOLE_PROJECT", "cmpnck95f00090hs67kq4n6fb")
ENV_ID = os.environ.get("CHALK_ENVIRONMENT_ID", "clk8fc4d2e1")
_SG_FALLBACK = "investigate-refund"


def trace_url(start_s: float, end_s: float) -> str:
    """Console flame-graph trace view for the agent, windowed ±5min around the call."""
    vi = getattr(investigate_refund, "version_info", None)
    sg = (getattr(vi, "scaling_group_name", "") if vi else "") or _SG_FALLBACK
    qs = urlencode({
        "v": "remote-call-traces",
        "ds": int(start_s * 1000) - 300_000,
        "de": int(end_s * 1000) + 300_000,
        "scalingGroupTraceView": "flame-graph",
    })
    return (f"{CONSOLE_BASE}/projects/{CONSOLE_PROJECT}"
            f"/environments/{ENV_ID}/scaling-groups/{sg}?{qs}")
