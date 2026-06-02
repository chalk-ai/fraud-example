"""The entire Chalk Compute integration for the refund-abuse demo.

Three things happen here, and nothing else:
  1. Resolve Elliot's deployed agent by name        -> get_agent()
  2. Call it over the wire and get its text back     -> investigate()
  3. Read that text into a verdict + the tool calls  -> split_verdict() / parse_steps()

The agent (`investigate_refund`) runs its whole agentic loop *inside Chalk
Compute* — discovering and querying features on its own, secrets injected by
Chalk, data never leaving the VPC. From here it's one function call.
"""

import os
import re
import time
from urllib.parse import urlencode

from chalkcompute import RemoteFunction
from dotenv import load_dotenv

# RemoteFunction.from_name authenticates with CHALK_* creds from the environment.
load_dotenv()

REMOTE_FN_NAME = "investigate_refund"

_agent = None


def get_agent() -> RemoteFunction:
    """Resolve a handle to the deployed agent by name (no URL, no client setup)."""
    global _agent
    if _agent is None:
        _agent = RemoteFunction.from_name(REMOTE_FN_NAME)
    return _agent


def investigate(user_id: int, reason: str) -> str:
    """Call the deployed agent and return its full response text.

    `.remote(...)` is the wire/RPC call (chalk-sandbox-sdk#180 split bare `fn(...)`
    off to run locally). It returns `"{trace}\n\n{verdict}"` — a list of the
    `get_chalk_features(...)` tool calls the agent made, then APPROVE/DENY/ESCALATE
    plus one line of reasoning.
    """
    fn = get_agent()
    call = getattr(fn, "remote", None) or fn
    return _collect_text(call(user_id, reason))


# ── Reading the agent's text response ────────────────────────────────────────

# The verdict keyword. Feature values in the trace never contain these words, so
# the first match marks the boundary between the trace and the decision.
_VERDICT_RE = re.compile(r"\b(APPROVE|DENY|ESCALATE)\b")

# A trace line is `  name(args) → result`, where result may span multiple lines.
# Match each call's result up to the next call's `name(` or end-of-trace.
_STEP_RE = re.compile(r"^ {2}(\w+)\((.*?)\)\s*→\s*(.*?)(?=\n {2}\w+\(|\Z)",
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


def _collect_text(result) -> str:
    """Drain the remote call into one string.

    `.remote(...)` returns a scalar string or an iterator of text deltas. NB: the
    compute transport currently buffers a generator's output and delivers every
    yield at once on completion, so iterating here is not incremental.
    """
    if isinstance(result, (str, bytes, bytearray)):
        return _coerce_text(result)
    return "".join(_coerce_text(item) for item in result)


def _coerce_text(item) -> str:
    if isinstance(item, (bytes, bytearray)):
        return item.decode()
    if isinstance(item, str):
        return item
    if hasattr(item, "to_pydict"):  # defensively unwrap an Arrow batch
        vals = item.to_pydict().get("result") or []
        return "".join(v if isinstance(v, str) else str(v) for v in vals)
    return str(item)


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
    vi = getattr(get_agent(), "version_info", None)
    sg = (getattr(vi, "scaling_group_name", "") if vi else "") or _SG_FALLBACK
    qs = urlencode({
        "v": "remote-call-traces",
        "ds": int(start_s * 1000) - 300_000,
        "de": int(end_s * 1000) + 300_000,
        "scalingGroupTraceView": "flame-graph",
    })
    return (f"{CONSOLE_BASE}/projects/{CONSOLE_PROJECT}"
            f"/environments/{ENV_ID}/scaling-groups/{sg}?{qs}")
