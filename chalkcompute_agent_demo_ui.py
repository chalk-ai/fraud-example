#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["fastapi", "uvicorn[standard]", "openai", "chalkpy", "python-dotenv"]
# ///
"""Refund-abuse agent demo UI — investigation plan edition.

Agent discovers that an individually clean refund claim is part of a broader
suspicious spike, builds an investigation plan, executes it step by step, and
ESCALATEs with suggested next actions.

Run:
  ./chalkcompute_agent_demo_ui.py
  open http://localhost:8000
"""

import asyncio
import json
import os
import queue
import re
import sys
import threading
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Search for .env from cwd upward so it works regardless of where uv places __file__
_dir = os.getcwd()
for _ in range(5):
    _candidate = os.path.join(_dir, ".env")
    if os.path.exists(_candidate):
        load_dotenv(_candidate, override=True)
        break
    _dir = os.path.dirname(_dir)

from openai import OpenAI
from chalk.client import ChalkClient

_model = "claude-sonnet-4-6"
_llm:   OpenAI      | None = None
_chalk: ChalkClient | None = None


def _clients() -> tuple[OpenAI, ChalkClient]:
    global _llm, _chalk
    if _llm is None:
        _llm = OpenAI(
            base_url="https://api.anthropic.com/v1",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            default_headers={"anthropic-version": "2023-06-01"},
        )
    if _chalk is None:
        _chalk = ChalkClient()
    return _llm, _chalk

_sessions: dict[str, list] = {}

SYSTEM_PROMPT = (
    "You investigate refund claims for potential fraud. "
    "Always follow this investigation sequence — do not skip steps: "
    "1. Call get_fraud_prediction to check the user's individual fraud signals. "
    "2. Call check_refund_volume_trend to check for a broader refund anomaly — do this for every claim regardless of individual signals. "
    "3. If a volume spike is detected, call investigate_spike_pattern to characterise the suspicious cohort. "
    "4. Call check_cohort_match to determine how closely this user fits the cohort. "
    "Issue a verdict only after all four steps. "
    "If the user is individually clean but matches a suspicious broader pattern (2+ cohort factors), ESCALATE — do not APPROVE in isolation. "
    "Format your final response exactly as (no markdown, no bullet symbols, no --- separators):\n"
    "APPROVE|DENY|ESCALATE\n"
    "<one sentence of reasoning>\n"
    "NEXT_STEPS: [\"action 1\", \"action 2\", \"action 3\"]"
)

# ── Investigation plan ────────────────────────────────────────────────────────
# Hardcoded for now; swap to an LLM-generated plan by replacing INVESTIGATION_PLAN
# with a pre-tool chat.completions call that returns the same JSON shape.

INVESTIGATION_PLAN = [
    {"id": "h1", "label": "Check user fraud baseline",          "tool": "get_fraud_prediction"},
    {"id": "h2", "label": "Check broader refund context",       "tool": "check_refund_volume_trend"},
    {"id": "h3", "label": "Drill into spike pattern",           "tool": "investigate_spike_pattern"},
    {"id": "h4", "label": "Assess cohort match for this user",  "tool": "check_cohort_match"},
]

_TOOL_TO_HYP = {step["tool"]: step["id"] for step in INVESTIGATION_PLAN}

TOOLS = [
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
    # 3/4 match — account is new, email is new, filed at 2am; merchant differs
    1: {"matches": 3, "total": 4,
        "matched": ["account_age <45d", "email_age <21d", "filed 01:00–04:30 UTC"],
        "missed":  ["merchant category"]},
    # 2/4 match — account older but email suspiciously new (burner?), filing time matches
    2: {"matches": 2, "total": 4,
        "matched": ["email_age <21d", "filed 01:00–04:30 UTC"],
        "missed":  ["account_age", "merchant category"]},
    # 4/4 match — hits every pattern factor
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


def _agent_thread(messages: list, q: queue.Queue, followup: bool = False) -> None:
    try:
        if not followup:
            q.put({"type": "plan", "steps": INVESTIGATION_PLAN})
        llm, _ = _clients()

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
                if verdict_match:
                    verdict = verdict_match.group(1)
                    next_steps: list[str] = []
                    ns = re.search(r"NEXT_STEPS:\s*(\[.*?\])", text, re.DOTALL)
                    if ns:
                        try:
                            next_steps = json.loads(ns.group(1))
                        except Exception:
                            pass
                    # Strip NEXT_STEPS and find the reasoning after the verdict keyword
                    reasoning = re.sub(r"NEXT_STEPS:.*", "", text, flags=re.DOTALL).strip()
                    # Drop everything up to and including the verdict line
                    reasoning = re.sub(r"^.*?\b(?:APPROVE|DENY|ESCALATE)\b[^\n]*\n?", "", reasoning, flags=re.DOTALL).strip()
                    # Strip markdown bold/italic markers
                    body = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", reasoning).strip()
                    q.put({"type": "decision", "verdict": verdict,
                           "text": body, "next_steps": next_steps})
                else:
                    messages.append({"role": "assistant", "content": text})
                    q.put({"type": "question", "text": text})
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
                inp  = json.loads(tc.function.arguments)
                hyp  = _TOOL_TO_HYP.get(tc.function.name)

                q.put({"type": "tool_call", "id": tc.id, "name": tc.function.name, "args": inp})
                if hyp:
                    q.put({"type": "hypothesis_update", "id": hyp, "status": "running"})

                result = _run_tool(tc.function.name, inp)
                status = _hyp_status(tc.function.name, result)

                q.put({"type": "tool_result", "id": tc.id, "result": result})
                if hyp:
                    q.put({"type": "hypothesis_update", "id": hyp,
                           "status": status, "summary": result[:90]})

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    except Exception as e:
        q.put({"type": "error", "message": str(e)})
    finally:
        q.put(None)


async def _sse(q: queue.Queue):
    while True:
        try:
            event = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue
        if event is None:
            yield "data: [DONE]\n\n"
            return
        yield f"data: {json.dumps(event)}\n\n"


app = FastAPI()


class InvestigateRequest(BaseModel):
    user_id: int
    reason: str


class ReplyRequest(BaseModel):
    message: str


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(HTML)


@app.post("/investigate")
async def investigate(req: InvestigateRequest) -> StreamingResponse:
    session_id = str(uuid.uuid4())
    messages: list = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"User {req.user_id}. Refund reason: {req.reason!r}."},
    ]
    _sessions[session_id] = messages

    q: queue.Queue = queue.Queue()
    threading.Thread(target=_agent_thread, args=(messages, q), daemon=True).start()

    async def stream():
        yield f"data: {json.dumps({'type': 'session', 'id': session_id})}\n\n"
        async for chunk in _sse(q):
            yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/reply/{session_id}")
async def reply(session_id: str, req: ReplyRequest) -> StreamingResponse:
    messages = _sessions.get(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages.append({"role": "user", "content": req.message + "\n\n(Answer this follow-up question directly and conversationally. Do not re-run the investigation or issue a new verdict.)"})
    q: queue.Queue = queue.Queue()
    threading.Thread(target=_agent_thread, args=(messages, q, True), daemon=True).start()
    return StreamingResponse(_sse(q), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    _sessions.pop(session_id, None)
    return {"ok": True}


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Acme Corp. Refund Investigator</title>
<style>
  :root {
    --bg:          #0a0a0a;
    --surface:     #131313;
    --surface2:    #1c1c1c;
    --border:      #252525;
    --text:        #e8e8e8;
    --muted:       #555;
    --accent:      #6366f1;
    --green:       #22c55e;
    --red:         #ef4444;
    --amber:       #f59e0b;
    --tool-bg:     #0d0d1e;
    --tool-border: #2a2a5a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text);
    height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  }

  /* ── Top-level split layout ── */
  .app-body {
    display: flex; flex: 1; overflow: hidden;
  }

  /* ── Left panel ── */
  .left-panel {
    width: 440px; flex-shrink: 0;
    display: flex; flex-direction: column;
    border-right: 1px solid var(--border); overflow: hidden;
  }

  /* ── Right panel ── */
  .right-panel {
    flex: 1; display: flex; flex-direction: column;
    background-color: var(--bg);
    background-image: radial-gradient(circle, #282828 1px, transparent 1px);
    background-size: 24px 24px;
    overflow: hidden;
  }
  .right-header {
    padding: 15px 24px; border-bottom: 1px solid var(--border); flex-shrink: 0;
    font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted);
    /* match left header height — same padding as .header */
    display: flex; align-items: center;
  }
  .right-body {
    flex: 1; overflow: auto; position: relative;
    display: flex; align-items: flex-start; justify-content: center;
    padding: 48px 32px;
  }

  /* ── Tree empty state ── */
  .tree-empty {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    color: var(--muted); pointer-events: none;
  }
  .tree-empty-icon { font-size: 32px; opacity: 0.18; }
  .tree-empty-text { font-size: 13px; }

  /* ── Tree canvas ── */
  #treeCanvas {
    position: relative; width: 580px; height: 560px; flex-shrink: 0;
    display: none;
  }

  /* ── Tree nodes (shared base) ── */
  .tree-node {
    position: absolute; border-radius: 10px; padding: 10px 14px;
    font-size: 12px; line-height: 1.4;
    border: 1px solid var(--border);
    background: var(--surface);
    transition: background 0.25s, border-color 0.25s, opacity 0.25s;
  }
  .tree-node-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 4px;
  }
  .tree-node-title {
    font-size: 12px; font-weight: 600; color: var(--text);
    font-family: 'SF Mono', 'Menlo', monospace;
  }
  .tree-node-summary {
    font-size: 11px; color: var(--muted); margin-top: 4px;
    font-family: 'SF Mono', 'Menlo', monospace; line-height: 1.35;
  }

  /* ── SOURCE node ── */
  #tree-source {
    left: 150px; top: 0; width: 280px; height: 62px;
    background: rgba(239,68,68,.08); border-color: rgba(239,68,68,.3);
  }
  #tree-source .tree-node-label { color: rgba(239,100,100,.7); }
  #tree-source .tree-node-title { color: #fca5a5; font-size: 12px; }

  /* ── H nodes (hypothesis) ── */
  .tree-hyp {
    width: 130px; height: 90px;
    opacity: 0.35;
  }
  #tree-h1 { left: 5px;   top: 115px; }
  #tree-h2 { left: 150px; top: 115px; }
  #tree-h3 { left: 295px; top: 115px; }
  #tree-h4 { left: 440px; top: 115px; }

  .tree-hyp.hyp-running {
    opacity: 1;
    background: rgba(99,102,241,.1); border-color: rgba(99,102,241,.4);
  }
  .tree-hyp.hyp-done {
    opacity: 1;
    background: rgba(34,197,94,.07); border-color: rgba(34,197,94,.3);
  }
  .tree-hyp.hyp-alert {
    opacity: 1;
    background: rgba(245,158,11,.08); border-color: rgba(245,158,11,.3);
  }

  .hyp-icon {
    font-size: 16px; display: inline-block; margin-bottom: 4px;
  }
  .hyp-icon.spinning { animation: spin 1s linear infinite; }

  /* ── CONCLUSION node ── */
  #tree-conclusion {
    left: 150px; top: 290px; width: 280px; height: 85px;
    display: none;
  }
  #tree-conclusion.verdict-approve  { background: rgba(34,197,94,.08);  border-color: rgba(34,197,94,.3); }
  #tree-conclusion.verdict-deny     { background: rgba(239,68,68,.08);  border-color: rgba(239,68,68,.3); }
  #tree-conclusion.verdict-escalate { background: rgba(245,158,11,.08); border-color: rgba(245,158,11,.3); }
  #tree-conclusion .conc-label {
    font-size: 13px; font-weight: 700; letter-spacing: .04em; margin-bottom: 4px;
  }
  #tree-conclusion.verdict-approve  .conc-label { color: var(--green); }
  #tree-conclusion.verdict-deny     .conc-label { color: var(--red); }
  #tree-conclusion.verdict-escalate .conc-label { color: var(--amber); }
  #tree-conclusion .conc-text { font-size: 11px; color: var(--muted); line-height: 1.4; }

  /* ── NEXT STEP nodes ── */
  .tree-nextstep {
    left: 150px; width: 280px; height: 44px;
    background: var(--surface2); border-color: var(--border);
    display: none;
    align-items: center; gap: 8px;
    padding: 0 14px;
  }
  /* use flex display when visible */
  .tree-nextstep.visible { display: flex; }
  #tree-ns0 { top: 395px; }
  #tree-ns1 { top: 449px; }
  #tree-ns2 { top: 503px; }
  .ns-arrow { color: var(--muted); font-size: 11px; flex-shrink: 0; }
  .ns-text  { font-size: 12px; color: var(--text); }

  /* ── SVG connector lines ── */
  .tree-svg {
    position: absolute; top: 0; left: 0;
    width: 580px; height: 560px; pointer-events: none;
  }
  .tree-svg line, .tree-svg polyline {
    stroke: #333; stroke-width: 1.5; fill: none;
  }
  #svg-conclusion-line { display: none; }

  /* ── Header ── */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 15px 24px; border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  .header-left { display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
  .logo { width: 26px; height: 26px; background: var(--accent); border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; }
  .model-badge { font-size: 11px; color: var(--muted); background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; padding: 3px 10px; font-family: 'SF Mono', 'Menlo', monospace; }

  /* ── User selector ── */
  .user-sel { position: relative; flex-shrink: 0; }
  .user-sel-btn {
    display: flex; align-items: center; gap: 8px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 11px 14px;
    color: var(--text); font-size: 13px; font-family: 'SF Mono', 'Menlo', monospace;
    cursor: pointer; white-space: nowrap; transition: border-color 0.15s; user-select: none;
  }
  .user-sel-btn:hover:not(:disabled) { border-color: #333; }
  .user-sel-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .user-sel-chevron { color: var(--muted); font-size: 10px; transition: transform 0.15s; }
  .user-sel-btn.open .user-sel-chevron { transform: rotate(180deg); }
  .user-dropdown {
    position: absolute; bottom: calc(100% + 8px); left: 0;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 6px; min-width: 260px;
    box-shadow: 0 -8px 32px rgba(0,0,0,.5);
    display: flex; flex-direction: column; gap: 4px;
    z-index: 100; animation: rise 0.15s ease;
  }
  .user-option {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 12px; border-radius: 8px; cursor: pointer; transition: background 0.1s; user-select: none;
  }
  .user-option:hover { background: var(--surface2); }
  .user-option-id { font-size: 13px; font-weight: 600; font-family: 'SF Mono', 'Menlo', monospace; }
  .user-option-right { display: flex; align-items: center; gap: 7px; }
  .risk-badge { font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 2px 7px; border-radius: 4px; }
  .risk-high   { background: rgba(239,68,68,.15);  color: #f87171; }
  .risk-low    { background: rgba(34,197,94,.15);  color: #4ade80; }
  .risk-medium { background: rgba(245,158,11,.15); color: #fbbf24; }
  .user-desc   { font-size: 11px; color: var(--muted); }

  /* ── Chat ── */
  .chat { flex: 1; overflow-y: auto; padding: 28px 24px; display: flex; flex-direction: column; gap: 22px; }
  .chat::-webkit-scrollbar { width: 4px; }
  .chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  .empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; color: var(--muted); pointer-events: none; }
  .empty-icon { font-size: 36px; opacity: 0.25; }
  .empty-text { font-size: 13px; }

  /* Bubbles */
  .msg-user {
    align-self: flex-end; max-width: 420px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 16px 16px 4px 16px; padding: 12px 16px; animation: rise 0.2s ease;
  }
  .msg-user-ref    { font-size: 11px; color: var(--muted); font-family: monospace; margin-bottom: 4px; }
  .msg-user-reason { font-size: 14px; line-height: 1.4; }
  .msg-reply {
    align-self: flex-end; max-width: 420px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 16px 16px 4px 16px; padding: 12px 16px; font-size: 14px; line-height: 1.4; animation: rise 0.2s ease;
  }
  .msg-question {
    align-self: flex-start; max-width: 480px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px; padding: 12px 16px; font-size: 14px; line-height: 1.55; animation: rise 0.2s ease;
  }
  .msg-agent { align-self: flex-start; max-width: 620px; display: flex; flex-direction: column; gap: 8px; animation: rise 0.2s ease; }

  /* ── Plan card ── */
  .plan-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden; animation: rise 0.2s ease;
  }
  .plan-header {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
  }
  .plan-step {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 9px 14px; border-bottom: 1px solid var(--border); transition: background 0.2s;
  }
  .plan-step:last-child { border-bottom: none; }
  .plan-step.status-running { background: rgba(99,102,241,.05); }
  .plan-step.status-done    { background: rgba(34,197,94,.04); }
  .plan-step.status-alert   { background: rgba(245,158,11,.05); }
  .step-icon { font-size: 13px; min-width: 16px; margin-top: 1px; }
  .step-icon.pending  { opacity: 0.3; }
  .step-icon.running  { animation: spin 1s linear infinite; display: inline-block; }
  .step-body { display: flex; flex-direction: column; gap: 3px; }
  .step-label { font-size: 13px; color: var(--text); }
  .step-label.muted { color: var(--muted); }
  .step-summary { font-size: 11px; color: var(--muted); font-family: 'SF Mono', 'Menlo', monospace; line-height: 1.4; }
  .step-summary.alert  { color: #fbbf24; }
  .step-summary.done   { color: #4ade80; }

  /* ── Tool card ── */
  .tool-card {
    background: var(--tool-bg); border: 1px solid var(--tool-border);
    border-left: 3px solid var(--accent); border-radius: 8px;
    padding: 10px 14px; font-family: 'SF Mono', 'Menlo', monospace; font-size: 12px; animation: rise 0.15s ease;
  }
  .tool-header { display: flex; align-items: center; gap: 7px; color: #a5b4fc; margin-bottom: 3px; }
  .tool-fn   { font-weight: 600; }
  .tool-args { color: #4b5563; }
  .tool-result-row { margin-top: 7px; padding-top: 7px; border-top: 1px solid var(--tool-border); display: flex; align-items: center; gap: 7px; }
  .tool-result-row.pending { color: var(--muted); }
  .tool-result-row.done    { color: #86efac; }

  /* ── Thinking dots ── */
  .thinking { display: flex; align-items: center; gap: 5px; padding: 8px 2px; }
  .dot { width: 6px; height: 6px; background: var(--muted); border-radius: 50%; animation: pulse 1.2s ease-in-out infinite; }
  .dot:nth-child(2) { animation-delay: .2s; }
  .dot:nth-child(3) { animation-delay: .4s; }

  /* ── Verdict ── */
  .verdict-card { border-radius: 12px; padding: 16px 20px; border: 1px solid; animation: rise 0.25s ease; }
  .verdict-approve  { background: rgba(34,197,94,.08);  border-color: rgba(34,197,94,.25); }
  .verdict-deny     { background: rgba(239,68,68,.08);  border-color: rgba(239,68,68,.25); }
  .verdict-escalate { background: rgba(245,158,11,.08); border-color: rgba(245,158,11,.25); }
  .verdict-label { font-size: 18px; font-weight: 700; letter-spacing: .04em; margin-bottom: 6px; }
  .verdict-approve  .verdict-label { color: var(--green); }
  .verdict-deny     .verdict-label { color: var(--red);   }
  .verdict-escalate .verdict-label { color: var(--amber); }
  .verdict-text { font-size: 13px; color: #999; line-height: 1.55; }

  /* ── Next steps ── */
  .next-steps { display: flex; flex-direction: column; gap: 6px; margin-top: 4px; animation: rise 0.3s ease; }
  .next-steps-header { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); padding: 2px 0 4px; }
  .next-step-btn {
    display: flex; align-items: center; gap: 8px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 9px 14px; font-size: 13px; color: var(--text);
    font-family: inherit; cursor: default; text-align: left;
  }
  .next-step-arrow { color: var(--muted); font-size: 11px; }

  /* ── Error ── */
  .error-card { background: rgba(239,68,68,.07); border: 1px solid rgba(239,68,68,.2); border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #fca5a5; }

  /* ── Input bar ── */
  .input-bar { padding: 14px 24px; border-top: 1px solid var(--border); display: flex; gap: 10px; flex-shrink: 0; align-items: center; }
  .main-input { flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 11px 16px; color: var(--text); font-size: 14px; font-family: inherit; outline: none; transition: border-color 0.15s; }
  .main-input::placeholder { color: var(--muted); }
  .main-input:focus { border-color: #3a3a5a; }
  .main-input:disabled { opacity: 0.4; }
  .submit-btn { background: var(--accent); color: #fff; border: none; border-radius: 10px; padding: 11px 22px; font-size: 14px; font-weight: 600; font-family: inherit; cursor: pointer; transition: opacity 0.15s; white-space: nowrap; }
  .submit-btn:hover:not(:disabled) { opacity: .85; }
  .submit-btn:disabled { opacity: .35; cursor: not-allowed; }
  .dismiss-btn { background: transparent; color: var(--text); border: 1px solid #444; border-radius: 10px; padding: 11px 18px; font-size: 14px; font-family: inherit; cursor: pointer; transition: color 0.15s, border-color 0.15s, background 0.15s; white-space: nowrap; display: none; }
  .dismiss-btn:hover { background: var(--surface2); border-color: #666; }

  @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes pulse { 0%,80%,100% { transform: scale(.55); opacity: .35; } 40% { transform: scale(1); opacity: 1; } }
  @keyframes spin  { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
</style>
</head>
<body>

<!-- ── Top header spanning full width ── -->
<div class="header">
  <div class="header-left">
    <div class="logo">A</div>
    Acme Corp. Refund Investigator
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    <button id="newBtn" class="dismiss-btn" onclick="dismiss()" style="display:none">+ New</button>
    <div class="model-badge">built on Chalk Compute · claude-sonnet-4-6</div>
  </div>
</div>

<!-- ── Split body ── -->
<div class="app-body">

  <!-- Left panel: chat -->
  <div class="left-panel">
    <div class="chat" id="chat">
      <div class="empty-state" id="emptyState">
        <div class="empty-icon">⚖</div>
        <div class="empty-text">Select a user and describe the refund reason to begin</div>
      </div>
    </div>

    <div class="input-bar">
      <div class="user-sel" id="userSel">
        <button class="user-sel-btn" id="userSelBtn" onclick="toggleDropdown()">
          <span id="userSelLabel">Select user</span>
          <span class="user-sel-chevron">▾</span>
        </button>
        <div class="user-dropdown" id="userDropdown" style="display:none">
          <div class="user-option" onclick="selectUser(1,'medium')">
            <span class="user-option-id">user_id=1</span>
            <div class="user-option-right">
              <span class="risk-badge risk-medium">Medium</span>
              <span class="user-desc">acct age 38d</span>
            </div>
          </div>
          <div class="user-option" onclick="selectUser(2,'low')">
            <span class="user-option-id">user_id=2</span>
            <div class="user-option-right">
              <span class="risk-badge risk-low">Low</span>
              <span class="user-desc">acct age 4y</span>
            </div>
          </div>
          <div class="user-option" onclick="selectUser(3,'high')">
            <span class="user-option-id">user_id=3</span>
            <div class="user-option-right">
              <span class="risk-badge risk-high">New</span>
              <span class="user-desc">acct age 22d</span>
            </div>
          </div>
        </div>
      </div>
      <input id="mainInput" class="main-input" type="text"
             placeholder="Select a user first…"
             disabled
             onkeydown="if(event.key==='Enter')primaryAction()">
      <button id="dismissBtn" class="dismiss-btn" onclick="dismiss()">New investigation</button>
      <button id="submitBtn"  class="submit-btn"  onclick="primaryAction()" disabled>Investigate →</button>
    </div>
  </div><!-- /left-panel -->

  <!-- Right panel: hypothesis tree -->
  <div class="right-panel">
    <div class="right-header">Investigation Tree</div>
    <div class="right-body">

      <!-- Empty state -->
      <div class="tree-empty" id="treeEmpty">
        <div class="tree-empty-icon">⬡</div>
        <div class="tree-empty-text">Start an investigation to see the hypothesis tree</div>
      </div>

      <!-- Tree canvas — hidden until investigation starts -->
      <div id="treeCanvas">

        <!-- SVG connector lines
             Node geometry (absolute positions):
               SOURCE:     left=150 top=0   w=280 h=62  → center-bottom=(290,62)
               H1:         left=5   top=115 w=130 h=90  → center-top=(70,115)  center-bottom=(70,205)
               H2:         left=150 top=115 w=130 h=90  → center-top=(215,115) center-bottom=(215,205)
               H3:         left=295 top=115 w=130 h=90  → center-top=(360,115) center-bottom=(360,205)
               H4:         left=440 top=115 w=130 h=90  → center-top=(505,115) center-bottom=(505,205)
               CONCLUSION: left=150 top=290 w=280 h=85  → center-top=(290,290) center-bottom=(290,375)
               NS0:        left=150 top=395 w=280 h=44  → center-top=(290,395)
        -->
        <svg class="tree-svg" xmlns="http://www.w3.org/2000/svg">
          <!-- Source → fan junction (vertical drop to y=90) -->
          <line x1="290" y1="62" x2="290" y2="90"/>
          <!-- Horizontal fan rail from H1 center-x to H4 center-x at y=90 -->
          <line x1="70"  y1="90" x2="505" y2="90"/>
          <!-- Vertical drops from fan rail to each H node top -->
          <line x1="70"  y1="90" x2="70"  y2="115"/>
          <line x1="215" y1="90" x2="215" y2="115"/>
          <line x1="360" y1="90" x2="360" y2="115"/>
          <line x1="505" y1="90" x2="505" y2="115"/>
          <!-- Vertical rises from each H node bottom to collector rail at y=265 -->
          <line x1="70"  y1="205" x2="70"  y2="265"/>
          <line x1="215" y1="205" x2="215" y2="265"/>
          <line x1="360" y1="205" x2="360" y2="265"/>
          <line x1="505" y1="205" x2="505" y2="265"/>
          <!-- Horizontal collector rail at y=265 -->
          <line x1="70"  y1="265" x2="505" y2="265"/>
          <!-- Collector → conclusion top -->
          <line x1="290" y1="265" x2="290" y2="290"/>
          <!-- Conclusion bottom → first next-step top (shown only after verdict) -->
          <line id="svg-conclusion-line" x1="290" y1="375" x2="290" y2="395"/>
        </svg>

        <!-- SOURCE node -->
        <div class="tree-node" id="tree-source">
          <div class="tree-node-label">Source · Refund Claim</div>
          <div class="tree-node-title" id="tree-source-title">—</div>
        </div>

        <!-- H nodes -->
        <div class="tree-node tree-hyp" id="tree-h1">
          <div class="hyp-icon">○</div>
          <div class="tree-node-title" id="tree-h1-label">—</div>
          <div class="tree-node-summary" id="tree-h1-summary"></div>
        </div>
        <div class="tree-node tree-hyp" id="tree-h2">
          <div class="hyp-icon">○</div>
          <div class="tree-node-title" id="tree-h2-label">—</div>
          <div class="tree-node-summary" id="tree-h2-summary"></div>
        </div>
        <div class="tree-node tree-hyp" id="tree-h3">
          <div class="hyp-icon">○</div>
          <div class="tree-node-title" id="tree-h3-label">—</div>
          <div class="tree-node-summary" id="tree-h3-summary"></div>
        </div>
        <div class="tree-node tree-hyp" id="tree-h4">
          <div class="hyp-icon">○</div>
          <div class="tree-node-title" id="tree-h4-label">—</div>
          <div class="tree-node-summary" id="tree-h4-summary"></div>
        </div>

        <!-- CONCLUSION node -->
        <div class="tree-node" id="tree-conclusion">
          <div class="conc-label" id="tree-conc-label">—</div>
          <div class="conc-text"  id="tree-conc-text"></div>
        </div>

        <!-- NEXT STEP nodes -->
        <div class="tree-node tree-nextstep" id="tree-ns0">
          <span class="ns-arrow">→</span><span class="ns-text" id="tree-ns0-text"></span>
        </div>
        <div class="tree-node tree-nextstep" id="tree-ns1">
          <span class="ns-arrow">→</span><span class="ns-text" id="tree-ns1-text"></span>
        </div>
        <div class="tree-node tree-nextstep" id="tree-ns2">
          <span class="ns-arrow">→</span><span class="ns-text" id="tree-ns2-text"></span>
        </div>

      </div><!-- /treeCanvas -->

    </div><!-- /right-body -->
  </div><!-- /right-panel -->

</div><!-- /app-body -->

<script>
let selectedUser   = null;
let currentUserId  = null;
let currentReason  = null;
let sessionId      = null;
let mode           = 'idle';
let activeAgentMsg = null;
let activeThinking = null;
let planCard       = null;

// ── Dropdown ──────────────────────────────────────────────────────────────────

function toggleDropdown() {
  const btn  = document.getElementById('userSelBtn');
  const menu = document.getElementById('userDropdown');
  const open = menu.style.display !== 'none';
  menu.style.display = open ? 'none' : 'flex';
  btn.classList.toggle('open', !open);
}

function selectUser(id, risk) {
  if (sessionId) dismiss();
  selectedUser = id;

  const btn   = document.getElementById('userSelBtn');
  const label = document.getElementById('userSelLabel');
  label.textContent = `user_id=${id}`;
  btn.className = `user-sel-btn`;
  document.getElementById('userDropdown').style.display = 'none';

  const input  = document.getElementById('mainInput');
  const submit = document.getElementById('submitBtn');
  input.disabled    = false;
  input.placeholder = 'Describe the refund reason…';
  input.value       = 'Item not as described';
  submit.disabled   = false;
  input.focus(); input.select();
}

document.addEventListener('click', e => {
  const sel = document.getElementById('userSel');
  if (sel && !sel.contains(e.target)) {
    document.getElementById('userDropdown').style.display = 'none';
    document.getElementById('userSelBtn').classList.remove('open');
  }
});

// ── Mode ──────────────────────────────────────────────────────────────────────

function setMode(m) {
  mode = m;
  const input   = document.getElementById('mainInput');
  const submit  = document.getElementById('submitBtn');
  const dismiss = document.getElementById('dismissBtn');
  const selBtn  = document.getElementById('userSelBtn');

  if (m === 'idle') {
    selBtn.disabled   = false;
    input.disabled    = false;
    input.placeholder = selectedUser ? 'Describe the refund reason…' : 'Select a user first…';
    input.value       = selectedUser ? 'Item not as described' : '';
    submit.disabled   = !selectedUser;
    submit.textContent = 'Investigate →';
    dismiss.style.display = 'none';
    if (selectedUser) input.focus();
  } else if (m === 'thinking') {
    selBtn.disabled = true; input.disabled  = true; submit.disabled = true;
    dismiss.style.display = 'inline-block';
  } else if (m === 'reply') {
    selBtn.disabled   = true;
    input.disabled    = false; input.placeholder = 'Reply to agent…'; input.value = '';
    submit.disabled   = false; submit.textContent = 'Send →';
    dismiss.style.display = 'inline-block';
    input.focus();
  } else if (m === 'done') {
    selBtn.disabled   = true;
    input.disabled    = false;
    input.placeholder = 'Ask a follow-up question…';
    input.value       = '';
    submit.disabled   = false;
    submit.textContent = 'Send →';
    dismiss.style.display = 'inline-block';
    input.focus();
  }
}

// ── Actions ───────────────────────────────────────────────────────────────────

function primaryAction() {
  if (mode === 'idle')  startInvestigation();
  else if (mode === 'reply' || mode === 'done') sendReply();
}

function startInvestigation() {
  if (!selectedUser) return;
  const reason = document.getElementById('mainInput').value.trim();
  if (!reason) return;

  currentUserId = selectedUser;
  currentReason = reason;

  document.getElementById('emptyState')?.remove();
  setMode('thinking');
  document.getElementById('newBtn').style.display = 'inline-block';

  appendUserBubble(selectedUser, reason);
  activeAgentMsg = appendAgentMsg();
  scrollBottom();

  fetch('/investigate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_id: selectedUser, reason}),
  }).then(res => streamEvents(res)).catch(() => setMode('idle'));
}

function sendReply() {
  const input = document.getElementById('mainInput');
  const text  = input.value.trim();
  if (!text || !sessionId) return;

  setMode('thinking');
  appendReplyBubble(text);
  activeAgentMsg = appendAgentMsg();
  scrollBottom();

  fetch(`/reply/${sessionId}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text}),
  }).then(res => streamEvents(res)).catch(() => setMode('reply'));
}

function dismiss() {
  if (sessionId) { fetch(`/session/${sessionId}`, {method: 'DELETE'}).catch(() => {}); sessionId = null; }
  activeAgentMsg = null; activeThinking = null; planCard = null;
  currentUserId = null; currentReason = null;
  document.getElementById('chat').innerHTML =
    '<div class="empty-state" id="emptyState">' +
    '<div class="empty-icon">⚖</div>' +
    '<div class="empty-text">Select a user and describe the refund reason to begin</div></div>';
  document.getElementById('userSelBtn').disabled = false;
  document.getElementById('newBtn').style.display = 'none';
  resetTree();
  setMode('idle');
}

// ── SSE stream ────────────────────────────────────────────────────────────────

function streamEvents(res) {
  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  (function read() {
    reader.read().then(({done, value}) => {
      if (done) { setMode('done'); return; }
      buf += decoder.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') return;
        try { handleEvent(JSON.parse(raw)); } catch (_) {}
      }
      scrollBottom();
      read();
    });
  })();
}

// ── Event handler ─────────────────────────────────────────────────────────────

function handleEvent(ev) {
  if (ev.type === 'session') {
    sessionId = ev.id;

  } else if (ev.type === 'plan') {
    activeThinking.style.display = 'none';
    planCard = buildPlanCard(ev.steps);
    activeAgentMsg.insertBefore(planCard, activeThinking);
    activeThinking.style.display = '';
    initTree(ev.steps);

  } else if (ev.type === 'hypothesis_update') {
    if (!planCard) return;
    const row  = planCard.querySelector(`[data-hyp="${ev.id}"]`);
    if (!row) return;
    const icon    = row.querySelector('.step-icon');
    const label   = row.querySelector('.step-label');
    const summary = row.querySelector('.step-summary');
    row.className = `plan-step status-${ev.status}`;

    if (ev.status === 'running') {
      icon.textContent = '↻'; icon.className = 'step-icon running';
      label.className  = 'step-label';
    } else if (ev.status === 'done') {
      icon.textContent = '✓'; icon.className = 'step-icon';
      icon.style.color = 'var(--green)';
      label.className  = 'step-label';
    } else if (ev.status === 'alert') {
      icon.textContent = '⚠'; icon.className = 'step-icon';
      icon.style.color = 'var(--amber)';
      label.className  = 'step-label';
    }
    if (ev.summary) {
      summary.textContent = ev.summary;
      summary.className   = `step-summary ${ev.status === 'alert' ? 'alert' : 'done'}`;
    }
    updateTreeHyp(ev.id, ev.status, ev.summary || '');

  } else if (ev.type === 'tool_call') {
    activeThinking.style.display = 'none';
    const args = Object.entries(ev.args).map(([k,v]) => `${k}=${JSON.stringify(v)}`).join(', ');
    const card = document.createElement('div');
    card.className = 'tool-card'; card.id = 'tc-' + ev.id;
    card.innerHTML =
      `<div class="tool-header"><span>⚙</span><span class="tool-fn">${esc(ev.name)}</span>` +
      `<span class="tool-args">(${esc(args)})</span></div>` +
      `<div class="tool-result-row pending" id="tr-${esc(ev.id)}">${mkThinking().outerHTML}</div>`;
    activeAgentMsg.insertBefore(card, activeThinking);

  } else if (ev.type === 'tool_result') {
    const tr = document.getElementById('tr-' + ev.id);
    if (tr) { tr.className = 'tool-result-row done'; tr.innerHTML = `<span>→</span><span>${esc(ev.result)}</span>`; }
    activeThinking.style.display = '';

  } else if (ev.type === 'question') {
    activeThinking.remove(); activeThinking = null;
    const bubble = document.createElement('div');
    bubble.className = 'msg-question'; bubble.textContent = ev.text;
    activeAgentMsg.appendChild(bubble);
    setMode('reply');

  } else if (ev.type === 'decision') {
    activeThinking.remove(); activeThinking = null;
    const v = ev.verdict.toLowerCase();

    const card = document.createElement('div');
    card.className = `verdict-card verdict-${v}`;
    card.innerHTML = `<div class="verdict-label">${esc(ev.verdict)}</div><div class="verdict-text">${esc(ev.text)}</div>`;
    activeAgentMsg.appendChild(card);

    if (ev.next_steps && ev.next_steps.length) {
      const ns = document.createElement('div');
      ns.className = 'next-steps';
      ns.innerHTML = '<div class="next-steps-header">Suggested next steps</div>';
      ev.next_steps.forEach(step => {
        const btn = document.createElement('div');
        btn.className = 'next-step-btn';
        btn.innerHTML = `<span class="next-step-arrow">→</span><span>${esc(step)}</span>`;
        ns.appendChild(btn);
      });
      activeAgentMsg.appendChild(ns);
    }
    renderTreeConclusion(ev.verdict, ev.text, ev.next_steps || []);
    setMode('done');

  } else if (ev.type === 'error') {
    activeThinking?.remove();
    const card = document.createElement('div');
    card.className = 'error-card'; card.textContent = '⚠ ' + ev.message;
    activeAgentMsg.appendChild(card);
    setMode('done');
  }

  scrollBottom();
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function buildPlanCard(steps) {
  const card = document.createElement('div');
  card.className = 'plan-card';
  card.innerHTML = '<div class="plan-header">🔍 Investigation Plan</div>';
  steps.forEach(s => {
    const row = document.createElement('div');
    row.className = 'plan-step';
    row.dataset.hyp = s.id;
    row.innerHTML =
      `<span class="step-icon pending">○</span>` +
      `<div class="step-body">` +
        `<span class="step-label muted">${esc(s.label)}</span>` +
        `<span class="step-summary"></span>` +
      `</div>`;
    card.appendChild(row);
  });
  return card;
}

function appendUserBubble(userId, reason) {
  const el = document.createElement('div');
  el.className = 'msg-user';
  el.innerHTML = `<div class="msg-user-ref">user_id=${esc(userId)}</div><div class="msg-user-reason">${esc(reason)}</div>`;
  document.getElementById('chat').appendChild(el);
}

function appendReplyBubble(text) {
  const el = document.createElement('div');
  el.className = 'msg-reply'; el.textContent = text;
  document.getElementById('chat').appendChild(el);
}

function appendAgentMsg() {
  const el = document.createElement('div');
  el.className = 'msg-agent';
  const t = mkThinking(); el.appendChild(t); activeThinking = t;
  document.getElementById('chat').appendChild(el);
  return el;
}

function mkThinking() {
  const el = document.createElement('div'); el.className = 'thinking';
  el.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
  return el;
}

function scrollBottom() { const c = document.getElementById('chat'); c.scrollTop = c.scrollHeight; }

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Tree helpers ──────────────────────────────────────────────────────────────

function initTree(steps) {
  document.getElementById('treeEmpty').style.display  = 'none';
  document.getElementById('treeCanvas').style.display = 'block';

  // Set source node title
  const reason = currentReason ? `"${currentReason}"` : '';
  document.getElementById('tree-source-title').textContent =
    `user_id=${currentUserId} · ${reason}`;

  // Set H node labels from plan steps (up to 4)
  const ids = ['h1','h2','h3','h4'];
  ids.forEach((hid, i) => {
    const step = steps[i];
    const labelEl = document.getElementById(`tree-${hid}-label`);
    if (labelEl) labelEl.textContent = step ? step.label : '—';
    // Reset state
    const node = document.getElementById(`tree-${hid}`);
    if (node) { node.className = 'tree-node tree-hyp'; }
    const icon = node ? node.querySelector('.hyp-icon') : null;
    if (icon) { icon.textContent = '○'; icon.className = 'hyp-icon'; }
    const sumEl = document.getElementById(`tree-${hid}-summary`);
    if (sumEl) sumEl.textContent = '';
  });

  // Hide conclusion and next steps
  document.getElementById('tree-conclusion').style.display = 'none';
  document.getElementById('svg-conclusion-line').style.display = 'none';
  ['ns0','ns1','ns2'].forEach(id => {
    const el = document.getElementById(`tree-${id}`);
    if (el) { el.className = 'tree-node tree-nextstep'; }
  });
}

function updateTreeHyp(id, status, summary) {
  // id is e.g. "h1", "h2", etc.
  const node = document.getElementById(`tree-${id}`);
  if (!node) return;
  const icon   = node.querySelector('.hyp-icon');
  const sumEl  = document.getElementById(`tree-${id}-summary`);

  // Remove existing state classes
  node.classList.remove('hyp-running','hyp-done','hyp-alert');

  if (status === 'running') {
    node.classList.add('hyp-running');
    icon.textContent = '↻';
    icon.className   = 'hyp-icon spinning';
  } else if (status === 'done') {
    node.classList.add('hyp-done');
    icon.textContent = '✓';
    icon.className   = 'hyp-icon';
    icon.style.color = 'var(--green)';
  } else if (status === 'alert') {
    node.classList.add('hyp-alert');
    icon.textContent = '⚠';
    icon.className   = 'hyp-icon';
    icon.style.color = 'var(--amber)';
  }

  if (sumEl && summary) {
    sumEl.textContent = summary.length > 60 ? summary.slice(0, 57) + '…' : summary;
  }
}

function renderTreeConclusion(verdict, text, nextSteps) {
  const v    = verdict.toLowerCase();
  const node = document.getElementById('tree-conclusion');
  node.className = `tree-node verdict-${v}`;
  node.style.display = 'block';

  document.getElementById('tree-conc-label').textContent = verdict;
  const textEl = document.getElementById('tree-conc-text');
  textEl.textContent = text.length > 80 ? text.slice(0, 77) + '…' : text;

  // Show connector line from conclusion to first next step
  document.getElementById('svg-conclusion-line').style.display = '';

  // Show up to 3 next step nodes
  const nsIds = ['ns0','ns1','ns2'];
  nsIds.forEach((nsid, i) => {
    const el     = document.getElementById(`tree-${nsid}`);
    const textEl = document.getElementById(`tree-${nsid}-text`);
    if (nextSteps[i] && el) {
      textEl.textContent = nextSteps[i].length > 36 ? nextSteps[i].slice(0, 33) + '…' : nextSteps[i];
      el.className = 'tree-node tree-nextstep visible';
    }
  });
}

function resetTree() {
  document.getElementById('treeEmpty').style.display  = '';
  document.getElementById('treeCanvas').style.display = 'none';

  // Clear source
  document.getElementById('tree-source-title').textContent = '—';

  // Reset H nodes
  ['h1','h2','h3','h4'].forEach(hid => {
    const node = document.getElementById(`tree-${hid}`);
    if (node) node.className = 'tree-node tree-hyp';
    const icon = node ? node.querySelector('.hyp-icon') : null;
    if (icon) { icon.textContent = '○'; icon.className = 'hyp-icon'; icon.style.color = ''; }
    const lbl = document.getElementById(`tree-${hid}-label`);
    if (lbl) lbl.textContent = '—';
    const sum = document.getElementById(`tree-${hid}-summary`);
    if (sum) sum.textContent = '';
  });

  // Hide conclusion
  const conc = document.getElementById('tree-conclusion');
  conc.style.display = 'none';
  conc.className = 'tree-node';
  document.getElementById('tree-conc-label').textContent = '—';
  document.getElementById('tree-conc-text').textContent  = '';

  // Hide SVG connector line
  document.getElementById('svg-conclusion-line').style.display = 'none';

  // Hide next steps
  ['ns0','ns1','ns2'].forEach(id => {
    const el = document.getElementById(`tree-${id}`);
    if (el) { el.className = 'tree-node tree-nextstep'; }
    const t = document.getElementById(`tree-${id}-text`);
    if (t) t.textContent = '';
  });
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"\n  Refund Intelligence UI → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
