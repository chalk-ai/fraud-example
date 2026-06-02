#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["fastapi", "uvicorn[standard]", "openai", "chalkpy", "python-dotenv", "chalkcompute>=1.5.17"]
# ///
"""Refund-abuse agent demo UI — investigation plan edition.

Agent discovers that an individually clean refund claim is part of a broader
suspicious spike, builds an investigation plan, executes it step by step, and
ESCALATEs.

The agent loop itself lives in chalkcompute_agent_core.py and runs on Chalk
Compute (so it shows up in Chalk's function tracing). This file is just the web
UI: it builds the message history, invokes the deployed `investigate` function
(which runs the whole loop and returns all UI events as a JSON array), and
replays those events to the browser as SSE with light pacing so the run still
looks live. (chalkcompute 2.0.0's generator-streaming call path doesn't deliver
chunks back, so we run to completion and replay rather than true-stream.)

  AGENT_REMOTE=1 (default)  → run the loop on Chalk Compute (auto-deploys on
                              import; pre-deploy with ./chalkcompute_agent_core.py deploy)
  AGENT_REMOTE=0            → run the loop in-process (no deploy needed; handy
                              for local UI iteration). Nothing is traced.
  AGENT_REPLAY_DELAY=0.45   → seconds between replayed events in remote mode.

Run:
  ./chalkcompute_agent_demo_ui.py 8123
  open http://localhost:8123
"""

import asyncio
import json
import os
import queue
import random
import sys
import threading
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

import time

from chalkcompute_agent_core import (
    SYSTEM_PROMPT,
    investigate as run_investigate,
    _investigate_impl,
)

# Run the agent on Chalk Compute by default; AGENT_REMOTE=0 runs it in-process.
AGENT_REMOTE = os.environ.get("AGENT_REMOTE", "1") != "0"
# When running remotely the function returns all events at once (the 2.0.0 generator
# streaming path is broken); replay them with light pacing so the tree still animates.
# The delay is jittered ±AGENT_REPLAY_JITTER around the base so it feels less mechanical.
_REPLAY_DELAY = float(os.environ.get("AGENT_REPLAY_DELAY", "0.45"))
_REPLAY_JITTER = float(os.environ.get("AGENT_REPLAY_JITTER", "0.4"))


def _replay_pause() -> None:
    """Sleep ~_REPLAY_DELAY with jitter so replayed events don't tick at a fixed cadence."""
    low = _REPLAY_DELAY * (1.0 - _REPLAY_JITTER)
    high = _REPLAY_DELAY * (1.0 + _REPLAY_JITTER)
    time.sleep(random.uniform(max(0.0, low), high))

_sessions: dict[str, list] = {}


def _event_strings(messages: list, followup: bool):
    """Yield JSON event strings from the agent — remote on Chalk Compute, or in-process.

    Returns (iterator_of_event_strings, paced) where `paced` is True when the events
    arrived all at once (remote) and should be replayed with a small delay.
    """
    payload = json.dumps(messages)
    if AGENT_REMOTE:
        # The whole investigation runs on Chalk Compute and comes back as one JSON
        # array of event strings (see chalkcompute_agent_core.investigate).
        events = json.loads(run_investigate(payload, followup))
        return iter(events), True
    return _investigate_impl(payload, followup), False


def _producer(messages: list, followup: bool, q: queue.Queue, session_id: str) -> None:
    """Drive the agent events onto the SSE queue, capturing the final session state."""
    try:
        events, paced = _event_strings(messages, followup)
        for s in events:
            event = json.loads(s)
            if event.get("type") == "_state":
                _sessions[session_id] = event["messages"]
                continue
            q.put(event)
            if paced and event.get("type") in ("plan", "tool_result", "hypothesis_update", "decision", "question"):
                _replay_pause()
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
    threading.Thread(target=_producer, args=(messages, False, q, session_id), daemon=True).start()

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
    threading.Thread(target=_producer, args=(messages, True, q, session_id), daemon=True).start()
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
<title>ACME Corp. Refund Investigator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    /* Chalk console — light surfaces */
    --bg:          #f7f9f8;
    --surface:     #ffffff;
    --surface2:    #f4f6f5;
    --surface3:    #eef1f0;
    --border:      #e3e6e5;
    --border-strong:#cfd4d2;
    --text:        #173029;   /* black-green / type-primary */
    --text2:       #3c6e65;
    --muted:       #7a8a87;   /* type-secondary */
    --faint:       #a2a5a4;
    /* Chalk greens */
    --accent:      #16883e;   /* bright-green-700 — interactive */
    --accent-hi:   #2aa853;
    --accent-deep: #12654F;   /* primary green stroke */
    --accent-soft: #e5f4e9;   /* bright-green-50 */
    /* Semantic (CDS) */
    --green:       #16a34a;  --green-text:#166534; --green-bg:#dcfce7; --green-bd:#86efac;
    --red:         #e01c40;  --red-text:#991b1b;   --red-bg:#fee2e2;   --red-bd:#fca5a5;
    --amber:       #b45309;  --amber-text:#b45309; --amber-bg:#fef3c7; --amber-bd:#fcd34d;
    --tool-bg:     #f3f8f5;
    --tool-border: #d3e7db;
    /* Elevation (CDS) */
    --sh-s: 0 1px 2px 0 rgb(16 48 41 / .04);
    --sh-m: 0 2px 8px 0 rgb(16 48 41 / .08);
    --sh-l: 0 8px 24px 0 rgb(16 48 41 / .12);
    --dot:  #d3dbd8;
    --line: #cfd6d3;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text);
    height: 100vh; display: flex; flex-direction: column; overflow: hidden;
    -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
  }
  .mono { font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', monospace; }

  /* ── Top-level split layout ── */
  .app-body { display: flex; flex: 1; overflow: hidden; }

  /* ── Left panel ── */
  .left-panel {
    width: 480px; flex-shrink: 0;
    display: flex; flex-direction: column;
    border-right: 1px solid var(--border); overflow: hidden;
    background: var(--surface);
  }

  /* ── Right panel ── */
  .right-panel {
    flex: 1; display: flex; flex-direction: column;
    background-color: var(--bg);
    background-image: radial-gradient(circle, var(--dot) 1px, transparent 1px);
    background-size: 22px 22px;
    overflow: hidden;
  }
  .right-header {
    padding: 16px 24px; border-bottom: 1px solid var(--border); flex-shrink: 0;
    font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted);
    display: flex; align-items: center; gap: 8px;
    background: var(--surface);
  }
  .right-header .dotmark { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  .right-body {
    flex: 1; overflow: auto; position: relative;
    display: flex; align-items: flex-start; justify-content: center;
    padding: 56px 32px;
  }

  /* ── Tree pane: planning / idle hint ── */
  .tree-hint {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    display: flex; flex-direction: column; align-items: center; gap: 16px;
    color: var(--muted); pointer-events: none; text-align: center; max-width: 320px;
  }
  .tree-hint-mark {
    width: 44px; height: 44px; border-radius: 11px; border: 1px dashed var(--border-strong);
    display: flex; align-items: center; justify-content: center; font-size: 20px; color: var(--faint);
  }
  .tree-hint-text { font-size: 13px; line-height: 1.5; }
  .tree-hint-text b { color: var(--text2); font-weight: 600; }
  .tree-planning .tree-hint-mark { border-style: solid; border-color: var(--accent); color: var(--accent); animation: spin 1.4s linear infinite; }
  .tree-planning .tree-hint-text { color: var(--text2); }

  /* ── Tree canvas ── */
  #treeCanvas {
    position: relative; flex-shrink: 0;
    display: none;
  }
  #treeCanvas.show { animation: canvasIn 0.5s cubic-bezier(.16,1,.3,1); }

  /* ── Tree nodes (shared base) ── */
  .tree-node {
    position: absolute; border-radius: 8px; padding: 11px 14px;
    font-size: 12px; line-height: 1.4;
    border: 1px solid var(--border);
    background: var(--surface);
    box-shadow: var(--sh-s);
    transition: background .3s, border-color .3s, opacity .3s, box-shadow .3s, transform .3s;
  }
  #tree-source, #tree-conclusion { overflow: hidden; }
  .tree-node-label {
    font-size: 9.5px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 5px;
  }
  .tree-node-title { font-size: 12px; font-weight: 600; color: var(--text); letter-spacing: -0.01em; }
  .tree-node-summary {
    font-size: 11px; color: var(--muted); margin-top: 5px;
    line-height: 1.4; word-break: break-word; overflow-wrap: break-word;
  }

  /* ── SOURCE node (position + size set by JS) ── */
  #tree-source {
    background: var(--red-bg); border-color: var(--red-bd);
  }
  #tree-source .tree-node-label { color: #b3505f; }
  #tree-source .tree-node-title { color: var(--red-text); font-size: 13px; }
  #tree-source .tree-node-title .mono { font-size: 12px; }

  /* ── H nodes (hypothesis) — left/top set by JS ── */
  .tree-hyp {
    width: 158px; min-height: 100px;
    opacity: 0.4; transform: scale(.97);
  }
  .tree-hyp.active { opacity: 1; transform: scale(1); }
  .tree-hyp.hyp-running {
    opacity: 1; transform: scale(1);
    background: var(--accent-soft); border-color: var(--accent);
    box-shadow: var(--sh-m); animation: glow 1.5s ease-in-out infinite;
  }
  .tree-hyp.hyp-done {
    opacity: 1; transform: scale(1);
    background: var(--green-bg); border-color: var(--green-bd); box-shadow: var(--sh-s);
  }
  .tree-hyp.hyp-alert {
    opacity: 1; transform: scale(1);
    background: var(--amber-bg); border-color: var(--amber-bd); box-shadow: var(--sh-s);
  }

  .hyp-icon {
    font-size: 15px; display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; margin-bottom: 5px; color: var(--faint);
  }
  .hyp-icon.spinning { animation: spin 1s linear infinite; color: var(--accent); }

  /* ── CONCLUSION node (left/top set by JS) ── */
  #tree-conclusion {
    width: 320px; min-height: 86px;
    display: none;
  }
  #tree-conclusion.show { display: block; animation: popIn 0.45s cubic-bezier(.16,1,.3,1); }
  #tree-conclusion.verdict-approve  { background: var(--green-bg); border-color: var(--green-bd); }
  #tree-conclusion.verdict-deny     { background: var(--red-bg);   border-color: var(--red-bd); }
  #tree-conclusion.verdict-escalate { background: var(--amber-bg); border-color: var(--amber-bd); }
  #tree-conclusion .conc-label {
    font-size: 13px; font-weight: 700; letter-spacing: .06em; margin-bottom: 5px;
    display: flex; align-items: center; gap: 6px;
  }
  #tree-conclusion.verdict-approve  .conc-label { color: var(--green-text); }
  #tree-conclusion.verdict-deny     .conc-label { color: var(--red-text); }
  #tree-conclusion.verdict-escalate .conc-label { color: var(--amber-text); }
  #tree-conclusion .conc-text { font-size: 11px; color: var(--text2); line-height: 1.45; }

  /* ── SVG connector lines ── */
  .tree-svg {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%; pointer-events: none; overflow: visible;
  }
  .tree-svg line, .tree-svg path {
    stroke: var(--line); stroke-width: 1.5; fill: none;
    transition: stroke .4s, stroke-width .4s;
  }
  .tree-svg .edge-active  { stroke: var(--accent); stroke-width: 2; }
  .tree-svg .edge-done    { stroke: var(--green); stroke-width: 2; }
  .tree-svg .edge-alert   { stroke: var(--amber); stroke-width: 2; }
  .tree-svg .edge-deny    { stroke: var(--red);   stroke-width: 2; }

  /* ── Header ── */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 15px 24px; border-bottom: 1px solid var(--border); flex-shrink: 0;
    background: var(--surface);
  }
  .header-left { display: flex; align-items: center; gap: 11px; }
  .header-title { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
  .header-tagline {
    display: flex; align-items: center; gap: 7px;
    margin-left: 3px; padding-left: 14px; border-left: 1px solid var(--border);
    font-size: 12px; font-weight: 400; color: var(--text2); letter-spacing: 0;
    font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', monospace;
  }
  .header-tagline .dotmark { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  .logo {
    width: 28px; height: 28px;
    background: linear-gradient(160deg, #177F65 0%, #12654F 100%);
    border-radius: 7px; display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: #fff;
    box-shadow: var(--sh-s);
  }
  .header-right { display: flex; align-items: center; gap: 12px; }
  .new-inv-btn {
    background: transparent; color: var(--text2); border: 1px solid var(--border);
    border-radius: 8px; padding: 7px 14px; font-size: 13px; font-weight: 500;
    font-family: inherit; cursor: pointer; white-space: nowrap;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
  }
  .new-inv-btn:hover { background: var(--surface2); border-color: var(--border-strong); color: var(--text); }

  /* ── User selector ── */
  .user-sel { position: relative; flex-shrink: 0; }
  .user-sel-btn {
    display: flex; align-items: center; gap: 8px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 11px 14px;
    color: var(--text); font-size: 13px; font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', monospace;
    cursor: pointer; white-space: nowrap; transition: border-color 0.15s, box-shadow 0.15s; user-select: none;
  }
  .user-sel-btn:hover:not(:disabled) { border-color: var(--border-strong); box-shadow: var(--sh-s); }
  .user-sel-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .user-sel-chevron { color: var(--muted); font-size: 10px; transition: transform 0.15s; }
  .user-sel-btn.open .user-sel-chevron { transform: rotate(180deg); }
  .user-dropdown {
    position: absolute; top: calc(100% + 8px); left: 0; right: 0;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 6px; min-width: 268px;
    box-shadow: 0 12px 28px rgb(16 48 41 / .16);
    display: flex; flex-direction: column; gap: 2px;
    z-index: 100; animation: rise 0.15s ease;
  }
  .user-option {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 12px; border-radius: 7px; cursor: pointer; transition: background 0.1s; user-select: none;
  }
  .user-option:hover { background: var(--surface2); }
  .user-option-id { font-size: 13px; font-weight: 600; font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', monospace; }
  .user-option-right { display: flex; align-items: center; gap: 7px; }
  .risk-badge { font-size: 9.5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 2px 7px; border-radius: 4px; }
  .risk-high   { background: var(--red-bg);   color: var(--red-text); }
  .risk-low    { background: var(--green-bg); color: var(--green-text); }
  .risk-medium { background: var(--amber-bg); color: var(--amber-text); }
  .user-desc   { font-size: 11px; color: var(--muted); }

  /* ── Chat ── */
  .chat { flex: 1; overflow-y: auto; padding: 28px 24px; display: flex; flex-direction: column; gap: 22px; }
  .chat::-webkit-scrollbar { width: 5px; }
  .chat::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }

  /* ── Fresh-start view (full-width centered hero + composer) ── */
  .start-view {
    flex: 1; overflow: auto;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 40px 24px;
    background-color: var(--bg);
    background-image: radial-gradient(circle, var(--dot) 1px, transparent 1px);
    background-size: 22px 22px;
  }
  .start-inner { width: 100%; max-width: 560px; display: flex; flex-direction: column; align-items: center; }
  .start-mark {
    width: 56px; height: 56px; border-radius: 14px;
    background: linear-gradient(160deg, #177F65 0%, #12654F 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; color: #fff; box-shadow: var(--sh-m); margin-bottom: 20px;
    animation: rise .4s ease;
  }
  .start-title { font-size: 22px; font-weight: 600; color: var(--text); letter-spacing: -0.02em; margin-bottom: 9px; text-align: center; animation: rise .45s ease; }
  .start-sub { font-size: 14px; line-height: 1.55; color: var(--muted); text-align: center; max-width: 420px; margin-bottom: 28px; animation: rise .5s ease; }

  /* Composer card */
  .composer {
    width: 100%; background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; box-shadow: var(--sh-l); padding: 18px;
    display: flex; flex-direction: column; gap: 12px; animation: rise .55s ease;
  }
  .composer-row { display: flex; gap: 10px; align-items: stretch; }
  .composer .user-sel { flex-shrink: 0; }
  .composer .user-sel-btn { height: 100%; }
  .composer-input {
    flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: 9px;
    padding: 11px 16px; color: var(--text); font-size: 14px; font-family: inherit; outline: none;
    transition: border-color .15s, box-shadow .15s;
  }
  .composer-input::placeholder { color: var(--muted); }
  .composer-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgb(22 136 62 / .12); }
  .composer-input:disabled { opacity: .5; }
  .composer-go {
    background: linear-gradient(180deg, #177F65 0%, #12654F 100%); color: #fff; border: none;
    border-radius: 9px; padding: 12px; font-size: 14px; font-weight: 600; font-family: inherit;
    cursor: pointer; transition: filter .15s, box-shadow .15s; box-shadow: var(--sh-s);
  }
  .composer-go:hover:not(:disabled) { filter: brightness(1.08); box-shadow: var(--sh-m); }
  .composer-go:active:not(:disabled) { filter: brightness(.95); }
  .composer-go:disabled { opacity: .4; cursor: not-allowed; box-shadow: none; }

  /* Bubbles */
  .msg-user {
    align-self: flex-end; max-width: 420px;
    background: var(--accent-soft); border: 1px solid var(--green-bd);
    border-radius: 14px 14px 4px 14px; padding: 11px 15px; animation: rise 0.2s ease;
  }
  .msg-user-ref    { font-size: 11px; color: var(--text2); font-family: 'JetBrains Mono','SF Mono',monospace; margin-bottom: 4px; }
  .msg-user-reason { font-size: 14px; line-height: 1.4; color: var(--text); }
  .msg-reply {
    align-self: flex-end; max-width: 420px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 14px 14px 4px 14px; padding: 11px 15px; font-size: 14px; line-height: 1.4; animation: rise 0.2s ease;
  }
  .msg-question {
    align-self: flex-start; max-width: 480px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px 14px 14px 4px; padding: 12px 16px; font-size: 14px; line-height: 1.55; animation: rise 0.2s ease;
    box-shadow: var(--sh-s);
  }
  .msg-agent { align-self: flex-start; max-width: 620px; width: 100%; display: flex; flex-direction: column; gap: 8px; animation: rise 0.2s ease; }

  /* ── Tool card ── */
  .tool-card {
    background: var(--tool-bg); border: 1px solid var(--tool-border);
    border-left: 3px solid var(--accent); border-radius: 8px;
    padding: 10px 14px; font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', monospace; font-size: 12px; animation: rise 0.15s ease;
    word-break: break-word; overflow-wrap: anywhere;
  }
  .tool-header { display: flex; align-items: center; gap: 7px; color: var(--accent); margin-bottom: 3px; }
  .tool-fn   { font-weight: 600; }
  .tool-args { color: var(--muted); }
  .tool-result-row { margin-top: 7px; padding-top: 7px; border-top: 1px solid var(--tool-border); display: flex; align-items: center; gap: 7px; }
  .tool-result-row.pending { color: var(--muted); }
  .tool-result-row.done    { color: var(--green-text); }

  /* ── Thinking dots ── */
  .thinking { display: flex; align-items: center; gap: 5px; padding: 8px 2px; }
  .dot { width: 6px; height: 6px; background: var(--accent); border-radius: 50%; animation: pulse 1.2s ease-in-out infinite; }
  .dot:nth-child(2) { animation-delay: .2s; }
  .dot:nth-child(3) { animation-delay: .4s; }

  /* ── Verdict ── */
  .verdict-card { border-radius: 12px; padding: 16px 20px; border: 1px solid; animation: rise 0.25s ease; }
  .verdict-approve  { background: var(--green-bg); border-color: var(--green-bd); }
  .verdict-deny     { background: var(--red-bg);   border-color: var(--red-bd); }
  .verdict-escalate { background: var(--amber-bg); border-color: var(--amber-bd); }
  .verdict-label { font-size: 18px; font-weight: 700; letter-spacing: .04em; margin-bottom: 6px; }
  .verdict-approve  .verdict-label { color: var(--green-text); }
  .verdict-deny     .verdict-label { color: var(--red-text);   }
  .verdict-escalate .verdict-label { color: var(--amber-text); }
  .verdict-text { font-size: 13px; color: var(--text2); line-height: 1.55; }

  /* ── Error ── */
  .error-card { background: var(--red-bg); border: 1px solid var(--red-bd); border-radius: 8px; padding: 10px 14px; font-size: 13px; color: var(--red-text); }

  /* ── Input bar ── */
  .input-bar { padding: 14px 24px; border-top: 1px solid var(--border); display: flex; gap: 10px; flex-shrink: 0; align-items: center; background: var(--surface); }
  .main-input { flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: 9px; padding: 11px 16px; color: var(--text); font-size: 14px; font-family: inherit; outline: none; transition: border-color 0.15s, box-shadow 0.15s; }
  .main-input::placeholder { color: var(--muted); }
  .main-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgb(22 136 62 / .12); }
  .main-input:disabled { opacity: 0.45; }
  .submit-btn {
    background: linear-gradient(180deg, #177F65 0%, #12654F 100%); color: #fff; border: none;
    border-radius: 9px; padding: 11px 22px; font-size: 14px; font-weight: 600; font-family: inherit;
    cursor: pointer; transition: filter 0.15s, box-shadow 0.15s; white-space: nowrap; box-shadow: var(--sh-s);
  }
  .submit-btn:hover:not(:disabled) { filter: brightness(1.08); box-shadow: var(--sh-m); }
  .submit-btn:active:not(:disabled) { filter: brightness(.95); }
  .submit-btn:disabled { opacity: .4; cursor: not-allowed; box-shadow: none; }

  @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes pulse { 0%,80%,100% { transform: scale(.55); opacity: .35; } 40% { transform: scale(1); opacity: 1; } }
  @keyframes spin  { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  @keyframes glow  { 0%,100% { box-shadow: 0 0 0 0 rgb(22 136 62 / 0); } 50% { box-shadow: 0 0 0 5px rgb(22 136 62 / .14); } }
  @keyframes canvasIn { from { opacity: 0; transform: translateY(10px) scale(.99); } to { opacity: 1; transform: none; } }
  @keyframes popIn { from { opacity: 0; transform: scale(.9); } 60% { transform: scale(1.02); } to { opacity: 1; transform: scale(1); } }
  @keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
  @media (prefers-reduced-motion: reduce) {
    * { animation: none !important; }
  }
</style>
</head>
<body>

<!-- ── Top header spanning full width ── -->
<div class="header">
  <div class="header-left">
    <div class="logo">A</div>
    <span class="header-title">ACME Corp. Refund Investigator</span>
    <span class="header-tagline"><span class="dotmark"></span>Built on the Chalk AI Data Platform</span>
  </div>
  <div class="header-right">
    <button id="newInvBtn" class="new-inv-btn" onclick="dismiss()" style="display:none">New investigation</button>
  </div>
</div>

<!-- ── Fresh-start view (full-width hero + composer) ── -->
<div class="start-view" id="startView">
  <div class="start-inner">
    <div class="start-mark">⬡</div>
    <div class="start-title">ACME Corp. Refund Investigator</div>
    <div class="start-sub">Pick a user and describe their refund claim.</div>

    <div class="composer">
      <div class="composer-row">
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
        <input id="startInput" class="composer-input" type="text"
               placeholder="Select a user first…"
               disabled
               onkeydown="if(event.key==='Enter')startInvestigation()">
      </div>
      <button id="startBtn" class="composer-go" onclick="startInvestigation()" disabled>Investigate →</button>
    </div>
  </div>
</div>

<!-- ── Split body (hidden until investigation starts) ── -->
<div class="app-body" id="appBody" style="display:none">

  <!-- Left panel: chat -->
  <div class="left-panel">
    <div class="chat" id="chat"></div>

    <div class="input-bar">
      <input id="mainInput" class="main-input" type="text"
             placeholder="Ask a follow-up question…"
             disabled
             onkeydown="if(event.key==='Enter')sendReply()">
      <button id="submitBtn"  class="submit-btn"  onclick="sendReply()" disabled>Send →</button>
    </div>
  </div><!-- /left-panel -->

  <!-- Right panel: hypothesis tree -->
  <div class="right-panel">
    <div class="right-header"><span class="dotmark"></span>Investigation Tree</div>
    <div class="right-body">

      <!-- Idle / planning hint (absolute-centered) -->
      <div class="tree-hint" id="treeHint">
        <div class="tree-hint-mark" id="treeHintMark">⬡</div>
        <div class="tree-hint-text" id="treeHintText">The agent's <b>hypothesis tree</b> renders here as it investigates.</div>
      </div>

      <!-- Tree canvas — built dynamically from the agent's plan -->
      <div id="treeCanvas"></div>

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

// ── Dropdown ──────────────────────────────────────────────────────────────────

function toggleDropdown() {
  const btn  = document.getElementById('userSelBtn');
  const menu = document.getElementById('userDropdown');
  const open = menu.style.display !== 'none';
  menu.style.display = open ? 'none' : 'flex';
  btn.classList.toggle('open', !open);
}

function selectUser(id, risk) {
  selectedUser = id;

  const btn   = document.getElementById('userSelBtn');
  const label = document.getElementById('userSelLabel');
  label.textContent = `user_id=${id}`;
  btn.className = `user-sel-btn`;
  document.getElementById('userDropdown').style.display = 'none';

  const input  = document.getElementById('startInput');
  const start  = document.getElementById('startBtn');
  input.disabled    = false;
  input.placeholder = 'Describe the refund reason…';
  input.value       = 'Item not as described';
  start.disabled    = false;
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

// Follow-up bar only (idle is handled by the start view).
function setMode(m) {
  mode = m;
  const input   = document.getElementById('mainInput');
  const submit  = document.getElementById('submitBtn');

  if (m === 'thinking') {
    input.disabled  = true; submit.disabled = true;
  } else if (m === 'reply') {
    input.disabled    = false; input.placeholder = 'Reply to agent…'; input.value = '';
    submit.disabled   = false;
    input.focus();
  } else if (m === 'done') {
    input.disabled    = false;
    input.placeholder = 'Ask a follow-up question…';
    input.value       = '';
    submit.disabled   = false;
    input.focus();
  }
}

// ── Actions ───────────────────────────────────────────────────────────────────

function startInvestigation() {
  if (!selectedUser) return;
  const reason = document.getElementById('startInput').value.trim();
  if (!reason) return;

  currentUserId = selectedUser;
  currentReason = reason;

  // Swap from the fresh-start view to the split investigation view.
  document.getElementById('startView').style.display = 'none';
  document.getElementById('appBody').style.display   = 'flex';
  document.getElementById('chat').innerHTML          = '';
  document.getElementById('newInvBtn').style.display = 'inline-block';
  showTreeHint('planning');
  setMode('thinking');

  appendUserBubble(selectedUser, reason);
  activeAgentMsg = appendAgentMsg();
  scrollBottom();

  fetch('/investigate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_id: selectedUser, reason}),
  }).then(res => streamEvents(res)).catch(() => setMode('done'));
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
  activeAgentMsg = null; activeThinking = null;
  currentUserId = null; currentReason = null;

  resetTree();
  document.getElementById('chat').innerHTML = '';

  // Back to the fresh-start view.
  document.getElementById('newInvBtn').style.display = 'none';
  document.getElementById('appBody').style.display   = 'none';
  document.getElementById('startView').style.display = 'flex';

  // Reset composer.
  selectedUser = null;
  document.getElementById('userSelLabel').textContent = 'Select user';
  const si = document.getElementById('startInput');
  si.value = ''; si.placeholder = 'Select a user first…'; si.disabled = true;
  document.getElementById('startBtn').disabled = true;
  mode = 'idle';
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

  } else if (ev.type === 'planning') {
    showTreeHint('planning');

  } else if (ev.type === 'plan') {
    buildTree(ev.steps);

  } else if (ev.type === 'hypothesis_update') {
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

    renderTreeConclusion(ev.verdict, ev.text);
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

// ── Tree helpers (dynamic, N nodes from the agent's plan) ──────────────────────

const NODE_W = 158, NODE_GAP = 20, SRC_W = 300, SRC_H = 64, CONC_W = 320;
const ROW_TOP = 118;   // top y of the hypothesis row
const FAN_Y   = 90;    // y of the horizontal fan rail
const PAD     = 10;    // bottom padding of the canvas

let treeSteps = [];    // [{id,label,tool}, ...]
let edgeCls   = {};    // edge id → highlight class, persisted across redraws
let layout    = null;  // {n, centerX, cx:[...]}

function showTreeHint(state) {
  const hint = document.getElementById('treeHint');
  const mark = document.getElementById('treeHintMark');
  const text = document.getElementById('treeHintText');
  hint.style.display = 'flex';
  if (state === 'planning') {
    hint.classList.add('tree-planning');
    mark.textContent = '↻';
    text.innerHTML   = 'Drafting the investigation plan…';
  } else {
    hint.classList.remove('tree-planning');
    mark.textContent = '⬡';
    text.innerHTML   = "The agent's <b>hypothesis tree</b> renders here as it investigates.";
  }
}

function hideTreeHint() {
  const hint = document.getElementById('treeHint');
  hint.style.display = 'none';
  hint.classList.remove('tree-planning');
}

function buildTree(steps) {
  treeSteps = steps;
  edgeCls   = {};
  hideTreeHint();

  const canvas  = document.getElementById('treeCanvas');
  const n       = steps.length;
  const rowW    = n * NODE_W + (n - 1) * NODE_GAP;
  const centerX = rowW / 2;
  const cx      = steps.map((_, i) => i * (NODE_W + NODE_GAP) + NODE_W / 2);
  layout = { n, centerX, cx };

  canvas.style.width = rowW + 'px';

  const reason = currentReason ? `"${esc(currentReason)}"` : '';
  let html = '<svg class="tree-svg" id="treeSvg" xmlns="http://www.w3.org/2000/svg"></svg>';
  html += `<div class="tree-node" id="tree-source" style="left:${centerX - SRC_W / 2}px;top:0;width:${SRC_W}px;">`
        +   `<div class="tree-node-label">Source · Refund Claim</div>`
        +   `<div class="tree-node-title"><span class="mono">user_id=${esc(currentUserId)}</span> · ${reason}</div>`
        + `</div>`;
  steps.forEach((s, i) => {
    html += `<div class="tree-node tree-hyp" id="tree-${esc(s.id)}" style="left:${i * (NODE_W + NODE_GAP)}px;top:${ROW_TOP}px;">`
          +   `<div class="hyp-icon">○</div>`
          +   `<div class="tree-node-title">${esc(s.label)}</div>`
          +   `<div class="tree-node-summary" id="tree-${esc(s.id)}-summary"></div>`
          + `</div>`;
  });
  html += `<div class="tree-node" id="tree-conclusion" style="left:${centerX - CONC_W / 2}px;width:${CONC_W}px;">`
        +   `<div class="conc-label" id="tree-conc-label">—</div>`
        +   `<div class="conc-text"  id="tree-conc-text"></div>`
        + `</div>`;
  canvas.innerHTML = html;

  canvas.style.display = 'block';
  canvas.classList.remove('show'); void canvas.offsetWidth; canvas.classList.add('show');

  drawEdges();
}

function svgLine(id, x1, y1, x2, y2) {
  const idAttr = id ? ` id="${id}"` : '';
  const cls    = id && edgeCls[id] ? ` class="${edgeCls[id]}"` : '';
  return `<line${idAttr}${cls} x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
}

// Recompute geometry from measured node heights and (re)draw all connectors.
function drawEdges() {
  if (!layout) return;
  const { n, centerX, cx } = layout;

  const bottoms = treeSteps.map(s => {
    const el = document.getElementById(`tree-${s.id}`);
    return el ? el.offsetTop + el.offsetHeight : ROW_TOP + 100;
  });
  const maxBottom  = Math.max.apply(null, bottoms.concat([ROW_TOP + 100]));
  const collectorY = maxBottom + 24;
  const concTop    = collectorY + 34;

  const conc = document.getElementById('tree-conclusion');
  if (conc) conc.style.top = concTop + 'px';

  const concVisible = conc && conc.classList.contains('show');
  const totalH = (concVisible ? concTop + conc.offsetHeight : collectorY) + PAD;
  document.getElementById('treeCanvas').style.height = totalH + 'px';

  const first = cx[0], last = cx[n - 1];
  let s = '';
  s += svgLine(null, centerX, SRC_H, centerX, FAN_Y);   // source → fan junction
  s += svgLine(null, first, FAN_Y, last, FAN_Y);         // fan rail
  treeSteps.forEach((st, i) => { s += svgLine(`e-in-${st.id}`,  cx[i], FAN_Y,        cx[i], ROW_TOP);    });
  treeSteps.forEach((st, i) => { s += svgLine(`e-out-${st.id}`, cx[i], bottoms[i],   cx[i], collectorY); });
  s += svgLine(null, first, collectorY, last, collectorY); // collector rail
  s += svgLine('e-conc', centerX, collectorY, centerX, concTop);
  document.getElementById('treeSvg').innerHTML = s;
}

function setEdge(id, cls) {
  if (cls) edgeCls[id] = cls; else delete edgeCls[id];
  const e = document.getElementById(id);
  if (!e) return;
  e.classList.remove('edge-active','edge-done','edge-alert','edge-deny');
  if (cls) e.classList.add(cls);
}

function updateTreeHyp(id, status, summary) {
  const node = document.getElementById(`tree-${id}`);
  if (!node) return;
  const icon  = node.querySelector('.hyp-icon');
  const sumEl = document.getElementById(`tree-${id}-summary`);

  node.classList.remove('hyp-running','hyp-done','hyp-alert');

  if (status === 'running') {
    node.classList.add('hyp-running');
    icon.textContent = '↻'; icon.className = 'hyp-icon spinning';
    setEdge(`e-in-${id}`, 'edge-active');
  } else if (status === 'done') {
    node.classList.add('hyp-done');
    icon.textContent = '✓'; icon.className = 'hyp-icon'; icon.style.color = 'var(--green-text)';
    setEdge(`e-in-${id}`, 'edge-done');
    setEdge(`e-out-${id}`, 'edge-done');
  } else if (status === 'alert') {
    node.classList.add('hyp-alert');
    icon.textContent = '⚠'; icon.className = 'hyp-icon'; icon.style.color = 'var(--amber)';
    setEdge(`e-in-${id}`, 'edge-alert');
    setEdge(`e-out-${id}`, 'edge-alert');
  }

  if (sumEl && summary) {
    sumEl.textContent = summary.length > 60 ? summary.slice(0, 57) + '…' : summary;
  }
  drawEdges();
}

function renderTreeConclusion(verdict, text) {
  const v    = verdict.toLowerCase();
  const node = document.getElementById('tree-conclusion');
  if (!node) return;
  node.className = `tree-node verdict-${v}`;
  node.style.display = 'block';
  void node.offsetWidth; node.classList.add('show');

  const glyph = v === 'approve' ? '✓' : v === 'deny' ? '✕' : '⚠';
  document.getElementById('tree-conc-label').innerHTML =
    `<span>${glyph}</span><span>${esc(verdict)}</span>`;
  const textEl = document.getElementById('tree-conc-text');
  textEl.textContent = text.length > 120 ? text.slice(0, 117) + '…' : text;

  setEdge('e-conc', v === 'approve' ? 'edge-done' : v === 'deny' ? 'edge-deny' : 'edge-alert');
  drawEdges();
}

function resetTree() {
  treeSteps = []; edgeCls = {}; layout = null;
  const canvas = document.getElementById('treeCanvas');
  canvas.style.display = 'none'; canvas.classList.remove('show');
  canvas.innerHTML = ''; canvas.style.height = ''; canvas.style.width = '';
  showTreeHint('idle');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"\n  Refund Intelligence UI → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
