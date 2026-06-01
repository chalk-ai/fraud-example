#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["fastapi", "uvicorn[standard]", "openai", "chalkpy", "python-dotenv"]
# ///
"""Refund-abuse agent demo UI.

Runs the agent loop locally (for SSE streaming) and calls Chalk for tool
execution and the LLM server for inference. Supports multi-turn conversation:
the agent can ask clarifying questions before issuing a verdict.

Run:
  ./chalkcompute_agent_demo_ui.py
  open http://localhost:8000
"""

import asyncio
import json
import os
import queue
import re
import threading
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

load_dotenv()

from openai import OpenAI
from chalk.client import ChalkClient

# ── Option A: self-hosted Qwen2.5-7B on Chalk Compute ────────────────────────
# _llm = OpenAI(base_url=os.environ["VLLM_URL"] + "/v1", api_key="EMPTY")
# _model = "Qwen/Qwen2.5-7B-Instruct"

# ── Option B: hosted Claude API (active) ─────────────────────────────────────
_llm = OpenAI(
    base_url="https://api.anthropic.com/v1",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    default_headers={"anthropic-version": "2023-06-01"},
)
_model = "claude-sonnet-4-6"
_chalk = ChalkClient()

# Session store: session_id → messages list
_sessions: dict[str, list] = {}

SYSTEM_PROMPT = (
    "You investigate refund claims for potential abuse. "
    "You have access to real-time signals from the Chalk feature store — "
    "always call both get_risk_score and get_prior_refund_count before ruling. "
    "If you need context your tools don't provide, ask the customer a direct question. "
    "Only issue a final verdict when you have enough information — "
    "begin that response with APPROVE, DENY, or ESCALATE on the first line, "
    "followed by one sentence of reasoning."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_risk_score",
            "description": (
                "Fetch the real-time fraud risk score for an order "
                "(0.0 = low risk, 1.0 = high risk)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prior_refund_count",
            "description": "Look up how many prior refund claims this customer has submitted.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]


def _run_tool(name: str, inp: dict) -> str:
    if name == "get_risk_score":
        ctx = _chalk.query(
            input={"order.id": inp["order_id"]},
            output=["order.refund_risk_score"],
        )
        return f"{ctx.get_feature_value('order.refund_risk_score'):.2f}"
    if name == "get_prior_refund_count":
        ctx = _chalk.query(
            input={"order.id": inp["order_id"]},
            output=["order.customer_prior_refunds"],
        )
        return str(ctx.get_feature_value("order.customer_prior_refunds"))
    return "error: unknown tool"


def _agent_thread(messages: list, q: queue.Queue) -> None:
    """Run one agent turn. Emits tool_call/tool_result events, then either
    a 'question' (agent needs more info) or 'decision' (final verdict)."""
    try:
        while True:
            response = _llm.chat.completions.create(
                model=_model, max_tokens=1024, tools=TOOLS, messages=messages,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                text = (msg.content or "").strip()
                m = re.match(r"(APPROVE|DENY|ESCALATE)", text)
                if m:
                    q.put({"type": "decision", "text": text, "verdict": m.group(1)})
                else:
                    # Agent is asking a question — append to history and surface to user
                    messages.append({"role": "assistant", "content": text})
                    q.put({"type": "question", "text": text})
                break

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
                q.put({"type": "tool_call", "id": tc.id, "name": tc.function.name, "args": inp})
                result = _run_tool(tc.function.name, inp)
                q.put({"type": "tool_result", "id": tc.id, "result": result})
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
    order_id: str
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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Order {req.order_id}. Refund reason: {req.reason!r}."},
    ]
    _sessions[session_id] = messages

    q: queue.Queue = queue.Queue()
    threading.Thread(target=_agent_thread, args=(messages, q), daemon=True).start()

    # Prepend session_id so the client knows which session this is
    async def stream():
        yield f"data: {json.dumps({'type': 'session', 'id': session_id})}\n\n"
        async for chunk in _sse(q):
            yield chunk

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reply/{session_id}")
async def reply(session_id: str, req: ReplyRequest) -> StreamingResponse:
    messages = _sessions.get(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages.append({"role": "user", "content": req.message})

    q: queue.Queue = queue.Queue()
    threading.Thread(target=_agent_thread, args=(messages, q), daemon=True).start()

    return StreamingResponse(
        _sse(q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
<title>Chalk Refund Intelligence</title>
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
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Header ── */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 15px 24px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .header-left {
    display: flex; align-items: center; gap: 10px;
    font-size: 15px; font-weight: 600; letter-spacing: -0.01em;
  }
  .logo {
    width: 26px; height: 26px; background: var(--accent);
    border-radius: 7px; display: flex; align-items: center;
    justify-content: center; font-size: 13px; font-weight: 700;
  }
  .model-badge {
    font-size: 11px; color: var(--muted);
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 20px; padding: 3px 10px;
    font-family: 'SF Mono', 'Menlo', monospace; letter-spacing: 0.01em;
  }

  /* ── Order selector (inline dropdown) ── */
  .order-sel {
    position: relative; flex-shrink: 0;
  }
  .order-sel-btn {
    display: flex; align-items: center; gap: 8px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 11px 14px;
    color: var(--text); font-size: 13px; font-family: 'SF Mono', 'Menlo', monospace;
    cursor: pointer; white-space: nowrap; transition: border-color 0.15s;
    user-select: none;
  }
  .order-sel-btn:hover:not(:disabled) { border-color: #333; }
  .order-sel-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .order-sel-btn.sel-high   { border-color: rgba(239,68,68,.6); }
  .order-sel-btn.sel-low    { border-color: rgba(34,197,94,.6); }
  .order-sel-btn.sel-medium { border-color: rgba(245,158,11,.6); }
  .order-sel-chevron { color: var(--muted); font-size: 10px; transition: transform 0.15s; }
  .order-sel-btn.open .order-sel-chevron { transform: rotate(180deg); }

  .order-dropdown {
    position: absolute; bottom: calc(100% + 8px); left: 0;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 6px; min-width: 240px;
    box-shadow: 0 -8px 32px rgba(0,0,0,.5);
    display: flex; flex-direction: column; gap: 4px;
    z-index: 100; animation: rise 0.15s ease;
  }
  .order-option {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 12px; border-radius: 8px; cursor: pointer;
    transition: background 0.1s; user-select: none;
  }
  .order-option:hover { background: var(--surface2); }
  .order-option-id { font-size: 13px; font-weight: 600; font-family: 'SF Mono', 'Menlo', monospace; }
  .order-option-right { display: flex; align-items: center; gap: 7px; }
  .risk-badge { font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 2px 7px; border-radius: 4px; }
  .risk-high   { background: rgba(239,68,68,.15);  color: #f87171; }
  .risk-low    { background: rgba(34,197,94,.15);  color: #4ade80; }
  .risk-medium { background: rgba(245,158,11,.15); color: #fbbf24; }
  .order-desc  { font-size: 11px; color: var(--muted); }

  /* ── Chat ── */
  .chat {
    flex: 1; overflow-y: auto; padding: 28px 24px;
    display: flex; flex-direction: column; gap: 22px;
  }
  .chat::-webkit-scrollbar { width: 4px; }
  .chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  .empty-state {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 10px; color: var(--muted); pointer-events: none;
  }
  .empty-icon { font-size: 36px; opacity: 0.25; }
  .empty-text { font-size: 13px; }

  /* Bubbles */
  .msg-user {
    align-self: flex-end; max-width: 420px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 16px 16px 4px 16px; padding: 12px 16px;
    animation: rise 0.2s ease;
  }
  .msg-order-ref { font-size: 11px; color: var(--muted); font-family: monospace; margin-bottom: 4px; }
  .msg-reason    { font-size: 14px; line-height: 1.4; }

  /* Plain user reply bubble (no order ref) */
  .msg-reply {
    align-self: flex-end; max-width: 420px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 16px 16px 4px 16px; padding: 12px 16px;
    font-size: 14px; line-height: 1.4; animation: rise 0.2s ease;
  }

  /* Agent question bubble */
  .msg-question {
    align-self: flex-start; max-width: 480px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px; padding: 12px 16px;
    font-size: 14px; line-height: 1.55; animation: rise 0.2s ease;
  }

  /* Agent tool container */
  .msg-agent {
    align-self: flex-start; max-width: 580px;
    display: flex; flex-direction: column; gap: 8px; animation: rise 0.2s ease;
  }

  /* Tool card */
  .tool-card {
    background: var(--tool-bg); border: 1px solid var(--tool-border);
    border-left: 3px solid var(--accent); border-radius: 8px;
    padding: 10px 14px; font-family: 'SF Mono', 'Menlo', monospace;
    font-size: 12px; animation: rise 0.15s ease;
  }
  .tool-header { display: flex; align-items: center; gap: 7px; color: #a5b4fc; margin-bottom: 3px; }
  .tool-fn   { font-weight: 600; }
  .tool-args { color: #4b5563; }
  .tool-result-row {
    margin-top: 7px; padding-top: 7px;
    border-top: 1px solid var(--tool-border);
    display: flex; align-items: center; gap: 7px;
  }
  .tool-result-row.pending { color: var(--muted); }
  .tool-result-row.done    { color: #86efac; }

  /* Thinking dots */
  .thinking { display: flex; align-items: center; gap: 5px; padding: 8px 2px; }
  .dot { width: 6px; height: 6px; background: var(--muted); border-radius: 50%; animation: pulse 1.2s ease-in-out infinite; }
  .dot:nth-child(2) { animation-delay: .2s; }
  .dot:nth-child(3) { animation-delay: .4s; }

  /* Verdict */
  .verdict-card { border-radius: 12px; padding: 16px 20px; border: 1px solid; animation: rise 0.25s ease; }
  .verdict-approve  { background: rgba(34,197,94,.08);  border-color: rgba(34,197,94,.25); }
  .verdict-deny     { background: rgba(239,68,68,.08);  border-color: rgba(239,68,68,.25); }
  .verdict-escalate { background: rgba(245,158,11,.08); border-color: rgba(245,158,11,.25); }
  .verdict-label { font-size: 18px; font-weight: 700; letter-spacing: .04em; margin-bottom: 6px; }
  .verdict-approve  .verdict-label { color: var(--green); }
  .verdict-deny     .verdict-label { color: var(--red);   }
  .verdict-escalate .verdict-label { color: var(--amber); }
  .verdict-text { font-size: 13px; color: #999; line-height: 1.55; }

  /* Error */
  .error-card {
    background: rgba(239,68,68,.07); border: 1px solid rgba(239,68,68,.2);
    border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #fca5a5;
  }

  /* ── Input bar ── */
  .input-bar {
    padding: 14px 24px; border-top: 1px solid var(--border);
    display: flex; gap: 10px; flex-shrink: 0; align-items: center;
  }
  .main-input {
    flex: 1; background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 11px 16px; color: var(--text);
    font-size: 14px; font-family: inherit; outline: none; transition: border-color 0.15s;
  }
  .main-input::placeholder { color: var(--muted); }
  .main-input:focus { border-color: #3a3a5a; }
  .main-input:disabled { opacity: 0.4; }
  .submit-btn {
    background: var(--accent); color: #fff; border: none;
    border-radius: 10px; padding: 11px 22px; font-size: 14px; font-weight: 600;
    font-family: inherit; cursor: pointer; transition: opacity 0.15s; white-space: nowrap;
  }
  .submit-btn:hover:not(:disabled) { opacity: .85; }
  .submit-btn:disabled { opacity: .35; cursor: not-allowed; }
  .dismiss-btn {
    background: transparent; color: var(--muted);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 11px 18px; font-size: 14px; font-family: inherit;
    cursor: pointer; transition: color 0.15s, border-color 0.15s; white-space: nowrap;
    display: none;
  }
  .dismiss-btn:hover { color: var(--text); border-color: #444; }

  @keyframes rise {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulse {
    0%,80%,100% { transform: scale(.55); opacity: .35; }
    40%          { transform: scale(1);   opacity: 1; }
  }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo">◆</div>
    Chalk Refund Intelligence
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    <button id="newBtn" class="dismiss-btn" onclick="dismiss()" style="display:none">+ New</button>
    <div class="model-badge">claude-sonnet-4-6 · Anthropic</div>
  </div>
</div>

<div class="chat" id="chat">
  <div class="empty-state" id="emptyState">
    <div class="empty-icon">⚖</div>
    <div class="empty-text">Select an order and describe the refund reason to begin</div>
  </div>
</div>

<div class="input-bar">
  <div class="order-sel" id="orderSel">
    <button class="order-sel-btn" id="orderSelBtn" onclick="toggleDropdown()">
      <span id="orderSelLabel">Select order</span>
      <span class="order-sel-chevron">▾</span>
    </button>
    <div class="order-dropdown" id="orderDropdown" style="display:none">
      <div class="order-option" onclick="selectOrder('ORD-8823','high')">
        <span class="order-option-id">ORD-8823</span>
        <div class="order-option-right">
          <span class="risk-badge risk-high">High Risk</span>
          <span class="order-desc">4 prior claims</span>
        </div>
      </div>
      <div class="order-option" onclick="selectOrder('ORD-1001','low')">
        <span class="order-option-id">ORD-1001</span>
        <div class="order-option-right">
          <span class="risk-badge risk-low">Low Risk</span>
          <span class="order-desc">No prior claims</span>
        </div>
      </div>
      <div class="order-option" onclick="selectOrder('ORD-4242','medium')">
        <span class="order-option-id">ORD-4242</span>
        <div class="order-option-right">
          <span class="risk-badge risk-medium">Medium Risk</span>
          <span class="order-desc">2 prior claims</span>
        </div>
      </div>
    </div>
  </div>
  <input id="mainInput" class="main-input" type="text"
         placeholder="Select an order first…"
         value=""
         disabled
         onkeydown="if(event.key==='Enter')primaryAction()">
  <button id="dismissBtn" class="dismiss-btn" onclick="dismiss()">New investigation</button>
  <button id="submitBtn"  class="submit-btn"  onclick="primaryAction()" disabled>Investigate →</button>
</div>

<script>
let selectedOrder  = null;
let selectedRisk   = null;
let sessionId      = null;
let mode           = 'idle';   // 'idle' | 'thinking' | 'reply' | 'done'
let activeAgentMsg = null;
let activeThinking = null;

function toggleDropdown() {
  const btn  = document.getElementById('orderSelBtn');
  const menu = document.getElementById('orderDropdown');
  const open = menu.style.display !== 'none';
  menu.style.display = open ? 'none' : 'flex';
  btn.classList.toggle('open', !open);
}

function selectOrder(id, risk) {
  if (sessionId) dismiss();
  selectedOrder = id;
  selectedRisk  = risk;

  const btn   = document.getElementById('orderSelBtn');
  const label = document.getElementById('orderSelLabel');
  label.textContent = id;
  btn.className = `order-sel-btn sel-${risk}`;

  document.getElementById('orderDropdown').style.display = 'none';

  const input  = document.getElementById('mainInput');
  const submit = document.getElementById('submitBtn');
  input.disabled    = false;
  input.placeholder = 'Describe the refund reason…';
  input.value       = 'Item arrived damaged';
  submit.disabled   = false;
  input.focus();
  input.select();
}

// Close dropdown on outside click
document.addEventListener('click', e => {
  const sel = document.getElementById('orderSel');
  if (sel && !sel.contains(e.target)) {
    document.getElementById('orderDropdown').style.display = 'none';
    document.getElementById('orderSelBtn').classList.remove('open');
  }
});

function setMode(m) {
  mode = m;
  const input   = document.getElementById('mainInput');
  const submit  = document.getElementById('submitBtn');
  const dismiss = document.getElementById('dismissBtn');
  const selBtn  = document.getElementById('orderSelBtn');

  if (m === 'idle') {
    selBtn.disabled   = false;
    input.disabled    = false;
    input.placeholder = selectedOrder ? 'Describe the refund reason…' : 'Select an order first…';
    input.value       = selectedOrder ? 'Item arrived damaged' : '';
    submit.disabled   = !selectedOrder;
    submit.textContent = 'Investigate →';
    dismiss.style.display = 'none';
    if (selectedOrder) input.focus();
  } else if (m === 'thinking') {
    selBtn.disabled = true;
    input.disabled  = true;
    submit.disabled = true;
    dismiss.style.display = 'inline-block';
  } else if (m === 'reply') {
    selBtn.disabled   = true;
    input.disabled    = false;
    input.placeholder = 'Reply to agent…';
    input.value       = '';
    submit.disabled   = false;
    submit.textContent = 'Send →';
    dismiss.style.display = 'inline-block';
    input.focus();
  } else if (m === 'done') {
    selBtn.disabled = true;
    input.disabled  = true;
    submit.disabled = true;
    dismiss.style.display = 'inline-block';
  }
}

function primaryAction() {
  if (mode === 'idle')  startInvestigation();
  else if (mode === 'reply') sendReply();
}

function startInvestigation() {
  if (!selectedOrder) return;
  const reason = document.getElementById('mainInput').value.trim();
  if (!reason) return;

  const chat = document.getElementById('chat');
  document.getElementById('emptyState')?.remove();
  setMode('thinking');

  document.getElementById('newBtn').style.display = 'inline-block';
  appendUserBubble(selectedOrder, reason);
  activeAgentMsg = appendAgentMsg();
  scrollBottom();

  fetch('/investigate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({order_id: selectedOrder, reason}),
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
  }).then(res => streamEvents(res)).catch(() => setMode('done'));
}

function dismiss() {
  if (sessionId) {
    fetch(`/session/${sessionId}`, {method: 'DELETE'}).catch(() => {});
    sessionId = null;
  }
  activeAgentMsg = null;
  activeThinking = null;
  document.getElementById('chat').innerHTML =
    '<div class="empty-state" id="emptyState">' +
      '<div class="empty-icon">⚖</div>' +
      '<div class="empty-text">Select an order and describe the refund reason to begin</div>' +
    '</div>';
  document.getElementById('orderSelBtn').disabled = false;
  document.getElementById('newBtn').style.display = 'none';
  setMode('idle');
}

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

function handleEvent(ev) {
  if (ev.type === 'session') {
    sessionId = ev.id;

  } else if (ev.type === 'tool_call') {
    activeThinking.style.display = 'none';
    const args = Object.entries(ev.args).map(([k, v]) => `${k}='${v}'`).join(', ');
    const card = document.createElement('div');
    card.className = 'tool-card';
    card.id = 'tc-' + ev.id;
    card.innerHTML =
      `<div class="tool-header"><span>⚙</span>` +
      `<span class="tool-fn">${esc(ev.name)}</span>` +
      `<span class="tool-args">(${esc(args)})</span></div>` +
      `<div class="tool-result-row pending" id="tr-${esc(ev.id)}">` +
        mkThinking().outerHTML + `</div>`;
    activeAgentMsg.insertBefore(card, activeThinking);

  } else if (ev.type === 'tool_result') {
    const tr = document.getElementById('tr-' + ev.id);
    if (tr) { tr.className = 'tool-result-row done'; tr.innerHTML = `<span>→</span><span>${esc(ev.result)}</span>`; }
    activeThinking.style.display = '';

  } else if (ev.type === 'question') {
    activeThinking.remove();
    activeThinking = null;
    const bubble = document.createElement('div');
    bubble.className = 'msg-question';
    bubble.textContent = ev.text;
    activeAgentMsg.appendChild(bubble);
    setMode('reply');

  } else if (ev.type === 'decision') {
    activeThinking.remove();
    activeThinking = null;
    const lines = ev.text.split('\n').map(l => l.trim()).filter(Boolean);
    const body  = lines.slice(1).join(' ') || lines[0].replace(/^(APPROVE|DENY|ESCALATE)[:\s–-]*/i, '');
    const v     = ev.verdict.toLowerCase();
    const card  = document.createElement('div');
    card.className = `verdict-card verdict-${v}`;
    card.innerHTML = `<div class="verdict-label">${esc(ev.verdict)}</div><div class="verdict-text">${esc(body)}</div>`;
    activeAgentMsg.appendChild(card);
    setMode('done');

  } else if (ev.type === 'error') {
    activeThinking?.remove();
    const card = document.createElement('div');
    card.className = 'error-card';
    card.textContent = '⚠ ' + ev.message;
    activeAgentMsg.appendChild(card);
    setMode('done');
  }

  scrollBottom();
}

function appendUserBubble(orderId, reason) {
  const el = document.createElement('div');
  el.className = 'msg-user';
  el.innerHTML = `<div class="msg-order-ref">${esc(orderId)}</div><div class="msg-reason">${esc(reason)}</div>`;
  document.getElementById('chat').appendChild(el);
}

function appendReplyBubble(text) {
  const el = document.createElement('div');
  el.className = 'msg-reply';
  el.textContent = text;
  document.getElementById('chat').appendChild(el);
}

function appendAgentMsg() {
  const el = document.createElement('div');
  el.className = 'msg-agent';
  const t = mkThinking();
  el.appendChild(t);
  activeThinking = t;
  document.getElementById('chat').appendChild(el);
  return el;
}

function mkThinking() {
  const el = document.createElement('div');
  el.className = 'thinking';
  el.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
  return el;
}

function scrollBottom() { const c = document.getElementById('chat'); c.scrollTop = c.scrollHeight; }

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"\n  Refund Intelligence UI → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
