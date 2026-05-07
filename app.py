"""
app.py
──────
FastAPI web server for the Multi-Agent Customer Support demo UI.

Endpoints
─────────
  GET  /           → serves static/index.html
  POST /api/chat   → accepts {"message":"..."}, streams Server-Sent Events
  GET  /api/memory → returns current session memory snapshot as JSON
  POST /api/reset  → tears down the orchestrator and starts a fresh session

Event-stream protocol (SSE)
───────────────────────────
Each event is a JSON object on a  data: {...}\n\n  line:

  {"type": "agent_start",  "agent": "TechAgent", "text": "..."}
  {"type": "agent_end",    "text": "..."}
  {"type": "turn_start",   "turn": 1, "agent": "TechAgent", "text": "..."}
  {"type": "turn_end",     "text": "..."}
  {"type": "middleware",   "kind": "security|ratelimit|validation",
                           "blocked": bool, "throttled": bool, "text": "..."}
  {"type": "memory",       "kind": "user|summary", "direction": "in|out",
                           "text": "..."}
  {"type": "chat",         "direction": "send|recv", "text": "..."}
  {"type": "tool",         "direction": "call|result",  "text": "..."}
  {"type": "response",     "text": "<full agent reply>"}
  {"type": "done",         "memory": { session_state, transcript, audit_log }}
  {"type": "error",        "text": "<error message>"}
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from events import set_queue
from orchestrator import MultiAgentOrchestrator

# ─── Global orchestrator state ────────────────────────────────────────────────

_orch: MultiAgentOrchestrator | None = None


async def _get_orch() -> MultiAgentOrchestrator:
    global _orch
    if _orch is None:
        _orch = MultiAgentOrchestrator()
        await _orch.__aenter__()
    return _orch


async def _reset_orch() -> None:
    global _orch
    if _orch is not None:
        try:
            await _orch.__aexit__(None, None, None)
        except Exception:
            pass
    _orch = None


# ─── Event classification ─────────────────────────────────────────────────────

def _classify(text: str) -> dict[str, Any] | None:
    """
    Parse a single log line emitted via events.emit() into a typed dict
    that the frontend knows how to display.  Returns None for divider lines
    that should be suppressed.
    """
    t = text.strip()
    if not t:
        return None
    # Suppress pure visual divider lines
    if all(c in "═─━ \t" for c in t):
        return None

    # ── Agent timing ────────────────────────────────────────────────────────
    if "run started" in t:
        m = re.search(r"\[(\w+)\]", t)
        return {"type": "agent_start", "agent": m.group(1) if m else "Agent", "text": t}
    if "run finished" in t:
        return {"type": "agent_end", "text": t}

    # ── Orchestrator turn hooks ─────────────────────────────────────────────
    if "TURN" in t and "→" in t:
        m = re.search(r"TURN (\d+) → \[(\w+)\]", t)
        if m:
            return {
                "type": "turn_start",
                "turn": int(m.group(1)),
                "agent": m.group(2),
                "text": t,
            }
    if "TURN" in t and "DONE" in t:
        return {"type": "turn_end", "text": t}

    # ── Middleware ──────────────────────────────────────────────────────────
    if "SecurityMiddleware" in t:
        return {
            "type": "middleware",
            "kind": "security",
            "blocked": "Blocked" in t,
            "text": t,
        }
    if "RateLimitMiddleware" in t:
        return {
            "type": "middleware",
            "kind": "ratelimit",
            "throttled": "throttled" in t,
            "text": t,
        }
    if "ValidationMiddleware" in t:
        return {"type": "middleware", "kind": "validation", "text": t}

    # ── Memory ──────────────────────────────────────────────────────────────
    if "UserMemory" in t:
        direction = "in" if "▶" in t else "out"
        return {"type": "memory", "kind": "user", "direction": direction, "text": t}
    if "ConvSummary" in t:
        return {"type": "memory", "kind": "summary", "text": t}

    # ── Chat (LLM) middleware ────────────────────────────────────────────────
    if "ChatAudit ▶" in t:
        return {"type": "chat", "direction": "send", "text": t}
    if "ChatAudit ◀" in t:
        return {"type": "chat", "direction": "recv", "text": t}

    # ── Tool calls ───────────────────────────────────────────────────────────
    if "ToolCall ▶" in t:
        return {"type": "tool", "direction": "call", "text": t}
    if "ToolCall ◀" in t:
        return {"type": "tool", "direction": "result", "text": t}

    # Default
    return {"type": "info", "text": t}


# ─── Memory snapshot ──────────────────────────────────────────────────────────

def _memory_snapshot(orch: MultiAgentOrchestrator) -> dict[str, Any]:
    raw_state: dict = orch.session.state or {}
    return {
        "session_state": raw_state,
        "transcript": list(orch._summary_provider._transcript),
        "audit_log": orch.hook.get_audit_log(),
    }


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Multi-Agent Demo UI")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/chat")
async def chat_endpoint(request: Request) -> StreamingResponse:
    body = await request.json()
    user_msg = (body.get("message") or "").strip()
    if not user_msg:
        return StreamingResponse(iter([]), media_type="text/event-stream")

    orch = await _get_orch()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    # Bind the queue to the current async context so middleware emit() calls
    # inside the agent task (which inherits this context) put events here.
    set_queue(queue)

    async def generate() -> AsyncGenerator[str, None]:
        async def _run() -> None:
            try:
                result = await orch.chat(user_msg)
                await queue.put(f"__RESULT__:{result}")
            except Exception as exc:
                await queue.put(f"__ERROR__:{exc}")
            finally:
                await queue.put(None)  # sentinel — drain is complete

        # create_task copies the current context (which has our queue bound)
        task = asyncio.create_task(_run())

        while True:
            item = await queue.get()
            if item is None:
                break
            if item.startswith("__RESULT__:"):
                text = item[len("__RESULT__:"):]
                yield f"data: {json.dumps({'type': 'response', 'text': text})}\n\n"
            elif item.startswith("__ERROR__:"):
                err = item[len("__ERROR__:"):]
                yield f"data: {json.dumps({'type': 'error', 'text': str(err)})}\n\n"
            else:
                event = _classify(item)
                if event is not None:
                    yield f"data: {json.dumps(event)}\n\n"

        await task  # ensure task cleanup
        yield f"data: {json.dumps({'type': 'done', 'memory': _memory_snapshot(orch)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/memory")
async def memory_endpoint() -> JSONResponse:
    orch = await _get_orch()
    return JSONResponse(_memory_snapshot(orch))


@app.post("/api/reset")
async def reset_endpoint() -> JSONResponse:
    await _reset_orch()
    return JSONResponse({"status": "reset"})


# ─── Dev entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
