"""
events.py
─────────
Context-variable-based event bus that captures log lines emitted by
middleware and memory providers and routes them to the per-request
async queue used by the SSE streaming endpoint.

Usage
─────
  # In a FastAPI request handler / SSE generator:
  queue: asyncio.Queue[str | None] = asyncio.Queue()
  set_queue(queue)                     # bind queue to this task's context
  task = asyncio.create_task(agent_run())   # task inherits copied context
  # …drain queue as SSE…

  # In middleware / memory code (instead of print):
  from events import emit
  emit(f"  🔧 [ToolCall ▶] {fn_name}({args_repr})")
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

# Each asyncio task gets its own copy of this variable (via create_task context copy).
_queue: ContextVar[asyncio.Queue | None] = ContextVar("_event_queue", default=None)


def emit(text: str) -> None:
    """
    Emit a log/event string.

    • If a queue is bound to the current context (i.e. we're inside a
      web-request task), puts the text on the queue for SSE streaming.
    • Otherwise falls back to plain print() so CLI usage is unaffected.
    """
    q = _queue.get()
    if q is not None:
        try:
            q.put_nowait(text)
        except Exception:
            pass  # never block or raise
    else:
        print(text)


def set_queue(q: "asyncio.Queue[str | None]") -> None:
    """Bind an async queue to the current execution context."""
    _queue.set(q)
