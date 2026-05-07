"""
middleware.py
─────────────
All middleware implementations for the multi-agent system.

Microsoft Agent Framework supports three middleware types:
  1. AgentMiddleware  – wraps the entire agent.run() call
  2. FunctionMiddleware – wraps each individual tool call
  3. ChatMiddleware   – wraps the raw LLM inference call

Both function-based and class-based styles are demonstrated.
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from events import emit
from agent_framework import (
    AgentContext,
    AgentMiddleware,
    ChatContext,
    ChatMiddleware,
    FunctionInvocationContext,
    FunctionMiddleware,
)

# ─────────────────────────────────────────────────────────────
# 1.  AGENT-LEVEL MIDDLEWARE  (wraps each full agent run)
# ─────────────────────────────────────────────────────────────

async def timing_middleware(
    context: AgentContext,
    call_next: Callable[[], Awaitable[None]],
) -> None:
    """
    [Agent Middleware] Measures wall-clock latency for each agent run
    and prints a timestamped start/end banner.
    """
    agent_name = context.agent.name
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    emit(f"  ⏱  [{agent_name}] run started  @ {ts} UTC")
    t0 = time.perf_counter()
    await call_next()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    emit(f"  ⏱  [{agent_name}] run finished   {elapsed_ms:.1f} ms")


# PII / sensitive-word block list
_BLOCKED_KEYWORDS: frozenset[str] = frozenset({
    "password", "passwd", "ssn", "social security",
    "credit card", "cvv", "pin number", "secret key",
})

async def security_middleware(
    context: AgentContext,
    call_next: Callable[[], Awaitable[None]],
) -> None:
    """
    [Agent Middleware] Blocks any agent run whose last user message
    contains PII or security-sensitive keywords.
    Not calling call_next() terminates the pipeline silently.
    """
    last_msg = context.messages[-1] if context.messages else None
    text = ""
    if last_msg and hasattr(last_msg, "text") and last_msg.text:
        text = last_msg.text.lower()

    for kw in _BLOCKED_KEYWORDS:
        if kw in text:
            emit(f"  🚫 [SecurityMiddleware] Blocked — sensitive term '{kw}' detected")
            return   # Pipeline terminated; agent does NOT run

    emit("  ✅ [SecurityMiddleware] Passed")
    await call_next()


class RateLimitMiddleware(AgentMiddleware):
    """
    [Agent Middleware — Class-based] Simple in-memory token-bucket rate
    limiter per agent.  Refill rate: 1 token / 2 s, capacity: 10 tokens.
    Throttles (sleeps 500 ms) when the bucket is empty.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": 10.0, "last_refill": time.monotonic()}
        )

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        agent_name = context.agent.name
        b = self._buckets[agent_name]

        # Token refill
        now = time.monotonic()
        elapsed = now - b["last_refill"]
        b["tokens"] = min(10.0, b["tokens"] + elapsed * 0.5)
        b["last_refill"] = now

        if b["tokens"] < 1.0:
            emit(f"  ⚠️  [RateLimitMiddleware] '{agent_name}' throttled 500 ms")
            await asyncio.sleep(0.5)
        else:
            b["tokens"] -= 1.0
            remaining = round(b["tokens"], 1)
            emit(f"  🪣 [RateLimitMiddleware] '{agent_name}' token consumed (remaining: {remaining})")

        await call_next()


# ─────────────────────────────────────────────────────────────
# 2.  FUNCTION-LEVEL MIDDLEWARE  (wraps each tool call)
# ─────────────────────────────────────────────────────────────

async def logging_fn_middleware(
    context: FunctionInvocationContext,
    call_next: Callable[[], Awaitable[None]],
) -> None:
    """
    [Function Middleware] Logs every tool invocation with its name,
    arguments preview, result preview, and execution time.
    """
    fn_name   = context.function.name
    args_repr = str(context.arguments)[:120]
    emit(f"    🔧 [ToolCall ▶] {fn_name}({args_repr})")
    t0 = time.perf_counter()
    await call_next()
    ms = (time.perf_counter() - t0) * 1000
    result_repr = str(getattr(context, "result", "—"))[:120]
    emit(f"    ✔  [ToolCall ◀] {fn_name} → {result_repr}  [{ms:.1f} ms]")


class ValidationFnMiddleware(FunctionMiddleware):
    """
    [Function Middleware — Class-based] Validates tool arguments before
    execution and injects a provenance tag into tool results.
    """

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        fn_name = context.function.name

        # Guard: reject obviously invalid account IDs
        args = context.arguments or {}
        account_id = args.get("account_id", "")
        if account_id and not account_id.startswith("ACC-"):
            emit(f"    ⛔ [ValidationMiddleware] Invalid account_id '{account_id}' for {fn_name}")
            # Inject an error result instead of calling the real tool
            context.result = {"error": f"Invalid account_id format: '{account_id}'"}
            return

        await call_next()

        # Post-call: tag the result with provenance metadata
        if isinstance(context.result, dict):
            context.result["_source"] = fn_name
            context.result["_ts"]     = datetime.now(timezone.utc).isoformat() + "Z"


# ─────────────────────────────────────────────────────────────
# 3.  CHAT (INFERENCE) MIDDLEWARE  (wraps each raw LLM call)
# ─────────────────────────────────────────────────────────────

async def chat_audit_middleware(
    context: ChatContext,
    call_next: Callable[[], Awaitable[None]],
) -> None:
    """
    [Chat Middleware] Logs the number of messages sent to the LLM and
    captures finish reason from the response. Runs once per model call,
    including tool-result follow-up calls inside a multi-turn tool loop.
    """
    msg_count = len(context.messages) if context.messages else 0
    emit(f"    💬 [ChatAudit ▶] Sending {msg_count} msg(s) to model")
    await call_next()
    if context.result:
        finish = getattr(context.result, "finish_reason", "unknown")
        emit(f"    💬 [ChatAudit ◀] Model responded (finish={finish})")
