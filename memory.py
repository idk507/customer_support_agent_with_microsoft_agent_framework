"""
memory.py
─────────
Context providers supply agent memory, personalisation, and external
knowledge that get injected into (or extracted from) each agent run.

Microsoft Agent Framework ContextProvider lifecycle:
  before_run(agent, session, context, state) — inject context before LLM call
  after_run (agent, session, context, state) — extract info from response

The `state` dict is provider-scoped and persisted across turns in the session.
"""

from __future__ import annotations

import re
from typing import Any

from events import emit
from agent_framework import AgentSession, ContextProvider, SessionContext


# ─────────────────────────────────────────────────────────────
# 1.  USER MEMORY PROVIDER
#     Remembers user name, account tier, and turn count.
# ─────────────────────────────────────────────────────────────

class UserMemoryProvider(ContextProvider):
    """
    Short-term conversational memory that persists user identity
    and account tier inside the AgentSession.state dict.

    before_run: Injects personalisation instructions.
    after_run : Scans messages for "my name is …" patterns and
                account-tier information to extract and remember.
    """

    SOURCE_ID = "user_memory"

    def __init__(self) -> None:
        super().__init__(self.SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        turn = state.get("turn_count", 0) + 1
        state["turn_count"] = turn

        lines: list[str] = [f"[Memory | Turn {turn}]"]

        name = state.get("user_name")
        if name:
            lines.append(f"User's name: {name}. Address them by name.")
        else:
            lines.append("User's name is not known yet; be polite.")

        tier = state.get("user_tier")
        if tier == "premium":
            lines.append("PREMIUM customer — offer proactive, detailed assistance.")
        elif tier == "standard":
            lines.append("Standard tier customer.")

        if state.get("open_ticket"):
            lines.append(f"Open ticket on file: {state['open_ticket']}. Reference it if relevant.")

        instruction = " ".join(lines)
        context.extend_instructions(self.SOURCE_ID, instruction)
        emit(f"  🧠 [UserMemory ▶] Injected: {instruction}")

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        for msg in context.get_messages():
            text = (getattr(msg, "text", None) or "")
            if not isinstance(text, str) or not text.strip():
                continue
            tl = text.lower()

            # Extract name
            match = re.search(r"my name is\s+([a-z]+)", tl)
            if match and not state.get("user_name"):
                state["user_name"] = match.group(1).capitalize()
                emit(f"  🧠 [UserMemory ◀] Remembered name='{state['user_name']}'")

            # Extract tier from account lookup results in conversation
            if "premium" in tl:
                state.setdefault("user_tier", "premium")
                emit(f"  🧠 [UserMemory ◀] Remembered tier='premium'")
            elif '"tier": "standard"' in tl or "standard tier" in tl:
                state.setdefault("user_tier", "standard")

            # Remember if a ticket was created
            ticket_match = re.search(r"(TKT-\d{5})", text)
            if ticket_match:
                state["open_ticket"] = ticket_match.group(1)
                emit(f"  🧠 [UserMemory ◀] Remembered ticket='{state['open_ticket']}'")


# ─────────────────────────────────────────────────────────────
# 2.  CONVERSATION SUMMARY PROVIDER
#     Maintains a running plain-text transcript and injects the
#     last N exchanges so all agents share conversation context.
# ─────────────────────────────────────────────────────────────

class ConversationSummaryProvider(ContextProvider):
    """
    Keeps a rolling transcript of the last MAX_TURNS turns and
    injects it as additional context so every specialist agent
    knows what happened earlier in the conversation.
    """

    SOURCE_ID   = "conv_summary"
    MAX_TURNS   = 6   # last 6 exchanges (12 lines)

    def __init__(self) -> None:
        super().__init__(self.SOURCE_ID)
        self._transcript: list[str] = []

    def record(self, role: str, text: str) -> None:
        """Called externally by the orchestrator after each turn."""
        entry = f"{role}: {text[:300]}"
        self._transcript.append(entry)
        # Keep rolling window
        if len(self._transcript) > self.MAX_TURNS * 2:
            self._transcript = self._transcript[-(self.MAX_TURNS * 2):]

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        if not self._transcript:
            return
        history = "\n".join(self._transcript)
        instruction = f"Conversation history so far:\n{history}\n"
        context.extend_instructions(self.SOURCE_ID, instruction)
        emit(f"  📜 [ConvSummary] Injected {len(self._transcript)} history lines")

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        # Nothing to extract; transcript is updated externally.
        pass


# ─────────────────────────────────────────────────────────────
# 3.  CONVERSATION HOOK
#     Application-level lifecycle callbacks (not a framework type,
#     but a pattern for cross-cutting telemetry and audit logging).
# ─────────────────────────────────────────────────────────────

class ConversationHook:
    """
    Fires around each conversational turn.
    Useful for: telemetry, audit logging, A/B experiments, UI events.
    """

    def __init__(self, name: str = "App") -> None:
        self.name  = name
        self._turn = 0
        self._log: list[dict[str, Any]] = []

    # ── Public hooks ──────────────────────────────────────────

    def on_turn_start(self, user_input: str, agent_name: str) -> None:
        self._turn += 1
        entry = {
            "turn": self._turn,
            "agent": agent_name,
            "user_input": user_input[:200],
        }
        self._log.append(entry)
        emit(f"  🎯 [{self.name}] TURN {self._turn} → [{agent_name}]")
        emit(f"     User: {user_input[:80]!r}")

    def on_turn_end(self, response: str | None, elapsed_ms: float) -> None:
        preview = (response or "(no response)")[:120]
        if self._log:
            self._log[-1]["response_preview"] = preview
            self._log[-1]["elapsed_ms"]        = round(elapsed_ms)
        emit(f"\n  \u2705 [{self.name}] TURN {self._turn} DONE  ({elapsed_ms:.0f} ms)")
        emit(f"     Agent: {preview}")

    def on_error(self, error: Exception) -> None:
        emit(f"\n  \u274c [{self.name}] ERROR on turn {self._turn}: {error}")
        if self._log:
            self._log[-1]["error"] = str(error)

    def on_security_block(self, reason: str) -> None:
        emit(f"\n  🔒 [{self.name}] Security block on turn {self._turn}: {reason}")

    # ── Audit log access ──────────────────────────────────────

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self._log)

    def print_audit_log(self) -> None:
        print(f"\n{'━'*68}")
        print(f"  📋 AUDIT LOG ({len(self._log)} turns)")
        print(f"{'━'*68}")
        for entry in self._log:
            ok  = "✅" if "error" not in entry else "❌"
            ms  = entry.get("elapsed_ms", "?")
            usr = entry.get("user_input", "")[:60]
            agt = entry.get("agent", "?")
            print(f"  {ok} Turn {entry['turn']:2d} | {agt:<18} | {ms:>6} ms | {usr!r}")
        print(f"{'━'*68}")
