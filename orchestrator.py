"""
orchestrator.py
───────────────
Session-based multi-agent orchestrator.

Responsibilities:
  • Maintains a single AgentSession per conversation (shared memory / state)
  • Classifies each user turn to the right specialist agent
  • Drives the ConversationSummaryProvider transcript
  • Fires ConversationHook events for telemetry
  • Exposes a clean `chat(user_input)` coroutine

Design rationale
────────────────
The framework's Handoff workflow requires all agents to be ChatAgent
sub-types. Because we deliberately mix two instruction styles (agent-as-tool
vs conversation), we implement the routing in Python instead of relying on
the LLM to hand off — keeping the orchestration deterministic and testable
without live API calls.
"""

from __future__ import annotations

import time
from typing import Any

from agent_framework import Agent, AgentSession

from agents import build_agents
from memory import ConversationHook, ConversationSummaryProvider


# Intent keywords → agent role mapping
_BILLING_KW  = frozenset({
    "bill", "billing", "invoice", "refund", "charge", "payment", "balance",
    "account", "overdue", "paid", "cost", "price", "plan", "subscription",
})
_TECH_KW     = frozenset({
    "vpn", "email", "slow", "down", "outage", "error", "bug", "connect",
    "technical", "api", "broken", "storage",
    "ticket", "issue", "diagnos", "latency",
})
_SUMMARY_KW  = frozenset({
    "summary", "summarize", "summarise", "recap", "what happened",
    "overview", "wrap up", "end",
})


def _classify_intent(text: str) -> str:
    """Keyword-based intent classifier → agent role key."""
    words = set(text.lower().split())
    # Check each word and partial prefix matches
    if any(kw in text.lower() for kw in _SUMMARY_KW):
        return "summarizer"
    if words & _BILLING_KW:
        return "billing"
    if words & _TECH_KW or any(kw in text.lower() for kw in _TECH_KW):
        return "tech"
    return "triage"


class MultiAgentOrchestrator:
    """
    Orchestrates a fleet of specialised agents over a shared AgentSession.

    Usage:
        async with MultiAgentOrchestrator() as orch:
            reply = await orch.chat("My VPN keeps dropping")
            reply = await orch.chat("Can you refund my last charge?")
            reply = await orch.chat("Please summarize our conversation")
        orch.hook.print_audit_log()
    """

    def __init__(self) -> None:
        self._summary_provider = ConversationSummaryProvider()
        self._agents: dict[str, Agent] = build_agents(self._summary_provider)
        self._session = AgentSession()
        self.hook     = ConversationHook("Orchestrator")

    # ── Context manager ───────────────────────────────────────

    async def __aenter__(self) -> "MultiAgentOrchestrator":
        for agent in self._agents.values():
            if hasattr(agent, "__aenter__"):
                await agent.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        for agent in self._agents.values():
            if hasattr(agent, "__aexit__"):
                await agent.__aexit__(*args)

    # ── Public API ────────────────────────────────────────────

    async def chat(self, user_input: str) -> str:
        """
        Process one user turn.

        1. Classify intent → pick agent
        2. Fire on_turn_start hook
        3. Run the agent with the shared session (carries memory across turns)
        4. Update conversation transcript
        5. Fire on_turn_end hook
        6. Return response text
        """
        intent     = _classify_intent(user_input)
        agent      = self._agents[intent]
        agent_name = agent.name

        self.hook.on_turn_start(user_input, agent_name)
        t0 = time.perf_counter()

        try:
            result       = await agent.run(user_input, session=self._session)
            response_txt = result.text or "(no text response)"
        except Exception as exc:
            self.hook.on_error(exc)
            response_txt = f"[System error — {type(exc).__name__}: {exc}]"

        elapsed = (time.perf_counter() - t0) * 1000
        self.hook.on_turn_end(response_txt, elapsed)

        # Update rolling transcript for the ConversationSummaryProvider
        self._summary_provider.record("User",        user_input)
        self._summary_provider.record(agent_name,    response_txt)

        return response_txt

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def agents(self) -> dict[str, Agent]:
        return self._agents
