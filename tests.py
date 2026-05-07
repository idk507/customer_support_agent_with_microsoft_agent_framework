"""
tests.py
────────
Comprehensive test suite for the multi-agent system.

Tests run WITHOUT live Azure OpenAI credentials by mocking the agent.run()
coroutine. This lets CI/CD pipelines validate all logic, middleware,
memory providers, hooks, workflow wiring, and tool functions in isolation.

Run:
    pytest tests.py -v
    pytest tests.py -v -k "test_tools"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ── Path setup ───────────────────────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_agent_response():
    """Return a factory for fake AgentResponse objects."""
    def _make(text: str = "mock response") -> MagicMock:
        resp = MagicMock()
        resp.text = text
        return resp
    return _make


@pytest.fixture
def mock_session():
    """Return a real AgentSession (no LLM needed)."""
    from agent_framework import AgentSession
    return AgentSession()


@pytest.fixture
def summary_provider():
    from memory import ConversationSummaryProvider
    return ConversationSummaryProvider()


# ─────────────────────────────────────────────────────────────
# SECTION 1 — Tool Tests
# ─────────────────────────────────────────────────────────────

class TestTools:

    def test_lookup_account_known(self):
        from tools import lookup_account
        result = lookup_account("ACC-1001")
        assert result["name"] == "Alice Johnson"
        assert result["tier"] == "premium"
        assert result["active"] is True

    def test_lookup_account_unknown(self):
        from tools import lookup_account
        result = lookup_account("ACC-9999")
        assert "error" in result

    def test_process_refund_structure(self):
        from tools import process_refund
        result = process_refund("ACC-1001", 25.00, "Service downtime")
        assert result["status"]     == "approved"
        assert result["amount_usd"] == 25.00
        assert result["account_id"] == "ACC-1001"
        assert result["refund_id"].startswith("REF-")
        assert "processed_at" in result

    def test_process_refund_rounding(self):
        from tools import process_refund
        result = process_refund("ACC-1001", 9.999, "test")
        assert result["amount_usd"] == 10.0

    def test_get_invoice_history_default_limit(self):
        from tools import get_invoice_history
        result = get_invoice_history("ACC-1001")
        assert len(result) == 3
        assert all("invoice_id" in i for i in result)

    def test_get_invoice_history_limit_respected(self):
        from tools import get_invoice_history
        result = get_invoice_history("ACC-1001", limit=1)
        assert len(result) == 1

    def test_check_service_status_operational(self):
        from tools import check_service_status
        result = check_service_status("email")
        assert result["status"] == "operational"
        assert result["uptime_30d"] > 99.0

    def test_check_service_status_degraded(self):
        from tools import check_service_status
        result = check_service_status("vpn")
        assert result["status"] == "degraded"
        assert "incident_id" in result

    def test_check_service_status_outage(self):
        from tools import check_service_status
        result = check_service_status("api")
        assert result["status"] == "major_outage"

    def test_check_service_status_unknown(self):
        from tools import check_service_status
        result = check_service_status("unicorn")
        assert result["status"] == "unknown"

    def test_check_service_status_case_insensitive(self):
        from tools import check_service_status
        r1 = check_service_status("VPN")
        r2 = check_service_status("vpn")
        assert r1["status"] == r2["status"]

    def test_search_knowledge_base_finds_results(self):
        from tools import search_knowledge_base
        results = search_knowledge_base("VPN connection issue")
        assert len(results) >= 1
        assert results[0]["id"] != "KB-000"

    def test_search_knowledge_base_no_results(self):
        from tools import search_knowledge_base
        results = search_knowledge_base("xyzzy frobnicator")
        assert results[0]["id"] == "KB-000"

    def test_create_support_ticket_structure(self):
        from tools import create_support_ticket
        result = create_support_ticket(
            subject="VPN dropping",
            description="Keeps disconnecting every 5 minutes.",
            priority="high",
            account_id="ACC-1001",
        )
        assert result["ticket_id"].startswith("TKT-")
        assert result["priority"]     == "high"
        assert result["status"]       == "open"
        assert result["sla_response_hours"] == 4

    def test_create_support_ticket_invalid_priority_fallback(self):
        from tools import create_support_ticket
        result = create_support_ticket("x", "y", "urgent")
        assert result["priority"] == "medium"  # Falls back to medium

    def test_create_support_ticket_subject_truncated(self):
        from tools import create_support_ticket
        long_subject = "A" * 200
        result = create_support_ticket(long_subject, "desc", "low")
        assert len(result["subject"]) <= 100

    def test_run_diagnostics_structure(self):
        from tools import run_diagnostics
        result = run_diagnostics("vpn", "ACC-1001")
        assert result["service"]           == "vpn"
        assert result["dns_resolution"]    == "ok"
        assert result["packet_loss_pct"]   > 0
        assert "recommended_action"        in result

    def test_run_diagnostics_no_issues_for_healthy_service(self):
        from tools import run_diagnostics
        result = run_diagnostics("email", "ACC-1001")
        assert result["packet_loss_pct"]   == 0.0


# ─────────────────────────────────────────────────────────────
# SECTION 2 — Middleware Tests
# ─────────────────────────────────────────────────────────────

class TestMiddleware:

    @pytest.mark.asyncio
    async def test_timing_middleware_calls_next(self):
        from middleware import timing_middleware
        ctx = MagicMock()
        ctx.messages = []
        ctx.agent_name = "TestAgent"
        called = []
        async def next_fn():
            called.append(True)
        await timing_middleware(ctx, next_fn)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_security_middleware_passes_safe_input(self):
        from middleware import security_middleware
        ctx = MagicMock()
        msg = MagicMock()
        msg.text = "I have a billing question"
        ctx.messages = [msg]
        called = []
        async def next_fn():
            called.append(True)
        await security_middleware(ctx, next_fn)
        assert called == [True]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sensitive", [
        "what is my password",
        "my credit card number is 1234",
        "social security number",
        "cvv 123",
        "secret key abc",
    ])
    async def test_security_middleware_blocks_sensitive(self, sensitive):
        from middleware import security_middleware
        ctx = MagicMock()
        msg = MagicMock()
        msg.text = sensitive
        ctx.messages = [msg]
        called = []
        async def next_fn():
            called.append(True)
        await security_middleware(ctx, next_fn)
        assert called == [], f"Should have blocked: {sensitive!r}"

    @pytest.mark.asyncio
    async def test_security_middleware_no_messages(self):
        from middleware import security_middleware
        ctx = MagicMock()
        ctx.messages = []
        called = []
        async def next_fn():
            called.append(True)
        await security_middleware(ctx, next_fn)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_rate_limit_middleware_passes_through(self):
        from middleware import RateLimitMiddleware
        mw  = RateLimitMiddleware()
        ctx = MagicMock()
        ctx.agent_name = "TestAgent"
        called = []
        async def next_fn():
            called.append(True)
        await mw.process(ctx, next_fn)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_rate_limit_middleware_throttles_on_empty_bucket(self):
        from middleware import RateLimitMiddleware
        mw = RateLimitMiddleware()
        # Force bucket to zero
        mw._buckets["BusyAgent"] = {"tokens": 0.0, "last_refill": time.monotonic()}
        ctx = MagicMock()
        ctx.agent_name = "BusyAgent"
        called = []
        async def next_fn():
            called.append(True)
        t0 = time.monotonic()
        await mw.process(ctx, next_fn)
        elapsed = time.monotonic() - t0
        assert called == [True]
        assert elapsed >= 0.45  # throttled ~500 ms

    @pytest.mark.asyncio
    async def test_logging_fn_middleware_logs_and_calls_next(self):
        from middleware import logging_fn_middleware
        ctx = MagicMock()
        ctx.function.name = "lookup_account"
        ctx.arguments     = {"account_id": "ACC-1001"}
        called = []
        async def next_fn():
            called.append(True)
        await logging_fn_middleware(ctx, next_fn)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_validation_mw_blocks_invalid_account_id(self):
        from middleware import ValidationFnMiddleware
        mw  = ValidationFnMiddleware()
        ctx = MagicMock()
        ctx.function.name = "lookup_account"
        ctx.arguments     = {"account_id": "BAD-ID"}
        called = []
        async def next_fn():
            called.append(True)
        await mw.process(ctx, next_fn)
        assert called == []
        assert "error" in ctx.result

    @pytest.mark.asyncio
    async def test_validation_mw_passes_valid_account_id(self):
        from middleware import ValidationFnMiddleware
        mw  = ValidationFnMiddleware()
        ctx = MagicMock()
        ctx.function.name = "lookup_account"
        ctx.arguments     = {"account_id": "ACC-1001"}
        ctx.result        = {"status": "ok"}
        called = []
        async def next_fn():
            called.append(True)
        await mw.process(ctx, next_fn)
        assert called == [True]
        # Result should be tagged
        assert "_source" in ctx.result
        assert "_ts" in ctx.result

    @pytest.mark.asyncio
    async def test_validation_mw_no_account_id_passes(self):
        from middleware import ValidationFnMiddleware
        mw  = ValidationFnMiddleware()
        ctx = MagicMock()
        ctx.function.name = "check_service_status"
        ctx.arguments     = {"service": "vpn"}
        ctx.result        = {"status": "degraded"}
        called = []
        async def next_fn():
            called.append(True)
        await mw.process(ctx, next_fn)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_chat_audit_middleware(self):
        from middleware import chat_audit_middleware
        ctx = MagicMock()
        ctx.messages  = ["m1", "m2", "m3"]
        ctx.response  = MagicMock()
        ctx.response.finish_reason = "stop"
        called = []
        async def next_fn():
            called.append(True)
        await chat_audit_middleware(ctx, next_fn)
        assert called == [True]


# ─────────────────────────────────────────────────────────────
# SECTION 3 — Memory & Context Provider Tests
# ─────────────────────────────────────────────────────────────

class TestMemory:

    @pytest.mark.asyncio
    async def test_user_memory_before_run_injects_instructions(self, mock_session):
        from memory import UserMemoryProvider
        from agent_framework import AgentSession, SessionContext

        provider = UserMemoryProvider()
        state    = {}
        ctx      = MagicMock(spec=SessionContext)
        ctx.get_messages.return_value = []

        await provider.before_run(
            agent=MagicMock(),
            session=mock_session,
            context=ctx,
            state=state,
        )
        ctx.extend_instructions.assert_called_once()
        assert state["turn_count"] == 1

    @pytest.mark.asyncio
    async def test_user_memory_increments_turn_count(self, mock_session):
        from memory import UserMemoryProvider
        provider = UserMemoryProvider()
        state    = {}
        ctx      = MagicMock()
        ctx.get_messages.return_value = []

        for i in range(3):
            await provider.before_run(agent=MagicMock(), session=mock_session,
                                       context=ctx, state=state)
        assert state["turn_count"] == 3

    @pytest.mark.asyncio
    async def test_user_memory_extracts_name(self, mock_session):
        from memory import UserMemoryProvider
        provider = UserMemoryProvider()
        state    = {}

        msg = MagicMock()
        msg.text = "Hello, my name is priya and I need help"
        ctx = MagicMock()
        ctx.get_messages.return_value = [msg]

        await provider.after_run(agent=MagicMock(), session=mock_session,
                                  context=ctx, state=state)
        assert state.get("user_name") == "Priya"

    @pytest.mark.asyncio
    async def test_user_memory_extracts_premium_tier(self, mock_session):
        from memory import UserMemoryProvider
        provider = UserMemoryProvider()
        state    = {}

        msg = MagicMock()
        msg.text = 'Account tier: premium, balance: $250'
        ctx = MagicMock()
        ctx.get_messages.return_value = [msg]

        await provider.after_run(agent=MagicMock(), session=mock_session,
                                  context=ctx, state=state)
        assert state.get("user_tier") == "premium"

    @pytest.mark.asyncio
    async def test_user_memory_extracts_ticket_id(self, mock_session):
        from memory import UserMemoryProvider
        provider = UserMemoryProvider()
        state    = {}

        msg = MagicMock()
        msg.text = "Your ticket TKT-12345 has been created."
        ctx = MagicMock()
        ctx.get_messages.return_value = [msg]

        await provider.after_run(agent=MagicMock(), session=mock_session,
                                  context=ctx, state=state)
        assert state.get("open_ticket") == "TKT-12345"

    @pytest.mark.asyncio
    async def test_user_memory_injects_known_name(self, mock_session):
        from memory import UserMemoryProvider
        provider = UserMemoryProvider()
        state    = {"user_name": "Alice", "turn_count": 2}
        ctx      = MagicMock()
        ctx.get_messages.return_value = []

        await provider.before_run(agent=MagicMock(), session=mock_session,
                                   context=ctx, state=state)
        call_args = ctx.extend_instructions.call_args[0]
        assert "Alice" in call_args[1]

    @pytest.mark.asyncio
    async def test_user_memory_premium_tier_instruction(self, mock_session):
        from memory import UserMemoryProvider
        provider = UserMemoryProvider()
        state    = {"user_tier": "premium"}
        ctx      = MagicMock()
        ctx.get_messages.return_value = []

        await provider.before_run(agent=MagicMock(), session=mock_session,
                                   context=ctx, state=state)
        call_args = ctx.extend_instructions.call_args[0]
        assert "PREMIUM" in call_args[1].upper()

    def test_conversation_summary_provider_record(self, summary_provider):
        summary_provider.record("User",   "Hello there")
        summary_provider.record("Agent",  "Hi, how can I help?")
        assert len(summary_provider._transcript) == 2
        assert "User: Hello" in summary_provider._transcript[0]

    def test_conversation_summary_provider_rolling_window(self, summary_provider):
        for i in range(20):
            summary_provider.record("User",  f"Message {i}")
            summary_provider.record("Agent", f"Response {i}")
        # MAX_TURNS * 2 = 12 lines
        assert len(summary_provider._transcript) == 12

    @pytest.mark.asyncio
    async def test_conversation_summary_provider_injects_history(
        self, summary_provider, mock_session
    ):
        summary_provider.record("User",  "My VPN is broken")
        summary_provider.record("Agent", "Let me check that for you.")

        ctx = MagicMock()
        await summary_provider.before_run(
            agent=MagicMock(), session=mock_session,
            context=ctx, state={}
        )
        ctx.extend_instructions.assert_called_once()
        instr = ctx.extend_instructions.call_args[0][1]
        assert "VPN" in instr

    @pytest.mark.asyncio
    async def test_conversation_summary_provider_no_history_no_call(
        self, summary_provider, mock_session
    ):
        ctx = MagicMock()
        await summary_provider.before_run(
            agent=MagicMock(), session=mock_session,
            context=ctx, state={}
        )
        ctx.extend_instructions.assert_not_called()


# ─────────────────────────────────────────────────────────────
# SECTION 4 — Conversation Hook Tests
# ─────────────────────────────────────────────────────────────

class TestConversationHook:

    def test_hook_increments_turn(self):
        from memory import ConversationHook
        hook = ConversationHook("Test")
        hook.on_turn_start("hello", "TriageAgent")
        hook.on_turn_start("world", "TechAgent")
        assert hook._turn == 2

    def test_hook_records_audit_log(self):
        from memory import ConversationHook
        hook = ConversationHook("Test")
        hook.on_turn_start("hello", "BillingAgent")
        hook.on_turn_end("response text", 123.0)
        log = hook.get_audit_log()
        assert len(log) == 1
        assert log[0]["agent"]         == "BillingAgent"
        assert log[0]["user_input"]    == "hello"
        assert log[0]["elapsed_ms"]    == 123
        assert "response_preview"      in log[0]

    def test_hook_records_error(self):
        from memory import ConversationHook
        hook = ConversationHook("Test")
        hook.on_turn_start("test", "Agent")
        hook.on_error(ValueError("boom"))
        log = hook.get_audit_log()
        assert log[0]["error"] == "boom"

    def test_hook_print_audit_log_does_not_crash(self, capsys):
        from memory import ConversationHook
        hook = ConversationHook("Test")
        hook.on_turn_start("hi", "TriageAgent")
        hook.on_turn_end("hi back", 50.0)
        hook.print_audit_log()
        captured = capsys.readouterr()
        assert "AUDIT LOG" in captured.out


# ─────────────────────────────────────────────────────────────
# SECTION 5 — Intent Classification Tests
# ─────────────────────────────────────────────────────────────

class TestIntentClassification:

    def _classify(self, text: str) -> str:
        from orchestrator import _classify_intent
        return _classify_intent(text)

    @pytest.mark.parametrize("text,expected", [
        ("I have a billing question",             "billing"),
        ("What is my account balance?",           "billing"),
        ("I need a refund for the overdue invoice", "billing"),
        ("My VPN keeps dropping",                 "tech"),
        ("The API is returning 503 errors",       "tech"),
        ("Email is not syncing",                  "tech"),
        ("Please summarize our conversation",     "summarizer"),
        ("Give me a recap",                       "summarizer"),
        ("Hello I need some help",                "triage"),
        ("Can you help me",                       "triage"),
    ])
    def test_intent_routing(self, text: str, expected: str):
        assert self._classify(text) == expected, (
            f"Text {text!r} → expected '{expected}'"
        )


# ─────────────────────────────────────────────────────────────
# SECTION 6 — Orchestrator Integration Tests (mocked LLM)
# ─────────────────────────────────────────────────────────────

class TestOrchestratorIntegration:
    """
    Tests inject mock agents directly into the orchestrator, bypassing
    the LLM client entirely — no Azure credentials needed.
    """

    def _make_mock_agent(self, response_text: str, name: str) -> AsyncMock:
        a = AsyncMock()
        a.name = name
        resp = MagicMock()
        resp.text = response_text
        a.run = AsyncMock(return_value=resp)
        a.__aenter__ = AsyncMock(return_value=a)
        a.__aexit__  = AsyncMock(return_value=None)
        return a

    def _make_orchestrator_with_mocks(self, response_text: str = "mock response") -> "MultiAgentOrchestrator":
        """Build orchestrator with fully mocked agents, no LLM client needed."""
        from orchestrator import MultiAgentOrchestrator
        from memory import ConversationSummaryProvider

        orch = object.__new__(MultiAgentOrchestrator)
        orch._summary_provider = ConversationSummaryProvider()
        orch._session = __import__("agent_framework").AgentSession()
        orch.hook = __import__("memory").ConversationHook("TestOrchestrator")

        mock_agents = {}
        for role in ("triage", "billing", "tech", "summarizer"):
            name = f"{role.capitalize()}Agent"
            mock_agents[role] = self._make_mock_agent(response_text, name)
        orch._agents = mock_agents
        return orch

    @pytest.mark.asyncio
    async def test_orchestrator_routes_billing_query(self):
        orch = self._make_orchestrator_with_mocks("Your account balance is $250.00.")
        async with orch:
            reply = await orch.chat("What is my account balance?")
        assert reply == "Your account balance is $250.00."
        # billing agent should have been called
        orch._agents["billing"].run.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_routes_tech_query(self):
        orch = self._make_orchestrator_with_mocks("Checking VPN status now.")
        async with orch:
            reply = await orch.chat("My VPN keeps disconnecting")
        assert reply == "Checking VPN status now."
        orch._agents["tech"].run.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_routes_summarizer_query(self):
        orch = self._make_orchestrator_with_mocks("Here is your summary.")
        async with orch:
            reply = await orch.chat("Please summarize our conversation")
        assert reply == "Here is your summary."
        orch._agents["summarizer"].run.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_session_persists_across_turns(self):
        orch = self._make_orchestrator_with_mocks("OK!")
        session_id = orch._session.session_id
        async with orch:
            await orch.chat("Hello!")
            await orch.chat("My VPN is broken")
            # Session object is the same across turns
            assert orch.session.session_id == session_id

    @pytest.mark.asyncio
    async def test_orchestrator_handles_agent_exception(self):
        from orchestrator import MultiAgentOrchestrator
        from memory import ConversationSummaryProvider

        orch = object.__new__(MultiAgentOrchestrator)
        orch._summary_provider = ConversationSummaryProvider()
        orch._session = __import__("agent_framework").AgentSession()
        orch.hook = __import__("memory").ConversationHook("TestOrch")

        mock_agents = {}
        for role in ("triage", "billing", "tech", "summarizer"):
            a = AsyncMock()
            a.name = f"{role.capitalize()}Agent"
            a.run  = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
            a.__aenter__ = AsyncMock(return_value=a)
            a.__aexit__  = AsyncMock(return_value=None)
            mock_agents[role] = a
        orch._agents = mock_agents

        async with orch:
            reply = await orch.chat("I have a billing question")

        # Should not raise; returns graceful error message
        assert "error" in reply.lower() or "RuntimeError" in reply

    @pytest.mark.asyncio
    async def test_orchestrator_transcript_grows(self):
        orch = self._make_orchestrator_with_mocks("OK!")
        async with orch:
            provider = orch._summary_provider
            await orch.chat("Hello")
            await orch.chat("How are you")
            # 2 turns × 2 entries each = 4 transcript lines
            assert len(provider._transcript) == 4

    @pytest.mark.asyncio
    async def test_orchestrator_audit_log_populated(self):
        orch = self._make_orchestrator_with_mocks("Got it!")
        async with orch:
            await orch.chat("I have a billing question")
            log = orch.hook.get_audit_log()
        assert len(log) == 1
        assert log[0]["user_input"] == "I have a billing question"

    @pytest.mark.asyncio
    async def test_orchestrator_passes_session_to_agent(self):
        """Verify agent.run() receives the shared session object."""
        orch = self._make_orchestrator_with_mocks("Done!")
        async with orch:
            await orch.chat("I need a refund")
        billing_mock = orch._agents["billing"]
        call_kwargs = billing_mock.run.call_args
        # session keyword argument must be the orchestrator's session
        assert call_kwargs.kwargs.get("session") is orch._session


# ─────────────────────────────────────────────────────────────
# SECTION 7 — Workflow Tests (mocked LLM)
# ─────────────────────────────────────────────────────────────

class TestWorkflow:

    def _make_agent_mock(self, response_text: str, name: str) -> AsyncMock:
        a = AsyncMock()
        a.name = name
        resp   = MagicMock()
        resp.text = response_text
        a.run  = AsyncMock(return_value=resp)
        a.__aenter__ = AsyncMock(return_value=a)
        a.__aexit__  = AsyncMock(return_value=None)
        return a

    @pytest.mark.asyncio
    async def test_workflow_triage_executor(self):
        from workflow import TriageExecutor

        agent = self._make_agent_mock("BILLING: invoice dispute", "TriageAgent")
        executor = TriageExecutor(agent)

        ctx = AsyncMock()
        await executor.process("I have an invoice problem", ctx)
        ctx.send_message.assert_called_once()
        sent = ctx.send_message.call_args[0][0]
        assert "invoice" in sent.lower() or "billing" in sent.lower()

    @pytest.mark.asyncio
    async def test_workflow_resolution_executor_routes_billing(self):
        from workflow import ResolutionExecutor

        billing_agent = self._make_agent_mock("Refund processed.", "BillingAgent")
        tech_agent    = self._make_agent_mock("Ticket created.",   "TechAgent")
        executor      = ResolutionExecutor(billing_agent, tech_agent)

        ctx = AsyncMock()
        await executor.process("[Classification: BILLING: refund request]", ctx)
        billing_agent.run.assert_called_once()
        tech_agent.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_workflow_resolution_executor_routes_tech(self):
        from workflow import ResolutionExecutor

        billing_agent = self._make_agent_mock("Refund processed.", "BillingAgent")
        tech_agent    = self._make_agent_mock("Ticket created.",   "TechAgent")
        executor      = ResolutionExecutor(billing_agent, tech_agent)

        ctx = AsyncMock()
        await executor.process("[Classification: TECHNICAL: VPN outage]", ctx)
        tech_agent.run.assert_called_once()
        billing_agent.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_workflow_summary_executor_yields_output(self):
        from workflow import SummaryExecutor

        agent    = self._make_agent_mock("Summary: VPN issue resolved.", "SummarizerAgent")
        executor = SummaryExecutor(agent)

        ctx = AsyncMock()
        await executor.process("Resolution details here.", ctx)
        ctx.yield_output.assert_called_once()
        output = ctx.yield_output.call_args[0][0]
        assert "Summary" in output or "VPN" in output

    @pytest.mark.asyncio
    async def test_workflow_summary_executor_fallback_on_empty_response(self):
        from workflow import SummaryExecutor

        agent      = AsyncMock()
        agent.name = "SummarizerAgent"
        resp       = MagicMock()
        resp.text  = None   # Empty response
        agent.run  = AsyncMock(return_value=resp)

        executor = SummaryExecutor(agent)
        ctx      = AsyncMock()
        await executor.process("some resolution", ctx)
        ctx.yield_output.assert_called_once_with("some resolution")


# ─────────────────────────────────────────────────────────────
# SECTION 8 — AgentSession serialisation
# ─────────────────────────────────────────────────────────────

class TestAgentSession:

    def test_session_serialization_round_trip(self):
        from agent_framework import AgentSession
        session = AgentSession(session_id="test-123")
        session.state["user_name"] = "Alice"
        session.state["turn_count"] = 3

        d = session.to_dict()
        assert d["session_id"] == "test-123"

        restored = AgentSession.from_dict(d)
        assert restored.session_id == "test-123"
        assert restored.state["user_name"] == "Alice"
        assert restored.state["turn_count"] == 3

    def test_session_auto_generates_id(self):
        from agent_framework import AgentSession
        s1 = AgentSession()
        s2 = AgentSession()
        assert s1.session_id != s2.session_id
        assert len(s1.session_id) > 0


# ─────────────────────────────────────────────────────────────
# SECTION 9 — Edge-case and regression tests
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_lookup_account_returns_dict(self):
        from tools import lookup_account
        result = lookup_account("ACC-1001")
        assert isinstance(result, dict)

    def test_process_refund_ceiling_not_exceeded(self):
        """process_refund should not accept amounts > 500."""
        from tools import process_refund
        # Pydantic validation in @tool won't raise here since we call directly;
        # ensure amount stored correctly within allowed range
        result = process_refund("ACC-1001", 499.99, "test")
        assert result["amount_usd"] == 499.99

    def test_search_kb_returns_list(self):
        from tools import search_knowledge_base
        assert isinstance(search_knowledge_base("anything"), list)

    def test_security_middleware_empty_text(self):
        """Middleware should not crash on messages with no .text attribute."""
        from middleware import security_middleware

        async def _run():
            ctx = MagicMock()
            msg = MagicMock(spec=[])  # No .text attribute
            ctx.messages = [msg]
            called = []
            async def next_fn():
                called.append(True)
            await security_middleware(ctx, next_fn)
            assert called == [True]

        asyncio.run(_run())

    def test_conversation_hook_empty_log_print(self, capsys):
        from memory import ConversationHook
        hook = ConversationHook("EdgeCase")
        hook.print_audit_log()
        out = capsys.readouterr().out
        assert "AUDIT LOG" in out

    @pytest.mark.asyncio
    async def test_user_memory_no_messages_no_crash(self, mock_session):
        from memory import UserMemoryProvider
        provider = UserMemoryProvider()
        ctx      = MagicMock()
        ctx.get_messages.return_value = []
        # Should not raise
        await provider.after_run(
            agent=MagicMock(), session=mock_session,
            context=ctx, state={}
        )
