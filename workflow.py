"""
Graph-based sequential workflow using WorkflowBuilder.

Pipeline:  TriageExecutor → ResolutionExecutor → SummaryExecutor → output

This demonstrates the framework's Executor / handler / WorkflowContext
pattern for deterministic, step-by-step multi-agent processing — distinct
from the conversational orchestrator in orchestrator.py.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import Never

from agent_framework import (
    Agent,
    Executor,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)

from agents import build_agents
from memory import ConversationSummaryProvider


# ─────────────────────────────────────────────────────────────
# Workflow Executors
# ─────────────────────────────────────────────────────────────

class TriageExecutor(Executor):
    """
    Step 1 — Classify and acknowledge the incoming support request.
    Sends the classification string to ResolutionExecutor.
    """

    def __init__(self, agent: Agent) -> None:
        super().__init__(id="triage_step")
        self._agent = agent

    @handler
    async def process(self, message: str, ctx: WorkflowContext[str]) -> None:
        print("  [Workflow] ➡ Step 1: TriageExecutor")
        result = await self._agent.run(
            f"Classify this support request in one sentence, stating whether it is "
            f"BILLING or TECHNICAL and the main topic:\n\n\"{message}\""
        )
        classification = result.text or message
        print(f"  [Workflow]   Triage: {classification[:80]}")
        # Send classification + original message downstream
        await ctx.send_message(f"[Classification: {classification}]\n\nOriginal request: {message}")


class ResolutionExecutor(Executor):
    """
    Step 2 — Route to the correct specialist and attempt resolution.
    Sends the resolution text to SummaryExecutor.
    """

    def __init__(self, billing_agent: Agent, tech_agent: Agent) -> None:
        super().__init__(id="resolution_step")
        self._billing = billing_agent
        self._tech    = tech_agent

    @handler
    async def process(self, triage_result: str, ctx: WorkflowContext[str]) -> None:
        print("  [Workflow] ➡ Step 2: ResolutionExecutor")
        tr_lower = triage_result.lower()

        if "billing" in tr_lower or "invoice" in tr_lower or "refund" in tr_lower:
            agent = self._billing
        else:
            agent = self._tech

        print(f"  [Workflow]   Routing to {agent.name}")
        result = await agent.run(
            f"You received this classified support request. Please resolve it:\n\n{triage_result}"
        )
        resolution = result.text or "Resolution unavailable."
        print(f"  [Workflow]   Resolution: {resolution[:80]}…")
        await ctx.send_message(resolution)


class SummaryExecutor(Executor):
    """
    Step 3 — Produce a final customer-facing summary.
    Yields the summary as the workflow output.
    """

    def __init__(self, agent: Agent) -> None:
        super().__init__(id="summary_step")
        self._agent = agent

    @handler
    async def process(self, resolution: str, ctx: WorkflowContext[Never, str]) -> None:
        print("  [Workflow] ➡ Step 3: SummaryExecutor")
        result = await self._agent.run(
            f"Summarize this support interaction for the customer:\n\n{resolution}"
        )
        summary = result.text or resolution
        print(f"  [Workflow]   Summary: {summary[:80]}…")
        await ctx.yield_output(summary)


# ─────────────────────────────────────────────────────────────
# Workflow builder
# ─────────────────────────────────────────────────────────────

def build_support_workflow() -> tuple[Workflow, dict[str, Agent]]:
    """
    Construct the 3-step support pipeline workflow and return it
    together with the underlying agents (for resource cleanup).
    """
    summary_provider = ConversationSummaryProvider()
    agents           = build_agents(summary_provider)

    triage_ex     = TriageExecutor(agents["triage"])
    resolution_ex = ResolutionExecutor(agents["billing"], agents["tech"])
    summary_ex    = SummaryExecutor(agents["summarizer"])

    workflow = (
        WorkflowBuilder(start_executor=triage_ex, name="SupportPipeline")
        .add_edge(triage_ex,     resolution_ex)
        .add_edge(resolution_ex, summary_ex)
        .build()
    )

    return workflow, agents


async def run_workflow_demo(user_request: str) -> str:
    """
    Run one request through the full 3-step pipeline.
    Returns the final customer-facing summary.
    """
    workflow, agents = build_support_workflow()

    print(f"\n  [Workflow] Running pipeline for: {user_request!r}")
    print(f"  {'─'*60}")

    # Enter agent context managers
    for agent in agents.values():
        await agent.__aenter__()

    try:
        result  = await workflow.run(user_request)
        outputs = result.get_outputs()
        return outputs[0] if outputs else "(no output)"
    finally:
        for agent in agents.values():
            await agent.__aexit__(None, None, None)
