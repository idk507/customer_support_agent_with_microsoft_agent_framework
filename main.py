"""
main.py
───────
Entry point for the Microsoft Agent Framework multi-agent demo.

Runs two demonstration modes:

Mode 1 — Conversational Orchestrator
    A multi-turn session showing memory persistence, agent routing,
    middleware pipelines, and conversation hooks.

Mode 2 — Workflow Pipeline
    A single-shot request through the deterministic 3-step workflow:
    Triage → Resolution → Summary.

Usage:
    python main.py            # Both modes
    python main.py --orch     # Orchestrator only
    python main.py --workflow # Workflow only
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _check_env() -> bool:
    """Verify Azure OpenAI credentials are set."""
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("   Set them in a .env file (see .env.example) or as shell variables.")
        print("   Running in MOCK mode (agents will fail gracefully).\n")
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Demo 1 — Conversational Orchestrator
# ─────────────────────────────────────────────────────────────

async def run_orchestrator_demo() -> None:
    from orchestrator import MultiAgentOrchestrator

    print("\n" + "━"*68)
    print("  DEMO 1: Conversational Multi-Agent Orchestrator")
    print("━"*68)

    # Multi-turn conversation that exercises all agents and memory
    turns = [
        "Hi! My name is Priya. I'm having trouble with my VPN — it keeps disconnecting.",
        "I looked it up — my account ID is ACC-1002.",
        "Can you check if there's a known outage for the VPN service?",
        "Please also check my account balance.",
        "I'd like a refund of $15.50 for the service degradation.",
        "Please give me a summary of our conversation.",
    ]

    async with MultiAgentOrchestrator() as orch:
        for user_input in turns:
            response = await orch.chat(user_input)
            print(f"\n  📤 Final Response:\n  {response[:400]}\n")

        # Print the full audit log at the end
        orch.hook.print_audit_log()

        # Show what the memory provider stored
        state = orch.session.state
        print(f"\n  🧠 Session State: {state}")


# ─────────────────────────────────────────────────────────────
# Demo 2 — Sequential Workflow Pipeline
# ─────────────────────────────────────────────────────────────

async def run_workflow_demo() -> None:
    from workflow import run_workflow_demo as _run

    print("\n" + "━"*68)
    print("  DEMO 2: Sequential Workflow Pipeline")
    print("━"*68)

    request = (
        "I'm getting 503 errors from your API. "
        "My account is ACC-1001. I need this fixed urgently."
    )
    print(f"\n  📥 Input request:\n  {request}\n")

    summary = await _run(request)

    print(f"\n  📋 Pipeline Output:\n{'─'*68}")
    print(f"  {summary}")
    print("─"*68)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

async def main() -> None:
    has_creds = _check_env()

    args = sys.argv[1:]
    run_orch     = "--orch"     in args or not args
    run_workflow = "--workflow" in args or not args

    if run_orch:
        await run_orchestrator_demo()

    if run_workflow:
        await run_workflow_demo()


if __name__ == "__main__":
    asyncio.run(main())
