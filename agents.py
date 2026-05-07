"""

Factory functions that construct the four specialised agents:
  • TriageAgent      — classifies and routes requests
  • BillingAgent     — handles invoices, refunds, account queries
  • TechAgent        — handles outages, diagnostics, tickets
  • SummarizerAgent  — produces end-of-session summaries

Each agent shares:
  - Azure OpenAI backend (via OpenAIChatClient)
  - UserMemoryProvider + ConversationSummaryProvider
  - Full middleware stack (timing, security, rate-limit, tool-log, chat-audit)
"""

from __future__ import annotations

import os

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from memory import ConversationSummaryProvider, UserMemoryProvider
from middleware import (
    RateLimitMiddleware,
    ValidationFnMiddleware,
    chat_audit_middleware,
    logging_fn_middleware,
    security_middleware,
    timing_middleware,
)
from tools import (
    check_service_status,
    create_support_ticket,
    get_invoice_history,
    lookup_account,
    process_refund,
    run_diagnostics,
    search_knowledge_base,
)


def _make_client() -> OpenAIChatClient:
    """
    Build an Azure OpenAI chat client from environment variables.

    Required env vars:
        AZURE_OPENAI_API_KEY     – your Azure OpenAI key
        AZURE_OPENAI_ENDPOINT    – e.g. https://<resource>.openai.azure.com
        AZURE_OPENAI_DEPLOYMENT  – deployment name, e.g. gpt-4o-mini
        AZURE_OPENAI_API_VERSION – e.g. 2024-12-01-preview
    """
    return OpenAIChatClient(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        base_url=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    )


def build_agents(
    summary_provider: ConversationSummaryProvider,
) -> dict[str, Agent]:
    """
    Create all four agents and return them keyed by role name.
    Agents share the same LLM client and provider instances.
    """
    client        = _make_client()
    user_memory   = UserMemoryProvider()
    rate_limiter  = RateLimitMiddleware()
    fn_validator  = ValidationFnMiddleware()

    # Shared middleware applied to every run of every agent
    shared_mw = [
        timing_middleware,      # wall-clock timing (function-based)
        security_middleware,    # PII / secret blocker (function-based)
        rate_limiter,           # token-bucket throttle (class-based)
        logging_fn_middleware,  # tool call logger (function-based)
        fn_validator,           # argument validator + result tagger (class-based)
        chat_audit_middleware,  # LLM call auditor (function-based)
    ]

    shared_providers = [user_memory, summary_provider]

    # ── Triage Agent ─────────────────────────────────────────
    triage_agent = Agent(
        client=client,
        name="TriageAgent",
        description="Routes incoming support requests to the correct specialist.",
        instructions=(
            "You are the first point of contact in a multi-agent support system. "
            "Your ONLY job is to classify the user's request and greet them warmly. "
            "Identify whether the issue is:\n"
            "  • BILLING — invoices, charges, refunds, account balances\n"
            "  • TECHNICAL — service outages, VPN, email, API, connectivity\n"
            "  • SUMMARY — user wants a recap of the conversation\n"
            "Respond with a brief, friendly acknowledgement and state which team "
            "will handle their request. Do NOT attempt to resolve the issue yourself."
        ),
        context_providers=shared_providers,
        middleware=shared_mw,
    )

    # ── Billing Agent ────────────────────────────────────────
    billing_agent = Agent(
        client=client,
        name="BillingAgent",
        description="Handles billing, invoices, refunds, and account enquiries.",
        instructions=(
            "You are the billing specialist. Help customers with:\n"
            "  • Account lookups — use lookup_account\n"
            "  • Invoice history — use get_invoice_history\n"
            "  • Refund requests — use process_refund (only when explicitly asked)\n"
            "Steps:\n"
            "  1. If you have an account ID, call lookup_account first.\n"
            "  2. Review the account status and balance.\n"
            "  3. For refunds, confirm the amount and reason before processing.\n"
            "  4. Always confirm the resolution before closing.\n"
            "Be concise and professional. If the issue is technical, say so politely."
        ),
        tools=[lookup_account, get_invoice_history, process_refund],
        context_providers=shared_providers,
        middleware=shared_mw,
    )

    # ── Tech Support Agent ───────────────────────────────────
    tech_agent = Agent(
        client=client,
        name="TechAgent",
        description="Handles technical issues, outages, diagnostics, and tickets.",
        instructions=(
            "You are the technical support specialist. Help customers with:\n"
            "  • Service status checks — use check_service_status\n"
            "  • Knowledge base search — use search_knowledge_base\n"
            "  • Diagnostics — use run_diagnostics for connectivity issues\n"
            "  • Ticket creation — use create_support_ticket for unresolved issues\n"
            "Troubleshooting workflow:\n"
            "  1. Check service status first.\n"
            "  2. Search the knowledge base for guidance.\n"
            "  3. If still unresolved, run diagnostics.\n"
            "  4. If diagnostics show a persistent issue, create a ticket.\n"
            "  5. Provide the customer with clear next steps.\n"
            "If the issue is billing, say so politely and defer to billing."
        ),
        tools=[check_service_status, search_knowledge_base, run_diagnostics, create_support_ticket],
        context_providers=shared_providers,
        middleware=shared_mw,
    )

    # ── Summarizer Agent ─────────────────────────────────────
    summarizer_agent = Agent(
        client=client,
        name="SummarizerAgent",
        description="Produces a clear end-of-conversation summary for the customer.",
        instructions=(
            "You are the conversation summarizer. Produce a tidy summary with:\n"
            "  1. **Issue reported** — what the customer needed\n"
            "  2. **Actions taken** — what was looked up, run, or created\n"
            "  3. **Resolution** — was it resolved? If not, what is open?\n"
            "  4. **Next steps** — any follow-up actions for the customer\n"
            "Keep the total response under 200 words. "
            "Close with a polite thank-you and invite them to contact us again."
        ),
        context_providers=shared_providers,
        middleware=shared_mw,
    )

    return {
        "triage":    triage_agent,
        "billing":   billing_agent,
        "tech":      tech_agent,
        "summarizer": summarizer_agent,
    }
