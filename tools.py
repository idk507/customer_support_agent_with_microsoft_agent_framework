"""
All @tool-decorated functions available to agents.

Tools load data dynamically from CSV files in the data/ directory so that
adding, editing, or removing records requires no code changes.  A lightweight
TF-IDF scoring engine drives the knowledge-base search without any external ML
dependencies.
"""

from __future__ import annotations

import csv
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from agent_framework import tool
from pydantic import Field

# ─────────────────────────────────────────────────────────────
# Data directory (resolved relative to this file)
# ─────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data"


# ─────────────────────────────────────────────────────────────
# CSV loading helpers  (cached so files are read once per process)
# ─────────────────────────────────────────────────────────────

def _load_csv(filename: str) -> list[dict[str, str]]:
    """Read a CSV from data/ and return rows as plain string dicts."""
    path = _DATA_DIR / filename
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@lru_cache(maxsize=None)
def _accounts() -> list[dict[str, str]]:
    return _load_csv("accounts.csv")


@lru_cache(maxsize=None)
def _invoices() -> list[dict[str, str]]:
    return _load_csv("invoices.csv")


@lru_cache(maxsize=None)
def _services() -> list[dict[str, str]]:
    return _load_csv("services.csv")


@lru_cache(maxsize=None)
def _knowledge_base() -> list[dict[str, str]]:
    return _load_csv("knowledge_base.csv")


# ─────────────────────────────────────────────────────────────
# Row coercion helpers
# ─────────────────────────────────────────────────────────────

def _cast_account(row: dict[str, str]) -> dict[str, Any]:
    return {
        "name":         row["name"],
        "tier":         row["tier"],
        "balance":      float(row["balance"]),
        "active":       row["active"].strip().lower() in ("true", "1", "yes"),
        "email":        row["email"],
        "plan":         row["plan"],
        "joined":       row["joined"],
        "phone":        row.get("phone", ""),
        "region":       row.get("region", ""),
        "usage_gb":     float(row.get("usage_gb", 0) or 0),
        "support_tier": row.get("support_tier", "standard"),
    }


def _cast_invoice(row: dict[str, str]) -> dict[str, Any]:
    return {
        "invoice_id":  row["invoice_id"],
        "account_id":  row["account_id"],
        "date":        row["date"],
        "amount":      float(row["amount"]),
        "status":      row["status"],
        "due_date":    row.get("due_date", ""),
        "description": row.get("description", ""),
        "currency":    row.get("currency", "USD"),
    }


def _cast_service(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "service":      row["service_name"].lower(),
        "status":       row["status"],
        "uptime_30d":   float(row["uptime_30d"]),
        "region":       row.get("region", "global"),
        "category":     row.get("category", ""),
        "last_checked": row.get("last_checked", ""),
    }
    if row.get("incident_id"):
        result["incident_id"] = row["incident_id"]
    if row.get("message"):
        result["message"] = row["message"]
    return result


# ─────────────────────────────────────────────────────────────
# Lightweight TF-IDF search engine for the knowledge base
# ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _build_idf(corpus: tuple[dict[str, str], ...]) -> dict[str, float]:
    """Compute inverse-document-frequency weights for all terms in the KB."""
    N = len(corpus)
    df: dict[str, int] = defaultdict(int)
    for doc in corpus:
        text = " ".join([
            doc.get("title", ""),
            doc.get("summary", ""),
            doc.get("tags", ""),
            doc.get("category", ""),
        ])
        for token in set(_tokenize(text)):
            df[token] += 1
    return {term: math.log((N + 1) / (count + 1)) + 1.0 for term, count in df.items()}


@lru_cache(maxsize=None)
def _kb_idf() -> dict[str, float]:
    return _build_idf(tuple(_knowledge_base()))  # type: ignore[arg-type]


def _kb_score(query_tokens: list[str], doc: dict[str, str]) -> float:
    """TF-IDF dot-product score for a single KB document."""
    text = " ".join([
        doc.get("title", ""),
        doc.get("summary", ""),
        doc.get("tags", ""),
        doc.get("category", ""),
    ])
    doc_tokens = _tokenize(text)
    if not doc_tokens:
        return 0.0
    tf: dict[str, float] = defaultdict(float)
    for t in doc_tokens:
        tf[t] += 1.0 / len(doc_tokens)

    idf = _kb_idf()
    title_tokens = set(_tokenize(doc.get("title", "")))
    score = 0.0
    for qt in query_tokens:
        tfidf = tf.get(qt, 0.0) * idf.get(qt, 1.0)
        # Boost terms that appear in the title
        score += tfidf * (1.5 if qt in title_tokens else 1.0)
    return score


# ─────────────────────────────────────────────────────────────
# Dynamic diagnostics model
# ─────────────────────────────────────────────────────────────

def _diagnostics_for(service_name: str, account_id: str) -> dict[str, Any]:
    """Derive realistic diagnostics from live service-status data."""
    svc_rows = [r for r in _services() if r["service_name"].lower() == service_name.lower()]
    if not svc_rows:
        return {
            "service":    service_name,
            "account_id": account_id,
            "error":      f"Service '{service_name}' is not monitored.",
        }
    row     = svc_rows[0]
    status  = row["status"]
    uptime  = float(row["uptime_30d"])
    incident = row.get("incident_id", "")

    # Simulate metrics based on current service health
    packet_loss = 0.0
    latency_ms  = 45
    cert_valid  = True
    recommended = "No issues detected. All systems nominal."

    if status == "degraded":
        packet_loss = round((100 - uptime) * 0.4, 2)
        latency_ms  = 250
        recommended = (
            f"Service is degraded ({incident}). "
            "Retry with back-off; consider failover to an alternative region."
        )
    elif status in ("major_outage", "partial_outage"):
        packet_loss = round((100 - uptime) * 0.8, 2)
        latency_ms  = 1800
        cert_valid  = uptime >= 90
        recommended = (
            f"Active outage detected ({incident}). "
            "Avoid further requests; subscribe to status-page notifications for ETA."
        )

    return {
        "service":            service_name,
        "account_id":         account_id,
        "current_status":     status,
        "dns_resolution":     "ok" if status != "major_outage" else "timeout",
        "latency_ms":         latency_ms,
        "packet_loss_pct":    packet_loss,
        "certificate_valid":  cert_valid,
        "uptime_30d_pct":     uptime,
        "recommended_action": recommended,
        "ran_at":             datetime.now(timezone.utc).isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────
# Billing Tools
# ─────────────────────────────────────────────────────────────

@tool(approval_mode="never_require")
def lookup_account(
    account_id: Annotated[str, Field(description="Customer account ID, e.g. ACC-1234")],
) -> dict[str, Any]:
    """Retrieve full account details for a given customer account ID from the live accounts database."""
    normalised = account_id.strip().upper()
    for row in _accounts():
        if row["account_id"].strip().upper() == normalised:
            return {"account_id": row["account_id"], **_cast_account(row)}
    return {"error": f"Account '{account_id}' not found in the system."}


@tool(approval_mode="never_require")
def process_refund(
    account_id: Annotated[str, Field(description="Account to credit the refund to")],
    amount: Annotated[float, Field(description="Refund amount in USD (0.01–500.00)", ge=0.01, le=500.0)],
    reason: Annotated[str, Field(description="Brief reason for the refund")],
) -> dict[str, Any]:
    """Issue a refund to a customer account and return the refund receipt."""
    account = lookup_account(account_id)
    if "error" in account:
        return {"status": "rejected", "error": account["error"]}
    if not account.get("active"):
        return {
            "status":     "rejected",
            "account_id": account_id,
            "error":      "Refunds cannot be issued to inactive accounts.",
        }
    return {
        "status":           "approved",
        "refund_id":        f"REF-{int(time.time()) % 1_000_000:06d}",
        "account_id":       account_id,
        "account_name":     account["name"],
        "amount_usd":       round(amount, 2),
        "reason":           reason,
        "processed_at":     datetime.now(timezone.utc).isoformat() + "Z",
        "expected_posting": "3–5 business days",
    }


@tool(approval_mode="never_require")
def get_invoice_history(
    account_id: Annotated[str, Field(description="Account ID to retrieve invoices for")],
    limit: Annotated[int, Field(description="Number of most recent invoices to return", ge=1, le=24)] = 3,
    status_filter: Annotated[
        str,
        Field(description="Optional status filter: all | paid | overdue | pending | cancelled (default: all)"),
    ] = "all",
) -> list[dict[str, Any]]:
    """Return the last N invoices for a customer account, optionally filtered by status."""
    normalised = account_id.strip().upper()
    rows = [r for r in _invoices() if r["account_id"].strip().upper() == normalised]

    if status_filter.lower() != "all":
        rows = [r for r in rows if r["status"].lower() == status_filter.lower()]

    # Sort descending by date so newest invoices come first
    rows.sort(key=lambda r: r["date"], reverse=True)

    if not rows:
        return [{"message": f"No invoices found for account '{account_id}' with filter '{status_filter}'."}]
    return [_cast_invoice(r) for r in rows[:limit]]


# ─────────────────────────────────────────────────────────────
# Technical Support Tools
# ─────────────────────────────────────────────────────────────

@tool(approval_mode="never_require")
def check_service_status(
    service: Annotated[
        str,
        Field(description="Service name to check, e.g. email, vpn, storage, api, auth, cdn, dns, database, webhook, monitoring"),
    ],
) -> dict[str, Any]:
    """Return the live operational status of a named service from the services database."""
    key = service.strip().lower()
    for row in _services():
        if row["service_name"].lower() == key:
            return _cast_service(row)
    available = [r["service_name"].lower() for r in _services()]
    return {
        "status":    "unknown",
        "service":   service,
        "message":   "Service not monitored.",
        "available": available,
    }


@tool(approval_mode="never_require")
def list_all_services() -> list[dict[str, Any]]:
    """Return the status of every monitored service — useful for a system-wide health overview."""
    return [_cast_service(row) for row in _services()]


@tool(approval_mode="never_require")
def search_knowledge_base(
    query: Annotated[str, Field(description="Natural-language search query")],
    top_k: Annotated[int, Field(description="Maximum number of articles to return", ge=1, le=10)] = 3,
) -> list[dict[str, str]]:
    """
    Search the internal knowledge base using TF-IDF ranking.

    Returns the most relevant troubleshooting articles ranked by relevance score.
    """
    kb = _knowledge_base()
    if not kb:
        return [{"id": "KB-000", "title": "Knowledge base unavailable", "summary": "CSV data not found."}]

    query_tokens = _tokenize(query)
    if not query_tokens:
        return [{"id": "KB-000", "title": "No results", "summary": "Please provide a search query."}]

    scored = [(_kb_score(query_tokens, doc), doc) for doc in kb]
    scored.sort(key=lambda x: x[0], reverse=True)

    results = [
        {
            "id":       doc["id"],
            "title":    doc["title"],
            "category": doc.get("category", ""),
            "summary":  doc["summary"],
            "score":    round(score, 4),
        }
        for score, doc in scored
        if score > 0
    ]

    if not results:
        return [{"id": "KB-000", "title": "No results found", "summary": "Try rephrasing your search."}]
    return results[:top_k]


@tool(approval_mode="never_require")
def create_support_ticket(
    subject: Annotated[str, Field(description="Short ticket subject (max 100 chars)")],
    description: Annotated[str, Field(description="Full description of the issue")],
    priority: Annotated[str, Field(description="Priority level: low | medium | high | critical")],
    account_id: Annotated[str, Field(description="Associated customer account ID")] = "unknown",
) -> dict[str, Any]:
    """Create a support ticket for issues that cannot be resolved immediately."""
    valid_priorities = {"low", "medium", "high", "critical"}
    safe_priority = priority.lower() if priority.lower() in valid_priorities else "medium"
    sla_hours = {"low": 72, "medium": 24, "high": 4, "critical": 1}

    account_name = "unknown"
    if account_id != "unknown":
        acct = lookup_account(account_id)
        if "name" in acct:
            account_name = acct["name"]

    return {
        "ticket_id":          f"TKT-{int(time.time()) % 100_000:05d}",
        "subject":            subject[:100],
        "description":        description[:500],
        "priority":           safe_priority,
        "account_id":         account_id,
        "account_name":       account_name,
        "status":             "open",
        "sla_response_hours": sla_hours[safe_priority],
        "created_at":         datetime.now(timezone.utc).isoformat() + "Z",
        "assigned_team":      "L2-Support",
    }


@tool(approval_mode="never_require")
def run_diagnostics(
    service: Annotated[str, Field(description="Service to run diagnostics on")],
    account_id: Annotated[str, Field(description="Customer account ID for targeted diagnostics")],
) -> dict[str, Any]:
    """
    Run automated diagnostics for a service on a specific account.

    Diagnostic metrics are derived dynamically from live service-status data,
    so results always reflect the current operational state of the service.
    """
    return _diagnostics_for(service.strip().lower(), account_id.strip())
