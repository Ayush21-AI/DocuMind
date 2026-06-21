"""
Lightweight, dependency-free document-type classifier.

Powers the single `/extract` endpoint: callers no longer need to know up front
whether they are sending a formal supplier invoice (→ 8-field schema) or a
till receipt / transaction slip (→ 3-field schema). A keyword-weighted
heuristic routes the document, with a confidence score so the API can flag
low-confidence routing for review.

It is intentionally a fast pure-Python pass rather than an extra LLM call —
classification this coarse does not need a 3B model, and keeping it local
means the routing decision is deterministic and testable.
"""
from __future__ import annotations

import re

INVOICE = "invoice"
EXPENSE = "expense"

# Signals weighted by how strongly they indicate a formal invoice vs a receipt.
_INVOICE_SIGNALS = {
    r"\binvoice\s*(no|number|#)": 3,
    r"\binvoice\b": 2,
    r"\bdue\s*date\b": 2,
    r"\bpayment\s*due\b": 2,
    r"\bsort\s*code\b": 3,
    r"\baccount\s*(no|number|#)": 2,
    r"\bvat\s*(no|number|reg)": 2,
    r"\bbill\s*to\b": 2,
    r"\bpurchase\s*order\b": 2,
    r"\bremittance\b": 2,
    r"\bsupplier\b": 1,
}
_EXPENSE_SIGNALS = {
    r"\breceipt\b": 3,
    r"\bchange\s*due\b": 3,
    r"\bcard\s*payment\b": 2,
    r"\bcontactless\b": 2,
    r"\bcashier\b": 2,
    r"\btill\b": 2,
    r"\bthank\s*you\s*for\s*shopping\b": 3,
    r"\bsubtotal\b": 1,
    r"\bauth\s*code\b": 2,
    r"\bmerchant\b": 1,
    r"\bvisa\b|\bmastercard\b|\bamex\b": 1,
}


def _score(text: str, signals: dict[str, int]) -> int:
    total = 0
    for pattern, weight in signals.items():
        if re.search(pattern, text):
            total += weight
    return total


def detect_document_type(text: str) -> tuple[str, float]:
    """
    Classify `text` as an invoice or an expense/receipt.

    Returns (document_type, confidence) where confidence is in [0, 1].
    Ties and empty input default to INVOICE (the richer schema) with low
    confidence so the caller knows routing was uncertain.
    """
    if not text or not text.strip():
        return INVOICE, 0.0

    lowered = text.lower()
    inv = _score(lowered, _INVOICE_SIGNALS)
    exp = _score(lowered, _EXPENSE_SIGNALS)

    if inv == 0 and exp == 0:
        return INVOICE, 0.2

    doc_type = INVOICE if inv >= exp else EXPENSE
    winner, loser = max(inv, exp), min(inv, exp)
    # Confidence grows with the margin between the two scores.
    confidence = round((winner - loser) / (winner + loser + 1) * 0.8 + 0.2, 2)
    return doc_type, min(confidence, 1.0)
