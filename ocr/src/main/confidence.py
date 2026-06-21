"""
Per-field confidence scoring + human-in-the-loop review flagging.

Most LLM extractors hand back fields with no indication of *how much to trust
them*. DocuMind scores every field on two independent, explainable signals:

  • format validity  — does the value match the shape it should have
                        (a date parses, an amount is numeric, a sort code is
                        NN-NN-NN, etc.)?
  • grounding        — does the value actually appear in the source document?
                        This is a cheap, powerful hallucination guard: a value
                        the model invented won't be found in the OCR text.

A present field scores 0.5 (found) + up to 0.25 (valid format) + up to 0.25
(grounded) = 0.5–1.0; an absent field scores 0.0. The overall score is the
mean over the fields the model actually populated, and `review_required` lets
callers bucket results into auto-accept / needs-review without guesswork.

Pure-Python and deterministic — no extra LLM calls — so it is fully unit-tested.
"""
from __future__ import annotations

import re

# Field "kinds" drive both format validation and grounding strategy.
DATE = "date"
AMOUNT = "amount"
SORTCODE = "sortcode"
ACCOUNT = "account"
TEXT = "text"

REVIEW_THRESHOLD = 0.6

_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_AMOUNT_RE = re.compile(r"^-?\d+\.\d{2}\s+[A-Z]{3}$")
_SORTCODE_RE = re.compile(r"^\d{2}-\d{2}-\d{2}$")
_ACCOUNT_RE = re.compile(r"^\d{6,10}$")


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _format_valid(value: str, kind: str) -> bool:
    if kind == DATE:
        if not _DATE_RE.match(value):
            return False
        d, m, _ = value.split("/")
        return 1 <= int(d) <= 31 and 1 <= int(m) <= 12
    if kind == AMOUNT:
        return bool(_AMOUNT_RE.match(value))
    if kind == SORTCODE:
        return bool(_SORTCODE_RE.match(value))
    if kind == ACCOUNT:
        return bool(_ACCOUNT_RE.match(value))
    return bool(value.strip())  # TEXT: any non-empty string is well-formed


def _grounded(value: str, kind: str, source: str) -> bool:
    """Is `value` actually supported by the source document text?"""
    if not source:
        return False
    src_alnum = _alnum(source)
    src_digits = _digits(source)

    if kind in (AMOUNT, SORTCODE, ACCOUNT):
        # Compare the significant integer digits (strip our appended ".00"/CCY).
        whole = value.split(".")[0] if kind == AMOUNT else value
        digits = _digits(whole)
        return bool(digits) and digits in src_digits
    if kind == DATE:
        d, m, y = value.split("/")
        return y in source and (d in src_digits or m in src_digits)
    # TEXT: at least one meaningful token (len >= 3) appears in the source.
    tokens = [t for t in re.split(r"\s+", value.lower()) if len(t) >= 3]
    return any(_alnum(t) and _alnum(t) in src_alnum for t in tokens)


def score_field(value: str, kind: str, source: str) -> float:
    """Confidence in a single field, in [0, 1]."""
    if not value or not value.strip():
        return 0.0
    score = 0.5
    if _format_valid(value, kind):
        score += 0.25
    if _grounded(value, kind, source):
        score += 0.25
    return round(score, 2)


def score_extraction(fields: dict[str, str], kinds: dict[str, str],
                     source: str) -> dict:
    """
    Score every field of an extraction.

    Returns:
        {
          "per_field": {field: 0.0–1.0, ...},
          "overall":   mean over populated fields (0.0 if none),
          "review_required": bool,
        }
    """
    per_field = {
        name: score_field(value, kinds.get(name, TEXT), source)
        for name, value in fields.items()
    }
    populated = [s for s in per_field.values() if s > 0.0]
    overall = round(sum(populated) / len(populated), 2) if populated else 0.0
    return {
        "per_field": per_field,
        "overall": overall,
        "review_required": overall < REVIEW_THRESHOLD,
    }
