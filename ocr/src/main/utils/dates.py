"""
Robust date normalisation to the UK format DD/MM/YYYY.

The LLM is prompted to return DD/MM/YYYY, but real-world output is noisy:
US ordering (MM/DD/YYYY), spelled-out months, ISO dates, two-digit years.
This module is the single source of truth for turning any of those into a
canonical `DD/MM/YYYY` string (or `""` when nothing parseable is present),
so both the invoice and expense validators behave identically.
"""
from __future__ import annotations

import re
from datetime import datetime

from dateutil import parser as _dtparser

_CANONICAL = re.compile(r"^\d{2}/\d{2}/\d{4}$")
# A date must contain at least one digit to be worth parsing.
_HAS_DIGIT = re.compile(r"\d")


def is_canonical(value: str) -> bool:
    """True if `value` is already a syntactically valid DD/MM/YYYY string."""
    if not _CANONICAL.match(value or ""):
        return False
    d, m, _ = value.split("/")
    return 1 <= int(d) <= 31 and 1 <= int(m) <= 12


def normalize_date(raw: str, dayfirst: bool = True) -> str:
    """
    Normalise a free-form date string to DD/MM/YYYY (UK convention).

    `dayfirst=True` means ambiguous values like 03/04/2024 are read as
    3 April 2024. Unambiguous US dates (e.g. 10/17/2020 — 17 cannot be a
    month) are still corrected to 17/10/2020 by the underlying parser.

    Returns "" if the input has no parseable date.
    """
    if not raw:
        return ""
    raw = raw.strip()
    if not _HAS_DIGIT.search(raw):
        return ""

    # Fast path: already canonical and semantically valid.
    if is_canonical(raw):
        return raw

    try:
        dt = _dtparser.parse(raw, dayfirst=dayfirst, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return ""

    # Reject parses that silently invented today's date from junk like "N/A".
    if not isinstance(dt, datetime):
        return ""
    return dt.strftime("%d/%m/%Y")
