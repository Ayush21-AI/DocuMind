"""
Currency-aware amount normalisation.

Turns noisy money strings ("£1,234.5", "$28,040 USD", "(1.234,50)") into a
canonical `"<X.XX> <CCY>"` form, detecting the currency where possible and
defaulting to GBP. Commas are treated as thousands separators (the dominant
UK/US convention); parentheses and leading minus signs denote negatives.

This replaces the previous naïve `re.sub(r'\\s*[A-Za-z]+$', '', amount)` which
neither stripped commas nor enforced two decimal places.
"""
from __future__ import annotations

import re

# Symbol -> ISO 4217 code.
_SYMBOLS = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY", "₹": "INR"}
# Currency codes/words we recognise as trailing or leading tokens.
_CODES = {"GBP", "USD", "EUR", "JPY", "INR", "CAD", "AUD", "CHF", "CNY",
          "POUNDS", "POUND", "DOLLARS", "DOLLAR", "EUROS", "EURO"}
_WORD_TO_CODE = {"POUNDS": "GBP", "POUND": "GBP", "DOLLARS": "USD",
                 "DOLLAR": "USD", "EUROS": "EUR", "EURO": "EUR"}

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def detect_currency(raw: str, default: str = "GBP") -> str:
    """Return the ISO currency code implied by `raw`, else `default`."""
    if not raw:
        return default
    upper = raw.upper()
    for sym, code in _SYMBOLS.items():
        if sym in raw:
            return code
    for token in re.findall(r"[A-Z]+", upper):
        if token in _WORD_TO_CODE:
            return _WORD_TO_CODE[token]
        if token in _CODES:
            return token
    return default


def normalize_amount(raw: str, default_currency: str = "GBP") -> tuple[str, str, bool]:
    """
    Normalise a money string.

    Returns (formatted, currency, ok):
      formatted -> "1234.50 GBP" (always two decimals), or "" if no number.
      currency  -> detected ISO code, or `default_currency`.
      ok        -> True if a numeric amount was found.

    Examples:
        "£0.50"        -> ("0.50 GBP", "GBP", True)
        "$28,040 USD"  -> ("28040.00 USD", "USD", True)
        "1,234"        -> ("1234.00 GBP", "GBP", True)
        "(1,234.50)"   -> ("-1234.50 GBP", "GBP", True)
        "N/A"          -> ("", "GBP", False)
    """
    if not raw:
        return "", default_currency, False
    raw = raw.strip()
    currency = detect_currency(raw, default_currency)

    negative = raw.startswith("-") or ("(" in raw and ")" in raw)

    match = _NUMBER.search(raw)
    if not match:
        return "", currency, False

    number = match.group(0).replace(",", "")  # commas = thousands separators
    try:
        value = abs(float(number))
    except ValueError:
        return "", currency, False
    if negative:
        value = -value

    return f"{value:.2f} {currency}", currency, True
