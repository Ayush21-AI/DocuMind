"""
Edge-case / data-hygiene suite.

These tests document the messy real-world inputs DocuMind is hardened against —
the kind of cases that break naive extractors. They double as living
documentation of the validation guarantees.
"""
import pytest

from ocr.src.main.confidence import AMOUNT, score_field
from ocr.src.main.model.expense_data import ExpenseData
from ocr.src.main.model.invoice_data import InvoiceData
from ocr.src.main.utils.currency import normalize_amount
from ocr.src.main.utils.dates import normalize_date


@pytest.mark.parametrize("raw,expected", [
    ("December 31, 2024", "31/12/2024"),
    ("31 December 2024", "31/12/2024"),
    ("2024-12-31", "31/12/2024"),
    ("10/17/2020", "17/10/2020"),   # US order, unambiguous -> corrected
    ("03/04/2024", "03/04/2024"),   # ambiguous -> UK day-first
    ("1/2/24", "01/02/2024"),       # two-digit year
    ("13/13/2024", ""),             # impossible -> rejected
    ("N/A", ""),                    # junk -> rejected, never guesses "today"
    ("", ""),
])
def test_date_edge_cases(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("£0.50", "0.50 GBP"),
    ("1,234", "1234.00 GBP"),        # thousands separator stripped
    ("$28,040.00 USD", "28040.00 USD"),  # multi-currency preserved
    ("(1,234.50)", "-1234.50 GBP"),  # accounting negative
    ("12.5", "12.50 GBP"),           # two decimals enforced
    ("100 pounds", "100.00 GBP"),    # currency word
    ("N/A", ""),                     # no number -> empty, not a crash
])
def test_amount_edge_cases(raw, expected):
    assert normalize_amount(raw)[0] == expected


def test_hallucination_guard_lowers_confidence():
    # A well-formed amount that does NOT appear in the source is suspicious:
    grounded = score_field("12.00 GBP", AMOUNT, "Total £12.00")
    invented = score_field("9999.00 GBP", AMOUNT, "Total £12.00")
    assert grounded > invented   # grounding catches the unsupported value


def test_sortcode_normalised_from_messy_input():
    assert InvoiceData(bankAccountSortCode="20 15 82").bankAccountSortCode == "20-15-82"
    assert InvoiceData(bankAccountSortCode="201582").bankAccountSortCode == "20-15-82"
    assert InvoiceData(bankAccountSortCode="garbage").bankAccountSortCode == ""


def test_extractors_never_raise_on_empty_or_junk():
    # Missing/garbage fields degrade gracefully to "" — never an exception.
    inv = InvoiceData(invoiceDate="not a date", totalVatAmount="lots")
    assert inv.invoiceDate == "" and inv.totalVatAmount == ""
    exp = ExpenseData()
    assert exp.transactionAmount == "" and exp.detectedCurrency == "GBP"
