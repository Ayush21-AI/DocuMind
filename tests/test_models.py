from ocr.src.main.model.expense_data import ExpenseData
from ocr.src.main.model.invoice_data import InvoiceData


def test_invoice_normalisation():
    inv = InvoiceData(
        invoiceDate="December 31, 2024",
        invoiceDueDate="2025-01-15",
        totalInvoiceAmount="£1,500.50",
        totalVatAmount="abc",            # no number -> ""
        bankAccountSortCode="201582",    # -> NN-NN-NN
    )
    assert inv.invoiceDate == "31/12/2024"
    assert inv.invoiceDueDate == "15/01/2025"
    assert inv.totalInvoiceAmount == "1500.50 GBP"
    assert inv.totalVatAmount == ""
    assert inv.bankAccountSortCode == "20-15-82"


def test_invoice_sortcode_passthrough_and_reject():
    assert InvoiceData(bankAccountSortCode="20-15-82").bankAccountSortCode == "20-15-82"
    assert InvoiceData(bankAccountSortCode="not a code").bankAccountSortCode == ""


def test_expense_normalisation_and_currency():
    exp = ExpenseData(invoiceDate="10/17/2020", transactionAmount="$28,040 USD")
    assert exp.invoiceDate == "17/10/2020"       # US ordering corrected
    assert exp.transactionAmount == "28040.00 USD"
    assert exp.detectedCurrency == "USD"


def test_expense_defaults_gbp():
    exp = ExpenseData(transactionAmount="28040")
    assert exp.transactionAmount == "28040.00 GBP"
    assert exp.detectedCurrency == "GBP"


def test_field_kinds_exist():
    assert set(InvoiceData.field_kinds()) == set(InvoiceData.model_fields)
    assert "transactionAmount" in ExpenseData.field_kinds()
