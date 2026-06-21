from ocr.src.main.classifier import EXPENSE, INVOICE, detect_document_type


def test_detects_invoice():
    text = "INVOICE Invoice Number: INV-001\nDue Date: 31/12/2024\nSort Code 11-22-33"
    doc_type, conf = detect_document_type(text)
    assert doc_type == INVOICE
    assert conf > 0.2


def test_detects_receipt():
    text = "RECEIPT\nSubtotal 5.00\nCard Payment Contactless\nChange Due 0.00\nThank you for shopping"
    doc_type, conf = detect_document_type(text)
    assert doc_type == EXPENSE
    assert conf > 0.2


def test_empty_defaults_to_invoice_low_confidence():
    assert detect_document_type("") == (INVOICE, 0.0)


def test_no_signals_defaults_to_invoice():
    doc_type, conf = detect_document_type("some random unrelated text")
    assert doc_type == INVOICE
    assert conf == 0.2
