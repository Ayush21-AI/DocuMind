from demo.pipeline import run_pipeline
from demo.samples import SAMPLES


def test_clean_invoice_sample():
    s = SAMPLES[0]
    r = run_pipeline(s["text"], s["model_output"])
    assert r["document_type"] == "invoice"
    assert r["fields"]["bankAccountSortCode"] == "20-15-82"   # normalised
    assert r["fields"]["totalInvoiceAmount"] == "1500.50 GBP"  # commas stripped
    assert r["review_required"] is False
    assert r["overall_confidence"] >= 0.9


def test_receipt_sample_currency_and_date():
    s = SAMPLES[1]
    r = run_pipeline(s["text"], s["model_output"])
    assert r["document_type"] == "expense"
    assert r["fields"]["invoiceDate"] == "17/10/2020"          # US order corrected
    assert r["fields"]["transactionAmount"] == "28040.00 USD"
    assert r["currency"] == "USD"


def test_low_quality_scan_flagged_for_review():
    s = SAMPLES[2]
    r = run_pipeline(s["text"], s["model_output"])
    assert r["overall_confidence"] == 0.0
    assert r["review_required"] is True
    assert r["routing_confidence"] <= 0.2   # classifier is honestly unsure
