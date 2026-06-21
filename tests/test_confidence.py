from ocr.src.main.confidence import (AMOUNT, DATE, TEXT, score_extraction,
                                     score_field)


def test_absent_field_is_zero():
    assert score_field("", AMOUNT, "anything") == 0.0


def test_valid_and_grounded_scores_full():
    assert score_field("31/12/2024", DATE, "Issued 31/12/2024") == 1.0
    assert score_field("1500.50 GBP", AMOUNT, "Total £1,500.50") == 1.0


def test_valid_but_not_grounded():
    # Well-formed amount that does not appear in the source = possible hallucination.
    assert score_field("9999.00 GBP", AMOUNT, "Total £12.00") == 0.75


def test_text_grounding_by_token():
    assert score_field("Acme Corp", TEXT, "From Acme Corp Ltd") == 1.0
    assert score_field("Imaginary Vendor", TEXT, "totally unrelated source") == 0.75


def test_score_extraction_overall_and_review():
    fields = {
        "invoiceDate": "31/12/2024",
        "totalInvoiceAmount": "100.00 GBP",
        "supplierName": "",          # absent
        "bankAccountSortCode": "",   # absent
    }
    kinds = {
        "invoiceDate": DATE,
        "totalInvoiceAmount": AMOUNT,
        "supplierName": TEXT,
        "bankAccountSortCode": "sortcode",
    }
    source = "Date 31/12/2024 Total 100.00"
    report = score_extraction(fields, kinds, source)
    assert report["per_field"]["supplierName"] == 0.0
    assert report["overall"] == 1.0          # mean over the two populated fields
    assert report["review_required"] is False


def test_review_required_when_low():
    fields = {"transactionAmount": "9999.00 GBP"}
    kinds = {"transactionAmount": AMOUNT}
    report = score_extraction(fields, kinds, "unrelated source with no amount")
    # 0.75 (valid, ungrounded) is above threshold; force a low case:
    low = score_extraction({"description": ""}, {"description": TEXT}, "x")
    assert low["overall"] == 0.0
    assert low["review_required"] is True
