from ocr.src.main.invoice_processor.ollama_processor import parse_ollama_response
from ocr.src.main.model.invoice_data import InvoiceData


def test_clean_json_content():
    result = {"message": {"content": '{"invoiceNumber": "INV-1", "supplierName": "Acme"}'}}
    parsed = parse_ollama_response(result, InvoiceData)
    assert parsed.invoiceNumber == "INV-1"
    assert parsed.supplierName == "Acme"


def test_markdown_fenced_json_fallback():
    result = {"message": {"content": '```json\n{"invoiceNumber": "INV-2"}\n```'}}
    parsed = parse_ollama_response(result, InvoiceData)
    assert parsed.invoiceNumber == "INV-2"


def test_generate_style_response_key():
    result = {"response": '{"invoiceNumber": "INV-3"}'}
    parsed = parse_ollama_response(result, InvoiceData)
    assert parsed.invoiceNumber == "INV-3"


def test_garbage_returns_empty_model():
    result = {"message": {"content": "the model said no json today"}}
    parsed = parse_ollama_response(result, InvoiceData)
    assert parsed.invoiceNumber == ""
    assert parsed.totalInvoiceAmount == ""
