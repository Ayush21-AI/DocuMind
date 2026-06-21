"""
Demo pipeline glue.

Runs the *real* DocuMind post-extraction stack — document-type classification,
Pydantic normalisation, and per-field confidence scoring — against a piece of
document text plus a model output. The only thing the demo stubs out is the LLM
call itself (cached per sample), so a free CPU Hugging Face Space can showcase
the production validation/confidence code with no GPU or Ollama required.

Kept free of any Gradio import so it can be unit-tested directly.
"""
from __future__ import annotations

import os
import sys

# Make the project's `ocr` package importable whether run from the repo root
# or from inside demo/ (e.g. on a Hugging Face Space).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ocr.src.main.classifier import INVOICE, detect_document_type  # noqa: E402
from ocr.src.main.confidence import score_extraction  # noqa: E402
from ocr.src.main.model.expense_data import ExpenseData  # noqa: E402
from ocr.src.main.model.invoice_data import InvoiceData  # noqa: E402


def run_pipeline(text: str, raw_model_output: dict, forced_type: str | None = None) -> dict:
    """
    Produce a confidence-scored ExtractionResult-shaped dict from document text
    and a (cached or live) model output.

    This mirrors the API's `_build_result`: classify -> validate/normalise ->
    score confidence (format + grounding) -> envelope.
    """
    if forced_type:
        document_type, routing_confidence = forced_type, 1.0
    else:
        document_type, routing_confidence = detect_document_type(text)

    Model = InvoiceData if document_type == INVOICE else ExpenseData
    model = Model(**raw_model_output)

    fields = model.model_dump()
    currency = fields.pop("detectedCurrency", "GBP")
    conf = score_extraction(fields, Model.field_kinds(), text)

    return {
        "document_type": document_type,
        "routing_confidence": routing_confidence,
        "fields": fields,
        "confidence": conf["per_field"],
        "overall_confidence": conf["overall"],
        "review_required": conf["review_required"],
        "currency": currency,
    }
