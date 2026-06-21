import json
from pathlib import Path

from evaluate import compute_metrics, format_markdown

FIXTURES = Path(__file__).parent / "fixtures" / "eval"


def _load():
    gold = json.loads((FIXTURES / "gold.json").read_text())
    pred = json.loads((FIXTURES / "predictions.json").read_text())
    return gold, pred


def test_perfect_predictions_score_one():
    gold, _ = _load()
    metrics = compute_metrics(gold, gold)
    assert metrics["overall"]["exact_match"] == 1.0
    assert metrics["overall"]["f1"] == 1.0


def test_known_errors_lower_scores():
    gold, pred = _load()
    metrics = compute_metrics(gold, pred)
    # invoiceNumber & total are all correct; vat/date/supplier/sortcode each have one error.
    assert metrics["per_field"]["invoiceNumber"]["exact_match"] == 1.0
    assert metrics["per_field"]["invoiceDate"]["exact_match"] == 0.75
    assert metrics["per_field"]["supplierName"]["exact_match"] == 0.75
    assert 0.0 < metrics["overall"]["f1"] < 1.0


def test_markdown_renders():
    gold, pred = _load()
    table = format_markdown(compute_metrics(gold, pred))
    assert "Exact match" in table and "macro avg" in table
