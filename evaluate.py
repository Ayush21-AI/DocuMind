#!/usr/bin/env python3
"""
DocuMind extraction evaluation harness.

Computes field-level accuracy metrics by comparing model predictions against a
hand-labelled gold set, and prints a Markdown report. This is what turns
"intelligent extraction" claims into numbers — exact-match accuracy plus
precision / recall / F1 per field.

Usage:
    # Score existing predictions against the gold set:
    python evaluate.py --gold tests/fixtures/eval/gold.json \\
                       --pred tests/fixtures/eval/predictions.json

    # (Optional) regenerate predictions by calling a running DocuMind API,
    # then score them — requires the live service + Ollama:
    python evaluate.py --gold gold.json --api http://localhost:8000 --docs ./samples

The scoring core (`compute_metrics`) is pure-Python and unit-tested, so the
metrics table can be regenerated in CI from cached prediction fixtures without
needing the LLM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _index_by_file(records: list[dict]) -> dict[str, dict]:
    return {r["file"]: r.get("fields", {}) for r in records}


def compute_metrics(gold: list[dict], pred: list[dict]) -> dict:
    """
    Compare predictions to gold annotations.

    For each field, a prediction is "correct" when it exactly equals the gold
    value. Precision is over non-empty predictions, recall over non-empty gold
    values; exact-match accuracy counts every document (empty==empty is correct).

    Returns {"per_field": {field: {...}}, "overall": {...}}.
    """
    gold_by_file = _index_by_file(gold)
    pred_by_file = _index_by_file(pred)
    files = list(gold_by_file)

    fields = sorted({f for fields in gold_by_file.values() for f in fields})
    per_field: dict[str, dict] = {}

    for field in fields:
        correct = pred_nonempty = gold_nonempty = total = 0
        for file in files:
            g = (gold_by_file[file].get(field) or "").strip()
            p = (pred_by_file.get(file, {}).get(field) or "").strip()
            total += 1
            if g:
                gold_nonempty += 1
            if p:
                pred_nonempty += 1
            if p == g:
                correct += 1

        correct_nonempty = sum(
            1 for file in files
            if (gold_by_file[file].get(field) or "").strip()
            and (gold_by_file[file].get(field) or "").strip()
            == (pred_by_file.get(file, {}).get(field) or "").strip()
        )
        precision = correct_nonempty / pred_nonempty if pred_nonempty else 0.0
        recall = correct_nonempty / gold_nonempty if gold_nonempty else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_field[field] = {
            "exact_match": round(correct / total, 3) if total else 0.0,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "n": total,
        }

    n_fields = len(per_field) or 1
    overall = {
        "exact_match": round(sum(m["exact_match"] for m in per_field.values()) / n_fields, 3),
        "f1": round(sum(m["f1"] for m in per_field.values()) / n_fields, 3),
        "documents": len(files),
        "fields": len(per_field),
    }
    return {"per_field": per_field, "overall": overall}


def format_markdown(metrics: dict) -> str:
    lines = ["| Field | Exact match | Precision | Recall | F1 | N |",
             "|---|---|---|---|---|---|"]
    for field, m in metrics["per_field"].items():
        lines.append(f"| `{field}` | {m['exact_match']:.3f} | {m['precision']:.3f} "
                     f"| {m['recall']:.3f} | {m['f1']:.3f} | {m['n']} |")
    o = metrics["overall"]
    lines.append(f"| **macro avg** | **{o['exact_match']:.3f}** | — | — | **{o['f1']:.3f}** | {o['documents']} docs |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="DocuMind evaluation harness")
    ap.add_argument("--gold", default="tests/fixtures/eval/gold.json")
    ap.add_argument("--pred", default="tests/fixtures/eval/predictions.json")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text())
    pred = json.loads(Path(args.pred).read_text())
    metrics = compute_metrics(gold, pred)
    print(format_markdown(metrics))
    print(f"\nOverall: exact-match {metrics['overall']['exact_match']:.1%}, "
          f"macro-F1 {metrics['overall']['f1']:.3f} "
          f"over {metrics['overall']['documents']} documents.")


if __name__ == "__main__":
    main()
