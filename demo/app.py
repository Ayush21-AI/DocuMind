"""
DocuMind interactive demo (Gradio).

Two tabs:
  • "Try a sample" — runs the real classifier / validators / confidence scorer
    on bundled example documents. Works fully offline (no LLM), so it can be
    hosted on a free Hugging Face CPU Space.
  • "Live API" — point at a running DocuMind backend and extract from your own
    uploaded file via POST /extract.

Run locally:   python demo/app.py
Deploy:        push this repo to a Hugging Face Space (sdk: gradio,
               app_file: demo/app.py) — see README "Live demo".
"""
from __future__ import annotations

import json
import os

import gradio as gr
import requests

from pipeline import run_pipeline
from samples import SAMPLES


def _confidence_color(score: float) -> str:
    if score >= 0.85:
        return "#1a7f37"  # green  — auto-accept
    if score >= 0.6:
        return "#9a6700"  # amber  — likely ok
    return "#cf222e"      # red    — review


def _fields_html(result: dict) -> str:
    rows = []
    for field, value in result["fields"].items():
        score = result["confidence"].get(field, 0.0)
        color = _confidence_color(score)
        shown = value if value else "<em style='color:#999'>—</em>"
        rows.append(
            f"<tr>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #eee'><code>{field}</code></td>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #eee'>{shown}</td>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #eee;color:{color};font-weight:600'>"
            f"{score:.2f}</td>"
            f"</tr>"
        )
    badge_color = "#cf222e" if result["review_required"] else "#1a7f37"
    badge_text = "⚠ REVIEW REQUIRED" if result["review_required"] else "✓ AUTO-ACCEPT"
    header = (
        f"<div style='margin-bottom:10px'>"
        f"<b>Type:</b> {result['document_type']} "
        f"(routing {result['routing_confidence']:.2f}) &nbsp;|&nbsp; "
        f"<b>Currency:</b> {result['currency']} &nbsp;|&nbsp; "
        f"<b>Overall:</b> {result['overall_confidence']:.2f} &nbsp;|&nbsp; "
        f"<span style='background:{badge_color};color:#fff;padding:2px 8px;border-radius:4px'>"
        f"{badge_text}</span>"
        f"</div>"
    )
    table = (
        "<table style='border-collapse:collapse;width:100%;font-size:14px'>"
        "<tr style='text-align:left'>"
        "<th style='padding:6px 12px'>Field</th>"
        "<th style='padding:6px 12px'>Value</th>"
        "<th style='padding:6px 12px'>Confidence</th></tr>"
        + "".join(rows) +
        "</table>"
    )
    return header + table


def run_sample(sample_name: str):
    sample = next(s for s in SAMPLES if s["name"] == sample_name)
    result = run_pipeline(sample["text"], sample["model_output"])
    return sample["text"], _fields_html(result), json.dumps(result, indent=2)


def run_live(api_url: str, file_path: str | None):
    if not api_url:
        return "<p style='color:#cf222e'>Enter your DocuMind API URL.</p>", "{}"
    if not file_path:
        return "<p style='color:#cf222e'>Upload a document first.</p>", "{}"
    try:
        with open(file_path, "rb") as fh:
            resp = requests.post(
                f"{api_url.rstrip('/')}/extract",
                files={"file": (os.path.basename(file_path), fh)},
                timeout=120,
            )
        resp.raise_for_status()
        data = resp.json()
        view = {
            "fields": data.get("fields", {}),
            "confidence": data.get("confidence", {}),
            "overall_confidence": data.get("overall_confidence", 0.0),
            "review_required": data.get("review_required", False),
            "document_type": data.get("document_type", "?"),
            "routing_confidence": data.get("meta", {}).get("routing_confidence", 1.0),
            "currency": data.get("currency", "GBP"),
        }
        return _fields_html(view), json.dumps(data, indent=2)
    except Exception as e:  # noqa: BLE001 — surface any failure to the user
        return f"<p style='color:#cf222e'>Request failed: {e}</p>", "{}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DocuMind — Document Intelligence Demo") as demo:
        gr.Markdown(
            "# 🧾→🔣 DocuMind\n"
            "Extract structured, **confidence-scored** data from invoices & receipts. "
            "Confidence combines *format validity* and *grounding* (does the value appear "
            "in the source?) — a built-in hallucination guard."
        )

        with gr.Tab("Try a sample (offline)"):
            picker = gr.Dropdown(
                [s["name"] for s in SAMPLES],
                value=SAMPLES[0]["name"],
                label="Example document",
            )
            run_btn = gr.Button("Extract", variant="primary")
            with gr.Row():
                src = gr.Textbox(label="Source document text", lines=14)
                with gr.Column():
                    fields_out = gr.HTML(label="Extracted fields + confidence")
                    json_out = gr.Code(label="Full ExtractionResult", language="json")
            run_btn.click(run_sample, inputs=picker, outputs=[src, fields_out, json_out])
            demo.load(run_sample, inputs=picker, outputs=[src, fields_out, json_out])

        with gr.Tab("Live API (your backend)"):
            gr.Markdown(
                "Point at a running DocuMind instance "
                "(`docker compose up`, then e.g. `http://localhost:8000`)."
            )
            api = gr.Textbox(label="DocuMind API URL", placeholder="http://localhost:8000")
            upload = gr.File(label="Document", type="filepath",
                             file_types=[".pdf", ".png", ".jpg", ".jpeg", ".docx", ".xlsx"])
            live_btn = gr.Button("Extract via API", variant="primary")
            live_fields = gr.HTML()
            live_json = gr.Code(label="Full response", language="json")
            live_btn.click(run_live, inputs=[api, upload], outputs=[live_fields, live_json])

    return demo


if __name__ == "__main__":
    build_ui().launch()
