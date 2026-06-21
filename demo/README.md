---
title: DocuMind
emoji: 🧾
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# DocuMind — interactive demo

A Gradio front-end for [DocuMind](https://github.com/Ayush21-AI/DocuMind),
self-hosted document intelligence (OCR + local LLM) with per-field confidence
scoring.

- **Try a sample (offline):** runs the real classifier, Pydantic validators and
  confidence scorer on bundled invoices/receipts — no LLM needed, so it runs on
  a free CPU Space. One sample deliberately includes an unsupported value so you
  can see the grounding check flag it for review.
- **Live API:** point at a running DocuMind backend (`docker compose up`) and
  extract from your own uploaded document.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Deploy free on Hugging Face Spaces

The `demo/` folder is a self-contained Gradio Space (note the YAML card above).
Easiest path:

```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli repo create documind-demo --type space --space_sdk gradio
# copy the demo files + the project's `ocr/` package into the Space repo, then:
git add . && git commit -m "DocuMind demo" && git push
```

The offline sample tab needs only the four deps in `requirements.txt` plus the
`ocr/` package (imported for the real validation/confidence code).
