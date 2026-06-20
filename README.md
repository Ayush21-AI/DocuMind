# DocuMind

**AI-powered document intelligence** — a production-grade microservice that automatically extracts structured financial data from unstructured documents (invoices, receipts, PDFs, images, DOCX, XLSX), eliminating manual data entry for UK accounting and expense workflows.

---

## What it does

Upload a document → DocuMind returns clean, validated JSON.

It runs a **two-stage deterministic pipeline**:

1. **Text extraction (OCR)** — PyMuPDF for native PDFs (with a per-page RapidOCR fallback for scanned pages), RapidOCR + OpenCV for images (with table/column layout reconstruction), python-docx for Word, openpyxl for Excel.
2. **LLM field extraction** — a locally hosted Llama 3.2 model (via Ollama) extracts the target fields under a strict, anti-hallucination system prompt at `temperature=0`, `format=json`.

Results are then normalised and validated with Pydantic (dates → `DD/MM/YYYY`, amounts → `X.XX GBP`), with graceful degradation: unparseable fields return `""` rather than raising errors to the caller.

## Two extraction modes

| Endpoint | Fields | Use case |
|---|---|---|
| `POST /process/invoice/` | 8 (invoice no., dates, totals, VAT, supplier, sort code, account no.) | Merchant invoices |
| `POST /process/income/expenses/invoice/` | 3 (date, amount, description) | Receipts / transaction slips |
| `GET /status` | — | Health check |

**Supported file types:** `.pdf`, `.png`, `.jpg`, `.jpeg`, `.docx`, `.xlsx`

## Architecture at a glance

Two independent, containerised microservices:

- **`documind-ocr`** — FastAPI app (async, 2 Uvicorn workers) handling extraction + LLM orchestration.
- **`documind-ollama`** — Ollama runtime with `llama3.2:latest` pre-baked into the image at build time.

Designed for deployment to Kubernetes via Helm.

### Engineering highlights

- **Generic LLM processor** — one `OllamaProcessor` class parameterised by Pydantic model, reused across both extraction modes.
- **Performance** — shared async HTTP client, singleton OCR engine per worker, LRU caching on inference, a background warmup loop to keep the model hot, and concurrent multi-page PDF processing via `asyncio.gather`.
- **Resilience** — `tenacity` retry with exponential backoff on transient network errors; typed HTTP error mapping (502/503/504) for LLM failures.
- **Containerisation** — CPU-thread and memory tuning for inference workloads (`OMP_NUM_THREADS`, `MALLOC_ARENA_MAX`, etc.).

## Tech stack

FastAPI · Uvicorn (uvloop) · Pydantic · RapidOCR · ONNX Runtime · OpenCV · PyMuPDF · python-docx · openpyxl · httpx · Ollama · Llama 3.2 · Docker · Kubernetes / Helm

## Running locally

```bash
# 1. Start Ollama and pull the model
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:0.15.6
docker exec -it ollama ollama pull llama3.2:latest

# 2. Set up the OCR service
cd ocr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the API
uvicorn ocr.src.main.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# Test it
curl http://localhost:8000/status
curl -X POST "http://localhost:8000/process/invoice/" -F "file=@/path/to/invoice.pdf"
curl -X POST "http://localhost:8000/process/income/expenses/invoice/" -F "file=@/path/to/receipt.png"
```

## Project layout

```
documind/
├── ocr/                    # FastAPI + OCR + LLM client service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/main/
│       ├── main.py                     # App, endpoints, lifespan, warmup loop
│       ├── invoice_processor/          # Generic Ollama LLM processor
│       ├── model/                      # Pydantic schemas (invoice + expense)
│       ├── text_extractor/             # PDF / image / DOCX / XLSX extractors
│       └── image_processing_utils/     # Table/column layout analysis
└── ollama/                 # Ollama + Llama 3.2 service (model pre-baked)
```

---

*Built as a portfolio demonstration of an end-to-end AI document-processing system.*
