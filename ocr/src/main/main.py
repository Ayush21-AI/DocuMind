import io
import os
import asyncio
import httpx
import logging
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import ORJSONResponse
from ocr.src.main.invoice_processor.ollama_processor import OllamaProcessor
from ocr.src.main.model.invoice_data import InvoiceData
from ocr.src.main.model.expense_data import ExpenseData
from ocr.src.main.text_extractor.extract_text_from_docx import extract_text_from_docx
from ocr.src.main.text_extractor.extract_text_from_image import extract_text_from_images
from ocr.src.main.text_extractor.extract_text_from_pdf import extract_text_from_pdf
from ocr.src.main.text_extractor.extract_text_from_xlsx import extract_text_from_xlsx

# ------------------ Environment Config -------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:latest")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
NUM_CORES = int(os.getenv("NUM_CORES", os.cpu_count() or 2))
max_invoice_pdf_pages = int(os.getenv("MAX_INVOICE_PDF_PAGES", 15))
if max_invoice_pdf_pages <= 0:
    logging.warning("MAX_INVOICE_PDF_PAGES must be positive. Defaulting to 15.")
    max_invoice_pdf_pages = 15

# System Prompt
DEFAULT_SYSTEM_PROMPT = """\
You are an expert system for extracting and validating key fields from merchant invoices. You will receive raw OCR or parsed text from invoice documents.
Your task is to extract exactly 8 specific fields from the input text. These fields may appear with various label variants or across different lines. 
You must match labels using their common variants, handle whitespace or newline-separated values, and return only values explicitly found in the input.
Do not guess, hallucinate or generate any values that are not explicitly present in the text.

Extract the following fields:

1. **invoiceNumber**:  
   - The unique reference number for the invoice.
   - Look for labels like: "Invoice Number", "Billing Number", "Invoice #", "Ref #", "Reference #", "Document #", or "Bill #" etc.
   - Extract the invoice number following the label, even if it appears on the next line or after whitespace.

2. **invoiceDate**:  
   - The date when the invoice was issued.
   - Look for labels like: "Date", "Invoice Date", "Date Issued", or "Invoiced On" etc.
   - Normalize all date formats to: `dd/MM/yyyy`
   - For example, convert "December 31, 2024" to "31/12/2024".
   - If the date is in a different format (e.g., "31 December, 2024"), convert it to the format "31/12/2024".

3. **invoiceDueDate**:  
   - The deadline date by which an invoice must be paid.
   - Look for labels like: "Invoice Due Date", "Due Date", "Payment Due By", "Payment Due Date", "Settlement Date", "Due On", "Due By" etc.
   - Normalize all date formats to: `dd/MM/yyyy`
   - If the date is in a different format (e.g., "December 31, 2024"), convert it to the format "31/12/2024".

4. **totalInvoiceAmount**:  
   - The total amount to be paid.
   - Look for labels like: "Total Due", "Total Amount", "Total", "Amount Due", or "Invoice Total" etc.
   - Extract only the overall total amount (not line item totals) 
   - Do not confuse individual line item totals with the overall total.
   - Format: "X.XX GBP". Remove if any "£" currency symbol and append append " GBP" as a suffix.
     For example, if the text reads "£0.50", your output should be "0.50 GBP".

5. **totalVatAmount**:  
   - The total VAT (tax) amount on the invoice.
   - Look for labels like: "VAT", "Total VAT", "VAT Amount", "Tax Amount", "Total Tax", or "Total VAT" etc.
   - Extract only the overall tax/VAT value, not line item taxes/vat.
   - Do not confuse individual line item totals with the overall total.
   - Format: "X.XX GBP". Remove if any "£" currency symbol and append append " GBP" as a suffix.
     For example, if the text reads "£0.50", your output should be "0.50 GBP".

6. **supplierName**:  
   - The name of the company or individual who issued the invoice.
   - Extract the most likely supplier name or company issuing the invoice.

7. **bankAccountSortCode**:  
   - UK Sort Code format: `NN-NN-NN`.
   - Look for these labels only: "Sort Code", "Bank Sort Code", "SC" not other than these.
   - If not found, return "".

8. **bankAccountNumber**:  
   - The supplier's bank account number.
   - Look for labels like: "Account Number", "Account #", "Acc No.", "Acc #", "Account No.", or "Acc Number".
   - If not found, return "".

**Rules:**  
- If a field is not found, return an empty string "" for that field.
- Your final output must be a valid JSON object with exactly the 8 keys specified below, and no additional text or commentary.
- Extract only the data that exists; Do not guess, hallucinate or generate any values that are not explicitly present in the text.
- Strictly adhere to the JSON format only.

### OUTPUT FORMAT:
Return ONLY a valid JSON object with exactly these 8 keys with values if available, no extra text at all:
```json
{
  "invoiceNumber": "",
  "invoiceDate": "",
  "invoiceDueDate": "",
  "totalInvoiceAmount": "",
  "totalVatAmount": "",
  "supplierName": "",
  "bankAccountSortCode": "",
  "bankAccountNumber": ""
}
"""
SYSTEM_PROMPT = os.getenv("SystemPrompt", DEFAULT_SYSTEM_PROMPT)

# Expense System Prompt (for 3-field extraction)
DEFAULT_EXPENSE_SYSTEM_PROMPT = """\
You are an expert information-extraction system for extracting and validating key fields from specialized in UK merchant invoices and receipts. 
You will receive raw input text may be noisy, unstructured, or OCR-generated or parsed text from any type of document (invoice, receipt, or transaction slip).
Your task is to read the whole input carefully and extract exactly 3 specific fields from the input text. These fields may appear with various label variants or across different lines. 
You must match labels using their common variants, handle whitespace or newline-separated values and values may appear on the same line, next line, or after whitespace, and return only values explicitly found in the input.

CRITICAL: Do not guess, hallucinate, infer, calculate or generate any values that are not explicitly present in the text.
If a value is missing, unclear, or not explicitly present → return an empty string "". 

Extract the following 3 fields:

1. **invoiceDate**:  
   - The date when the invoice/receipt was issued, Date of the invoice, receipt, or transaction.
   - Look for labels like: "Date", "Invoice Date", "Date Issued", "Order Date", "Receipt Date", "Issued On", "Purchase Date", "Transaction Date" or "Invoiced On" etc.
   - If multiple dates exist, choose the one most clearly associated with the transaction (label precedence above). Ignore metadata dates (printed on top, system date unless labeled).
   - Normalize all date formats strictly to DD/MM/YYYY (two-digit day/two-digit month/four-digit year)
   - NEVER return MM/DD/YYYY (US format). The day MUST come first: DD/MM/YYYY. For example, October 17 2020 → 17/10/2020, NOT 10/17/2020.
   - For example, see the conversations below:  
        - December 31, 2024 → 31/12/2024
        - 31 December 2024 → 31/12/2024
        - 2024-12-31 → 31/12/2024
        - 1/2/24 → 01/02/2024 (assume two-digit year maps to 2000+ if ambiguous)
        OCR errors: 120ctober 2073 → 12/10/2073 if clearly decipherable
   - If the date is in a different format (e.g., "31 December, 2024"), convert it to the format "31/12/2024" or from this format "120ctober 2073" to this format "12/10/2073"
   - If ambiguous between day/month (e.g., 03/04/2024) and context suggests UK format, interpret as DD/MM/YYYY.
   - If no valid/parsable date is present, return ""

2. **transactionAmount**:
   - The overall Total Amount or Final amount paid/payable for the transaction (NOT VAT or line-item totals).
   - Look for labels like: "Total", "Total Due", "Total Amount", "Amount", "Amount Due", "Grand Total", "Invoice Total", "Balance Due", "Total Paid", "Payment Amount" or "Amount Paid" etc.
   - MUST extract only the overall total or transaction amount (NOT VAT alone or line item totals).
   - If multiple totals found, prefer in order: Total Paid > Amount Due > Grand Total > Total.
   - Do not confuse individual line item totals with the overall total.
   - Remove any £ or $ or USD or GBP or any other currency symbol or name. 
   - ALWAYS append " GBP" as a suffix. Format strictly as: X.XX GBP (two decimal places).
   - IMPORTANT: Commas in numbers are THOUSAND separators, NOT decimals. Remove commas before formatting.
   - For example:
        - £0.50 → 0.50 GBP
        - £1,234 → 1234.00 GBP
        - $28,040 USD → 28040.00 GBP
        - Total: 12.5 → 12.50 GBP (assume GBP if no currency given)
        - £(1,234.50) → -1234.50 GBP (preserve negative if indicated).
   - If amount includes trailing words like USD, GBP, GBP total, pounds, normalize to X.XX GBP.
   - If the document explicitly uses another currency (e.g., USD, EUR), include that currency code instead (e.g., 10.00 USD) — otherwise assume GBP.
   - If no clear overall total found, return "".

3. **Description**:
   - A short textual description of the transaction — what was purchased / merchant + context.
   - Look for label fields: "Description", "Item", "Details", "Transaction Details", "Merchant", "Purchase", "Service", "Product" etc.
   - If an explicit field exists, take that short text (prefer the single most descriptive short line).
   - If no explicit description, use the clearest short merchant name or merchant name + purchase context (e.g., "TESCO - GROCERIES" or "Starbucks London - coffee"), but do NOT invent, summarize, or expand beyond text present.
   - Keep it short (preferably one short phrase or 2–8 words extracted exactly as present). Do not add punctuation or qualifiers.
   - If nothing usable is found, return "".
   - Do NOT generate or summarize beyond what is explicitly stated.
   
### OUTPUT FORMAT CONSTRAINTS (STRICT):
No additional fields, no comments, no explanation text.
Strings must obey the normalizations above (date format dd/MM/yyyy, amount format X.XX GBP).

Output exactly this JSON structure object with these EXACT 3 keys with their values if available (keys must match exactly):
```json
{
  "invoiceDate": "",
  "transactionAmount": "",
  "description": ""
}
"""
EXPENSE_SYSTEM_PROMPT = os.getenv("ExpenseSystemPrompt", DEFAULT_EXPENSE_SYSTEM_PROMPT)

# ------------------ Logging Config ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

# ------------------ Globals ----------------------------
shared_httpx_client: httpx.AsyncClient | None = None
shared_invoice_processor: OllamaProcessor | None = None
shared_expense_processor: OllamaProcessor | None = None

# ------------------ Lifespan Context Manager ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global shared_httpx_client, shared_invoice_processor, shared_expense_processor
    shared_httpx_client = httpx.AsyncClient(timeout=30.0)

    shared_invoice_processor = OllamaProcessor(
        model_name=MODEL_NAME,
        ollama_host=OLLAMA_HOST,
        system_prompt=SYSTEM_PROMPT,
        http_client=shared_httpx_client,
        response_model=InvoiceData
    )

    shared_expense_processor = OllamaProcessor(
        model_name=MODEL_NAME,
        ollama_host=OLLAMA_HOST,
        system_prompt=EXPENSE_SYSTEM_PROMPT,
        http_client=shared_httpx_client,
        response_model=ExpenseData,
        num_predict=150
    )

    logging.info("HTTP client and processors initialized.")
    logging.info(f"OCR Engine using model: {MODEL_NAME} at {OLLAMA_HOST}")

    warmup_task = asyncio.create_task(_llm_model_warmup_loop())
    yield
    warmup_task.cancel()
    await shared_httpx_client.aclose()
    logging.info("HTTP client closed.")

async def _llm_model_warmup_loop():
    """Send a lightweight ping to Ollama every 10s to keep the model hot."""
    await asyncio.sleep(10)  # initial delay for Ollama to be ready
    while True:
        try:
            await shared_httpx_client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": DEFAULT_EXPENSE_SYSTEM_PROMPT},
                        {"role": "user", "content": "ping"}
                    ],
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=30.0
            )
            logging.debug("Ollama warmup ping sent.")
        except Exception as e:
            logging.warning(f"Ollama warmup ping failed: {e}")
        await asyncio.sleep(10)


# ------------------ App Setup ----------------------------
app = FastAPI(
    title="Invoice Processing API for OCR",
    default_response_class=ORJSONResponse,
    lifespan=lifespan
)


# ------------------ Extractor Routing --------------------------
async def route_to_extractor(extension: str, stream: io.BytesIO) -> str:
    stream.seek(0)
    match extension:
        case ".pdf":
            return await extract_text_from_pdf(stream, max_invoice_pdf_pages)
        case ".png" | ".jpg" | ".jpeg":
            return await extract_text_from_images(stream)
        case ".docx":
            return await extract_text_from_docx(stream)
        case ".xlsx":
            return await extract_text_from_xlsx(stream)
        case _:
            raise ValueError("Unsupported file type. Only PDF, Images, DOCX, or XLSX allowed.")


# ------------------ Main Endpoint ------------------------------
@app.post(
    "/process/invoice/",
    response_model=InvoiceData,
    summary="Process an invoice file",
    response_description="Extracts structured invoice data using OCR and LLM",
    responses={
        200: {"description": "Successfully extracted invoice data"},
        400: {"description": "Invalid file or extraction failure"},
        500: {"description": "Unexpected server error"},
    }
)
async def process_invoice(file: UploadFile):
    filename = file.filename
    extension = Path(filename).suffix.lower()
    content_type = file.content_type

    logging.info(f"Processing file: {filename} (type: {content_type})")

    try:
        if extension not in [".pdf", ".png", ".jpg", ".jpeg", ".docx", ".xlsx"]:
            raise HTTPException(status_code=400, detail="Invalid file type.")

        file_content = await file.read()
        file_stream = io.BytesIO(file_content)

        extracted_text = await route_to_extractor(extension, file_stream)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from file")
        logging.info(f"Extracted text length: {len(extracted_text)}")
        logging.info(f"Extracted text data: \n {extracted_text}")

        assert shared_invoice_processor is not None, "Invoice processor is not initialized."
        result = await shared_invoice_processor.extract_data(extracted_text)
        logging.info(f"Extracted invoice data: {result.model_dump()}")
        return result

    except ValueError as ve:
        logging.error(f"Extractor failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("Unexpected error during invoice processing")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

    finally:
        await file.close()


# ------------------ Expense Endpoint ------------------------------
@app.post(
    "/process/income/expenses/invoice/",
    response_model=ExpenseData,
    summary="Process an expense document",
    response_description="Extracts 3 fields (invoiceDate, transactionAmount, Description) using OCR and LLM",
    responses={
        200: {"description": "Successfully extracted expense data"},
        400: {"description": "Invalid file or extraction failure"},
        500: {"description": "Unexpected server error"},
    }
)
async def process_expense(file: UploadFile):
    filename = file.filename
    extension = Path(filename).suffix.lower()
    content_type = file.content_type

    logging.info(f"Processing expense file: {filename} (type: {content_type})")

    try:
        if extension not in [".pdf", ".png", ".jpg", ".jpeg", ".docx", ".xlsx"]:
            raise HTTPException(status_code=400, detail="Invalid file type.")

        file_content = await file.read()
        file_stream = io.BytesIO(file_content)

        extracted_text = await route_to_extractor(extension, file_stream)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from file")
        logging.info(f"Extracted text length: {len(extracted_text)}")
        logging.info(f"Extracted text data: \n {extracted_text}")

        assert shared_expense_processor is not None, "Expense processor is not initialized."
        result = await shared_expense_processor.extract_data(extracted_text)
        logging.info(f"Extracted expense data: {result.model_dump()}")
        return result

    except ValueError as ve:
        logging.error(f"Extractor failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("Unexpected error during expense processing")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

    finally:
        await file.close()


# ------------------ Health Check ------------------------------
@app.get("/status")
async def health_check():
    return {
        "status": "ok", "message": "OCR Processor Service is up and ready to serve!" if shared_invoice_processor else "error",
        "ollama_host": OLLAMA_HOST,
        "model": MODEL_NAME,
        "cores": NUM_CORES,
        "initialized": bool(shared_invoice_processor)
    }


# --------------------------------------------------------------------------
# Launch Server if Run as Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        workers=2,
        loop="uvloop",
        reload=False
    )
