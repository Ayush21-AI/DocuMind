"""
Bundled example documents for the offline demo.

Each sample carries the raw document text (as the OCR/parse stage would emit it)
and a cached model output (what Llama 3.2 returns for it). The demo feeds these
through the real classifier / validators / confidence scorer, so it works with
no LLM while still exercising the production code paths. `SAMPLES[2]` is crafted
so the grounding check catches an unsupported value and flags it for review.
"""

SAMPLES = [
    {
        "name": "Supplier invoice (clean)",
        "text": (
            "ACME CORP LTD\n"
            "INVOICE\n"
            "Invoice Number: INV-2024-001\n"
            "Invoice Date: 15/03/2024\n"
            "Due Date: 30/03/2024\n"
            "Description                 Qty   Amount\n"
            "Consulting services          10   1250.50\n"
            "VAT @ 20%                          300.10\n"
            "Total Due: £1,500.50\n"
            "Sort Code: 20-15-82   Account Number: 12345678\n"
        ),
        "model_output": {
            "invoiceNumber": "INV-2024-001",
            "invoiceDate": "15/03/2024",
            "invoiceDueDate": "30/03/2024",
            "totalInvoiceAmount": "£1,500.50",
            "totalVatAmount": "300.10",
            "supplierName": "ACME CORP LTD",
            "bankAccountSortCode": "201582",
            "bankAccountNumber": "12345678",
        },
    },
    {
        "name": "Till receipt (US-dated, USD)",
        "text": (
            "TESCO EXPRESS\n"
            "RECEIPT\n"
            "Order Date: 10/17/2020\n"
            "Groceries\n"
            "Card Payment  Contactless\n"
            "TOTAL  $28,040.00 USD\n"
            "Change Due 0.00\n"
            "Thank you for shopping with us\n"
        ),
        "model_output": {
            "invoiceDate": "10/17/2020",
            "transactionAmount": "$28,040.00 USD",
            "description": "TESCO EXPRESS - Groceries",
        },
    },
    {
        "name": "Low-quality scan (flagged for review)",
        "text": (
            "▓▓ R3CE!PT ▓▓\n"
            "░░░░░░░░░░░░\n"
            "T0T@L  ?? ???\n"
            "blurred / unreadable\n"
        ),
        # OCR yielded noise; the model couldn't ground anything, so it returns
        # empties. DocuMind flags the result for review (overall confidence 0)
        # and reports low routing confidence — it doesn't invent values.
        "model_output": {
            "invoiceNumber": "",
            "invoiceDate": "",
            "invoiceDueDate": "",
            "totalInvoiceAmount": "",
            "totalVatAmount": "",
            "supplierName": "",
            "bankAccountSortCode": "",
            "bankAccountNumber": "",
        },
    },
]
