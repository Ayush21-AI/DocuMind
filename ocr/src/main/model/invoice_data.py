import re

from pydantic import BaseModel, Field, model_validator

from ocr.src.main.confidence import ACCOUNT, AMOUNT, DATE, SORTCODE, TEXT
from ocr.src.main.utils.currency import normalize_amount
from ocr.src.main.utils.dates import normalize_date

_SORTCODE_DIGITS = re.compile(r"^\d{6}$")


def _normalize_sortcode(value: str) -> str:
    """Coerce UK sort codes to NN-NN-NN; drop anything that isn't 6 digits."""
    digits = re.sub(r"\D", "", value or "")
    if _SORTCODE_DIGITS.match(digits):
        return f"{digits[0:2]}-{digits[2:4]}-{digits[4:6]}"
    return value if re.match(r"^\d{2}-\d{2}-\d{2}$", value or "") else ""


class InvoiceData(BaseModel):
    """8-field supplier-invoice schema. Field descriptions are sent to the LLM
    as part of the JSON Schema (constrained decoding) to ground extraction."""

    invoiceNumber: str = Field(default="", description="Unique invoice/reference number, e.g. INV-2024-001")
    invoiceDate: str = Field(default="", description="Issue date, normalised to DD/MM/YYYY")
    invoiceDueDate: str = Field(default="", description="Payment due date, normalised to DD/MM/YYYY")
    totalInvoiceAmount: str = Field(default="", description="Overall total payable, e.g. 1500.50 GBP")
    totalVatAmount: str = Field(default="", description="Total VAT/tax amount, e.g. 300.10 GBP")
    supplierName: str = Field(default="", description="Name of the company that issued the invoice")
    bankAccountSortCode: str = Field(default="", description="UK sort code in NN-NN-NN format")
    bankAccountNumber: str = Field(default="", description="Supplier bank account number")

    @staticmethod
    def field_kinds() -> dict[str, str]:
        return {
            "invoiceNumber": TEXT,
            "invoiceDate": DATE,
            "invoiceDueDate": DATE,
            "totalInvoiceAmount": AMOUNT,
            "totalVatAmount": AMOUNT,
            "supplierName": TEXT,
            "bankAccountSortCode": SORTCODE,
            "bankAccountNumber": ACCOUNT,
        }

    @model_validator(mode="before")
    def normalise(cls, values: dict) -> dict:
        # Coerce everything to strings first.
        for field in ("invoiceNumber", "invoiceDate", "invoiceDueDate",
                      "totalInvoiceAmount", "totalVatAmount", "supplierName",
                      "bankAccountSortCode", "bankAccountNumber"):
            val = values.get(field)
            if not isinstance(val, str):
                values[field] = str(val) if val is not None else ""

        # Dates -> DD/MM/YYYY (or "").
        for date_field in ("invoiceDate", "invoiceDueDate"):
            values[date_field] = normalize_date(values.get(date_field, ""))

        # Amounts -> "X.XX GBP" (commas stripped, two decimals enforced).
        for amount_field in ("totalInvoiceAmount", "totalVatAmount"):
            formatted, _, ok = normalize_amount(values.get(amount_field, ""))
            values[amount_field] = formatted if ok else ""

        # Sort code -> NN-NN-NN.
        values["bankAccountSortCode"] = _normalize_sortcode(values.get("bankAccountSortCode", ""))

        return values
