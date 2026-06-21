from pydantic import BaseModel, Field, model_validator

from ocr.src.main.confidence import AMOUNT, DATE, TEXT
from ocr.src.main.utils.currency import normalize_amount
from ocr.src.main.utils.dates import normalize_date


class ExpenseData(BaseModel):
    """3-field receipt/expense schema (date, amount, description)."""

    invoiceDate: str = Field(default="", description="Transaction/receipt date, normalised to DD/MM/YYYY")
    transactionAmount: str = Field(default="", description="Overall total paid, e.g. 28040.00 GBP")
    description: str = Field(default="", description="Short description: merchant and/or what was purchased")

    # Populated by the validator; surfaced by the API but not asked of the LLM.
    detectedCurrency: str = Field(default="GBP", description="Detected ISO currency code", exclude=False)

    @staticmethod
    def field_kinds() -> dict[str, str]:
        return {
            "invoiceDate": DATE,
            "transactionAmount": AMOUNT,
            "description": TEXT,
        }

    @model_validator(mode="before")
    def normalise(cls, values: dict) -> dict:
        for field in ("invoiceDate", "transactionAmount", "description"):
            val = values.get(field)
            if not isinstance(val, str):
                values[field] = str(val) if val is not None else ""

        # Robust date handling (US ordering, spelled-out months, ISO, etc.).
        values["invoiceDate"] = normalize_date(values.get("invoiceDate", ""))

        # Currency-aware amount: strips symbols/commas, enforces two decimals,
        # keeps a detected non-GBP currency instead of forcing GBP blindly.
        formatted, currency, ok = normalize_amount(values.get("transactionAmount", ""))
        values["transactionAmount"] = formatted if ok else ""
        values["detectedCurrency"] = currency

        return values
