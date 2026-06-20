import re
from pydantic import BaseModel, model_validator


class ExpenseData(BaseModel):
    invoiceDate: str = ""
    transactionAmount: str = ""
    description: str = ""

    @model_validator(mode="before")
    def default_invalid_fields(cls, values: dict) -> dict:
        # Ensure all values are strings
        for field in cls.model_fields:
            val = values.get(field)
            if not isinstance(val, str):
                values[field] = str(val) if val is not None else ""

        # Validate and normalise invoiceDate to DD/MM/YYYY
        date_str = values.get("invoiceDate", "").strip()
        date_pattern = re.compile(r"^\d{2}/\d{2}/\d{4}$")
        if date_pattern.match(date_str):
            p1, p2, yyyy = date_str.split("/")
            d, m = int(p1), int(p2)
            if m > 12 <= d:
                # MM/DD/YYYY detected — swap to DD/MM/YYYY
                date_str = f"{p2}/{p1}/{yyyy}"
            elif not (1 <= d <= 31 and 1 <= m <= 12):
                date_str = ""
            values["invoiceDate"] = date_str
        else:
            values["invoiceDate"] = ""

        # Ensure transactionAmount always ends with GBP
        amount = values.get("transactionAmount", "").strip()
        if amount:
            # Strip any trailing currency text and replace with GBP
            amount = re.sub(r'\s*[A-Za-z]+$', '', amount).strip()
            values["transactionAmount"] = f"{amount} GBP"

        return values
