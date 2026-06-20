import re
from pydantic import BaseModel, model_validator


class InvoiceData(BaseModel):
    invoiceNumber: str = ""
    invoiceDate: str = ""
    invoiceDueDate: str = ""
    totalInvoiceAmount: str = ""
    totalVatAmount: str = ""
    supplierName: str = ""
    bankAccountSortCode: str = ""
    bankAccountNumber: str = ""

    @model_validator(mode="before")
    def default_invalid_fields(cls, values: dict) -> dict:
        # Ensure all values are strings
        for field in cls.model_fields:
            val = values.get(field)
            if not isinstance(val, str):
                values[field] = str(val) if val is not None else ""

        # Validate date fields with DD/MM/YYYY format
        date_pattern = re.compile(r"^\d{2}/\d{2}/\d{4}$")
        for date_field in ["invoiceDate", "invoiceDueDate"]:
            if not date_pattern.match(values.get(date_field, "")):
                values[date_field] = ""

        return values
