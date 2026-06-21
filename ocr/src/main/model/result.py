from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    """
    Rich, self-describing extraction envelope returned by all endpoints.

    Beyond the raw fields, it surfaces *how much to trust them*: per-field and
    overall confidence, a human-review flag, the detected document type and
    currency, and processing metadata. This is what lets a consumer build an
    auto-accept / needs-review / manual-entry workflow on top of DocuMind.
    """

    document_type: str = Field(description="'invoice' or 'expense'")
    fields: dict[str, str] = Field(description="Extracted, normalised field values")
    confidence: dict[str, float] = Field(description="Per-field confidence in [0, 1]")
    overall_confidence: float = Field(description="Mean confidence over populated fields")
    review_required: bool = Field(description="True if a human should review the result")
    currency: str = Field("GBP", description="Detected ISO currency code")
    meta: dict = Field(default_factory=dict, description="Processing metadata (timings, model, cache)")
