# app/api/granulate.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.granulate import granulate_text

router = APIRouter(prefix="/v1", tags=["granulate"])


class GranulateRequest(BaseModel):
    text: str
    taxonomy: dict[str, list[str]] | None = None
    top_k_evidence: int = Field(default=6, ge=1, le=12)
    min_similarity: float = Field(default=0.05, ge=0.0, le=1.0)


@router.post("/granulate")
def granulate(body: GranulateRequest):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        return granulate_text(
            text=text,
            aspects=body.taxonomy,
            top_k_evidence=body.top_k_evidence,
            min_similarity=body.min_similarity,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
