"""Application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_analysis import router as analysis_router
from services.config import load_cors_allowed_origins

app = FastAPI(title="NLP Analysis Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=load_cors_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/v1")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple healthcheck endpoint."""

    return {"status": "ok"}
