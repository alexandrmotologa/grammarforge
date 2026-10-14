"""FastAPI server configuration and static mount for GrammarForge Studio."""

from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from grammarforge.studio.routes import router as api_router

static_dir = os.path.join(os.path.dirname(__file__), "static")
index_html_path = os.path.join(static_dir, "index.html")

app = FastAPI(
    title="GrammarForge Studio",
    description="Interactive Web Studio for GBNF Constrained Decoding and Token Masking.",
    version="0.1.0",
)

# Include API endpoints
app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the single page application."""
    return FileResponse(index_html_path)


# Mount static assets
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
