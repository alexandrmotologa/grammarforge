"""OpenAI-compatible reverse proxy gateway with transparent GBNF grammar injection and stream healing."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from grammarforge.compiler.json_schema import compile_json_schema
from grammarforge.core.config import CompilerConfig, WhitespaceMode
from grammarforge.engine.repair import heal_partial_json


def create_gateway_proxy(upstream_url: str = "http://127.0.0.1:11434") -> FastAPI:
    """Create a FastAPI application acting as an OpenAI-compatible constrained decoding proxy."""
    app = FastAPI(
        title="GrammarForge Gateway Proxy",
        description="Transparent OpenAI-compatible reverse proxy with GBNF grammar enforcement.",
        version="0.1.0",
    )

    clean_upstream = upstream_url.rstrip("/")

    @app.api_route("/v1/models", methods=["GET"])
    async def proxy_models(request: Request) -> Response:
        """Transparently proxy model listing."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"{clean_upstream}/v1/models")
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"),
                )
            except Exception as err:
                raise HTTPException(status_code=502, detail=f"Failed to connect to upstream: {err}")

    @app.api_route("/v1/chat/completions", methods=["POST"])
    async def proxy_chat_completions(request: Request) -> Response:
        """Intercept chat completions, compile JSON Schema to GBNF, and heal truncated outputs."""
        try:
            body: Dict[str, Any] = await request.json()
        except Exception as err:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {err}")

        # Extract schema from response_format if present
        schema: Optional[Dict[str, Any]] = None
        resp_fmt = body.get("response_format")
        if isinstance(resp_fmt, dict):
            if "json_schema" in resp_fmt and isinstance(resp_fmt["json_schema"], dict):
                schema = resp_fmt["json_schema"].get("schema")
            elif "schema" in resp_fmt and isinstance(resp_fmt["schema"], dict):
                schema = resp_fmt["schema"]

        # If tools provided, generate tool call schema
        tools = body.get("tools")
        if not schema and isinstance(tools, list) and tools:
            # First tool schema
            first_tool = tools[0]
            if isinstance(first_tool, dict) and "function" in first_tool:
                schema = first_tool["function"].get("parameters")

        # Compile schema to GBNF if detected
        if schema:
            gbnf_str = compile_json_schema(schema, config=CompilerConfig(whitespace_mode=WhitespaceMode.LENIENT))
            # Inject native grammar parameters for llama.cpp / vLLM / Ollama
            body["grammar"] = gbnf_str
            body["guided_grammar"] = gbnf_str

        is_streaming = body.get("stream", False)

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                upstream_resp = await client.post(
                    f"{clean_upstream}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
            except Exception as err:
                raise HTTPException(status_code=502, detail=f"Upstream communication error: {err}")

            if not is_streaming:
                try:
                    resp_data = upstream_resp.json()
                    # Check if response content was truncated and needs healing
                    if schema and "choices" in resp_data and resp_data["choices"]:
                        choice = resp_data["choices"][0]
                        msg = choice.get("message", {})
                        raw_content = msg.get("content", "")
                        if raw_content:
                            repair_res = heal_partial_json(raw_content, schema=schema)
                            if repair_res.suffix_added:
                                msg["content"] = repair_res.repaired_text
                                msg["_grammarforge_healed"] = True
                    return JSONResponse(content=resp_data, status_code=upstream_resp.status_code)
                except Exception:
                    return Response(content=upstream_resp.content, status_code=upstream_resp.status_code)

            # Streaming response
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
            )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_all(request: Request, path: str) -> Response:
        """Catch-all reverse proxy route."""
        url = f"{clean_upstream}/{path}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                body = await request.body()
                resp = await client.request(
                    method=request.method,
                    url=url,
                    headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                    content=body,
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"),
                )
            except Exception as err:
                raise HTTPException(status_code=502, detail=f"Proxy error: {err}")

    return app
