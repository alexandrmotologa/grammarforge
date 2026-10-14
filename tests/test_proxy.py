"""Unit tests for OpenAI-compatible gateway proxy."""

import pytest
from fastapi.testclient import TestClient
from grammarforge.gateway.proxy import create_gateway_proxy


def test_proxy_app_creation() -> None:
    app = create_gateway_proxy("http://127.0.0.1:11434")
    client = TestClient(app)

    # Missing body / bad request
    res = client.post("/v1/chat/completions", content="invalid-json")
    assert res.status_code == 400


def test_proxy_route_registration() -> None:
    app = create_gateway_proxy("http://127.0.0.1:11434")
    routes = [route.path for route in app.routes]
    assert "/v1/chat/completions" in routes
    assert "/v1/models" in routes
