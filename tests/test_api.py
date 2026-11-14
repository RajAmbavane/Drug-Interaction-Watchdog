"""Tests for the FastAPI app: health, auth guard, CORS, and error handling."""

import sys
import types

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_current_user
from api.main import app

FAKE_USER = {"patient_id": "patient-123", "email": "test@example.com"}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def authed_client():
    app.dependency_overrides[get_current_user] = lambda: FAKE_USER
    yield TestClient(app)
    app.dependency_overrides.clear()


# ── Public endpoints ─────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_points_to_docs(client):
    assert client.get("/").json()["docs"] == "/docs"


# ── Auth guard ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "method, path, body",
    [
        ("post", "/analyse", {"drug_list": ["warfarin", "aspirin"]}),
        ("post", "/intake/image", {"image_b64": "abc"}),
        ("get", "/patients/me", None),
    ],
)
def test_protected_routes_require_token(client, method, path, body):
    resp = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
    assert resp.status_code == 401


# ── CORS ─────────────────────────────────────────────────────────────────────

def _preflight(client, origin):
    return client.options(
        "/analyse",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


def test_cors_allows_configured_origin(client):
    resp = _preflight(client, "http://localhost:3000")
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_unknown_origin(client):
    resp = _preflight(client, "https://evil.example.com")
    assert resp.headers.get("access-control-allow-origin") is None


# ── Error handling ───────────────────────────────────────────────────────────

def _install_fake(monkeypatch, module_name, **attrs):
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)


def test_analyse_hides_internal_errors(authed_client, monkeypatch):
    def broken_orchestrator():
        raise RuntimeError("password=hunter2 at db.internal:5432")

    _install_fake(monkeypatch, "agents", __path__=[])
    _install_fake(monkeypatch, "agents.orchestrator", get_orchestrator=broken_orchestrator)

    resp = authed_client.post("/analyse", json={"drug_list": ["warfarin", "aspirin"]})
    assert resp.status_code == 500
    assert "hunter2" not in resp.text
    assert "db.internal" not in resp.text


def test_intake_hides_internal_errors(authed_client, monkeypatch):
    class BrokenAgent:
        def intake_from_b64(self, *_):
            raise RuntimeError("GROQ_API_KEY=sk-secret rejected")

    _install_fake(monkeypatch, "agents", __path__=[])
    _install_fake(monkeypatch, "agents.vision_intake_agent", VisionIntakeAgent=BrokenAgent)

    resp = authed_client.post("/intake/image", json={"image_b64": "abc"})
    assert resp.status_code == 500
    assert "sk-secret" not in resp.text


def test_analyse_validates_request_body(authed_client):
    resp = authed_client.post("/analyse", json={"drugs": "warfarin"})
    assert resp.status_code == 422
