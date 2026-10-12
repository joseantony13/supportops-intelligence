"""Lightweight API contract tests; the first model call may download the model."""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_sample_tickets():
    response = client.get("/tickets/sample")
    assert response.status_code == 200
    assert len(response.json()["tickets"]) >= 5

def test_ticket_validation():
    response = client.post("/tickets/analyze", json={"subject": "x", "description": ""})
    assert response.status_code == 422
