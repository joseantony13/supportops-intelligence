# Docstring: says these tests only check the API's contract (status codes, response shape)
# The note about downloading applies to tests that trigger the model; none of these three do
"""Lightweight API contract tests; the first model call may download the model."""
# TestClient lets you call the FastAPI app in-process, with no uvicorn and no network
from fastapi.testclient import TestClient
# Import the FastAPI app object defined in backend/main.py
from backend.main import app

# Create one test client wrapping the app; every test reuses it
client = TestClient(app)

# pytest collects any function whose name starts with "test_"
def test_health():
    # Send GET /health to the app and capture the response
    response = client.get("/health")
    # Fail the test unless the HTTP status code is 200 (OK)
    assert response.status_code == 200
    # Fail the test unless the JSON body has "status" equal to "ok"
    assert response.json()["status"] == "ok"

# Second test: checks the sample-tickets endpoint
def test_sample_tickets():
    # Send GET /tickets/sample
    response = client.get("/tickets/sample")
    # Must return 200 (OK)
    assert response.status_code == 200
    # The "tickets" list must contain at least 5 items (the app has 8, so this passes)
    assert len(response.json()["tickets"]) >= 5

# Third test: checks that invalid input is rejected
def test_ticket_validation():
    # POST a body whose description is empty (""), which breaks the min_length=3 rule in TicketRequest
    response = client.post("/tickets/analyze", json={"subject": "x", "description": ""})
    # FastAPI/Pydantic should reject it automatically with 422 (Unprocessable Entity)
    assert response.status_code == 422
