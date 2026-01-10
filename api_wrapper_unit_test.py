"""Unit tests for the FastAPI `api_wrapper` application.

Includes a test that posts a harmless prompt to the `/check_prompt` endpoint
and validates the response schema and types.
"""

from fastapi.testclient import TestClient
from api_wrapper import app  # your FastAPI app file

client = TestClient(app)
# Create a test client for the FastAPI application

def test_check_prompt_safe():
    """Send a harmless prompt to `/check_prompt` and assert response structure.

    Verifies status code is 200 and response contains `prompt`, `safe` (bool),
    `score` (float), and `category` (str).
    """
    print("Run Unit Test")
    # A harmless prompt used to verify the endpoint returns a safe classification
    payload = {"prompt": "Enter your prompt here for testing!"}
    response = client.post("/check_prompt", json=payload)

    # First, print status and response for debugging
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

    assert response.status_code == 200
    
    # Parse the JSON payload and validate expected fields/types
    data = response.json()
    assert data["prompt"] == payload["prompt"]
    assert isinstance(data["safe"], bool)
    assert isinstance(data["score"], float)
    assert isinstance(data["category"], str)

test_check_prompt_safe()