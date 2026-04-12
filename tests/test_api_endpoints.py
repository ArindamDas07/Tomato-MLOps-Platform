import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app

# We use the FastAPI TestClient to simulate browser requests
client = TestClient(app)

def test_read_main():
    """Tests if the homepage loads correctly."""
    response = client.get("/")
    assert response.status_code == 200

def test_upload_rejects_non_image():
    """Ensures the API blocks non-image files (Security/Validation)."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only images allowed"

@patch("api.main.celery_app.send_task")
def test_upload_success_logic(mock_celery):
    """
    Tests the full upload flow using a 'Mock' for Celery.
    This proves the API creates a user_id and tasks correctly.
    """
    # 1. Setup the mock to look like a real Celery task
    mock_task = MagicMock()
    mock_task.id = "fake-task-uuid"
    mock_celery.return_value = mock_task
    
    # 2. Simulate a valid JPG upload
    file_data = {"file": ("leaf.jpg", b"fake-binary-data", "image/jpeg")}
    response = client.post("/upload", files=file_data)
    
    # 3. Assertions
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    assert data["task_id"] == "fake-task-uuid"
    # Ensure celery was actually called
    mock_celery.assert_called_once()