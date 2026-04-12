from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Set environment to testing to skip real DB pings
os.environ["ENV"] = "testing"

# Senior Move: Mock redis_client BEFORE importing app
with patch("shared.redis_conn.redis_client") as mock_redis:
    from api.main import app
    client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_upload_rejects_non_image():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400

@patch("api.main.celery_app.send_task")
def test_upload_success_logic(mock_celery):
    mock_task = MagicMock()
    mock_task.id = "fake-task-uuid"
    mock_celery.return_value = mock_task
    
    file_data = {"file": ("leaf.jpg", b"fake-data", "image/jpeg")}
    # Use a mock for the redis_client.setex if needed, but here we only check API logic
    response = client.post("/upload", files=file_data)
    
    assert response.status_code == 200
    assert "user_id" in response.json()