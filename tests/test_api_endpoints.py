from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Setup environment for CI
os.environ["ENV"] = "testing"
# Pointing UPLOAD_DIR to a local relative path that is always writable in CI
os.environ["UPLOAD_DIR"] = "test_uploads"

with patch("shared.redis_conn.redis_client") as mock_redis:
    from api.main import app
    client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_upload_rejects_non_image():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400

@patch("api.main.celery_app.send_task")
def test_upload_success_logic(mock_celery, tmp_path):
    """
    Uses pytest's tmp_path to ensure the test has a writable folder
    in the GitHub runner environment.
    """
    # 1. Setup mock
    mock_task = MagicMock()
    mock_task.id = "fake-task-uuid"
    mock_celery.return_value = mock_task
    
    # 2. Force the API to use the temporary folder provided by pytest
    with patch("api.main.UPLOAD_DIR", tmp_path):
        file_data = {"file": ("leaf.jpg", b"fake-data", "image/jpeg")}
        response = client.post("/upload", files=file_data)
        
        # 3. Validation
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "task_id" in data
        mock_celery.assert_called_once()