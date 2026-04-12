from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
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
    assert response.json()["detail"] == "Only images allowed"

@patch("api.main.celery_app.send_task")
def test_upload_success_logic(mock_celery):
    mock_task = MagicMock()
    mock_task.id = "fake-task-uuid"
    mock_celery.return_value = mock_task
    
    file_data = {"file": ("leaf.jpg", b"fake-binary-data", "image/jpeg")}
    response = client.post("/upload", files=file_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    assert data["task_id"] == "fake-task-uuid"
    mock_celery.assert_called_once()