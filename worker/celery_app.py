import os
from dotenv import load_dotenv
from celery import Celery

# Load environment variables for infrastructure configuration
load_dotenv()

# --- Infrastructure Config ---
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

# DB 0 is the message broker (pending tasks)
REDIS_DB_TASKS = os.getenv("REDIS_DB_TASKS", "0")
# DB 1 is the Celery internal backend (task state tracking)
REDIS_DB_RESULTS = os.getenv("REDIS_DB_RESULTS", "1")

# --- Celery App Initialization ---
# Standardized App ID: 'tomato_app' to match the Producer (main.py)
celery_app = Celery(
    "tomato_app",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_TASKS}",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_RESULTS}",
    # 'include' ensures the worker auto-registers the tasks in worker/worker.py
    include=["worker.worker"]
)

# --- Optional Optimization Settings ---
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Security: Only accept JSON payloads
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)