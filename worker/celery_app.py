import os
from dotenv import load_dotenv
from celery import Celery
from loguru import logger

# Load environment variables for infrastructure configuration
load_dotenv()

# --- Infrastructure Config ---
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

# DB 0: The Message Broker (Where tasks wait in line)
REDIS_DB_TASKS = os.getenv("REDIS_DB_TASKS", "0")
# DB 1: Celery Internal Backend (Celery's internal tracking: PENDING/SUCCESS)
# Note: Our actual App results are saved to DB 2 via shared/redis_conn.py
REDIS_DB_RESULTS = os.getenv("REDIS_DB_RESULTS", "1")

# --- Celery App Initialization ---
celery_app = Celery(
    "tomato_app",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_TASKS}",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_RESULTS}",
    include=["worker.worker"]
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # --- SENIOR RESILIENCE SETTINGS ---
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,

    # --- RELIABILITY ---
    task_acks_late=True, 
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    
    # --- NEW SENIOR MOVE: ZOMBIE PREVENTION ---
    # Soft limit: Raises an exception in the worker so it can clean up gracefully
    task_soft_time_limit=60, 
    # Hard limit: Kills the worker process if it completely freezes
    task_time_limit=75,
    
    # Tracks when a task actually starts processing (useful for monitoring latency)
    task_track_started=True
)

logger.info("⚙️ Celery App initialized with Time Limits and Late Acks.")