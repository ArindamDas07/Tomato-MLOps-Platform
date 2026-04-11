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
    
    # If Redis is still starting up, don't crash. Wait and try again.
    broker_connection_retry_on_startup=True,
    
    # How many times to try reconnecting if the broker drops
    broker_connection_max_retries=10,

    # --- RELIABILITY (You already had these, keeping them) ---
    # Task is only deleted from Redis AFTER it successfully finishes
    task_acks_late=True, 
    # If the worker crashes, the task is returned to the queue automatically
    task_reject_on_worker_lost=True,
    
    # Optimization: Don't let one worker hog all tasks. 
    # This helps with Auto-scaling (HPA).
    worker_prefetch_multiplier=1 
)