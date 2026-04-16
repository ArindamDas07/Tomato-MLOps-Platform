from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
import aiofiles
from loguru import logger
import uuid
import shutil
from pathlib import Path
from celery import Celery
from contextlib import asynccontextmanager
import os
import json
import sys

# Contract Imports
from shared.redis_conn import redis_client, check_redis_health
from shared.schemas import InferenceResult, TaskResponse 

# -------------------- Config --------------------
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB_TASKS = os.getenv("REDIS_DB_TASKS", "0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
BASE_DIR = Path(__file__).resolve().parent
MAX_FILE_SIZE = 10 * 1024 * 1024 # 10MB in bytes
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup logic and directory creation."""
    logger.info(f"🚀 Starting Tomato API Gateway (Storage: {UPLOAD_DIR})")
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Critical: Upload directory not writable: {e}")

    if os.getenv("ENV") != "testing":
        if not check_redis_health():
            logger.error("Could not establish Redis connection. Exiting.")
            sys.exit(1)
    yield
    logger.info("🛑 Shutting down...")

app = FastAPI(title="Tomato MLOps Gateway", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

celery_app = Celery("tomato_app", broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_TASKS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/")
async def read_index(request: Request):
    # Senior Fix: Correct argument order (request must be first)
    return templates.TemplateResponse(request, "index.html")

# --- THE CRITICAL FIX: Add the missing Health Check ---
@app.get("/health")
async def health_check():
    """Endpoint for Kubernetes Liveness/Readiness probes."""
    return {"status": "healthy", "redis": check_redis_health()}

@app.post('/upload')
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only images allowed")
    
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"Image too large. Maximum allowed size is 10MB. Got {file.size / (1024*1024):.2f}MB")
    
    user_id = str(uuid.uuid4())
    user_folder = UPLOAD_DIR / user_id
    
    try:
        user_folder.mkdir(parents=True, exist_ok=True)
        file_path = user_folder / "image.jpg"
        content = await file.read()
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        task = celery_app.send_task("worker.gatekeeper", args=[user_id, str(file_path)])
        return {"user_id": user_id, "task_id": task.id}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if user_folder.exists():
            shutil.rmtree(user_folder)
        raise HTTPException(status_code=500, detail="Upload failed")

@app.get('/leaf_checker/{user_id}/{task_id}', response_model=TaskResponse)
async def check_leaf(user_id: str, task_id: str):
    try:
        result = redis_client.get(task_id)
        if result == "tomato":
            return TaskResponse(status="done", valid=True)
        elif result == "invalid":
            user_folder = UPLOAD_DIR / user_id
            if user_folder.exists():
                shutil.rmtree(user_folder)
            return TaskResponse(status="done", valid=False, message="Not a tomato leaf")
        return TaskResponse(status="processing")
    except Exception:
        return TaskResponse(status="error", message="Redis error")

@app.post('/predict/{user_id}')
async def trigger_prediction(user_id: str):
    file_path = UPLOAD_DIR / user_id / "image.jpg"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Session expired")
    try:
        task = celery_app.send_task("worker.classifier", args=[user_id, str(file_path)])
        return {"task_id": task.id}
    except Exception:
        raise HTTPException(status_code=503, detail="Worker overloaded")

@app.get('/result/{user_id}/{task_id}', response_model=TaskResponse)
async def get_final_result(user_id: str, task_id: str):
    try:
        raw_result = redis_client.get(task_id)
        if raw_result:
            # Result from Redis is a JSON string, convert to dict
            prediction_data = json.loads(raw_result)
            # Validate dict into InferenceResult object
            validated_inference = InferenceResult(**prediction_data)
            
            user_folder = UPLOAD_DIR / user_id
            if user_folder.exists():
                shutil.rmtree(user_folder)
            
            return TaskResponse(status="done", prediction=validated_inference)
            
        return TaskResponse(status="processing")
    except Exception as e:
        logger.error(f"Data parsing error: {e}")
        return TaskResponse(status="error", message="Processing result failed")