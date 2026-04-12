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

# --- Contract Imports ---
from shared.redis_conn import redis_client, check_redis_health
from shared.schemas import InferenceResult 

# -------------------- Config --------------------
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB_TASKS = os.getenv("REDIS_DB_TASKS", "0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
UPLOAD_DIR = Path("/app/uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup/shutdown. In production, we verify DB connectivity here.
    """
    logger.info("🚀 Starting Tomato API Gateway...")
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Only perform the fatal health check if we aren't in a test environment
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

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/upload')
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only images allowed")
    
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
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
    except Exception:
        if user_folder.exists():
            shutil.rmtree(user_folder)
        raise HTTPException(status_code=500, detail="Upload failed")

@app.get('/leaf_checker/{user_id}/{task_id}')
async def check_leaf(user_id: str, task_id: str):
    try:
        result = redis_client.get(task_id)
        if result:
            if result == "tomato":
                return {"status": "done", "valid": True}
            else:
                user_folder = UPLOAD_DIR / user_id
                if user_folder.exists():
                    shutil.rmtree(user_folder)
                return {"status": "done", "valid": False, "message": "Not a tomato leaf"}
        return {"status": "processing"}
    except Exception:
        return {"status": "error", "message": "Redis error"}

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

@app.get('/result/{user_id}/{task_id}')
async def get_final_result(user_id: str, task_id: str):
    try:
        raw_result = redis_client.get(task_id)
        if raw_result:
            prediction_json = json.loads(raw_result)
            validated_result = InferenceResult(**prediction_json)
            user_folder = UPLOAD_DIR / user_id
            if user_folder.exists():
                shutil.rmtree(user_folder)
            return {"status": "done", "prediction": validated_result}
        return {"status": "processing"}
    except Exception:
        return {"status": "error", "message": "Parsing error"}