from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import aiofiles
from loguru import logger
import uuid
import shutil
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from app.redis_conn import redis_client
from celery import Celery
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------- Config --------------------
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB_TASKS = os.getenv("REDIS_DB_TASKS", "0")
UPLOAD_DIR = Path('uploads')
UPLOAD_DIR.mkdir(exist_ok=True)

# -------------------- Initialization --------------------
app = FastAPI(title="Tomato MLOps Gateway")
Instrumentator().instrument(app).expose(app)

celery_app = Celery("tomato_app", broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_TASKS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STEP 1: UPLOAD ---
@app.post('/upload')
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only images allowed")
    
    user_id = str(uuid.uuid4())
    user_folder = UPLOAD_DIR / user_id
    
    try:
        user_folder.mkdir(parents=True, exist_ok=True)
        file_path = user_folder / "image.jpg"

        content = await file.read()
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # PRO MOVE: Wrap task triggering in a try-except to catch Redis connection issues
        task = celery_app.send_task("worker.gatekeeper", args=[user_id, str(file_path)])
        
        logger.info(f"User {user_id} | Upload successful | TaskID: {task.id}")
        return {"user_id": user_id, "task_id": task.id}

    except Exception as e:
        # PRO MOVE: If anything fails during upload, clean up the folder immediately
        if user_folder.exists():
            shutil.rmtree(user_folder)
        logger.exception(f"Upload failed for user {user_id}") # Note the .exception()
        raise HTTPException(status_code=500, detail="Failed to process upload")

# --- STEP 2: GATEKEEPER POLLING ---
@app.get('/leaf_checker/{user_id}/{task_id}')
async def check_leaf(user_id: str, task_id: str):
    try:
        result = redis_client.get(task_id)
        if result:
            if result == "tomato":
                return {"status": "done", "valid": True}
            else:
                # PRO MOVE: Use a utility for cleanup to keep route clean
                if (UPLOAD_DIR / user_id).exists():
                    shutil.rmtree(UPLOAD_DIR / user_id)
                return {"status": "done", "valid": False, "message": "Not a tomato leaf"}
        
        return {"status": "processing"}
    except Exception:
        logger.exception(f"Error polling gatekeeper for {user_id}")
        return {"status": "error", "message": "Redis connection failure"}

# --- STEP 3: PREDICT ---
@app.post('/predict/{user_id}')
async def trigger_prediction(user_id: str):
    file_path = UPLOAD_DIR / user_id / "image.jpg"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Session expired or file not found")

    try:
        task = celery_app.send_task("worker.classifier", args=[user_id, str(file_path)])
        logger.info(f"User {user_id} | Prediction started | TaskID: {task.id}")
        return {"task_id": task.id}
    except Exception:
        logger.exception(f"Failed to queue classifier for {user_id}")
        raise HTTPException(status_code=500, detail="Worker service unavailable")

# --- STEP 4: FINAL RESULT ---
@app.get('/result/{user_id}/{task_id}')
async def get_final_result(user_id: str, task_id: str):
    try:
        result = redis_client.get(task_id)
        if result:
            # Atomic cleanup
            user_folder = UPLOAD_DIR / user_id
            if user_folder.exists():
                shutil.rmtree(user_folder)
            
            logger.info(f"User {user_id} | Prediction delivered | Folder purged")
            return {"status": "done", "prediction": result}
            
        return {"status": "processing"}
    except Exception:
        logger.exception(f"Error retrieving result for {task_id}")
        return {"status": "error", "message": "Internal error"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")