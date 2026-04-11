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

# Import our robust redis client and the TTL value we defined
from shared.redis_conn import redis_client, RESULT_TTL

# -------------------- Config & Paths --------------------
# Senior Tip: We use environment variables for EVERYTHING so we can change 
# behavior in K8s without touching code.
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB_TASKS = os.getenv("REDIS_DB_TASKS", "0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path("/app/uploads")

# -------------------- Lifespan Management --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI pattern to handle startup and shutdown logic.
    This ensures the upload directory exists BEFORE the first request arrives.
    """
    logger.info("🚀 Starting Tomato API Gateway...")
    UPLOAD_DIR.mkdir(exist_ok=True)
    yield
    logger.info("🛑 Shutting down Tomato API Gateway...")

# -------------------- Initialization --------------------
app = FastAPI(title="Tomato MLOps Gateway", lifespan=lifespan)

# Prometheus Metrics
Instrumentator().instrument(app).expose(app)

# Celery Initialization
celery_app = Celery("tomato_app", broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_TASKS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, # Professional: Restricted via ConfigMap
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        
        # Defensive Programming: Wrap Celery calls in try-except
        try:
            task = celery_app.send_task("worker.gatekeeper", args=[user_id, str(file_path)])
        except Exception as celery_err:
            logger.error(f"Celery Broker Error: {celery_err}")
            raise HTTPException(status_code=503, detail="Task queue is currently unavailable")
            
        logger.info(f"User {user_id} | Upload successful | TaskID: {task.id}")
        return {"user_id": user_id, "task_id": task.id}

    except HTTPException:
        raise
    except Exception as e:
        if user_folder.exists():
            shutil.rmtree(user_folder)
        logger.exception(f"Upload failed for user {user_id}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")

# --- STEP 2: GATEKEEPER POLLING ---
@app.get('/leaf_checker/{user_id}/{task_id}')
async def check_leaf(user_id: str, task_id: str):
    try:
        result = redis_client.get(task_id)
        if result:
            if result == "tomato":
                return {"status": "done", "valid": True}
            else:
                # Cleanup logic remains, but we add logging
                if (UPLOAD_DIR / user_id).exists():
                    shutil.rmtree(UPLOAD_DIR / user_id)
                logger.warning(f"User {user_id} | Rejected by Gatekeeper")
                return {"status": "done", "valid": False, "message": "Not a tomato leaf"}
        
        return {"status": "processing"}
    except Exception:
        logger.exception(f"Redis polling error for {user_id}")
        return {"status": "error", "message": "Database connection flickering"}

# --- STEP 3: PREDICT ---
@app.post('/predict/{user_id}')
async def trigger_prediction(user_id: str):
    file_path = UPLOAD_DIR / user_id / "image.jpg"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Session expired. Please upload again.")

    try:
        task = celery_app.send_task("worker.classifier", args=[user_id, str(file_path)])
        return {"task_id": task.id}
    except Exception:
        logger.error(f"Failed to queue classifier for {user_id}")
        raise HTTPException(status_code=503, detail="Worker service overloaded")

# --- STEP 4: FINAL RESULT ---
@app.get('/result/{user_id}/{task_id}')
async def get_final_result(user_id: str, task_id: str):
    try:
        result = redis_client.get(task_id)
        if result:
            # The result is found! 
            # Senior Move: The Worker will actually set the result with an EXPIRE (TTL)
            # so we don't need to manually delete it from Redis here.
            # But we MUST clean up the disk folder.
            user_folder = UPLOAD_DIR / user_id
            if user_folder.exists():
                shutil.rmtree(user_folder)
            
            return {"status": "done", "prediction": result}
            
        return {"status": "processing"}
    except Exception:
        return {"status": "error", "message": "Result retrieval failed"}