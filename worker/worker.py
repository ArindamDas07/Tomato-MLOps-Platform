from worker.celery_app import celery_app
from worker.models import TomatoModelSuite
from worker.utils import get_raw_array, preprocess, calculate_drift
from worker.metrics import log_inference_result, log_gatekeeper_result, push_metrics
import random
import time
import socket
import shutil
from pathlib import Path
from app.redis_conn import redis_client 
from loguru import logger
import numpy as np
import os 
from dotenv import load_dotenv
import json

# -------------------- Initialization --------------------
load_dotenv()
CELERY_MAX_RETRIES = int(os.getenv("CELERY_MAX_RETRIES", 3))

# -------------------- Class Labels --------------------
CLASS_LABELS = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
    "Septoria_leaf_spot", "Spider_mites", "Target_Spot",
    "Yellow_Leaf_Curl_Virus", "Mosaic_virus", "Healthy"
]

# -------------------- Task 1: Gatekeeper --------------------
@celery_app.task(
    bind=True,
    name="worker.gatekeeper",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": CELERY_MAX_RETRIES}
)
def task_gatekeeper(self, user_id: str, image_path: str):
    """
    Stage 1: Validates if the input image contains a tomato leaf.
    Cleans up the directory immediately if the image is rejected.
    """
    start_time = time.time()
    worker_name = socket.gethostname()
    
    try:
        model = TomatoModelSuite.load_model(model_name="gate_keeper")
        
        arr = get_raw_array(image_path)
        image = preprocess(arr, model_name="gate_keeper")
        
        score = model.predict(image, verbose=0)
        latency = round(time.time() - start_time, 3)
        
        status = 'tomato' if score[0][0] > 0.7 else 'invalid'
        
        # Log and update Redis
        log_gatekeeper_result(user_id, status, latency)
        redis_client.set(self.request.id, status)
        
        # --- EXIT POINT A: Cleanup on rejection ---
        if status == 'invalid':
            user_folder = Path(image_path).parent
            if user_folder.exists():
                shutil.rmtree(user_folder)
                logger.info(f"♻️ Workspace purged after gatekeeper rejection: {user_id}")

        push_metrics(worker_name)
        logger.info(f"Gatekeeper complete for {user_id} | Result: {status}")

    except Exception as e:
        logger.exception(f"Critical error in gatekeeper for user {user_id}")
        # Even on error, we might want to keep the file for debugging, 
        # but in production, we usually delete it or move it to an 'error' folder.
        raise e

# -------------------- Task 2: Classifier --------------------
@celery_app.task(
    bind=True,
    name="worker.classifier",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": CELERY_MAX_RETRIES}
)
def task_classifier(self, user_id: str, image_path: str):
    """
    Stage 2: Performs classification with A/B testing and Drift Detection.
    Guarantees cleanup of the shared volume workspace upon completion.
    """
    start_time = time.time()
    worker_name = socket.gethostname()
    user_folder = Path(image_path).parent # Identify the folder to delete later

    try:
        # A/B Testing Logic (70/30 split)
        model_name = "efficient" if random.random() < 0.7 else "resnet"
        model = TomatoModelSuite.load_model(model_name=model_name)
        
        arr = get_raw_array(image_path)
        stats, data_drift = calculate_drift(arr)
        
        image = preprocess(arr, model_name=model_name)
        score = model.predict(image, verbose=0)
        
        index = int(np.argmax(score))
        confidence = float(np.max(score))
        disease = CLASS_LABELS[index]
        confidence_pct = round(confidence * 100, 2)
        
        latency = round(time.time() - start_time, 3)

        # Telemetry
        log_inference_result(
            user_id=user_id, 
            model_name=model_name, 
            label=disease, 
            conf=confidence, 
            stats=stats, 
            drift=data_drift, 
            latency=latency
        )
        
        result_json = json.dumps({"disease": disease, "confidence": confidence_pct, "model": model_name})
        redis_client.set(self.request.id, result_json)
        
        push_metrics(worker_name)
        logger.success(f"Classification successful for {user_id} | Model: {model_name}")

    except Exception as e:
        logger.exception(f"Critical error in classifier for user {user_id}")
        raise e
    
    finally:
        # --- EXIT POINT B: Final Cleanup ---
        # This block runs NO MATTER WHAT. If the code above succeeds OR crashes,
        # the folder is deleted to prevent storage leaks.
        if user_folder.exists():
            shutil.rmtree(user_folder)
            logger.info(f"♻️ Final workspace cleanup complete for user: {user_id}")