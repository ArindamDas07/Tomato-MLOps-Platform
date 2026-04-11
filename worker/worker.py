from worker.celery_app import celery_app
from worker.models import TomatoModelSuite
from worker.utils import get_raw_array, preprocess, calculate_drift
from worker.metrics import log_inference_result, log_gatekeeper_result, push_metrics
import random
import time
import socket
import shutil
from pathlib import Path
# Senior Tip: Import RESULT_TTL from our shared redis config
from shared.redis_conn import redis_client, RESULT_TTL 
from loguru import logger
import numpy as np
from dotenv import load_dotenv
import json
import os

# -------------------- Initialization --------------------
load_dotenv()
CELERY_MAX_RETRIES = int(os.getenv("CELERY_MAX_RETRIES", 3))

# Using a standard list for labels - consistent across all inference steps
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
    Uses pathlib for cross-platform compatibility and setex for memory safety.
    """
    start_time = time.time()
    worker_name = socket.gethostname()
    img_file = Path(image_path)
    
    # --- SENIOR MOVE: Defensive Check ---
    # Ensure the file wasn't deleted by a Janitor or a previous failed task.
    if not img_file.exists():
        logger.error(f"Gatekeeper Aborted: {img_file} not found. Possible race condition.")
        return

    try:
        # Load model using the Singleton Suite
        model = TomatoModelSuite.load_model(model_name="gate_keeper")
        
        # Image processing
        arr = get_raw_array(str(img_file))
        image = preprocess(arr, model_name="gate_keeper")
        
        # Inference
        score = model.predict(image, verbose=0)
        latency = round(time.time() - start_time, 3)
        
        # Business Logic: Is it a tomato leaf? (Threshold 0.7)
        status = 'tomato' if score[0][0] > 0.7 else 'invalid'
        
        # --- FIX: Memory Safety ---
        # Store result in Redis with a Time-To-Live (TTL) of 1 hour.
        # This prevents Redis from filling up forever.
        redis_client.setex(self.request.id, RESULT_TTL, status)
        
        # Observability
        log_gatekeeper_result(user_id, status, latency)
        
        # --- EXIT POINT A: Immediate cleanup on rejection ---
        if status == 'invalid':
            user_folder = img_file.parent
            if user_folder.exists():
                shutil.rmtree(user_folder)
                logger.info(f"♻️ Gatekeeper: {user_id} rejected. Storage purged.")

        push_metrics(worker_name)
        logger.info(f"Gatekeeper complete for {user_id} | Result: {status}")

    except Exception as e:
        logger.exception(f"Critical error in gatekeeper for user {user_id}")
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
    Stage 2: Performs disease classification with A/B testing and Drift Detection.
    Guarantees cleanup of the shared volume workspace upon completion.
    """
    start_time = time.time()
    worker_name = socket.gethostname()
    img_file = Path(image_path)
    user_folder = img_file.parent

    # Defensive Check
    if not img_file.exists():
        logger.error(f"Classifier Aborted: {img_file} missing.")
        return

    try:
        # A/B Testing Logic (70/30 split)
        model_name = "efficient" if random.random() < 0.7 else "resnet"
        model = TomatoModelSuite.load_model(model_name=model_name)
        
        # Drift Detection (Data Quality Check)
        arr = get_raw_array(str(img_file))
        stats, data_drift = calculate_drift(arr)
        
        # Inference
        image = preprocess(arr, model_name=model_name)
        score = model.predict(image, verbose=0)
        
        index = int(np.argmax(score))
        confidence = float(np.max(score))
        disease = CLASS_LABELS[index]
        
        latency = round(time.time() - start_time, 3)

        # Unified Observability (MLflow & Prometheus)
        log_inference_result(
            user_id=user_id, 
            model_name=model_name, 
            label=disease, 
            conf=confidence, 
            stats=stats, 
            drift=data_drift, 
            latency=latency
        )
        
        # Create JSON response for the Frontend
        result_payload = json.dumps({
            "disease": disease, 
            "confidence": round(confidence * 100, 2), 
            "model": model_name
        })
        
        # --- FIX: Memory Safety ---
        # Set result in Redis with 1 hour expiration
        redis_client.setex(self.request.id, RESULT_TTL, result_payload)
        
        push_metrics(worker_name)
        logger.success(f"Classification delivered for {user_id} | Model: {model_name}")

    except Exception as e:
        logger.exception(f"Classification failed for {user_id}")
        raise e
    
    finally:
        # --- SENIOR MOVE: The "Guarantee" Cleanup ---
        # This block runs even if the model crashes. 
        # Prevents storage leaks on the shared Kubernetes Volume.
        if user_folder.exists():
            shutil.rmtree(user_folder)
            logger.info(f"♻️ Workspace cleared for user: {user_id}")