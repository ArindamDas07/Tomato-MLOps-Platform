from worker.celery_app import celery_app
from worker.models import TomatoModelSuite
from worker.utils import get_raw_array, preprocess, calculate_drift
from worker.metrics import log_inference_result, log_gatekeeper_result, push_metrics

# Senior Fix: Import from shared package
from shared.redis_conn import redis_client, RESULT_TTL 
from shared.schemas import InferenceResult

import random
import time
import socket
import shutil
from pathlib import Path
from loguru import logger
import numpy as np
from dotenv import load_dotenv
import json
import os

# -------------------- Initialization --------------------
load_dotenv()
CELERY_MAX_RETRIES = int(os.getenv("CELERY_MAX_RETRIES", 3))

# Standard labels used for mapping model output indices
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
    Validates if the image is a tomato leaf.
    Cleans up immediately ONLY on rejection.
    """
    start_time = time.time()
    worker_name = socket.gethostname()
    img_file = Path(image_path)
    
    if not img_file.exists():
        logger.error(f"Gatekeeper Aborted: {img_file} missing. Possible storage sync issue.")
        return

    try:
        model = TomatoModelSuite.load_model(model_name="gate_keeper")
        
        arr = get_raw_array(str(img_file))
        image = preprocess(arr, model_name="gate_keeper")
        
        score = model.predict(image, verbose=0)
        latency = round(time.time() - start_time, 3)
        
        # Binary Classification Logic
        status = 'tomato' if score[0][0] > 0.7 else 'invalid'
        
        # Store simple string status in Redis
        redis_client.setex(self.request.id, RESULT_TTL, status)
        
        log_gatekeeper_result(user_id, status, latency)
        
        # Cleanup ONLY if rejected. If valid, we keep the file for Task 2.
        if status == 'invalid':
            user_folder = img_file.parent
            if user_folder.exists():
                shutil.rmtree(user_folder)
                logger.info(f"♻️ Gatekeeper Reject: {user_id} purged from storage.")

        push_metrics(worker_name)
        logger.info(f"Gatekeeper finished for {user_id} | Result: {status}")

    except Exception as e:
        logger.exception(f"Gatekeeper process failure for user {user_id}")
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
    Performs Inference with A/B Testing.
    Contract-based result delivery. No disk cleanup (API-owned).
    """
    start_time = time.time()
    worker_name = socket.gethostname()
    img_file = Path(image_path)

    if not img_file.exists():
        logger.error(f"Classifier Aborted: {img_file} missing. Cannot retry.")
        return

    try:
        # A/B Testing Logic
        model_name = "efficient" if random.random() < 0.7 else "resnet"
        model = TomatoModelSuite.load_model(model_name=model_name)
        
        # Data Drift Calculation
        arr = get_raw_array(str(img_file))
        stats, data_drift = calculate_drift(arr)
        
        # Inference
        image = preprocess(arr, model_name=model_name)
        score = model.predict(image, verbose=0)
        
        index = int(np.argmax(score))
        confidence = float(np.max(score))
        disease_label = CLASS_LABELS[index]
        
        # --- SENIOR MOVE: Contract Enforcement ---
        # Construct result using the Shared Pydantic Schema
        result_obj = InferenceResult(
            disease=disease_label,
            confidence=round(confidence * 100, 2),
            model=model_name
        )
        
        latency = round(time.time() - start_time, 3)

        # Telemetry
        log_inference_result(
            user_id=user_id, 
            model_name=model_name, 
            label=disease_label, 
            conf=confidence, 
            stats=stats, 
            drift=data_drift, 
            latency=latency
        )
        
        # Set result in Redis using the model's built-in JSON export
        redis_client.setex(self.request.id, RESULT_TTL, result_obj.model_dump_json())
        
        push_metrics(worker_name)
        logger.success(f"Classification successful for {user_id}")

    except Exception as e:
        logger.exception(f"Inference failure for {user_id}")
        raise e
    
    # NOTE: The 'finally' cleanup block is REMOVED. 
    # The API's /result endpoint now owns image deletion to ensure 100% safety.