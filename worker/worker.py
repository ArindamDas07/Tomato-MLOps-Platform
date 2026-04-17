from worker.celery_app import celery_app
from worker.models import TomatoModelSuite
from worker.utils import get_raw_array, preprocess, calculate_drift
from worker.metrics import log_inference_result, log_gatekeeper_result, push_metrics
from shared.redis_conn import redis_client, RESULT_TTL 
from shared.schemas import InferenceResult
import hashlib  
import time
import socket
import shutil
from pathlib import Path
from loguru import logger
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
CELERY_MAX_RETRIES = int(os.getenv("CELERY_MAX_RETRIES", 3))

CLASS_LABELS = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
    "Septoria_leaf_spot", "Spider_mites", "Target_Spot",
    "Yellow_Leaf_Curl_Virus", "Mosaic_virus", "Healthy"
]

@celery_app.task(
    bind=True,
    name="worker.gatekeeper",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": CELERY_MAX_RETRIES}
)
def task_gatekeeper(self, user_id: str, image_path: str):
    start_time = time.time()
    worker_name = socket.gethostname()
    img_file = Path(image_path)
    
    if not img_file.exists():
        logger.error(f"Gatekeeper Aborted: {img_file} missing.")
        return

    try:
        model = TomatoModelSuite.load_model(model_name="gate_keeper")
        arr = get_raw_array(str(img_file))
        image = preprocess(arr, model_name="gate_keeper")
        score = model.predict(image, verbose=0)
        latency = round(time.time() - start_time, 3)
        status = 'tomato' if score[0][0] > 0.7 else 'invalid'
        
        redis_client.setex(self.request.id, RESULT_TTL, status)
        log_gatekeeper_result(user_id, status, latency)
        
        if status == 'invalid':
            user_folder = img_file.parent
            if user_folder.exists():
                shutil.rmtree(user_folder)
        
        push_metrics(worker_name)
    except Exception as e:
        logger.exception(f"Gatekeeper failure for user {user_id}")
        raise e

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
    Inference task with deterministic A/B testing via User ID hashing.
    """
    start_time = time.time()
    worker_name = socket.gethostname()
    img_file = Path(image_path)

    if not img_file.exists():
        logger.error(f"Classifier Aborted: {img_file} missing.")
        return

    try:
        #  Deterministic Hashing ---
        # Instead of random, we hash the user_id to get a consistent assignment.
        # EfficientNet: 70% | ResNet: 30%
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        model_name = "efficient" if user_hash < 70 else "resnet"
        
        model = TomatoModelSuite.load_model(model_name=model_name)
        
        arr = get_raw_array(str(img_file))
        stats, data_drift = calculate_drift(arr)
        image = preprocess(arr, model_name=model_name)
        score = model.predict(image, verbose=0)
        
        index = int(np.argmax(score))
        disease_label = CLASS_LABELS[index]
        confidence = float(np.max(score))
        
        result_obj = InferenceResult(
            disease=disease_label,
            confidence=round(confidence * 100, 2),
            model=model_name
        )
        
        latency = round(time.time() - start_time, 3)
        log_inference_result(user_id, model_name, disease_label, confidence, stats, data_drift, latency)
        
        # Consistent with Phase 1 Fix: Saving JSON-serialized schema
        redis_client.setex(self.request.id, RESULT_TTL, result_obj.model_dump_json())
        
        push_metrics(worker_name)
        logger.success(f"Task complete: User {user_id} -> {model_name}")

    except Exception as e:
        logger.exception(f"Inference failure for {user_id}")
        raise e