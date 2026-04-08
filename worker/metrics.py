from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Histogram, push_to_gateway, Gauge
import mlflow
import os
from dotenv import load_dotenv

# -------------------- Initialization --------------------
# Load environment variables for infrastructure endpoints
load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")
PUSHGATEWAY = os.getenv("PUSHGATEWAY", "http://pushgateway:9091")

# -------------------- MLflow Setup --------------------
# Centralized logic to handle concurrent experiment initialization across multiple workers
mlflow.set_tracking_uri(MLFLOW_URI)
experiment_name = "Tomato Leaf Disease Detector"

def setup_monitoring(name):
    """
    Safely retrieves or creates the MLflow experiment ID.
    Handles race conditions during horizontal scaling.
    """
    try:
        exp = mlflow.get_experiment_by_name(name)
        if exp:
            return exp.experiment_id
        return mlflow.create_experiment(name)
    except Exception:
        # Fallback if another worker created it simultaneously
        return mlflow.get_experiment_by_name(name).experiment_id

# Global Experiment ID for this worker session
EXPERIMENT_ID = setup_monitoring(experiment_name)

# -------------------- Prometheus Setup --------------------
# Using a custom Registry to isolate our application metrics
registry = CollectorRegistry()

# Counter for tracking the ingestion funnel (Total uploads)
TOMATO_REQUESTS = Counter("tomato_uploads_total", "Total images uploaded", registry=registry)

# Counter with labels to track Gatekeeper filtering performance
TOMATO_GATEKEEPER = Counter(
    "tomato_gatekeeper_checks_total", 
    "Total Gatekeeper passes/fails", 
    ['status'], 
    registry=registry
)

# Multi-dimensional Counter to track A/B testing results and disease distribution
TOMATO_DISEASE = Counter(
    'tomato_disease_predictions_total', 
    "Total successful predictions", 
    ['disease', 'model_variant'], 
    registry=registry
)

# Labeled Histogram for high-resolution performance monitoring (P95/P99 latency)
TOMATO_LATENCY = Histogram(
    "tomato_inference_latency_seconds", 
    "Time taken for model inference", 
    ['model_variant'], 
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0], 
    registry=registry
)

# Multi-dimensional Gauge for tracking data drift vs training baseline
DRIFT_GAUGE = Gauge(
    "tomato_drift_percentage", 
    "Deviation from training baseline", 
    ["drift_type"], 
    registry=registry
)

# -------------------- Helper Functions --------------------

def update_drift(drift_result: dict):
    """Updates all 5 drift dimensions (Brightness, Contrast, RGB) in Prometheus."""
    for key, value in drift_result.items():
        DRIFT_GAUGE.labels(drift_type=key).set(value)

def push_metrics(worker_name: str):
    """
    Pushes the local registry to the Pushgateway.
    Uses worker_instance as a grouping key to support horizontal scaling.
    """
    try:
        push_to_gateway(
            PUSHGATEWAY, 
            job='tomato_worker_pipeline', 
            grouping_key={'worker_instance': worker_name}, 
            registry=registry
        )
    except Exception as e:
        logger.error(f"Could not push to gateway: {e}")

# -------------------- Unified Observability Reporters --------------------

def log_inference_result(user_id, model_name, label, conf, stats, drift, latency):
    """
    Atomic reporter for the classification stage.
    Synchronizes Infrastructure telemetry (Prometheus) with Model telemetry (MLflow).
    """
    # 1. Update Prometheus internal state
    TOMATO_DISEASE.labels(disease=label, model_variant=model_name).inc()
    TOMATO_LATENCY.labels(model_variant=model_name).observe(latency)
    update_drift(drift)
    
    # 2. Record full trace to MLflow
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=f"user_{user_id}"):
        mlflow.set_tag("user_id", user_id)
        mlflow.log_param("model_used", model_name)
        mlflow.log_param("prediction", label)
        
        mlflow.log_metric("confidence", conf)
        mlflow.log_metric("latency_seconds", latency)
        
        # Iterative logging for dynamic metadata (Drift and Stats)
        for key, value in stats.items():
            mlflow.log_metric(key, value)
        for key, value in drift.items():
            mlflow.log_metric(key, value)
            
    logger.success(f"Inference logged for User {user_id} | Model: {model_name} | Latency: {latency}s")

def log_gatekeeper_result(user_id, status, latency):
    """
    Specifically logs the Gatekeeper's performance and filter rate.
    """
    # Increment total ingress counter
    TOMATO_REQUESTS.inc()
    
    # Update status and performance metrics
    TOMATO_GATEKEEPER.labels(status=status).inc()
    TOMATO_LATENCY.labels(model_variant="gatekeeper").observe(latency)
    
    # Log gatekeeper performance to MLflow for pipeline auditing
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=f"gk_{user_id}"):
        mlflow.set_tag("user_id", user_id)
        mlflow.log_param("step", "gatekeeper")
        mlflow.log_param("gatekeeper_status", status)
        mlflow.log_metric("latency", latency)

    logger.info(f"Gatekeeper finished for {user_id} | Status: {status} | Latency: {latency}s")