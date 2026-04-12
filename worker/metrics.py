from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Histogram, push_to_gateway, Gauge
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

# These endpoints are provided by your k8s/tomato-config.yaml
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")
PUSHGATEWAY = os.getenv("PUSHGATEWAY", "http://pushgateway-service:9091")

# -------------------- MLflow Setup --------------------
mlflow.set_tracking_uri(MLFLOW_URI)
EXPERIMENT_NAME = "Tomato Leaf Disease Detector"
_cached_experiment_id = None 

def get_experiment_id():
    global _cached_experiment_id
    if _cached_experiment_id:
        return _cached_experiment_id

    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp:
            _cached_experiment_id = exp.experiment_id
        else:
            _cached_experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        return _cached_experiment_id
    except Exception as e:
        logger.warning(f"📡 MLflow not reachable: {e}. AI will continue without logging.")
        return None

# -------------------- Prometheus Setup --------------------
registry = CollectorRegistry()
TOMATO_REQUESTS = Counter("tomato_uploads_total", "Total images uploaded", registry=registry)
TOMATO_GATEKEEPER = Counter("tomato_gatekeeper_checks_total", "Total Gatekeeper passes/fails", ['status'], registry=registry)
TOMATO_DISEASE = Counter('tomato_disease_predictions_total', "Total successful predictions", ['disease', 'model_variant'], registry=registry)
TOMATO_LATENCY = Histogram("tomato_inference_latency_seconds", "Time taken for model inference", ['model_variant'], buckets=[0.1, 0.5, 1.0, 2.0, 5.0], registry=registry)
DRIFT_GAUGE = Gauge("tomato_drift_percentage", "Deviation from training baseline", ["drift_type"], registry=registry)

def update_drift(drift_result: dict):
    for key, value in drift_result.items():
        DRIFT_GAUGE.labels(drift_type=key).set(value)

def push_metrics(worker_name: str):
    """
    Senior Move: Non-blocking telemetry.
    In an auto-scaled environment, the Pushgateway might be under load.
    We set a 3-second timeout so the AI worker never hangs.
    """
    try:
        push_to_gateway(
            PUSHGATEWAY, 
            job='tomato_worker_pipeline', 
            grouping_key={'worker_instance': worker_name}, 
            registry=registry,
            timeout=3 # Crucial for scaling stability
        )
    except Exception as e:
        # We log the error but do NOT raise it. 
        # The User's prediction is more important than the dashboard.
        logger.error(f"📊 Telemetry push deferred: {e}")

def log_inference_result(user_id, model_name, label, conf, stats, drift, latency):
    # 1. Update local metrics
    TOMATO_DISEASE.labels(disease=label, model_variant=model_name).inc()
    TOMATO_LATENCY.labels(model_variant=model_name).observe(latency)
    update_drift(drift)
    
    # 2. Log to MLflow if available
    exp_id = get_experiment_id()
    if exp_id:
        try:
            with mlflow.start_run(experiment_id=exp_id, run_name=f"user_{user_id}"):
                mlflow.set_tag("user_id", user_id)
                mlflow.log_param("model_used", model_name)
                mlflow.log_param("prediction", label)
                mlflow.log_metric("confidence", conf)
                mlflow.log_metric("latency_seconds", latency)
                for key, value in stats.items():
                    mlflow.log_metric(key, value)
                for key, value in drift.items():
                    mlflow.log_metric(key, value)
        except Exception as e:
            logger.error(f"❌ MLflow logging failed: {e}")
            
    logger.success(f"✅ Metrics updated for user {user_id}")

def log_gatekeeper_result(user_id, status, latency):
    TOMATO_REQUESTS.inc()
    TOMATO_GATEKEEPER.labels(status=status).inc()
    TOMATO_LATENCY.labels(model_variant="gatekeeper").observe(latency)
    
    exp_id = get_experiment_id()
    if exp_id:
        try:
            with mlflow.start_run(experiment_id=exp_id, run_name=f"gk_{user_id}"):
                mlflow.set_tag("user_id", user_id)
                mlflow.log_param("step", "gatekeeper")
                mlflow.log_param("status", status)
                mlflow.log_metric("latency", latency)
        except Exception as e:
            logger.error(f"❌ MLflow Gatekeeper logging failed: {e}")