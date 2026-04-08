import numpy as np
from PIL import Image
from io import BytesIO

# Import the specific preprocessing functions with aliased names for clarity
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# --- DRIFT DETECTION BASELINES ---
# These values are derived from 18,339 training images and act as the "Source of Truth"
TRAINING_BASELINES = {
    "brightness": 116.536,
    "contrast": 44.831,
    "r_channel": 118.486,
    "g_channel": 121.477,
    "b_channel": 109.645
}

# --- IMAGE PROCESSING UTILITIES ---

def get_raw_array(image_path: str):
    """
    Decodes an image from a file path into a standardized NumPy array (224x224x3).
    This raw array (pixels 0-255) is the input for drift detection.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    return np.array(image)

def preprocess(arr: np.ndarray, model_name: str):
    """
    Applies model-specific preprocessing to a NumPy array.
    Uses .copy() to prevent in-place modification and data corruption.
    """
    # Defensive copy to prevent in-place modifications by Keras functions
    arr = arr.copy()
    
    if model_name == "gate_keeper":
        preprocessed_arr = mobilenet_preprocess(arr)
    elif model_name == 'resnet':
        preprocessed_arr = resnet_preprocess(arr)
    elif model_name == 'efficient':
        preprocessed_arr = eff_preprocess(arr)  
    else:
        raise ValueError(f"Unknown model name for preprocessing: {model_name}")
    
    # Keras models expect a batch dimension (1, height, width, channels)
    return np.expand_dims(preprocessed_arr, axis=0)

def calculate_drift(arr: np.ndarray):
    """
    Calculates both absolute stats and percentage drift from training baselines.
    This provides actionable telemetry for MLflow and Prometheus.
    """
    # 1. Calculate the statistics for the live image
    live_stats = {
        "brightness": np.mean(arr),
        "contrast": np.std(arr),
        "r_channel": np.mean(arr[:, :, 0]),
        "g_channel": np.mean(arr[:, :, 1]),
        "b_channel": np.mean(arr[:, :, 2])
    }
    
    # 2. Calculate the percentage drift from the training baseline
    drift_percentages = {}
    for key, baseline_value in TRAINING_BASELINES.items():
        live_value = live_stats[key]
        drift = ((live_value - baseline_value) / baseline_value) * 100
        drift_percentages[f"drift_{key}_pct"] = drift
    
    return live_stats, drift_percentages