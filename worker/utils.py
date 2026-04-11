import numpy as np
from PIL import Image, UnidentifiedImageError
from loguru import logger

# Model-specific preprocessing imports
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# Baselines derived from the 18,339 training images
TRAINING_BASELINES = {
    "brightness": 116.536,
    "contrast": 44.831,
    "r_channel": 118.486,
    "g_channel": 121.477,
    "b_channel": 109.645
}

def get_raw_array(image_path: str):
    """
    Safely decodes an image. 
    Senior Tip: Always catch UnidentifiedImageError for user uploads.
    """
    try:
        # We use a context manager to ensure the file handle is closed
        with Image.open(image_path) as img:
            image = img.convert("RGB").resize((224, 224))
            return np.array(image)
    except UnidentifiedImageError:
        logger.error(f"❌ Corrupt image file provided: {image_path}")
        raise ValueError("Provided file is not a valid image.")
    except Exception as e:
        logger.error(f"❌ Unexpected image error: {e}")
        raise

def preprocess(arr: np.ndarray, model_name: str):
    """Applies specific normalization based on the model architecture."""
    # Work on a copy to ensure the raw array stays clean for drift detection
    temp_arr = arr.copy().astype('float32')
    
    if model_name == "gate_keeper":
        preprocessed = mobilenet_preprocess(temp_arr)
    elif model_name == 'resnet':
        preprocessed = resnet_preprocess(temp_arr)
    elif model_name == 'efficient':
        # EfficientNet usually expects [0, 255] or scaling depending on the version
        preprocessed = eff_preprocess(temp_arr)  
    else:
        raise ValueError(f"Unknown preprocessing variant: {model_name}")
    
    return np.expand_dims(preprocessed, axis=0)

def calculate_drift(arr: np.ndarray):
    """
    Calculates statistical deviation from the training set.
    Essential for MLOps monitoring.
    """
    # 1. Real-time stats
    live_stats = {
        "brightness": np.mean(arr),
        "contrast": np.std(arr),
        "r_channel": np.mean(arr[:, :, 0]),
        "g_channel": np.mean(arr[:, :, 1]),
        "b_channel": np.mean(arr[:, :, 2])
    }
    
    # 2. Percentage Drift
    drift_percentages = {}
    for key, baseline in TRAINING_BASELINES.items():
        live_val = live_stats[key]
        # Formula: ((Actual - Expected) / Expected) * 100
        drift_percentages[f"drift_{key}_pct"] = ((live_val - baseline) / baseline) * 100
    
    return live_stats, drift_percentages