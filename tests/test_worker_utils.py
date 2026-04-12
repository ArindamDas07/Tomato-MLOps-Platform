import pytest
import numpy as np
from worker.utils import calculate_drift, TRAINING_BASELINES

def test_calculate_drift_math():
    """
    SENIOR TEST: Proves that the drift percentage formula is correct.
    If the training baseline for brightness is 100, and our image is 110,
    the drift must be exactly 10.0%.
    """
    # 1. Create a fake image array (224x224x3) where every pixel is 116.536
    # (Matches the training baseline exactly)
    baseline_val = TRAINING_BASELINES["brightness"]
    perfect_image = np.full((224, 224, 3), baseline_val, dtype=np.uint8)
    
    stats, drift = calculate_drift(perfect_image)
    
    # Drift should be 0.0% if the image matches the baseline
    assert drift["drift_brightness_pct"] == pytest.approx(0.0, abs=1e-2)

def test_calculate_drift_high_deviation():
    """
    Tests that the system correctly identifies very bright images (+100% drift).
    """
    baseline_val = TRAINING_BASELINES["brightness"]
    # Create an image twice as bright as the baseline
    bright_val = baseline_val * 2
    bright_image = np.full((224, 224, 3), bright_val, dtype=np.uint8)
    
    stats, drift = calculate_drift(bright_image)
    
    # Should report 100% drift
    assert drift["drift_brightness_pct"] == pytest.approx(100.0, abs=1e-2)

def test_calculate_drift_channels():
    """
    Ensures that R, G, and B channel drift are calculated independently.
    """
    # Create an image that is ONLY red
    red_image = np.zeros((224, 224, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255 # Max Red
    
    stats, drift = calculate_drift(red_image)
    
    assert stats["r_channel"] == 255.0
    assert stats["g_channel"] == 0.0
    assert "drift_r_channel_pct" in drift