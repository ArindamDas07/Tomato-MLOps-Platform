import pytest
import numpy as np
from worker.utils import calculate_drift, TRAINING_BASELINES

def test_calculate_drift_math():
    """
    SENIOR TEST: Proves that the drift percentage formula is correct.
    Using float32 dummy image so pixels can match the baseline (116.536) exactly.
    """
    baseline_val = TRAINING_BASELINES["brightness"]
    # Senior Fix: Use float32 to avoid uint8 rounding errors in tests
    perfect_image = np.full((224, 224, 3), baseline_val, dtype=np.float32)
    
    stats, drift = calculate_drift(perfect_image)
    
    # Drift should be 0.0% if the image matches the baseline
    assert drift["drift_brightness_pct"] == pytest.approx(0.0, abs=1e-5)

def test_calculate_drift_high_deviation():
    """
    Tests that the system correctly identifies +100% drift.
    """
    baseline_val = TRAINING_BASELINES["brightness"]
    # Image twice as bright as the baseline
    bright_image = np.full((224, 224, 3), baseline_val * 2, dtype=np.float32)
    
    stats, drift = calculate_drift(bright_image)
    
    assert drift["drift_brightness_pct"] == pytest.approx(100.0, abs=1e-5)

def test_calculate_drift_channels():
    """Ensures R, G, and B channel drift calculations are accurate."""
    red_image = np.zeros((224, 224, 3), dtype=np.float32)
    red_image[:, :, 0] = 255.0 # Max Red
    
    stats, drift = calculate_drift(red_image)
    
    assert stats["r_channel"] == 255.0
    assert stats["g_channel"] == 0.0
    assert "drift_r_channel_pct" in drift