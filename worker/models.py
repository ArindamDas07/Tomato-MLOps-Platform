import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from loguru import logger
import numpy as np
import threading  # Senior Addition: For thread safety

class TomatoModelSuite:
    """
    Centralized Model Hub using the Singleton Pattern.
    Ensures heavy TensorFlow models are loaded into memory exactly once.
    """
    _models = {}
    _lock = threading.Lock() # Senior Move: Prevents race conditions during loading

    @classmethod
    def load_model(cls, model_name):
        """
        Thread-safe lazy loader for model variants.
        """
        # Double-Checked Locking Pattern
        if model_name not in cls._models:
            with cls._lock:
                if model_name not in cls._models:
                    logger.info(f"💾 Initializing model: {model_name}")
                    try:
                        if model_name == 'resnet':
                            cls._models['resnet'] = tf.keras.models.load_model(
                                "/app/models/resnet50/v1/tomato_resnet_model.h5", 
                                compile=False
                            )
                        elif model_name == 'gate_keeper':
                            cls._models["gate_keeper"] = tf.keras.models.load_model(
                                "/app/models/gatekeeper/v1/tomato_leaf_validator.h5", 
                                compile=False
                            )
                        elif model_name == 'efficient':    
                            cls._models['efficient'] = cls._load_efficient()
                        else:
                            raise ValueError(f"Unknown Model Identifier: {model_name}")
                        
                        # Trigger warmup to optimize TF graph kernels
                        cls._warmup(model_name)
                        logger.success(f"✅ {model_name} ready for inference")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load model {model_name}: {e}")
                        raise e
                
        return cls._models[model_name]        
    
    @classmethod
    def _load_efficient(cls):
        """Reconstructs EfficientNetB0 and loads fine-tuned weights."""
        base = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        base.trainable = False
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(10, activation="softmax")(x)
        
        model = models.Model(inputs=base.input, outputs=output)
        model.load_weights("/app/models/efficientnetb0/v1/efficientnetb0_tomato_96pct_weights.h5")
        return model

    @classmethod
    def _warmup(cls, model_name):
        """Dry run to prevent 'First Request Penalty'."""
        shape = (224, 224, 3)
        img = np.expand_dims(np.random.randint(0, 256, shape, dtype=np.uint8), axis=0)
        _ = cls._models[model_name].predict(img, verbose=0)
        logger.info(f"🔥 {model_name} warmup completed")