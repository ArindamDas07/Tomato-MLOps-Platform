import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from loguru import logger
import numpy as np
import threading
import mlflow
import os

class TomatoModelSuite:
    """
    Centralized Model Hub using the Singleton Pattern.
    Priority: 1. MLflow Model Registry | 2. Local Container Filesystem
    """
    _models = {}
    _lock = threading.Lock()

    @classmethod
    def load_model(cls, model_name):
        """Thread-safe lazy loader for dynamic model management."""
        if model_name not in cls._models:
            with cls._lock:
                if model_name not in cls._models:
                    logger.info(f"💾 Initializing model: {model_name}")
                    
                    # Setup MLflow Tracking
                    tracking_uri = os.getenv("MLFLOW_URI", "http://mlflow:5000")
                    mlflow.set_tracking_uri(tracking_uri)

                    try:
                        if model_name == 'resnet':
                            try:
                                # Try pulling from Registry (Model Name: resnet_tomato, Alias: latest)
                                cls._models['resnet'] = mlflow.tensorflow.load_model("models:/resnet_tomato@latest")
                                logger.info("✅ ResNet loaded from MLflow Registry")
                            except Exception:
                                logger.warning("⚠️ MLflow Registry unreachable/empty. Using local ResNet fallback.")
                                cls._models['resnet'] = tf.keras.models.load_model(
                                    "/app/models/resnet50/v1/tomato_resnet_model.h5", 
                                    compile=False
                                )
                        
                        elif model_name == 'gate_keeper':
                            # Gatekeeper is always local for edge-speed validation
                            cls._models["gate_keeper"] = tf.keras.models.load_model(
                                "/app/models/gatekeeper/v1/tomato_leaf_validator.h5", 
                                compile=False
                            )
                            
                        elif model_name == 'efficient':    
                            try:
                                cls._models['efficient'] = mlflow.tensorflow.load_model("models:/efficient_tomato@latest")
                                logger.info("✅ EfficientNet loaded from MLflow Registry")
                            except Exception:
                                logger.warning("⚠️ MLflow Registry unreachable/empty. Using local EfficientNet fallback.")
                                cls._models['efficient'] = cls._load_efficient_local()
                        else:
                            raise ValueError(f"Unknown Model Identifier: {model_name}")
                        
                        cls._warmup(model_name)
                        logger.success(f"✅ {model_name} is ready for live traffic")
                        
                    except Exception as e:
                        logger.error(f"❌ Critical failure loading {model_name}: {e}")
                        raise e
                
        return cls._models[model_name]        
    
    @classmethod
    def _load_efficient_local(cls):
        """Fallback architecture reconstruction for local weight files."""
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
        """Executes a dry-run to optimize the TensorFlow computation graph."""
        shape = (224, 224, 3)
        img = np.expand_dims(np.random.randint(0, 256, shape, dtype=np.uint8), axis=0)
        _ = cls._models[model_name].predict(img, verbose=0)
        logger.info(f"🔥 {model_name} warmup completed")