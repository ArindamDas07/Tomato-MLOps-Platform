import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from loguru import logger
import numpy as np

class TomatoModelSuite:
    """
    Centralized Model Hub using the Singleton Pattern.
    Ensures heavy TensorFlow models are loaded into memory exactly once 
    and shared across all Celery tasks.
    """
    _models = {}

    @classmethod
    def load_model(cls, model_name):
        """
        Thread-safe lazy loader for model variants.
        Uses 'compile=False' to optimize memory usage and loading speed for inference.
        """
        if model_name not in cls._models:
            logger.info(f"💾 Initializing model: {model_name}")
            
            if model_name == 'resnet':
                # Load ResNet50 Full Model (Architecture + Weights)
                cls._models['resnet'] = tf.keras.models.load_model(
                    "/app/models/resnet50/v1/tomato_resnet_model.h5", 
                    compile=False
                )
                cls._warmup('resnet')
                logger.success("✅ ResNet50 loaded successfully")
            
            elif model_name == 'gate_keeper':
                # Load Binary Filter Model (Architecture + Weights)
                cls._models["gate_keeper"] = tf.keras.models.load_model(
                    "/app/models/gatekeeper/v1/tomato_leaf_validator.h5", 
                    compile=False
                )
                cls._warmup('gate_keeper')
                logger.success("✅ Gatekeeper (MobileNetV2) loaded successfully")
                
            elif model_name == 'efficient':    
                # EfficientNetB0 requires Architecture Reconstruction before loading weights
                cls._models['efficient'] = cls._load_efficient()
                cls._warmup('efficient')
                logger.success("✅ EfficientNetB0 loaded successfully")
            else:
                raise ValueError(f"Unknown Model Identifier: {model_name}")
                
        return cls._models[model_name]        
    
    @classmethod
    def _load_efficient(cls):
        """
        Reconstructs the EfficientNetB0 computational graph.
        Specifically needed when only weight files (.h5) are provided.
        """
        base = EfficientNetB0(
            weights=None, # Prevents downloading ImageNet weights
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base.trainable = False

        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(10, activation="softmax")(x)

        model = models.Model(inputs=base.input, outputs=output)
        
        # Load the custom fine-tuned weights into the reconstructed architecture
        model.load_weights("/app/models/efficientnetb0/v1/efficientnetb0_tomato_96pct_weights.h5")
        return model

    @classmethod
    def _warmup(cls, model_name):
        """
        Executes a 'Dry Run' inference to trigger TensorFlow graph optimization.
        Prevents the 'First Request Penalty' for real-world users.
        """
        try:
            # Generate a random dummy image of the expected input shape
            shape = (224, 224, 3)
            rgb_array = np.random.randint(0, 256, shape, dtype=np.uint8)
            img = np.expand_dims(rgb_array, axis=0)
            
            # Run one inference to initialize C++ kernels and memory buffers
            _ = cls._models[model_name].predict(img, verbose=0)

            logger.info(f"🔥 {model_name} warmup completed")

        except Exception as e:
            logger.warning(f"⚠️ {model_name} warmup failed: {e}")