from pydantic import BaseModel, Field
from typing import Optional

class InferenceResult(BaseModel):
    """The standardized response format for AI predictions."""
    disease: str = Field(..., description="Detected disease name")
    confidence: float = Field(..., description="Confidence percentage (0-100)")
    model: str = Field(..., description="Model used (resnet/efficient)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "disease": "Late_blight",
                    "confidence": 95.42,
                    "model": "efficient"
                }
            ]
        }
    }

class TaskResponse(BaseModel):
    """Standardized response for polling endpoints."""
    status: str = Field(..., description="processing | done | error")
    valid: Optional[bool] = None
    message: Optional[str] = None
    prediction: Optional[InferenceResult] = None