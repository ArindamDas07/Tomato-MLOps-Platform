from pydantic import BaseModel, Field
from typing import Optional

class InferenceResult(BaseModel):
    """
    Standardized Schema for AI Model outputs.
    This is the 'Contract' that ensures the Worker sends exactly 
    what the API expects.
    """
    disease: str = Field(
        ..., 
        description="The name of the detected tomato leaf disease",
        example="Late_blight"
    )
    confidence: float = Field(
        ..., 
        description="Confidence score as a percentage (0.0 - 100.0)",
        example=95.42
    )
    model: str = Field(
        ..., 
        description="Identifier of the model used (e.g., 'resnet' or 'efficient')",
        example="efficient"
    )

class TaskResponse(BaseModel):
    """
    Standardized Schema for the Gateway polling endpoints.
    Used for /leaf_checker and /result endpoints.
    """
    status: str = Field(..., description="Current state: processing | done | error")
    valid: Optional[bool] = Field(None, description="True if a tomato leaf was detected")
    message: Optional[str] = Field(None, description="Error or status message")
    prediction: Optional[InferenceResult] = Field(None, description="The final AI prediction result")