"""API wrapper for adversarial prompt defense.

Provides a FastAPI application exposing a `/check_prompt` endpoint that
classifies text prompts as safe or adversarial using a Hugging Face
transformers pipeline.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


app = FastAPI(title="Adversarial Prompt Defense API")
# Create FastAPI application instance

# --------------- MODEL LOAD -----------------
# Initialize the text-classification pipeline (model + tokenizer)
classifier = pipeline(
    "text-classification",
    model="defense_classifier_final_V2",
    tokenizer="defense_classifier_final_V2",
    device=0  # GPU if available, otherwise -1 for CPU
)

# --------------- REQUEST SCHEMA ------------
class PromptRequest(BaseModel):
    """Request schema for prompt checking.

    Attributes
    ----------
    prompt: str
        The text prompt to be classified.
    """
    prompt: str

# --------------- RESPONSE SCHEMA -----------
class PromptResponse(BaseModel):
    """Response schema for prompt checking.

    Attributes
    ----------
    prompt: str
        The original prompt text.
    safe: bool
        Whether the prompt is considered safe.
    category: Optional[str]
        The classification label (if not safe).
    score: Optional[float]
        Confidence score for the classification.
    """
    prompt: str
    safe: bool
    category: Optional[str] = None
    score: Optional[float] = None

# Pydantic models above define the request and response JSON schemas

# --------------- ENDPOINT ------------------
@app.post("/check_prompt", response_model=PromptResponse)
def check_prompt(request: PromptRequest):
    """Classify a prompt and return a structured response.

    Parameters
    ----------
    request: PromptRequest
        The request body containing the prompt to check.

    Returns
    -------
    PromptResponse
        Structured result including original prompt, safety flag, category, and score.
    """
    # Run the classifier on the incoming prompt
    result = classifier(request.prompt)[0]
    label = result["label"]
    score = result["score"]

    # Format and return the structured response
    return PromptResponse(
        prompt=request.prompt,
        safe=(label.lower() == "safe"),
        category=label if label.lower() != "safe" else "none",
        score=score,
    )
