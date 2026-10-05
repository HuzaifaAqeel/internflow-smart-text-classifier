"""
FastAPI service for the Smart Text Classifier.

Endpoints:
  GET  /health   -> {"status": "ok", "model": ...}
  POST /predict  -> {"text":..., "category":..., "confidence":..., "all_scores":{...}}

The trained pipeline (cleaning + TF-IDF + LogisticRegression) is loaded
once at startup from artifacts/model.joblib.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
LABELS = ["Complaint", "Inquiry", "Feedback", "Other"]

# The pickled pipeline references src.preprocess.clean_batch, so make the
# src package importable before unpickling.
import sys

sys.path.insert(0, str(ROOT / "src"))
import preprocess  # noqa: F401,E402  (imported for its side effect on unpickling)

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run src/train.py first."
        )
    pipeline = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Smart Text Classifier", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictResponse(BaseModel):
    text: str
    category: str
    confidence: float
    all_scores: dict


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "tfidf-logistic-regression",
        "labels": LABELS,
        "model_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    proba = pipeline.predict_proba([text])[0]
    classes = list(pipeline.named_steps["clf"].classes_)
    scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    best = max(scores, key=scores.get)
    return PredictResponse(
        text=req.text,
        category=best,
        confidence=scores[best],
        all_scores=scores,
    )
