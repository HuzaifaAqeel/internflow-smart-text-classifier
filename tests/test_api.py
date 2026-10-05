"""
API tests for the Smart Text Classifier (FastAPI TestClient).

Run from the project root:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


LABELS = {"Complaint", "Inquiry", "Feedback", "Other"}

DEMO_SAMPLES = [
    ("My order arrived damaged, I want a refund", "Complaint"),
    ("The delivery driver left my parcel in the rain and everything is ruined", "Complaint"),
    ("I was charged twice for the same purchase, this is unacceptable", "Complaint"),
    ("How long does standard delivery take to Lahore?", "Inquiry"),
    ("Do you offer cash on delivery for orders over Rs 5000?", "Inquiry"),
    ("What is your return policy for sale items?", "Inquiry"),
    ("The new checkout flow is so much faster, great job on the update", "Feedback"),
    ("Your support team resolved my issue within minutes, very impressed", "Feedback"),
    ("I love the eco friendly packaging you use for all orders", "Feedback"),
    ("Just wanted to say hi, hope you're having a great day", "Other"),
    ("Could you share your press kit for a media feature?", "Other"),
    ("I noticed a typo on your about us page, just letting you know", "Other"),
]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert set(body["labels"]) == LABELS


def test_predict_valid(client):
    r = client.post("/predict", json={"text": "My order arrived damaged, I want a refund"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Complaint"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["all_scores"].keys()) == LABELS
    assert abs(sum(body["all_scores"].values()) - 1.0) < 1e-3


def test_predict_empty_text_rejected(client):
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 422


def test_predict_missing_field_rejected(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_wrong_type_rejected(client):
    r = client.post("/predict", json={"text": 123})
    assert r.status_code == 422


def test_demo_samples_all_correct_categories(client):
    for text, expected in DEMO_SAMPLES:
        r = client.post("/predict", json={"text": text})
        assert r.status_code == 200, text
        body = r.json()
        assert body["category"] in LABELS, text
        assert body["category"] == expected, text
