"""
CLI demo for the Smart Text Classifier.

Loads artifacts/model.joblib and prints the predicted category +
confidence for a list of sample customer-support texts.

Usage:
    python demo.py                # run on the built-in samples
    python demo.py "your text"     # classify your own text(s)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

# The pickled pipeline references src.preprocess.clean_batch, so make the
# src package importable before unpickling.
sys.path.insert(0, str(ROOT / "src"))
import preprocess  # noqa: F401,E402  (imported for its side effect on unpickling)

SAMPLES = [
    "My order arrived damaged, I want a refund",
    "The delivery driver left my parcel in the rain and everything is ruined",
    "I was charged twice for the same purchase, this is unacceptable",
    "How long does standard delivery take to Lahore?",
    "Do you offer cash on delivery for orders over Rs 5000?",
    "What is your return policy for sale items?",
    "The new checkout flow is so much faster, great job on the update",
    "Your support team resolved my issue within minutes, very impressed",
    "I love the eco friendly packaging you use for all orders",
    "Just wanted to say hi, hope you're having a great day",
    "Could you share your press kit for a media feature?",
    "I noticed a typo on your about us page, just letting you know",
]


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"Model artifact not found at {MODEL_PATH}. Run src/train.py first.")
        sys.exit(1)
    pipeline = joblib.load(MODEL_PATH)
    texts = sys.argv[1:] if len(sys.argv) > 1 else SAMPLES

    print("Smart Text Classifier — demo predictions")
    print("=" * 60)
    for text in texts:
        proba = pipeline.predict_proba([text])[0]
        classes = list(pipeline.named_steps["clf"].classes_)
        best_idx = int(proba.argmax())
        print(f"input     : {text}")
        print(f"predicted : {classes[best_idx]}  (confidence {proba[best_idx]:.4f})")
        print("-" * 60)


if __name__ == "__main__":
    main()
