"""
demo_predict.py — quick command-line demo of the trained scam classifiers.

No server, no setup beyond `pip install -r requirements.txt`. Good for a
live viva/demo: shows the SVM (fast, default) and DistilBERT (high-recall)
backends side by side on a handful of example messages, then drops into an
interactive prompt so you can type your own.

Usage:
    python demo_predict.py
"""
import sys

from src.models.predictor import ScamPredictor

EXAMPLES = [
    "Congratulations! You have WON $5000. Click here now to claim your prize: bit.ly/claim99",
    "Hey, are we still on for lunch tomorrow at 12?",
    "URGENT: Your bank account will be suspended. Verify your OTP immediately: hxxp://fake-bank.co",
    "Your Amazon package has shipped and will arrive Friday. Track it here: amazon.in/track",
    "Dear winner, you've been selected for a $10,000 lottery prize. Reply with your bank details to claim.",
]


def show(text, predictor):
    svm_result = predictor.predict(text, backend="svm")
    print(f"\nText: {text}")
    print(f"  SVM        -> {'SCAM' if svm_result['is_scam'] else 'safe':<5} "
          f"(prob={svm_result['probability']:.4f}, risk={svm_result['risk']})")

    try:
        bilstm_result = predictor.predict(text, backend="bilstm")
        print(f"  BiLSTM     -> {'SCAM' if bilstm_result['is_scam'] else 'safe':<5} "
              f"(prob={bilstm_result['probability']:.4f}, risk={bilstm_result['risk']})")
    except Exception as e:
        print(f"  BiLSTM     -> unavailable ({e})")

    # DistilBERT is loaded lazily on first use — only invoked here for the
    # demo; comment this out if you want SVM-only speed during testing.
    try:
        db_result = predictor.predict(text, backend="distilbert")
        print(f"  DistilBERT -> {'SCAM' if db_result['is_scam'] else 'safe':<5} "
              f"(prob={db_result['probability']:.4f}, risk={db_result['risk']})")
    except Exception as e:
        print(f"  DistilBERT -> unavailable ({e})")


def main():
    print("Loading models...")
    predictor = ScamPredictor(backend="svm")

    print("=" * 70)
    print("ScamShield NLP — Example Predictions")
    print("=" * 70)
    for text in EXAMPLES:
        show(text, predictor)

    print("\n" + "=" * 70)
    print("Interactive mode — type a message to classify it, or 'quit' to exit.")
    print("=" * 70)
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("quit", "exit"):
            break
        show(text, predictor)


if __name__ == "__main__":
    sys.exit(main())
