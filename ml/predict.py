import os, pickle
import numpy as np

MODEL_PATH  = os.getenv("MODEL_PATH", "artifacts/model.pkl")
_bundle     = None

def _load():
    global _bundle
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)

_load()

def predict(input_dict: dict) -> dict:
    global _bundle
    if _bundle is None:
        _load()
    if _bundle is None:
        return {"prediction": None, "confidence": 0.0, "warning": "Model not loaded"}

    model    = _bundle.get("model")
    scaler   = _bundle.get("scaler") or _bundle.get("preprocessor")
    features = _bundle.get("features", list(input_dict.keys()))

    X = np.array([[input_dict.get(f, 0.0) for f in features]], dtype=float)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    pred   = model.predict(X)[0]
    proba  = model.predict_proba(X)[0]
    conf   = round(float(max(proba)), 4)

    # Decode label encoder if present
    le = _bundle.get("label_encoder")
    if le is not None:
        pred = le.inverse_transform([int(pred)])[0]

    return {"prediction": str(pred), "confidence": conf}
