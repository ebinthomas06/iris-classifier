"""
v1 — RandomForest baseline classifier.
AutoMLOps dispatches to this file based on framework: sklearn in mlops.yaml.
"""
import os, time, json, yaml, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

def train():
    start = time.time()

    with open("mlops.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg.get("dataset_path", "data/iris.csv"))
    features = cfg.get("numerical_features", [])
    target   = cfg.get("target_column", "species")

    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "task_type":          "classification",
        "framework":          "sklearn",
        "model":              "RandomForestClassifier",
        "accuracy":           round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":          round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall":             round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1":                 round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "loss":               round(float(log_loss(y_test, y_proba)), 4),
        "training_time_sec":  round(time.time() - start, 2),
        "commit_sha":         os.getenv("COMMIT_SHA", "local"),
        "model_artifact_path": "artifacts/model.pkl",
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": features, "framework": "sklearn"}, f)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[v1 sklearn] acc={metrics['accuracy']} f1={metrics['f1']} loss={metrics['loss']}")
    return metrics

if __name__ == "__main__":
    train()
