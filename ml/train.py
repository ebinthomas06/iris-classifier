"""
v2 — XGBoost + Optuna hyperparameter-tuned classifier.
AutoMLOps dispatches to this file based on framework: xgboost in mlops.yaml.
"""
import os, time, json, yaml, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train():
    start = time.time()

    with open("mlops.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg.get("dataset_path", "data/iris.csv"))
    features = cfg.get("numerical_features", [])
    target   = cfg.get("target_column", "species")

    X = df[features].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df[target].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgb_cfg = cfg.get("xgboost", {})
    n_trials     = int(xgb_cfg.get("n_trials", 30))
    depth_range  = xgb_cfg.get("max_depth_range", [3, 8])
    lr_range     = xgb_cfg.get("lr_range", [0.01, 0.3])
    n_classes    = len(le.classes_)

    def objective(trial):
        params = {
            "objective":        "multi:softprob",
            "num_class":        n_classes,
            "eval_metric":      "mlogloss",
            "verbosity":        0,
            "max_depth":        trial.suggest_int("max_depth", depth_range[0], depth_range[1]),
            "learning_rate":    trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
        model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    final_params = {
        "objective":        "multi:softprob",
        "num_class":        n_classes,
        "eval_metric":      "mlogloss",
        "verbosity":        0,
        "max_depth":        best["max_depth"],
        "learning_rate":    best["learning_rate"],
        "n_estimators":     best["n_estimators"],
        "subsample":        best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "min_child_weight": best["min_child_weight"],
        "reg_alpha":        best["reg_alpha"],
        "reg_lambda":       best["reg_lambda"],
        "random_state":     42,
        "n_jobs":           -1,
    }

    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "task_type":          "classification",
        "framework":          "xgboost",
        "model":              "XGBClassifier",
        "accuracy":           round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":          round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall":             round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1":                 round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "loss":               round(float(log_loss(y_test, y_proba)), 4),
        "training_time_sec":  round(time.time() - start, 2),
        "commit_sha":         os.getenv("COMMIT_SHA", "local"),
        "model_artifact_path": "artifacts/model.pkl",
        "best_params":        best,
        "optuna_best_value":  round(study.best_value, 4),
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "label_encoder": le,
            "features": features,
            "framework": "xgboost",
        }, f)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[v2 xgboost] acc={metrics['accuracy']} f1={metrics['f1']} loss={metrics['loss']} "
          f"(optuna best_cv={metrics['optuna_best_value']})")
    return metrics


if __name__ == "__main__":
    train()
