"""
model_monitor.py
────────────────
Model monitoring: drift detection + performance tracking.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, precision_recall_curve, auc
)


MONITOR_DIR = "monitoring"
os.makedirs(MONITOR_DIR, exist_ok=True)

METRICS_HISTORY_PATH = os.path.join(MONITOR_DIR, "metrics_history.json")
DRIFT_LOG_PATH       = os.path.join(MONITOR_DIR, "drift_log.json")


# ──────────────────────────────────────────────────────────────────────────────
# Performance Tracking
# ──────────────────────────────────────────────────────────────────────────────

def _load_history():
    if os.path.exists(METRICS_HISTORY_PATH):
        with open(METRICS_HISTORY_PATH, "r") as f:
            return json.load(f)
    return []


def _save_history(history):
    with open(METRICS_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def track_performance(model, X_test, y_test, model_name="Random Forest", extra_tags=None):
    """Compute metrics and log them with a timestamp."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(pr_auc, 4),
        "test_size": int(len(y_test)),
        "fraud_count": int(y_test.sum()),
    }
    if extra_tags:
        entry.update(extra_tags)

    history = _load_history()
    history.append(entry)
    _save_history(history)
    print(f"[Monitor] Performance logged at {entry['timestamp']}")
    return entry


def show_performance_history(n_last=None):
    """Display the last n performance records."""
    history = _load_history()
    if not history:
        print("[Monitor] No performance history yet.")
        return

    if n_last:
        history = history[-n_last:]

    df = pd.DataFrame(history)
    cols = ["timestamp", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "test_size", "fraud_count"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "="*60)
    print("PERFORMANCE HISTORY")
    print("="*60)
    print(df[cols].to_string(index=False, float_format="%.4f"))

    if len(df) > 1:
        print(f"\nTrend (latest vs first):")
        latest = df.iloc[-1]
        first  = df.iloc[0]
        for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            delta = latest[col] - first[col]
            arrow = "+" if delta >= 0 else ""
            print(f"  {col:>10}: {first[col]:.4f} -> {latest[col]:.4f} ({arrow}{delta:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# Data Drift Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame,
                 features: list, threshold: float = 0.05) -> dict:
    """
    Compare two datasets using KS-test per feature.
    Returns drift flags + KS statistics.
    """
    from sklearn.model_selection import train_test_split

    results = {}
    drifted_features = []

    for feat in features:
        if feat not in reference_data or feat not in current_data:
            continue
        stat, p_value = ks_2samp(reference_data[feat].dropna(), current_data[feat].dropna())
        is_drift = p_value < threshold
        results[feat] = {
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "drift": bool(is_drift),
        }
        if is_drift:
            drifted_features.append(feat)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_features": len(features),
        "drifted_features": len(drifted_features),
        "drift_ratio": round(len(drifted_features) / len(features), 4),
        "drift_threshold": threshold,
        "drifted_feature_names": drifted_features,
        "feature_details": results,
    }

    # Log drift
    log = []
    if os.path.exists(DRIFT_LOG_PATH):
        with open(DRIFT_LOG_PATH, "r") as f:
            log = json.load(f)
    log.append(summary)
    with open(DRIFT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n[Drift Detect] Scanned {len(features)} features, "
          f"{len(drifted_features)} drifted (p<{threshold})")
    if drifted_features:
        print(f"Drifted: {', '.join(drifted_features[:10])}")

    return summary


def show_drift_history(n_last=None):
    """Display drift detection history."""
    if not os.path.exists(DRIFT_LOG_PATH):
        print("[Drift Detect] No drift history yet.")
        return

    with open(DRIFT_LOG_PATH, "r") as f:
        log = json.load(f)

    if n_last:
        log = log[-n_last:]

    print("\n" + "="*60)
    print("DRIFT DETECTION HISTORY")
    print("="*60)
    for entry in log:
        ts = entry["timestamp"][:19]
        drift_pct = f"{entry['drift_ratio']*100:.1f}%"
        names = entry.get("drifted_feature_names", [])
        sample = ", ".join(names[:5])
        if len(names) > 5:
            sample += f" ... (+{len(names)-5} more)"
        print(f"\n  {ts} | Drift: {drift_pct} ({entry['drifted_features']}/{entry['total_features']})")
        if names:
            print(f"       Features: {sample}")


# ──────────────────────────────────────────────────────────────────────────────
# Full monitoring run
# ──────────────────────────────────────────────────────────────────────────────

def run_monitoring():
    print("="*60)
    print("MODEL MONITORING SUITE")
    print("="*60)

    # Load data
    df = pd.read_csv("DATA/creditcard.csv")
    features = [f"V{i}" for i in range(1, 29)]
    X = df[features]
    y = df["Class"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load model
    model = joblib.load("rf_pipeline.pkl")
    print(f"\nLoaded model: rf_pipeline.pkl")

    # 1. Track performance
    print("\n--- Performance Tracking ---")
    track_performance(model, X_test, y_test)
    show_performance_history(n_last=5)

    # 2. Drift detection (compare test to train reference)
    print("\n--- Data Drift Detection ---")
    detect_drift(X_train, X_test, features)
    show_drift_history(n_last=5)

    print("\n[Monitor] Monitoring run complete.\n")


if __name__ == "__main__":
    run_monitoring()