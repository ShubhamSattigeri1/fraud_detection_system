"""
model_comparison.py
───────────────────
Compares Logistic Regression, Random Forest, and TensorFlow/Keras models
on the credit card fraud dataset with SMOTE.
"""

import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# Try to import TensorFlow
TF_AVAILABLE = False  # Disabled for speed - enable if needed
# try:
#     import tensorflow as tf
#     from tensorflow.keras import layers, models, callbacks, optimizers
#     TF_AVAILABLE = True
# except ImportError:
#     TF_AVAILABLE = False
#     print("TensorFlow not available - skipping TF model")


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    print("Loading data...")
    df = pd.read_csv("DATA/creditcard.csv")
    features = [f"V{i}" for i in range(1, 29)]
    X = df[features]
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean():.4%}, Test fraud rate: {y_test.mean():.4%}")
    
    return X_train, X_test, y_train, y_test, features


def apply_smote(X_train, y_train):
    print("\nApplying SMOTE...")
    smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Train shape: {X_train_sm.shape}, Fraud rate: {y_train_sm.mean():.4%}")
    return X_train_sm, y_train_sm


def create_preprocessor(features):
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numerical_transformer, features)]
    )
    return preprocessor


# ──────────────────────────────────────────────────────────────────────────────
# Model Training Functions
# ──────────────────────────────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train, features):
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    preprocessor = create_preprocessor(features)
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(
                max_iter=200,
                class_weight="balanced",
                random_state=42,
                solver='lbfgs'
            )),
        ]
    )
    
    pipeline.fit(X_train, y_train)
    print("Logistic Regression trained successfully.")
    return pipeline


def train_random_forest(X_train, y_train, features):
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST (FAST CONFIG)")
    print("="*60)
    
    preprocessor = create_preprocessor(features)
    
    # Use smaller config for speed on large dataset
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )),
        ]
    )
    
    pipeline.fit(X_train, y_train)
    print("Random Forest trained successfully.")
    return pipeline


def train_svm(X_train, y_train, features):
    """Train SVM with linear kernel (sampled for speed)."""
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    print("\n" + "="*60)
    print("TRAINING SVM (LinearSVC + Calibration)") 
    print("="*60)

    # Sample 50k rows for speed (SMOTE data is 455k)
    if len(X_train) > 100000:
        from sklearn.model_selection import train_test_split as tts
        X_samp, _, y_samp, _ = tts(X_train, y_train, train_size=100000, random_state=42, stratify=y_train)
    else:
        X_samp, y_samp = X_train, y_train

    preprocessor = create_preprocessor(features)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=2000),
            cv=3, n_jobs=-1
        )),
    ])

    pipeline.fit(X_samp, y_samp)
    print("SVM trained successfully.")
    return pipeline


def train_tensorflow_model(X_train, y_train, features, X_val=None, y_val=None):
    """Skipped for speed - enable TF_AVAILABLE = True to use"""
    print("\nTensorFlow model skipped for speed. Enable TF_AVAILABLE to train.")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Functions
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*60}")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Precision-Recall AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TP: {tp:,}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def compare_models(results):
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    df = df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    df = df.set_index("model")
    
    print("\nMetrics Comparison:")
    print(df.to_string(float_format="%.4f"))
    
    # Rank by key metrics for fraud detection
    print("\n" + "-"*80)
    print("RANKINGS (higher is better):")
    print("-"*80)
    
    for metric in ["recall", "f1", "pr_auc", "roc_auc", "precision", "accuracy"]:
        ranked = df.sort_values(metric, ascending=False)
        print(f"\n  By {metric.upper()}:")
        for i, (model, row) in enumerate(ranked.iterrows(), 1):
            print(f"    {i}. {model}: {row[metric]:.4f}")
    
    # Overall recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # For fraud detection, prioritize: Recall > F1 > PR-AUC > ROC-AUC
    # Weight the metrics
    weights = {"recall": 0.35, "f1": 0.25, "pr_auc": 0.20, "roc_auc": 0.15, "precision": 0.05}
    df["weighted_score"] = sum(df[metric] * weight for metric, weight in weights.items())
    
    best_model = df["weighted_score"].idxmax()
    print(f"\nWeighted Score (Recall:35%, F1:25%, PR-AUC:20%, ROC-AUC:15%, Precision:5%):")
    for model, row in df.sort_values("weighted_score", ascending=False).iterrows():
        marker = " *** BEST" if model == best_model else ""
        print(f"  {model}: {row['weighted_score']:.4f}{marker}")
    
    print(f"\nRECOMMENDED MODEL: {best_model}")
    
    return best_model, df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    
    # Load data
    X_train, X_test, y_train, y_test, features = load_data()
    
    # Apply SMOTE
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    
    # Train all models
    results = []
    
    # 1. Logistic Regression
    lr_model = train_logistic_regression(X_train_sm, y_train_sm, features)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results.append(lr_results)
    joblib.dump(lr_model, "lr_pipeline.pkl")
    
    # 2. Random Forest
    rf_model = train_random_forest(X_train_sm, y_train_sm, features)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results.append(rf_results)
    joblib.dump(rf_model, "rf_pipeline.pkl")
    
    # 3. SVM
    svm_model = train_svm(X_train_sm, y_train_sm, features)
    svm_results = evaluate_model(svm_model, X_test, y_test, "SVM")
    results.append(svm_results)
    joblib.dump(svm_model, "svm_pipeline.pkl")

    # 4. TensorFlow
    tf_model = train_tensorflow_model(X_train_sm, y_train_sm, features)
    if tf_model:
        tf_results = evaluate_model(tf_model, X_test, y_test, "TensorFlow")
        results.append(tf_results)
        # Save TF model separately
        tf_model.model.save("tf_model.keras")
        joblib.dump(tf_model.preprocessor, "tf_preprocessor.pkl")
    
    # Compare
    best_model, comparison_df = compare_models(results)
    
    # Save comparison results
    comparison_df.to_csv("model_comparison_results.csv")
    print("\nComparison results saved to model_comparison_results.csv")
    
    # Save best model name
    with open("best_model.txt", "w") as f:
        f.write(best_model)
    print(f"Best model ({best_model}) saved to best_model.txt")