"""
cross_validation.py
───────────────────
Cross-validation for robust model evaluation with SMOTE.
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


def load_data():
    df = pd.read_csv("DATA/creditcard.csv")
    features = [f"V{i}" for i in range(1, 29)]
    X = df[features]
    y = df["Class"]
    return X, y, features


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


def cv_with_smote(model, X, y, preprocessor, cv=5, random_state=42):
    """
    Perform cross-validation with SMOTE applied within each fold
    to prevent data leakage.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Use imblearn pipeline to apply SMOTE within each fold
    imb_pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(sampling_strategy="auto", random_state=random_state, k_neighbors=5)),
        ("classifier", model)
    ])
    
    # Get out-of-fold predictions
    y_proba = cross_val_predict(imb_pipeline, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_proba)
    metrics["pr_auc"] = auc(recall_curve, precision_curve)
    
    return metrics, y_proba, y_pred


def run_cross_validation(sample_frac=0.1):
    print("="*60)
    print(f"CROSS-VALIDATION WITH SMOTE (5-Fold) - {sample_frac*100:.0f}% sample")
    print("="*60)
    
    X, y, features = load_data()
    
    # Use stratified sample for speed
    from sklearn.model_selection import train_test_split
    X, _, y, _ = train_test_split(X, y, train_size=sample_frac, random_state=42, stratify=y)
    print(f"Sample size: {len(X)}")
    
    preprocessor = create_preprocessor(features)
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=200, class_weight="balanced", random_state=42, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=2000),
            cv=3, n_jobs=-1
        ),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nRunning CV for {name}...")
        metrics, y_proba, y_pred = cv_with_smote(model, X, y, preprocessor, cv=5)
        results[name] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    
    # Summary table
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    df = pd.DataFrame(results).T
    print(df.to_string(float_format="%.4f"))
    
    # Weighted score (same as before)
    weights = {"recall": 0.35, "f1": 0.25, "pr_auc": 0.20, "roc_auc": 0.15, "precision": 0.05}
    df["weighted_score"] = sum(df[m]*w for m,w in weights.items())
    best = df["weighted_score"].idxmax()
    
    print(f"\nWeighted Score Ranking:")
    for model, row in df.sort_values("weighted_score", ascending=False).iterrows():
        marker = " *** BEST" if model == best else ""
        print(f"  {model}: {row['weighted_score']:.4f}{marker}")
    
    # Save results
    df.to_csv("cv_results.csv")
    print("\nCross-validation results saved to cv_results.csv")
    
    return df


if __name__ == "__main__":
    run_cross_validation()