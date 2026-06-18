import numpy as np
import joblib
import pandas as pd
import shap
from str_generator import generate_str_report
from imblearn.over_sampling import SMOTE
from plain_english import explain_in_plain_english
from sklearn.compose import ColumnTransformer
from FAISS.retriever import retrieve, print_retrieved_rules
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model_monitor import track_performance, detect_drift, show_performance_history, show_drift_history

# ──────────────────────────────────────────────────────────────────────────────
# Load or Train Best Model (Random Forest)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("DATA/creditcard.csv")

features = [f"V{i}" for i in range(1, 29)]
X = df[features]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data read successfully.")

# Try to load saved best model
try:
    clf_pipeline = joblib.load("rf_pipeline.pkl")
    print("Loaded saved Random Forest model (best model from comparison).")
except FileNotFoundError:
    print("Saved model not found. Training new model...")
    # Apply SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print("SMOTE applied successfully.")

    # Build pipeline
    print("Defining pipeline...")
    numerical_col = features

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numerical_transformer, numerical_col)]
    )

    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("rf_model", RandomForestClassifier(
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
    print("Pipeline defined successfully.")

    # Fit on SMOTE-resampled data
    print("Fitting model...")
    clf_pipeline.fit(X_train_sm, y_train_sm)
    print("Model fitted successfully.")

    # Save model
    joblib.dump(clf_pipeline, "rf_pipeline.pkl")
    print("Model saved.")

# ──────────────────────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────────────────
print("\nMaking predictions...")
y_pred = clf_pipeline.predict(X_test)
y_proba = clf_pipeline.predict_proba(X_test)[:, 1]
print("Predictions made successfully.")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
roc_auc = roc_auc_score(y_test, y_proba)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall_curve, precision_curve)
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Show Model Comparison Results
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
try:
    comp_df = pd.read_csv("model_comparison_results.csv")
    print(comp_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "weighted_score"]].to_string(index=False, float_format="%.4f"))
    best = comp_df.loc[comp_df["weighted_score"].idxmax(), "model"]
    print(f"\nBest Model: {best}")
except FileNotFoundError:
    print("No comparison results found. Run model_comparison.py first.")

# ──────────────────────────────────────────────────────────────────────────────
# SHAP Explanation (on sample for speed)
# ──────────────────────────────────────────────────────────────────────────────
print("\nRunning SHAP...")
rf_model = clf_pipeline.named_steps["clf"]

# Use only first 10 test samples for SHAP (much faster)
X_test_sample = X_test.iloc[:10]
X_test_transformed = pd.DataFrame(
    clf_pipeline.named_steps["preprocessor"].transform(X_test_sample),
    columns=features,
    index=X_test_sample.index,
)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer(X_test_transformed)

fraud_explanation = shap_values[0, :, 1]   # class-1 (fraud) SHAP for row 0
print("SHAP complete (waterfall plot skipped for headless environment).")

# ──────────────────────────────────────────────────────────────────────────────
# Plain-English explanation + FAISS rule retrieval
# ──────────────────────────────────────────────────────────────────────────────
plain = explain_in_plain_english(fraud_explanation)

query_text = plain["summary"] if isinstance(plain, dict) and "summary" in plain else plain
results = retrieve(query_text)
print_retrieved_rules(results)

# ──────────────────────────────────────────────────────────────────────────────
# Build the STR report
# ──────────────────────────────────────────────────────────────────────────────
sample_idx = X_test_sample.index[0]
raw_txn     = df.loc[sample_idx, features]
raw_amount  = df.loc[sample_idx, "Amount"]

# Build the dict that str_generator expects
output_score = float(np.asarray(fraud_explanation.base_values).flatten()[0]) + float(fraud_explanation.values.sum())
fraud_prob = min(max(output_score, 0), 1)
if fraud_prob >= 0.8:
    risk_tier = "CRITICAL"
elif fraud_prob >= 0.5:
    risk_tier = "HIGH"
elif fraud_prob >= 0.2:
    risk_tier = "MEDIUM"
else:
    risk_tier = "LOW"

shap_result_dict = {
    "risk_tier": risk_tier,
    "risk_score": round(fraud_prob * 100, 2),
    "fraud_probability": fraud_prob,
}

# Build llm_result dict
matched_rules = [r["rule_id"] for r in results]
llm_result_dict = {
    "llm_explanation": query_text,
    "matched_rules": matched_rules,
}

generate_str_report(
    transaction=raw_txn,
    shap_result=shap_result_dict,
    plain_english=plain,
    llm_result=llm_result_dict,
    raw_amount=raw_amount,
)

# ──────────────────────────────────────────────────────────────────────────────
# Model Monitoring
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL MONITORING")
print("="*60)
track_performance(clf_pipeline, X_test, y_test)
show_performance_history(n_last=3)
# Sample data for drift detection (speed)
X_train_drift = X_train.sample(n=10000, random_state=42)
X_test_drift = X_test.sample(n=10000, random_state=42)
detect_drift(X_train_drift, X_test_drift, features)
show_drift_history(n_last=3)

print("\nDone!")