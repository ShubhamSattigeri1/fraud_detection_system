"""
app.py
──────
Streamlit frontend — FraudShield AI dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="FraudShield AI", layout="wide", page_icon="🛡️")

MONITOR_DIR = "monitoring"
os.makedirs(MONITOR_DIR, exist_ok=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load("rf_pipeline.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("DATA/creditcard.csv")
    features = [f"V{i}" for i in range(1, 29)]
    X = df[features]
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return df, X_train, X_test, y_train, y_test, features

@st.cache_data
def load_cv_results():
    try:
        return pd.read_csv("cv_results.csv", index_col=0)
    except FileNotFoundError:
        return None

@st.cache_data
def load_comparison_results():
    try:
        return pd.read_csv("model_comparison_results.csv", index_col=0)
    except FileNotFoundError:
        return None

def load_monitoring_data():
    metrics = []
    if os.path.exists("monitoring/metrics_history.json"):
        with open("monitoring/metrics_history.json") as f:
            metrics = json.load(f)
    drift = []
    if os.path.exists("monitoring/drift_log.json"):
        with open("monitoring/drift_log.json") as f:
            drift = json.load(f)
    return metrics, drift


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=80)
st.sidebar.title("FraudShield AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Model Comparison",
    "Cross-Validation",
    "Monitoring",
    "Try It Now",
])
st.sidebar.markdown("---")
st.sidebar.caption("Canara Bank · Fraud Detection System")


# ── Page: Overview ───────────────────────────────────────────────────────────

if page == "Overview":
    st.title("🛡️ FraudShield AI — Fraud Detection System")
    st.markdown("""
    ### End-to-End Pipeline

    | Step | What it does |
    |------|-------------|
    | **1. Data** | Kaggle credit card fraud dataset (284k transactions, 0.17% fraud rate) |
    | **2. SMOTE** | Synthetic oversampling to balance classes (50/50) |
    | **3. Models** | Logistic Regression vs Random Forest vs SVM |
    | **4. SHAP** | Explain each prediction — why was this flagged? |
    | **5. RAG** | FAISS vector search against RBI, PMLA, UPI fraud KB |
    | **6. STR** | Generate RBI-compliant Suspicious Transaction Report |
    | **7. Monitor** | Track performance drift + data drift over time |
    """)

    cv = load_cv_results()
    if cv is not None and "weighted_score" in cv.columns:
        best_model = cv["weighted_score"].idxmax()
        best = cv.loc[best_model]
        st.metric("Best Model", best_model)

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "weighted_score"]
        cols = st.columns(len(metrics))
        for col, metric in zip(cols, metrics):
            with col:
                st.metric(metric.replace("_", "-").title(), f"{best[metric]:.4f}")

    st.divider()
    st.subheader("Architecture")
    st.markdown("""
    ```mermaid
    graph LR
        A[Kaggle Data] --> B[SMOTE]
        B --> C[Model: RF]
        C --> D[SHAP Explain]
        D --> E[FAISS KB]
        E --> F[STR Report]
    ```
    """)


# ── Page: Model Comparison ───────────────────────────────────────────────────

elif page == "Model Comparison":
    st.title("📊 Model Comparison")
    comp = load_comparison_results()
    if comp is not None:
        cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "weighted_score"]
        comp_display = comp[cols].copy()
        st.dataframe(comp_display.style.highlight_max(axis=0, color="#90EE90"), use_container_width=True)

        st.subheader("Metric Rankings")
        for metric in ["recall", "f1", "pr_auc", "roc_auc", "precision", "accuracy"]:
            ranked = comp.sort_values(metric, ascending=False)
            st.write(f"**By {metric.upper()}:**")
            for i, (model, row) in enumerate(ranked.iterrows(), 1):
                st.write(f"   {i}. {model}: **{row[metric]:.4f}**")

        best = comp["weighted_score"].idxmax()
        st.success(f"🏆 **Recommended Model: {best}** (weighted score: {comp.loc[best, 'weighted_score']:.4f})")
    else:
        st.warning("Run `model_comparison.py` first to generate comparison results.")


# ── Page: Cross-Validation ───────────────────────────────────────────────────

elif page == "Cross-Validation":
    st.title("🔬 Cross-Validation (5-Fold with SMOTE)")
    cv = load_cv_results()
    if cv is not None:
        cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "weighted_score"]
        st.dataframe(cv[cols].style.highlight_max(axis=0, color="#90EE90"), use_container_width=True)

        st.subheader("Weighted Score Formula")
        st.code("recall*0.35 + f1*0.25 + pr_auc*0.20 + roc_auc*0.15 + precision*0.05")

        best = cv["weighted_score"].idxmax()
        st.success(f"🏆 **{best}** wins cross-validation (score: {cv.loc[best, 'weighted_score']:.4f})")

        # Bar chart
        import altair as alt
        chart_data = cv.reset_index().melt(id_vars="index", value_vars=cols)
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("index:N", title="Model"),
            y=alt.Y("value:Q", title="Score"),
            color="index:N",
            column=alt.Column("variable:N", title=None),
        ).properties(width=120, height=200)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Run `cross_validation.py` first.")


# ── Page: Monitoring ─────────────────────────────────────────────────────────

elif page == "Monitoring":
    st.title("📈 Model Monitoring")
    metrics, drift = load_monitoring_data()

    tab1, tab2 = st.tabs(["Performance History", "Drift Detection"])

    with tab1:
        if metrics:
            df_m = pd.DataFrame(metrics)
            st.dataframe(df_m[["timestamp", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]], use_container_width=True)

            if len(df_m) > 1:
                st.subheader("Performance Trends")
                trend = df_m.set_index("timestamp")[["accuracy", "recall", "f1", "roc_auc"]].astype(float)
                st.line_chart(trend)

            latest = df_m.iloc[-1]
            st.info(f"Latest run: {latest['timestamp']} — ROC-AUC: {latest['roc_auc']:.4f}, Recall: {latest['recall']:.4f}")
        else:
            st.info("No performance history yet. Run `test.py` first.")

    with tab2:
        if drift:
            for entry in drift:
                ts = entry["timestamp"][:19]
                ratio = entry["drift_ratio"] * 100
                if entry["drifted_features"] > 0:
                    st.error(f"🔴 {ts} — {ratio:.0f}% features drifted ({entry['drifted_features']}/{entry['total_features']})")
                    st.write(f"Drifted: {', '.join(entry['drifted_feature_names'][:10])}")
                else:
                    st.success(f"🟢 {ts} — No drift detected")
        else:
            st.info("No drift history yet. Run `test.py` first.")


# ── Page: Try It Now ─────────────────────────────────────────────────────────

elif page == "Try It Now":
    st.title("🔍 Try a Transaction")
    st.markdown("Select a test transaction to see the full analysis pipeline.")

    _, X_train, X_test, y_train, y_test, features = load_data()
    model = load_model()

    idx = st.selectbox("Choose a test sample:", list(X_test.index[:100]))
    sample = X_test.loc[idx]
    true_label = y_test.loc[idx]

    if st.button("Analyze Transaction", type="primary"):
        with st.spinner("Running full analysis pipeline..."):
            st.session_state.clear()

            # Preprocess
            X_t = pd.DataFrame([sample], columns=features)
            X_t_t = pd.DataFrame(
                model.named_steps["preprocessor"].transform(X_t),
                columns=features, index=X_t.index
            )
            rf = model.named_steps["clf"]

            # Predict
            pred = model.predict(X_t)[0]
            proba = model.predict_proba(X_t)[0, 1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", "🚨 FRAUD" if pred else "✅ LEGIT")
            col2.metric("Fraud Probability", f"{proba:.2%}")
            col3.metric("Actual Label", "FRAUD" if true_label else "LEGIT")

            st.divider()

            # SHAP
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer(X_t_t)
            exp = shap_vals[0, :, 1]

            st.subheader("🔬 SHAP Explanation (Top 10 Features)")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            shap.plots.waterfall(exp, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

            # Plain English
            from plain_english import explain_in_plain_english
            plain = explain_in_plain_english(exp, top_n=5)
            st.subheader("📝 Plain English Summary")
            st.write(plain["summary"])
            for b in plain["bullets"]:
                st.write(b)

            # FAISS
            from FAISS.retriever import retrieve, print_retrieved_rules
            st.subheader("📚 Matched Policies (FAISS)")
            results = retrieve(plain["summary"])
            for r in results:
                st.markdown(f"**{r['rule_id']}** — {r['source']}  (score: {r['score']})")
                st.caption(r["text"][:300])

            # STR Report
            from str_generator import generate_str_report
            st.subheader("📄 Suspicious Transaction Report")
            output_score = float(np.asarray(exp.base_values).flatten()[0]) + float(exp.values.sum())
            fraud_prob = min(max(output_score, 0), 1)
            true_risk_tier = "CRITICAL" if fraud_prob >= 0.8 else "HIGH" if fraud_prob >= 0.5 else "MEDIUM" if fraud_prob >= 0.2 else "LOW"

            # Force HIGH so STR always generates for demo visibility
            force_risk_tier = "CRITICAL" if fraud_prob >= 0.5 else "HIGH"
            shap_result_dict = {"risk_tier": force_risk_tier, "risk_score": round(fraud_prob * 100, 2), "fraud_probability": fraud_prob}
            llm_result_dict = {"llm_explanation": plain["summary"], "matched_rules": [r["rule_id"] for r in results]}

            raw_df = pd.read_csv("DATA/creditcard.csv")
            raw_amount = raw_df.loc[idx, "Amount"]
            raw_txn = {
                "transaction_id": f"TXN-{idx}",
                "type": "UPI Transfer",
                "merchant": "Unknown",
                "device": "Unrecognised Device",
                "location": "Unknown Location",
            }
            report_out = generate_str_report(
                transaction=raw_txn,
                shap_result=shap_result_dict,
                plain_english=plain,
                llm_result=llm_result_dict,
                raw_amount=raw_amount,
            )
            if report_out:
                st.success(f"STR Report generated: `{report_out['filepath']}`")
                with open(report_out["filepath"], encoding="utf-8") as f:
                    st.code(f.read(), language="text")
            else:
                # Fallback: generate report manually for low-risk transactions
                st.info("Transaction is low risk. Showing demo STR report...")
                from str_generator import _build_report
                demo_shap = {"risk_tier": "LOW", "risk_score": 5.0, "fraud_probability": 0.05}
                demo_llm = {"llm_explanation": plain["summary"], "matched_rules": [r["rule_id"] for r in results]}
                fallback_text, _ = _build_report(raw_txn, demo_shap, plain, demo_llm, raw_amount)
                st.code(fallback_text, language="text")


# ── Footer ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.caption("FraudShield AI v1.0 | Canara Bank")
