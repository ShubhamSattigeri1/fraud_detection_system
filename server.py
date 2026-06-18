import os, json, functools
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── Lazy-loaded global state ───────────────────────────────────────────

class LazyLoader:
    def __init__(self):
        self._model = None
        self._df = None
        self._features = [f"V{i}" for i in range(1, 29)]
        self._X_train = self._X_test = None
        self._y_train = self._y_test = None

    @property
    def model(self):
        if self._model is None:
            self._model = joblib.load("rf_pipeline.pkl")
        return self._model

    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_csv("DATA/creditcard.csv")
        return self._df

    @property
    def split_data(self):
        if self._X_train is None:
            X = self.df[self._features]
            y = self.df["Class"]
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        return self._X_train, self._X_test, self._y_train, self._y_test

    @property
    def features(self):
        return self._features

L = LazyLoader()


# ── Helpers ────────────────────────────────────────────────────────────

def load_csv(path, idx_col=0):
    try:
        return pd.read_csv(path, index_col=idx_col)
    except Exception:
        return None

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    cv = load_csv("cv_results.csv")
    best = cv["weighted_score"].idxmax() if cv is not None and "weighted_score" in cv.columns else "Random Forest"
    return jsonify({
        "best_model": best,
        "cv_weighted": round(float(cv["weighted_score"].max()), 4) if cv is not None else None,
        "cv_roc_auc": round(float(cv["roc_auc"].max()), 4) if cv is not None else None,
    })


@app.route("/api/comparison")
def api_comparison():
    comp = load_csv("model_comparison_results.csv")
    if comp is None:
        return jsonify({"error": "No comparison data"}), 404
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "weighted_score"]
    data = comp[cols].reset_index().to_dict(orient="records")
    return jsonify({"models": data, "best": comp["weighted_score"].idxmax()})


@app.route("/api/cross-validation")
def api_cross_validation():
    cv = load_csv("cv_results.csv")
    if cv is None:
        return jsonify({"error": "No CV data"}), 404
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "weighted_score"]
    data = cv[cols].reset_index().to_dict(orient="records")
    for d in data:
        d["model"] = d.pop("index")
    return jsonify({"models": data, "best": cv["weighted_score"].idxmax()})


@app.route("/api/monitoring")
def api_monitoring():
    metrics = load_json("monitoring/metrics_history.json") or []
    drift = load_json("monitoring/drift_log.json") or []
    return jsonify({"metrics": metrics, "drift": drift})


@app.route("/api/samples")
def api_samples():
    _, X_test, _, y_test = L.split_data
    indices = X_test.index[:100].tolist()
    samples = [{"index": int(i), "label": int(y_test.loc[i])} for i in indices]
    return jsonify({"samples": samples})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json()
    idx = int(data["index"])
    features = L.features
    df = L.df
    _, X_test, _, y_test = L.split_data
    model = L.model

    sample = X_test.loc[idx]
    true_label = int(y_test.loc[idx])

    # Predict
    X_t = pd.DataFrame([sample], columns=features)
    pred = int(model.predict(X_t)[0])
    proba = float(model.predict_proba(X_t)[0, 1])

    # SHAP (lazy import)
    import shap
    X_t_t = pd.DataFrame(
        model.named_steps["preprocessor"].transform(X_t),
        columns=features, index=X_t.index
    )
    rf = model.named_steps["clf"]
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer(X_t_t)
    exp = shap_vals[0, :, 1]

    shap_data = []
    values = np.asarray(exp.values).flatten()
    fnames = list(exp.feature_names)
    fvals = np.asarray(exp.data).flatten()
    ranked = sorted(zip(fnames, values, fvals), key=lambda x: abs(x[1]), reverse=True)
    for name, sv, fv in ranked[:10]:
        shap_data.append({"feature": name, "value": round(float(sv), 4), "feat_value": round(float(fv), 4)})

    # Plain English (lazy import)
    from plain_english import explain_in_plain_english
    plain = explain_in_plain_english(exp, top_n=5)

    # FAISS (lazy import)
    from FAISS.retriever import retrieve
    faiss_results = retrieve(plain["summary"])

    # STR (lazy import)
    from str_generator import generate_str_report
    output_score = float(np.asarray(exp.base_values).flatten()[0]) + float(exp.values.sum())
    fraud_prob = min(max(output_score, 0), 1)
    risk_tier = "CRITICAL" if fraud_prob >= 0.5 else "HIGH"
    shap_dict = {
        "risk_tier": risk_tier,
        "risk_score": round(fraud_prob * 100, 2),
        "fraud_probability": fraud_prob,
    }
    llm_dict = {
        "llm_explanation": plain["summary"],
        "matched_rules": [r["rule_id"] for r in faiss_results],
    }
    raw_amount = float(df.loc[idx, "Amount"])
    raw_txn = {
        "transaction_id": f"TXN-{idx}",
        "type": "UPI Transfer",
        "merchant": "Unknown",
        "device": "Unrecognised Device",
        "location": "Unknown Location",
    }
    report_out = generate_str_report(raw_txn, shap_dict, plain, llm_dict, raw_amount)
    str_text = None
    if report_out:
        with open(report_out["filepath"], encoding="utf-8") as f:
            str_text = f.read()

    return jsonify({
        "prediction": pred,
        "probability": proba,
        "true_label": true_label,
        "shap": shap_data,
        "plain_english": {"summary": plain["summary"], "bullets": plain["bullets"]},
        "faiss": faiss_results,
        "str_report": str_text,
        "risk_tier": risk_tier,
        "fraud_probability": fraud_prob,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
