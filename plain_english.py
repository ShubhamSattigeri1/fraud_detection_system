import numpy as np


def explain_in_plain_english(explanation, top_n=5, class_label="fraud"):
    """Print a plain-English summary of a SHAP explanation for one prediction."""
    values = np.asarray(explanation.values).flatten()
    feature_names = list(explanation.feature_names)

    if hasattr(explanation, "data") and explanation.data is not None:
        feature_values = np.asarray(explanation.data).flatten()
    else:
        feature_values = np.full(len(feature_names), np.nan)

    base = None
    if hasattr(explanation, "base_values") and explanation.base_values is not None:
        base = float(np.asarray(explanation.base_values).flatten()[0])

    output = None
    if hasattr(explanation, "output_values") and explanation.output_values is not None:
        output = float(np.asarray(explanation.output_values).flatten()[0])
    elif base is not None:
        output = base + float(values.sum())

    ranked = sorted(
        zip(feature_names, values, feature_values),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    print("\n--- Plain English Explanation ---")
    if base is not None and output is not None:
        print(
            f"Starting from the model's average risk score ({base:.4f}), "
            f"this transaction's features push the score to {output:.4f}."
        )

    if output is not None and base is not None:
        direction = f"toward {class_label}" if output > base else f"away from {class_label}"
        print(f"Overall, the model is pushed {direction}.\n")

    print(f"Top {min(top_n, len(ranked))} reasons:")
    bullets = []
    for name, shap_val, feat_val in ranked[:top_n]:
        if shap_val > 0:
            impact = f"increases the {class_label} score"
        elif shap_val < 0:
            impact = f"decreases the {class_label} score"
        else:
            impact = "has little effect"

        if not np.isnan(feat_val):
            bullet = f"  - {name} (value: {feat_val:.4f}) {impact} by {abs(shap_val):.4f}."
        else:
            bullet = f"  - {name} {impact} by {abs(shap_val):.4f}."

        print(bullet)
        bullets.append(bullet)

    print("--- End Explanation ---\n")

    # Return structured data for STR report
    summary = (
        f"Model pushed toward {class_label}. "
        f"Top factors: {', '.join([r[0] for r in ranked[:3]])}."
    )
    return {
        "summary": summary,
        "bullets": bullets,
        "base_score": base,
        "output_score": output,
        "top_features": [(r[0], float(r[1]), float(r[2]) if not np.isnan(r[2]) else None) for r in ranked[:top_n]]
    }