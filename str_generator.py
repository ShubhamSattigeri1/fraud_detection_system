"""
str_generator.py
─────────────────
Step 6 --- STR Report Generator.

Takes the combined output of your full pipeline and saves a formatted
RBI-compliant Suspicious Transaction Report as a .txt file.

Usage --- add these lines after your generator output:
    from str_generator import generate_str_report

    generate_str_report(
        transaction=raw_txn,        # original transaction dict
        shap_result=shap_result,    # from explain_transaction()
        plain_english=plain,        # from explain_in_plain_english()
        llm_result=explanation,     # from generate_explanation()
        raw_amount=row["Amount"],   # original unscaled amount
    )
"""

import os
import uuid
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
REPORTS_DIR = "str_reports"   # folder where reports are saved


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_report_id() -> str:
    """Generates a unique report ID like STR-2026-00142."""
    short = str(uuid.uuid4().int)[:5]
    year  = datetime.now().year
    return f"STR-{year}-{short.zfill(5)}"


def _get_timestamp() -> str:
    return datetime.now().strftime("%d-%b-%Y %H:%M")


def _tier_to_action(risk_tier: str) -> str:
    actions = {
        "CRITICAL": "AUTO BLOCK transaction + File STR with FIU-IND immediately",
        "HIGH":     "Flag for manual review + Notify compliance officer",
        "MEDIUM":   "Trigger OTP re-verification + Monitor account",
        "LOW":      "Transaction allowed --- no action required",
    }
    return actions.get(risk_tier, "Review required")


def _should_generate(risk_tier: str) -> bool:
    """STR reports are only generated for HIGH and CRITICAL transactions."""
    return risk_tier in ("HIGH", "CRITICAL")


# ── Report Builder ─────────────────────────────────────────────────────────────

def _build_report(
    transaction: dict,
    shap_result: dict,
    plain_english: dict,
    llm_result: dict,
    raw_amount: float | None,
) -> str:
    """Formats all pipeline outputs into the STR report string."""

    report_id  = _get_report_id()
    timestamp  = _get_timestamp()
    risk_tier  = shap_result["risk_tier"]
    risk_score = shap_result["risk_score"]
    fraud_prob = shap_result["fraud_probability"]

    # Transaction fields --- use what's available, fall back to "N/A"
    txn_id     = transaction.get("transaction_id", f"TXN{str(uuid.uuid4().int)[:6]}")
    txn_type   = transaction.get("type", "UPI Transfer")
    merchant   = transaction.get("merchant", "Unknown")
    device     = transaction.get("device", "Unrecognised Device")
    location   = transaction.get("location", "Unknown Location")
    amount_str = f"INR {raw_amount:,.0f}" if raw_amount else "N/A"

    # Top 2 bullets for the report
    bullets = plain_english.get("bullets", [])
    bullets_text = "\n".join(f"  {b}" for b in bullets[:5])

    # Matched rules
    matched_rules = llm_result.get("matched_rules", [])
    rules_text = ", ".join(matched_rules) if matched_rules else "N/A"

    # LLM explanation --- wrap at 60 chars for clean formatting
    raw_explanation = llm_result.get("llm_explanation", "N/A")
    wrapped = []
    for line in raw_explanation.split("\n"):
        while len(line) > 60:
            split_at = line[:60].rfind(" ")
            split_at = split_at if split_at > 0 else 60
            wrapped.append("  " + line[:split_at])
            line = line[split_at:].strip()
        wrapped.append("  " + line)
    explanation_text = "\n".join(wrapped)

    action = _tier_to_action(risk_tier)

    report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        SUSPICIOUS TRANSACTION REPORT
        FraudShield AI --- Canara Bank
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Report ID       : {report_id}
Generated       : {timestamp}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSACTION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transaction ID  : {txn_id}
Amount          : {amount_str}
Type            : {txn_type}
Merchant        : {merchant}
Device          : {device}
Location        : {location}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Risk Tier       : {risk_tier}
Risk Score      : {risk_score} / 100
Fraud Prob      : {fraud_prob:.0%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL FINDINGS (SHAP Analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{bullets_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI EXPLANATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{explanation_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATCHED POLICIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{rules_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{action}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        END OF REPORT --- FraudShield AI v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()

    return report, report_id


# ── Main Function ──────────────────────────────────────────────────────────────

def generate_str_report(
    transaction: dict,
    shap_result: dict,
    plain_english: dict,
    llm_result: dict,
    raw_amount: float | None = None,
) -> dict | None:
    """
    Generates and saves an STR report for HIGH and CRITICAL transactions.
    Skips LOW and MEDIUM --- no report needed.

    Parameters
    ----------
    transaction   : dict --- raw transaction fields
    shap_result   : dict --- from explain_transaction()
    plain_english : dict --- from explain_in_plain_english()
    llm_result    : dict --- from generate_explanation()
    raw_amount    : float | None --- original unscaled amount

    Returns
    -------
    dict with report_id and filepath, or None if skipped
    """
    risk_tier = shap_result["risk_tier"]

    # Only generate STR for HIGH and CRITICAL
    if not _should_generate(risk_tier):
        print(f"[FraudShield] Risk tier is {risk_tier} --- no STR required.\n")
        return None

    # Build report text
    report_text, report_id = _build_report(
        transaction, shap_result, plain_english, llm_result, raw_amount
    )

    # Save to file
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = f"{report_id}.txt"
    filepath = os.path.join(REPORTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[FraudShield] STR Report saved -> {filepath}\n")

    return {
        "report_id": report_id,
        "filepath":  filepath,
        "risk_tier": risk_tier,
    }
