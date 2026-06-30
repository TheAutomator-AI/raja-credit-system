"""
AI Creditworthiness Prediction System - Backend API
Flask REST API serving the trained credit risk model.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
    MODEL_INFO = json.load(f)

FEATURES = MODEL_INFO["features"]

prediction_history = []  # in-memory store for demo purposes


def build_feature_row(payload):
    """Map a user-friendly prediction form into the model's feature schema."""
    age = float(payload.get("age", 35))
    income = float(payload.get("income", 4000))
    monthly_expenses = float(payload.get("monthlyExpenses", 1500))
    employment_years = float(payload.get("employmentYears", 3))
    loan_amount = float(payload.get("loanAmount", 10000))
    debt = float(payload.get("debt", 5000))
    num_loans = float(payload.get("numberOfLoans", 2))
    savings = float(payload.get("savings", 2000))
    credit_utilization = float(payload.get("creditUtilization", 30)) / 100.0
    existing_emi = float(payload.get("existingEMI", 200))
    default_history = float(payload.get("defaultHistory", 0))
    dependents = float(payload.get("dependents", 0))
    real_estate_loans = float(payload.get("realEstateLoans", 0))

    debt_ratio = (debt + existing_emi) / (income + 1)
    revolving_util = min(credit_utilization, 1.5)
    past_due_30_59 = default_history if default_history <= 2 else 2
    past_due_60_89 = max(default_history - 2, 0)
    times_90_late = max(default_history - 4, 0)

    row = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": past_due_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": num_loans,
        "NumberOfTimes90DaysLate": times_90_late,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": past_due_60_89,
        "NumberOfDependents": dependents,
        "DebtToIncome": debt_ratio * income,
        "TotalPastDue": past_due_30_59 + past_due_60_89 + times_90_late,
        "IncomePerDependent": income / (dependents + 1),
        "CreditLinesPerAge": num_loans / max(age, 1),
    }
    return pd.DataFrame([row])[FEATURES]


def score_to_band(score):
    if score >= 750:
        return "Excellent"
    if score >= 650:
        return "Good"
    if score >= 550:
        return "Fair"
    return "Poor"


def run_prediction(payload):
    X = build_feature_row(payload)
    X_scaled = scaler.transform(X)
    default_prob = float(model.predict_proba(X_scaled)[0][1])
    creditworthy_prob = 1 - default_prob

    credit_score = int(300 + creditworthy_prob * 550)  # map to 300-850 scale
    approved = creditworthy_prob >= 0.5

    if default_prob < 0.15:
        risk_category, interest_rate = "Low Risk", 6.5
    elif default_prob < 0.35:
        risk_category, interest_rate = "Moderate Risk", 10.5
    elif default_prob < 0.6:
        risk_category, interest_rate = "High Risk", 16.0
    else:
        risk_category, interest_rate = "Very High Risk", 22.0

    fi = sorted(MODEL_INFO["feature_importance"], key=lambda r: -r["importance"])[:5]
    top_features = [f["feature"] for f in fi]

    reasons = []
    if payload.get("creditUtilization", 30) and float(payload.get("creditUtilization", 30)) > 50:
        reasons.append("High credit utilization ratio")
    if float(payload.get("defaultHistory", 0)) > 0:
        reasons.append("History of late or missed payments")
    if float(payload.get("debt", 0)) > float(payload.get("income", 1)) * 0.5:
        reasons.append("High debt-to-income ratio")
    if float(payload.get("savings", 0)) < float(payload.get("loanAmount", 0)) * 0.1:
        reasons.append("Low savings relative to requested loan amount")
    if not reasons:
        reasons.append("Stable income and healthy repayment history")

    suggestions = [
        "Reduce credit utilization below 30%",
        "Maintain consistent on-time payments",
        "Build an emergency savings buffer",
        "Avoid taking on multiple new loans simultaneously",
    ]

    result = {
        "approved": approved,
        "creditScore": credit_score,
        "scoreBand": score_to_band(credit_score),
        "approvalProbability": round(creditworthy_prob * 100, 2),
        "defaultProbability": round(default_prob * 100, 2),
        "confidence": round(max(creditworthy_prob, default_prob) * 100, 2),
        "riskCategory": risk_category,
        "suggestedInterestRate": interest_rate,
        "topInfluencingFeatures": top_features,
        "reasons": reasons,
        "suggestions": suggestions,
        "explanation": (
            f"Based on the applicant's financial profile, the model estimates a "
            f"{round(default_prob * 100, 1)}% probability of serious delinquency within 2 years. "
            f"This places the applicant in the '{risk_category}' category."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    }
    return result


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "AI Creditworthiness Prediction System",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/metrics", "/model-info", "/feature-importance", "/batch-predict"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None, "timestamp": datetime.utcnow().isoformat()})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        result = run_prediction(payload)
        entry = {**result, "input": payload, "id": len(prediction_history) + 1}
        prediction_history.append(entry)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    try:
        payload = request.get_json(force=True)
        rows = payload.get("records", [])
        results = []
        for row in rows:
            res = run_prediction(row)
            results.append({**res, "input": row})
        return jsonify({"count": len(results), "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "all_models": MODEL_INFO["metrics"],
        "best_model": MODEL_INFO["best_model_name"],
        "best_metrics": MODEL_INFO["best_metrics"],
        "train_rows": MODEL_INFO["train_rows"],
        "test_rows": MODEL_INFO["test_rows"],
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": MODEL_INFO["best_model_name"],
        "features": MODEL_INFO["features"],
        "target_meaning": MODEL_INFO["target_meaning"],
        "metrics": MODEL_INFO["best_metrics"],
    })


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    return jsonify({"feature_importance": MODEL_INFO["feature_importance"]})


@app.route("/history", methods=["GET"])
def history():
    return jsonify({"count": len(prediction_history), "history": prediction_history[-50:]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
