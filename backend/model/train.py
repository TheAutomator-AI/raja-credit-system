"""
AI Creditworthiness Prediction System
Model Training Pipeline
Dataset: Give Me Some Credit (cs-training.csv)
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "..", "dataset", "cs-training.csv")
ARTIFACT_DIR = BASE
CHART_DIR = os.path.join(BASE, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)
df = pd.read_csv(DATA_PATH, index_col=0)
print(f"Shape: {df.shape}")
print(df.head())

# Target: SeriousDlqin2yrs == 1 means person defaulted (NOT creditworthy)
# We model "Creditworthy" = 1 - SeriousDlqin2yrs for intuitive interpretation,
# but we train directly on default risk and invert downstream.
TARGET = "SeriousDlqin2yrs"

print("\n" + "=" * 60)
print("STEP 2: CLEANING")
print("=" * 60)

before = len(df)
df = df.drop_duplicates()
print(f"Duplicates removed: {before - len(df)}")

print("Missing values:\n", df.isnull().sum())

# Missing value handling
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].mode()[0])

# Outlier handling (cap at 99th percentile) for known noisy columns
outlier_cols = [
    "RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome",
    "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]
for col in outlier_cols:
    cap = df[col].quantile(0.99)
    df[col] = np.where(df[col] > cap, cap, df[col])

# age outlier: age must be >0
df = df[df["age"] > 0]

print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

df["DebtToIncome"] = df["DebtRatio"] * df["MonthlyIncome"]
df["TotalPastDue"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"]
    + df["NumberOfTime60-89DaysPastDueNotWorse"]
    + df["NumberOfTimes90DaysLate"]
)
df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
df["CreditLinesPerAge"] = df["NumberOfOpenCreditLinesAndLoans"] / df["age"]

FEATURES = [c for c in df.columns if c != TARGET]
X = df[FEATURES]
y = df[TARGET]

print(f"Final feature set ({len(FEATURES)}): {FEATURES}")
print(f"Class balance:\n{y.value_counts(normalize=True)}")

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()

print("\n" + "=" * 60)
print("STEP 4: TRAIN/TEST SPLIT + SCALING")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("STEP 5: SMOTE CLASS BALANCING")
print("=" * 60)
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print(f"Before SMOTE: {dict(y_train.value_counts())}")
print(f"After SMOTE:  {dict(pd.Series(y_train_bal).value_counts())}")

print("\n" + "=" * 60)
print("STEP 6: TRAIN MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_bal, y_train_bal)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    results[name] = metrics
    trained_models[name] = model
    print(metrics)

print("\n" + "=" * 60)
print("STEP 7: MODEL COMPARISON & BEST MODEL SELECTION")
print("=" * 60)

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.sort_values("roc_auc", ascending=False)
print(comparison_df)

best_model_name = comparison_df.index[0]
best_model = trained_models[best_model_name]
print(f"\nBEST MODEL: {best_model_name}")

# Model comparison chart
plt.figure(figsize=(10, 6))
comparison_df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot(kind="bar", figsize=(12, 6))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "model_comparison.png"), dpi=120)
plt.close()

print("\n" + "=" * 60)
print("STEP 8: EVALUATION CHARTS FOR BEST MODEL")
print("=" * 60)

best_preds = best_model.predict(X_test_scaled)
best_probs = best_model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Creditworthy", "Default Risk"],
            yticklabels=["Creditworthy", "Default Risk"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "confusion_matrix.png"), dpi=120)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_probs)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={results[best_model_name]['roc_auc']})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "roc_curve.png"), dpi=120)
plt.close()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, best_probs)
plt.figure(figsize=(7, 6))
plt.plot(rec, prec, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "precision_recall_curve.png"), dpi=120)
plt.close()

report = classification_report(y_test, best_preds, target_names=["Creditworthy", "Default Risk"])
print(report)

# Feature importance
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
else:
    importances = np.ones(len(FEATURES))

fi_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis")
plt.title(f"Feature Importance - {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "feature_importance.png"), dpi=120)
plt.close()

print("\n" + "=" * 60)
print("STEP 9: SAVE ARTIFACTS")
print("=" * 60)

joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

model_info = {
    "best_model_name": best_model_name,
    "features": FEATURES,
    "metrics": results,
    "best_metrics": results[best_model_name],
    "classification_report": report,
    "feature_importance": fi_df.to_dict(orient="records"),
    "target_meaning": "0 = Creditworthy (low default risk), 1 = Not Creditworthy (high default risk)",
    "train_rows": int(X_train.shape[0]),
    "test_rows": int(X_test.shape[0]),
}

with open(os.path.join(ARTIFACT_DIR, "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)

print("Saved: best_model.pkl, scaler.pkl, model_info.json")
print(f"\nFINAL BEST MODEL: {best_model_name}")
print(json.dumps(results[best_model_name], indent=2))
