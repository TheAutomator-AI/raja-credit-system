# Raja Credit System — AI Creditworthiness Prediction System

A complete ML-powered credit risk assessment platform: a trained scikit-learn
pipeline served via a Flask REST API, with a premium fintech-styled React
dashboard.

## What's included

```
project/
├── dataset/
│   └── cs-training.csv          # "Give Me Some Credit" dataset (your upload)
├── backend/
│   ├── app.py                   # Flask REST API
│   ├── requirements.txt
│   └── model/
│       ├── train.py             # Full training pipeline
│       ├── best_model.pkl       # Trained Random Forest (auto-selected)
│       ├── scaler.pkl           # StandardScaler fit on training data
│       ├── model_info.json      # Metrics, features, feature importance
│       └── charts/              # Generated evaluation charts (PNG)
└── frontend/
    └── index.html                # Full React app (no build step required)
```

## Dataset & target

The dataset's label, `SeriousDlqin2yrs`, marks whether a borrower experienced
serious delinquency within two years. We model this directly as default risk,
and report **creditworthiness = 1 − default probability**. 14 features are
used after cleaning, outlier capping, and engineering (`DebtToIncome`,
`TotalPastDue`, `IncomePerDependent`, `CreditLinesPerAge`).

## Model results

Three models were trained and benchmarked on a held-out 20% test set
(150,000 rows, SMOTE-balanced training set):

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|----------------------|----------|-----------|--------|-------|---------|
| Logistic Regression   | 80.2%    | 21.7%     | 75.2%  | 33.7% | 86.0%   |
| Decision Tree         | 78.5%    | 20.2%     | 75.0%  | 31.8% | 84.6%   |
| **Random Forest** ✅   | **82.5%**| **23.9%** | 74.2%  | **36.2%** | **86.2%** |

**Random Forest** was auto-selected as the best model based on ROC-AUC.
Note: this is a highly imbalanced dataset (~7% default rate), so precision is
intentionally traded for recall — the model is tuned to catch genuine default
risk rather than maximize raw accuracy. This is a standard and defensible
choice for credit risk models, and is documented in `model_info.json`.

## Deploying for free (live public demo)

This repo is pre-configured for a zero-cost split deployment: the **backend**
on Render, the **frontend** on GitHub Pages.

### Backend → Render

1. Push this repo to GitHub (see steps below).
2. Go to [render.com](https://render.com) → **New** → **Blueprint** → connect
   your repo. Render reads `render.yaml` automatically and creates a free web
   service running `gunicorn app:app` out of `backend/`.
3. Wait for the build to finish, then copy the live URL, e.g.
   `https://raja-credit-system-api.onrender.com`.
4. Check it works: visit `https://your-url.onrender.com/health`.

   *(Free Render services spin down after inactivity — the first request
   after idling can take ~30s to wake up. That's normal.)*

### Frontend → GitHub Pages

1. In your GitHub repo: **Settings → Pages → Source → GitHub Actions**.
2. The included workflow (`.github/workflows/deploy-pages.yml`) deploys
   `frontend/index.html` automatically on every push to `main`. Trigger it
   manually the first time via **Actions → Deploy Frontend to GitHub Pages →
   Run workflow** if it doesn't fire automatically.
3. Your site will be live at `https://<your-username>.github.io/<repo-name>/`.
4. Open it, click **⚙ API Settings** in the sidebar, and paste your Render
   backend URL from above. This is stored in the browser's `localStorage`, so
   you only need to set it once per device.

If you skip backend setup entirely, the frontend still works standalone
using its built-in offline heuristic scorer — useful for a quick demo link
with zero backend cost.

### Pushing to GitHub

```bash
cd raja-credit-system
git init
git add .
git commit -m "Initial commit: Raja Credit System"
gh repo create raja-credit-system --public --source=. --remote=origin --push
# or, without gh CLI:
# git remote add origin https://github.com/<you>/raja-credit-system.git
# git branch -M main && git push -u origin main
```

## Running it locally

### 1. Backend (Flask API)

```bash
cd backend
pip install -r requirements.txt
python app.py
# API now running at http://localhost:5000
```

To retrain the model from scratch:
```bash
cd backend/model
python train.py
```

### 2. Frontend

The frontend is a single, dependency-free HTML file using React, Recharts,
and Tailwind via CDN — no `npm install` or build step needed.

```bash
# just open it
open frontend/index.html
# or serve it
cd frontend && python -m http.server 8080
```

By default it calls the API at `http://localhost:5000`. If the backend isn't
running, the UI automatically falls back to a client-side heuristic scorer so
the demo still works end-to-end — useful for offline presentations. To point
it at a different API URL, run in the browser console:
```js
localStorage.setItem('CREDIT_API_BASE', 'https://your-api-url.com')
```

## API Reference

| Method | Route               | Description                                  |
|--------|----------------------|-----------------------------------------------|
| GET    | `/`                  | Service info                                  |
| GET    | `/health`            | Health check                                  |
| POST   | `/predict`           | Single applicant prediction                   |
| POST   | `/batch-predict`     | Predict for an array of applicant records     |
| GET    | `/metrics`           | All model metrics + best model                |
| GET    | `/model-info`        | Active model name, features, target meaning   |
| GET    | `/feature-importance`| Ranked feature importance                     |
| GET    | `/history`           | In-memory prediction log (last 50)            |

### Example: `POST /predict`

```json
{
  "age": 35, "income": 5000, "monthlyExpenses": 1800,
  "employmentYears": 4, "loanAmount": 15000, "debt": 6000,
  "numberOfLoans": 2, "savings": 4000, "creditUtilization": 35,
  "existingEMI": 300, "defaultHistory": 0, "dependents": 0
}
```

Response includes `creditScore` (300–850 scale), `approved`,
`approvalProbability`, `riskCategory`, `suggestedInterestRate`,
`topInfluencingFeatures`, human-readable `reasons`, and `suggestions`.

## Frontend pages

Landing · Dashboard (live stat cards, charts, recent predictions) ·
New Prediction (14-field form) · Result (animated credit score gauge, risk
meter, AI explanation) · Analytics (feature importance, ROC curve, model
comparison) · History (search, sort, paginate, export CSV).

## Notes for the demo / write-up

- Charts from the training run (correlation heatmap, confusion matrix, ROC
  curve, precision-recall curve, feature importance, model comparison) are
  saved as PNGs in `backend/model/charts/` — drop these straight into a
  slide deck or project report.
- `model_info.json` contains the full classification report and per-feature
  importances for citing in documentation.
- The in-memory prediction history resets when the Flask server restarts;
  swap in a database (SQLite/Postgres) for persistence in a production
  deployment.
