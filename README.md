# Customer Churn Prediction 

This repo trains a customer **churn classifier** and serves real‑time predictions via a **Flask API** (Dockerized with Gunicorn). It includes preprocessing (ordinal encoding & scaling), model training, evaluation, and an HTTP endpoint for inference.

---

## 1) Problem Summary & Approach

**Goal:** Predict whether a telecom customer will churn (`Churn = Yes/No`).  
**Data shape:** Mixed categorical + numeric features (e.g., `Contract`, `PaymentMethod`, `tenure`, `MonthlyCharges`, plus several binary service flags).  
**Challenge:** Class imbalance (churners are a minority), mixed dtypes, business need to optimize for **recall** on the positive class.

### Approach
1. **Preprocessing**
   - Drop identifiers: `customerID`.
   - Normalize `Yes/No` columns to `1/0` (e.g., `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `MultipleLines`).  
     > The target `Churn` is **not** used at inference time.
   - Treat blanks as missing; impute **numeric** with median, **categorical** with most-frequent.
   - **Encoding:** `OrdinalEncoder` for categorical features (`handle_unknown='use_encoded_value', unknown_value=-1`).
   - **Scaling:** `StandardScaler` on numeric features.
   - All steps wrapped in a **`Pipeline`** so train & inference use the exact same transformations.
2. **Model**
   - **AdaBoost** classifier.
   - Tuned key params: `n_estimators`, `learning_rate`, and base tree depth.
3. **Evaluation**
   - Stratified train/valid split.
   - Metrics: **PR‑AUC** (primary), **ROC‑AUC**, **F1**, **Recall@threshold**.
4. **Artifacts**
   - Persist **OrdinalEncoder**, **Scaler**, and **Model** as pickles under `artifacts/` for reproducible inference.

---

## 2) Key Findings from EDA


- **Churn rate** typically around **25–30%**.
- **Contract type** is highly predictive: **month‑to‑month** customers churn far more than 1‑ or 2‑year contracts.
- **Payment method:** **Electronic check** shows higher churn vs credit card/bank transfer.
- **Tenure:** Low‑tenure customers churn more; probability drops as tenure increases.
- **Charges:** Higher **MonthlyCharges** correlate with higher churn; **TotalCharges** interacts with tenure.
- Value‑add services (**OnlineSecurity**, **TechSupport**) often correlate with **lower** churn.
- Some services (like **Fiber optic**) can be associated with **higher** churn if they come with higher charges and lower perceived value.

---

## 3) Model Performance Comparison


| Model                | ROC‑AUC | Precision | Accuracy   | Recall (thr=0.50)                     |
|---------------------|:------:|:-----:|:---:|:-----------------|
| Logistic Regression |  0.82  | 0.52  | 0.75 |       0.75 |      
| Random Forest       |  0.83  | 0.54  | 0.76 |       0.76        |
| Gradient Boosting       |  0.83  | 0.54  | 0.77 |       0.72        |
| KNN       |  0.79 | 0.49  | 0.72 |       0.77        |
| **AdaBoost (chosen)** | **0.83** | **0.51** | **0.74** |     **0.80**      |

**Why AdaBoost?** It produced the best **Recall** on validation. Hyperparameter tuning typically yields better recall.

---

## 4) Production Recommendations

- **Schema contract:** fix an explicit **feature list & order**. Reject/alert on unknown fields.
- **Input validation:** type checks, allowed values, ranges; default or reject missing.
- **Versioning:** tag and store `encoder.pkl`, `scaler.pkl`, `model.pkl`.
- **Monitoring:** log request features, scores, decisions; track drift, and monthly **PR‑AUC/ROC‑AUC** on holdout.
- **Serving:** use **Gunicorn** behind a reverse proxy.
- **Performance:** batch predictions where possible; ensure pickles load once at startup.
- **Security:** rate limits, auth (token/header), and secrets via env vars.
- **Containerization:** small Python image, non‑root user.
### Performance Analysis: Latency & Throughput


Using a tiny example (100 requests with latencies in ms):

Average Latency = 157 ms
Total duration = sum(latencies) = 16362 ms = 16.362 s

Throughput (RPS) ≈ requests / total time = 100 / 16.362 ≈ 6.11 RPS

### Edge Case Handling: Out‑of‑Distribution (OOD)

- **Schema & types:** Strict request model (like Pydantic). Reject with **422** on type/enum violations.
- **Unknown categories:** `OrdinalEncoder(..., handle_unknown="use_encoded_value", unknown_value=-1)` safeguards inference.  
- Log the **unknown rate** per categorical. If `unknown_rate > 1%` for any feature in a day → raise an alert.
- **Missing & defaults:** Impute per training strategy (median / most‑frequent). Emit a warning field in the response.

### Data Drift & Concept Drift Monitoring

- **Feature drift (input):**
  - **PSI** per feature.
  - **KS test** for numeric features; **Jensen–Shannon** for categoricals.
- **Performance drift (on labeled feedback):**
  - **Primary:** PR‑AUC, Recall@threshold, Precision@threshold.
  - **Guardrails:** p95 latency, non‑2xx rate, unknown‑category rate.

### A/B Testing for Model Updates

- **Goal:** Safely validate a new model (B) vs current (A) on real traffic.

- **Split:** start with 10% B / 90% A, gradually ramp (10→25→50%).

- **Randomization:** by stable customer key (to avoid cross-condition contamination).

- **Guardrail metrics (business):**

  - Retention uplift (Δ churn rate after outreach).

  - Offer cost per retained customer.

  - Net revenue lift.

- **Model metrics:** precision/recall@policy threshold, AUC, calibration.

- **Run duration:** long enough to reach power (e.g., 2–4 weeks depending on volume).

- **Stop conditions:** pre-registered; use sequential testing or Bayesian A/B to avoid p-hacking.

- **Fail-safe:** instant rollback to A if guardrails breach.

### Local Deployment Architecture

```
[ Client ] 
    │  HTTPS
    ▼
[ API Gateway / LB ]
    │  HTTP (internal)
    ▼
[ NGINX ]
    │            ┌───────────────────────────────┐
    ├──────────▶ │ Gunicorn workers (Flask app) │ ──▶ [ Preproc (Encoder/Scaler) ] ─▶ [ AdaBoost Model ]
                 └───────────────────────────────┘

[ Model Registry ] — versioned artifacts (encoder.pkl, scaler.pkl, model.pkl)
[ Data Lake ] — requests + outcomes for monitoring & retraining
```

### Cloud Deployment Architecture
```
[ Client ]
    │  HTTPS
    ▼
[ Amazon API Gateway ]  (or Lambda Function URL)
    │  Lambda proxy integration
    ▼
[ AWS Lambda  (Container Image from ECR) ]
    │   ┌─────────────────────────────────────────────────────────┐
    │   │  /var/task/app         │
    │   │  ├─ artifacts/: encoder.pkl, scaler.pkl, model.pkl     │
    │   │  ├─ code: build_features(), ordinal_encode(), predict()│
    │   │  └─ returns: { churn_probability, churn_label }        │
    │   └─────────────────────────────────────────────────────────┘
    │
    ├──► [ CloudWatch Logs & Metrics ]
    ├──► (Optional) S3: pull artifacts at cold start
    └──► (Optional) VPC: reach RDS

[ Model Registry / S3 ] — versioned artifacts
[ Data Lake (S3) ] — store requests & outcomes for drift & retraining
```

### Cost Analysis & Optimization

  I'll keep it parametric for cloud solutions.
  #### Assumptions:
- **Average request rate:** `R` requests/sec

- **Latency target:** `L` ms
- **Per-worker sustainable QPS:**  `Q` 
- **Required workers:** `W` = ceil(`peak_R` / `Q`)

#### Operational cost model:
- **Compute**: `W * instance_hour_cost * hours_per_month`
- **Storage**: artifacts (~MBs), logs/metrics (GBs/month)
- **Data transfer:** negligible for JSON

#### Optimizations:
- Keep the pipeline light.

- Batch predictions (e.g., nightly churn scores) to reduce online load.

- Reduce serialization overhead: keep model loaded; avoid reloading per request.

- Caching: if the same customer is scored multiple times/day, cache recent outputs with TTL.

---

## 5) Instructions to Run the Code

### A) Local (Python)

**Prereqs:** Python 3.10+

**Model Training**: Check the jupyter notebook file.

**Inference API:**
```bash
# Create env & install deps
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt

# Run API (Flask dev server)
python app.py  # serves on http://127.0.0.1:8000 
```

**Gunicorn (recommended):**
```bash
gunicorn -w 2 -k gthread -b 0.0.0.0:8000 app:app
```

### API Endpoints

- `POST /predict` — returns `{"customerID": string,"churn_probability": float, "churn_label": 0|1}`  
  **Body (raw features, example likely to predict churn=1):**
  ```json
  {
  "customerID": "CHURN-002",
  "gender": "Male",
  "SeniorCitizen": 1,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 17,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 45.05,
  "TotalCharges": 770.6
  }
  ```

### B) Docker

Build a small production image and run with Gunicorn.

```bash
# Build
docker build -t churn-api:latest .

# Run (artifacts baked into image)
docker run --rm -p 8000:8000 --name churn-api churn-api:latest
```

**Health check (if implemented):**
```bash
curl http://localhost:8000/health
```

**Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @examples/sample.json
```
Or send POST request to the same API endpoint on Postman

---


