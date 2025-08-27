# Critical Thinking Report

This document captures the key **engineering decisions**, **trade‑offs**, and **learned failures** from building the AdaBoost‑based churn predictor and its Flask/Gunicorn/Docker serving stack. Metrics referenced below are from the latest validation split (e.g., **precision=0.51**, **recall=0.80**, **ROC‑AUC=0.832**, **accuracy=0.74**).

---

## 1) Decision Journal 

1. **Model family: AdaBoost (shallow trees) as the primary classifier**
   - **Why:** Best **ROC‑AUC / recall** balance among fast, CPU‑friendly models while keeping latency tight. Decision stumps/short trees capture simple, high‑value interactions without heavy infra.
   - **Alternatives considered:**  Random Forest (competitive ROC‑AUC but less recall).

2. **Categorical handling with `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`**
   - **Why:** Guarantees **stable inference** when production sends **unseen categories**, preventing 500s and enabling drift monitoring via “unknown‑category rate”. Lower memory/latency than wide One‑Hot on every categorical.

3. **Uniform preprocessing pipeline + separate persisted artifacts**
   - **Why:** A single `Pipeline` (impute → encode → scale → model) ensures train/serve parity; persisting **encoder/scaler/model** separately enables **hot‑swaps** of the model while freezing encoders for auditability.
   - **Impact:** Faster redeploys and clearer versioning.

4. **Threshold tuning instead of default 0.50**
   - **Why:** Business prefers **higher recall** on churners (catch more at‑risk customers). We tune the decision threshold on the **PR curve** to hit recall ≈ **0.75–0.8**, accepting lower precision (~0.51) given outreach costs.
   - **Impact:** Confusion matrix aligns with retention capacity; fewer misses at the cost of more false positives.

5. **Flask + Gunicorn + Docker for serving (HTTP/JSON microservice)**
   - **Why:** Simple, reliable, fast to ship. **Gunicorn gthread** workers provide concurrency with low memory overhead; Docker standardizes runtime; easy to scale horizontally behind an LB.

---

## 2) Trade‑off Analysis 

1. **Model complexity vs. interpretability**
   - **Choice:** AdaBoost over Logistic Regression.
   - **Why:** Higher **recall** on minority class and better PR‑AUC at our operating point.
   - **Trade‑off:** Reduced plain‑language interpretability vs LR coefficients.

2. **Recall vs. Precision (operating threshold)**
   - **Choice:** Increase recall to ~0.80 with precision ~0.51.
   - **Why:** Missing churn (FN) is costlier than an unnecessary offer (FP).
   - **Trade‑off:** More false positives increases outreach cost. 

3. **Throughput/latency vs. potential top‑end accuracy**
   - **Choice:** AdaBoost over XGBoost/LightGBM.
   - **Why:** Lower operational footprint (no compiled deps), quick cold/warm start, predictable CPU latency.
   - **Trade‑off:** Might leave 1–3 PR‑AUC points on the table vs GBDTs.

---

## 3) Failure Analysis (what didn’t work & hypotheses)
- **Random Forest looked strong offline but regressed in recall**
  - **Symptoms:** ROC‑AUC competitive, but at recall ~0.76 it lagged AdaBoost by a few points.
  - **Hypothesis:** RF probability estimates were less sharp around the operating threshold.

- **Naive handling of unseen categories (early experiments) caused 500s.**
  - **Symptoms:** Production‑like payload with a new `PaymentMethod` crashed.
  - **Hypothesis:** Encoder lacked `unknown_value`; fixed by enabling safe fallback (−1) and logging unknown rates.

---

## 4) Alternative Approaches (what else & when it’s better)

- **Gradient Boosting Trees (LightGBM / CatBoost)**
  - **When better:** We need **maximum PR‑AUC** and can tolerate a slightly heavier stack. CatBoost excels with many categoricals; LightGBM is fast on CPU.


- **Time‑to‑Churn (Survival Analysis)**
  - **When better:** If retention actions depend on **when** a customer might churn, not just **if**. Use CoxPH or gradient‑boosted survival.


---

