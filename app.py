from __future__ import annotations
import os, json, pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify



ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
ORD_PATH   = os.path.join(ARTIFACT_DIR, "ordinal_encoder.pkl")
SCL_PATH   = os.path.join(ARTIFACT_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "ada_model.pkl")


with open(ORD_PATH, "rb") as f:
    ord_enc = pickle.load(f)
with open(SCL_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)


ELECT_METHODS = {
    "Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"
}

REQUIRED_FIELDS = [
    "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract",
    "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"
]

yes_no_cols = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "MultipleLines"
]

service_binary_cols = [
    "PhoneService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

ORD_COLS = ["InternetService", "Contract", "PaymentMethod", "gender", "NEW_CONTRACT_LENGTH", "NEW_PAYMENT_METHOD", "NEW_TENURE"]
NUM_SCALE_COLS = ['MonthlyCharges', 'TotalCharges', 'AvgChargesPerMonth']
PASS_COLS = [
    "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "MultipleLines", "ServicesCount"
]

FEATURE_ORDER = ['gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice',
       'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
       'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
       'contract', 'paperlessbilling', 'paymentmethod', 'monthlycharges',
       'totalcharges', 'new_contract_length', 'new_payment_method',
       'servicescount', 'avgchargespermonth', 'new_tenure']

def normalize_yes_no(v):
    """Map 'Yes'->1, 'No'->0; also 'No internet/phone service' -> 0; otherwise passthrough."""
    if isinstance(v, str):
        v_low = v.strip().lower()
        if v_low in ("yes", "no"):
            return 1 if v_low == "yes" else 0
        if "no internet" in v_low or "no phone" in v_low:
            return 0
    return v

def tenure_bucket(t):
    """Bucket tenure to your NEW_TENURE categories."""
    try:
        t = int(t)
    except Exception:
        return "unknown"
    if 0 <= t <= 12:  return "1-year"
    if 12 < t <= 24:  return "2-year"
    if 24 < t <= 36:  return "3-year"
    if 36 < t <= 48:  return "4-year"
    if 48 < t <= 60:  return "5-year"
    if 60 < t <= 72:  return "6-year"
    return "6-year"



def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df2 = df.copy()


    if "customerID" in df2.columns:
        df2 = df2.drop(columns=["customerID"])

    for c in ["MonthlyCharges", "TotalCharges"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    if "TotalCharges" in df2.columns:
        df2["TotalCharges"] = df2["TotalCharges"].fillna(0.0)

    # Normalize Yes/No-style columns -> 0/1
    for col in yes_no_cols:
        if col in df2.columns:
            df2[col] = df2[col].apply(normalize_yes_no)
            df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0).astype(int)

    # NEW_CONTRACT_LENGTH
    if "Contract" in df2.columns:
        df2["NEW_CONTRACT_LENGTH"] = df2["Contract"].apply(
            lambda x: "yearly" if str(x) in ["One year", "Two year"] else "monthly"
        )

    # NEW_PAYMENT_METHOD
    if "PaymentMethod" in df2.columns:
        df2["NEW_PAYMENT_METHOD"] = df2["PaymentMethod"].apply(
            lambda x: "elect" if str(x) in ELECT_METHODS else "no_elect"
        )

    
    exist = [c for c in service_binary_cols if c in df2.columns]
    df2["ServicesCount"] = df2[exist].sum(axis=1) if exist else 0


    if all(c in df2.columns for c in ["TotalCharges", "tenure", "MonthlyCharges"]):
        denom = df2["tenure"].replace(0, np.nan)
        df2["AvgChargesPerMonth"] = (df2["TotalCharges"] / denom).fillna(df2["MonthlyCharges"])
    else:
        df2["AvgChargesPerMonth"] = df2.get("MonthlyCharges", pd.Series(0, index=df2.index))

 
    if "tenure" in df2.columns:
        df2["NEW_TENURE"] = df2["tenure"].apply(tenure_bucket)
    else:
        df2["NEW_TENURE"] = "unknown"

    return df2

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "AdaBoost",
        "artifacts_dir": ARTIFACT_DIR
    })
    
@app.post("/predict")
def predict():

    try:
        payload = request.get_json(force=True, silent=False)
        if payload is None:
            return jsonify(error="Invalid or empty JSON body"), 400

        rows = payload if isinstance(payload, list) else [payload]

        missing_report = []
        for i, r in enumerate(rows):
            missing = [k for k in REQUIRED_FIELDS if k not in r]
            if missing:
                missing_report.append({"index": i, "missing": missing})
        if missing_report:
            return jsonify(error="Missing required fields", details=missing_report), 400

        df_raw = pd.DataFrame(rows)

        # 1) Build engineered features (stateless)
        feats = build_features(df_raw)

        # 2) Transform with saved OrdinalEncoder & StandardScaler
        #    (OrdinalEncoder was trained with handle_unknown='use_encoded_value', unknown_value=-1)
        feats[ORD_COLS] = ord_enc.transform(feats[ORD_COLS])
        feats[NUM_SCALE_COLS] = scaler.transform(feats[NUM_SCALE_COLS].astype(float))
        feats.columns = [x.lower() for x in feats.columns]
        # 3) Assemble in the exact training order
        X = feats[FEATURE_ORDER].to_numpy()

        # 4) Predict
        probs = clf.predict_proba(X)[:, 1]
        
        labels = (probs >= 0.5).astype(int)

        results = []
        for r, p, l in zip(rows, probs, labels):
            results.append({
                "customerID": r.get("customerID"),
                "churn_probability": float(round(p, 6)),
                "churn_label": int(l)
            })

        return jsonify(results=results)

    except Exception as e:
        return jsonify(error=f"Inference error: {str(e)}"), 400
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
