

import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import time

# Load .env
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
ALLOWED_EXT = {'csv', 'json'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Load models safely
def safe_load_model(path):
    if not os.path.exists(path):
        print(f"⚠️ Model file not found: {path}")
        return None
    try:
        model = joblib.load(path)
        print(f"✅ Loaded model: {path}")
        return model
    except Exception as e:
        print(f"⚠️ Could not load model {path}: {e}")
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
provider_model = safe_load_model(os.path.join(BASE_DIR, 'xgb_fraud_detection_model.joblib'))
claim_model = safe_load_model(os.path.join(BASE_DIR, 'claimlevel_xgb_model.joblib'))

# Features
PROVIDER_FEATURES = [
    "is_inpatient","is_groupcode","ChronicCond_rheumatoidarthritis",
    "Beneficiaries_Count","DeductibleAmtPaid","InscClaimAmtReimbursed",
    "ChronicCond_Alzheimer","ChronicCond_IschemicHeart",
    "Days_Admitted","ChronicCond_stroke"
]

CLAIM_FEATURES = [
    'Total_Claims_Per_Bene',
    'TimeInHptal',
    'Provider_Claim_Frequency',
    'ChronicCond_stroke_Yes',
    'DeductibleAmtPaid',
    'OPD_Flag_Yes',
    'Diagnosis_Count',
    'ChronicDisease_Count',
    'Age'
]

# Utils
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def preprocess_input_dict(data_dict, expected_features):
    row = {}
    for f in expected_features:
        v = data_dict.get(f, 0)
        try:
            row[f] = float(v)
        except:
            row[f] = 0.0
    return pd.DataFrame([row], columns=expected_features)

def model_predict_df(df, model):
    if model is None:
        raise ValueError("Model not loaded.")
    preds = model.predict(df)
    try:
        probs = model.predict_proba(df)[:,1]
    except:
        probs = np.zeros(len(preds))
    return preds, probs

def risk_level_from_prob(p):
    if p >= 0.85: return 'Very High'
    if p >= 0.65: return 'High'
    if p >= 0.4: return 'Medium'
    return 'Low'

def potential_saving_calc(prob, amount):
    return round(float(prob)*float(amount)*0.6,2)

# AI Summary
def generate_ai_summary(stats, analysis_type="Provider"):
    if not OPENROUTER_API_KEY:
        return "(AI Summary skipped, no API key provided.)"

    prompt = f"""
You are an experienced healthcare fraud analyst reviewing {analysis_type.lower()}-level data 
from a medical insurance system. Analyze the provided statistics carefully and create a 
structured, human-readable fraud investigation report with proper alignment, indentation, 
and tab spacing. 

Use all the provided statistics including:
- Fraud Probability: {stats['fraud_probability']}
- Risk Level: {stats['risk_level']}
- Potential Saving: {stats['potential_saving']}
- Claim Amount: {stats.get('claim_amount', 0)}
- Deductible Amount Paid: {stats.get('deductible_amt_paid', 0)}
- Inpatient Status: {stats.get('is_inpatient', 0)}
- Beneficiaries Count: {stats.get('beneficiaries_count', 0)}

Write a professional report highlighting:
1. Overall fraud situation
2. Likely fraud types
3. Patterns or causes
4. Recommended actions
5. Final verdict
"""

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-oss-20b",
        "messages":[{"role":"system","content":"You are a helpful fraud analyst."},
                    {"role":"user","content":prompt}],
        "temperature":0.7
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(AI Summary Error: {str(e)})"

# Routes
@app.route('/')
def index():
    return render_template('i.html')

@app.route('/predict_provider_single', methods=['POST'])
def predict_provider_single():
    try:
        data = request.get_json()
        df = preprocess_input_dict(data, PROVIDER_FEATURES)
        preds, probs = model_predict_df(df, provider_model)
        
        prob = float(probs[0])
        pred = int(preds[0])
        rl = risk_level_from_prob(prob)
        saving = potential_saving_calc(prob, data.get('InscClaimAmtReimbursed',0))

        stats = {
            "total_claims": 1,
            "fraud_claims": pred,
            "fraud_percentage": round(prob*100,2),
            "fraud_probability": round(prob,4),
            "risk_level": rl,
            "potential_saving": saving,
            "claim_amount": data.get("InscClaimAmtReimbursed",0),
            "is_inpatient": data.get("is_inpatient",0),
            "deductible_amt_paid": data.get("DeductibleAmtPaid",0),
            "beneficiaries_count": data.get("Beneficiaries_Count",0)
        }

        # Generate AI summary
        ai_summary = generate_ai_summary(stats,"Provider")

        # Save report
        report_file = f"provider_single_{uuid.uuid4().hex}.txt"
        report_path = os.path.join(REPORT_FOLDER, report_file)
        with open(report_path,'w',encoding='utf-8') as f:
            f.write("=== Statistics ===\n"+json.dumps(stats,indent=2)+"\n\n")
            f.write("=== AI Summary ===\n"+ai_summary)

        return jsonify({
            "prediction":"Fraud" if pred==1 else "No Fraud",
            "fraud_probability": round(prob,4),
            "risk_level": rl,
            "potential_saving": saving,
            "input": data,
            "ai_summary": ai_summary,
            "report_path": f"/reports/{report_file}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}),500

@app.route('/predict_claim_single', methods=['POST'])
def predict_claim_single():
    try:
        data = request.get_json()
        df = preprocess_input_dict(data, CLAIM_FEATURES)
        preds, probs = model_predict_df(df, claim_model)
        
        prob = float(probs[0])
        pred = int(preds[0])
        rl = risk_level_from_prob(prob)
        saving = potential_saving_calc(prob, data.get('DeductibleAmtPaid',0)*2)  # example calculation

        stats = {
            "total_claims": 1,
            "fraud_claims": pred,
            "fraud_percentage": round(prob*100,2),
            "fraud_probability": round(prob,4),
            "risk_level": rl,
            "potential_saving": saving,
            "claim_amount": data.get("DeductibleAmtPaid",0)*2,
            "deductible_amt_paid": data.get("DeductibleAmtPaid",0),
            "beneficiaries_count": 1
        }

        ai_summary = generate_ai_summary(stats,"Claim")

        report_file = f"claim_single_{uuid.uuid4().hex}.txt"
        report_path = os.path.join(REPORT_FOLDER, report_file)
        with open(report_path,'w',encoding='utf-8') as f:
            f.write("=== Statistics ===\n"+json.dumps(stats,indent=2)+"\n\n")
            f.write("=== AI Summary ===\n"+ai_summary)

        return jsonify({
            "prediction":"Fraud" if pred==1 else "No Fraud",
            "fraud_probability": round(prob,4),
            "risk_level": rl,
            "potential_saving": saving,
            "input": data,
            "ai_summary": ai_summary,
            "report_path": f"/reports/{report_file}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}),500

@app.route('/upload_provider_batch', methods=['POST'])
def upload_provider_batch():
    try:
        file = request.files.get('file')
        if file is None or file.filename=='':
            return jsonify({"error":"No file uploaded"}),400
        if not allowed_file(file.filename):
            return jsonify({"error":"File type not allowed."}),400
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER,filename)
        file.save(filepath)
        df = pd.read_csv(filepath) if filename.lower().endswith('.csv') else pd.read_json(filepath)
        for f in PROVIDER_FEATURES:
            if f not in df.columns: df[f]=0.0
        df_input = df[PROVIDER_FEATURES]
        preds, probs = model_predict_df(df_input, provider_model)
        df['prediction'] = ['Fraud' if p==1 else 'No Fraud' for p in preds]
        df['fraud_probability'] = probs
        df['risk_level'] = df['fraud_probability'].apply(risk_level_from_prob)
        df['flag_suspicious'] = df['fraud_probability'] >= 0.65
        df['potential_saving'] = df.apply(lambda r: potential_saving_calc(r['fraud_probability'], r.get('InscClaimAmtReimbursed',0)), axis=1)

        fraud_claims = int(df['prediction'].value_counts().get('Fraud',0))
        total_claims = len(df)
        fraud_percentage = round((fraud_claims/total_claims)*100,2) if total_claims>0 else 0

        stats = {
            "total_claims": total_claims,
            "fraud_claims": fraud_claims,
            "fraud_percentage": fraud_percentage,
            "fraud_probability": float(df['fraud_probability'].mean()),
            "risk_level": df['risk_level'].mode()[0] if not df.empty else 'Low',
            "potential_saving": df['potential_saving'].sum()
        }

        ai_summary = generate_ai_summary(stats,"Provider")

        report_file = f"provider_batch_{uuid.uuid4().hex}.txt"
        report_path = os.path.join(REPORT_FOLDER, report_file)
        with open(report_path,'w',encoding='utf-8') as f:
            f.write("=== Statistics ===\n"+json.dumps(stats,indent=2)+"\n\n")
            f.write("=== AI Summary ===\n"+ai_summary)

        return jsonify({
            "message":"Batch Analysis Completed",
            "summary_preview": df.head(50).to_dict(orient='records'),
            "total_records":len(df),
            "ai_summary": ai_summary,
            "report_path": f"/reports/{report_file}"
        })
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route('/reports/<path:filename>')
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__=='__main__':
    app.run(debug=True)
