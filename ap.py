import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import joblib
import pandas as pd
import numpy as np

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
ALLOWED_EXT = {'csv', 'json'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB upload limit

# ---- Load models (provider and claim) ----
def safe_load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Warning: could not load model {path}: {e}")
        return None

provider_model = safe_load_model('xgb_fraud_detection_model.joblib')
claim_model = safe_load_model('claimlevel_xgb_model.joblib') or provider_model

# ---- Feature lists (as per your message) ----
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



# ---- Utility functions ----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_input_dict(data_dict, expected_features):
    """Return DataFrame with expected_features in same order.
       Missing features will be filled with 0; non-numeric cast to float when possible."""
    row = {}
    for f in expected_features:
        v = data_dict.get(f, None)
        if v is None or v == '':
            row[f] = 0.0
        else:
            try:
                row[f] = float(v)
            except:
                row[f] = 0.0
    return pd.DataFrame([row], columns=expected_features)

def model_predict_df(df, model):
    """Return predictions and probabilities for given df and model."""
    if model is None:
        raise ValueError("Model not loaded on server.")
    # ensure columns match model input expectations — here we assume df matches features
    preds = model.predict(df)
    # predict_proba might be not present for some models
    try:
        probs = model.predict_proba(df)[:, 1]
    except:
        probs = np.zeros(len(preds))
    return preds, probs

def risk_level_from_prob(p):
    if p >= 0.85:
        return 'Very High'
    if p >= 0.65:
        return 'High'
    if p >= 0.4:
        return 'Medium'
    return 'Low'

def potential_saving_calc(prob, amount):
    # very rough heuristic: if fraud prob high, potential saving = prob * amount * 0.6
    return round(float(prob) * float(amount) * 0.6, 2)

# ---- Routes ----
@app.route('/')
def index():
    return render_template('i.html')

# ---- Single analysis: Provider ----
@app.route('/predict_provider_single', methods=['POST'])
def predict_provider_single():
    try:
        data = request.get_json()
        df = preprocess_input_dict(data, PROVIDER_FEATURES)
        preds, probs = model_predict_df(df, provider_model)
        prob = float(probs[0])
        pred = int(preds[0])
        rl = risk_level_from_prob(prob)
        saving = potential_saving_calc(prob, data.get('InscClaimAmtReimbursed', 0))
        return jsonify({
            "prediction": "Fraud" if pred == 1 else "No Fraud",
            "fraud_probability": round(prob, 4),
            "risk_level": rl,
            "confidence": round(prob, 4),
            "potential_saving": saving,
            "input": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Single analysis: Claim ----
@app.route('/predict_claim_single', methods=['POST'])
def predict_claim_single():
    try:
        data = request.get_json()
        df = preprocess_input_dict(data, CLAIM_FEATURES)
        preds, probs = model_predict_df(df, claim_model)
        prob = float(probs[0])
        pred = int(preds[0])
        rl = risk_level_from_prob(prob)
        saving = potential_saving_calc(prob, data.get('InscClaimAmtReimbursed', 0))
        return jsonify({
            "prediction": "Fraud" if pred == 1 else "No Fraud",
            "fraud_probability": round(prob, 4),
            "risk_level": rl,
            "confidence": round(prob, 4),
            "potential_saving": saving,
            "input": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Batch upload & analysis (Provider) ----
@app.route('/upload_provider_batch', methods=['POST'])
def upload_provider_batch():
    try:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({"error":"No file uploaded"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error":"File type not allowed. Use CSV or JSON."}), 400

        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # parse file
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_json(filepath, lines=False)

        # ensure we have expected features
        df_input = df.copy()
        for f in PROVIDER_FEATURES:
            if f not in df_input.columns:
                df_input[f] = 0.0
        df_input = df_input[PROVIDER_FEATURES]

        preds, probs = model_predict_df(df_input, provider_model)
        df['prediction'] = ['Fraud' if p==1 else 'No Fraud' for p in preds]
        df['fraud_probability'] = probs
        df['risk_level'] = df['fraud_probability'].apply(risk_level_from_prob)
        # flag suspicious if prob >= 0.65
        df['flag_suspicious'] = df['fraud_probability'] >= 0.65
        df['potential_saving'] = df.apply(lambda r: potential_saving_calc(r['fraud_probability'], r.get('InscClaimAmtReimbursed',0)), axis=1)

        # basic summary returned (first 50 rows to avoid huge responses)
        summary = df.head(200).to_dict(orient='records')
        analysis_id = uuid.uuid4().hex
        # save analysis to disk
        df.to_csv(os.path.join(REPORT_FOLDER, f"provider_analysis_{analysis_id}.csv"), index=False)

        return jsonify({
            "message": "Analysis Completed",
            "analysis_id": analysis_id,
            "summary_preview": summary,
            "total_records": len(df),
            "actions": ["generate_ai_report", "open_dashboard"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Batch upload & analysis (Claim) ----
@app.route('/upload_claim_batch', methods=['POST'])
def upload_claim_batch():
    try:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({"error":"No file uploaded"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error":"File type not allowed. Use CSV or JSON."}), 400

        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # parse
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_json(filepath, lines=False)

        df_input = df.copy()
        for f in CLAIM_FEATURES:
            if f not in df_input.columns:
                df_input[f] = 0.0
        df_input = df_input[CLAIM_FEATURES]

        preds, probs = model_predict_df(df_input, claim_model)
        df['prediction'] = ['Fraud' if p==1 else 'No Fraud' for p in preds]
        df['fraud_probability'] = probs
        df['risk_level'] = df['fraud_probability'].apply(risk_level_from_prob)
        df['flag_suspicious'] = df['fraud_probability'] >= 0.65
        df['potential_saving'] = df.apply(lambda r: potential_saving_calc(r['fraud_probability'], r.get('InscClaimAmtReimbursed',0)), axis=1)

        summary = df.head(200).to_dict(orient='records')
        analysis_id = uuid.uuid4().hex
        df.to_csv(os.path.join(REPORT_FOLDER, f"claim_analysis_{analysis_id}.csv"), index=False)

        return jsonify({
            "message": "Analysis Completed",
            "analysis_id": analysis_id,
            "summary_preview": summary,
            "total_records": len(df),
            "actions": ["generate_ai_report", "open_dashboard"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Generate a simple AI report (placeholder) ----
@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        payload = request.get_json()
        analysis_id = payload.get('analysis_id', uuid.uuid4().hex)
        analysis_type = payload.get('type', 'provider')
        summary = payload.get('summary', [])
        # Create a simple textual report summarizing top suspicious items
        suspicious = [s for s in summary if s.get('flag_suspicious') or s.get('fraud_probability',0) >= 0.65]
        top_susp = suspicious[:10]
        now = datetime.utcnow().isoformat()
        report_filename = f"{analysis_type}_report_{analysis_id}.txt"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"AI Fraud Report (Generated Locally) - {now}\n")
            f.write(f"Analysis ID: {analysis_id}\n")
            f.write(f"Analysis type: {analysis_type}\n")
            f.write(f"Total suspicious records: {len(suspicious)}\n\n")
            f.write("Top suspicious records (preview):\n")
            for r in top_susp:
                f.write(json.dumps(r, default=str) + "\n")
            f.write("\nNOTE: This is a local placeholder report. Replace with Grok/Gemini API integration if desired.\n")
        return jsonify({"message":"Report generated", "report_file": report_filename, "report_path": f"/reports/{report_filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Serve generated reports and dashboard ----
@app.route('/reports/<path:filename>')
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

@app.route('/dashboard')
def dashboard():
    # in a full implementation you'd pull real analysis files; here we show a simple page
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
