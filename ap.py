# ap.py
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

# -------------------------
# Load .env
# -------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# -------------------------
# Folders
# -------------------------
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
ALLOWED_EXT = {'csv', 'json'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# -------------------------
# Load models safely
# -------------------------
def safe_load_model(path):
    if not os.path.exists(path):
        print(f"⚠ Model file not found: {path}")
        return None
    try:
        model = joblib.load(path)
        print(f"✅ Loaded model: {path}")
        return model
    except Exception as e:
        print(f"⚠ Could not load model {path}: {e}")
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

provider_model = safe_load_model(os.path.join(BASE_DIR, 'xgb_fraud_detection_model.joblib'))
claim_model = safe_load_model(os.path.join(BASE_DIR, 'claimlevel_xgb_model.joblib'))

# -------------------------
# Features
# -------------------------
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

# -------------------------
# Utils
# -------------------------
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
    try:
        return round(float(prob)*float(amount)*0.6,2)
    except:
        return 0.0

def append_master_csv(df, master_filename):
    path = os.path.join(REPORT_FOLDER, master_filename)
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False)

def generate_ai_summary(stats, analysis_type="Provider"):
    if not OPENROUTER_API_KEY:
        return "(AI Summary skipped, no API key provided.)"

    prompt = f"""
You are an experienced healthcare fraud analyst reviewing {analysis_type.lower()}-level data 
from a medical insurance system. Analyze the provided statistics carefully and create a 
structured, human-readable fraud investigation report with proper alignment, indentation, 
and tab spacing. 

Use all the provided statistics including:
- Fraud Probability: {stats.get('fraud_probability')}
- Risk Level: {stats.get('risk_level')}
- Potential Saving: {stats.get('potential_saving')}
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
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(AI Summary Error: {str(e)})"

# -------------------------
# In-memory storage for latest single results
# -------------------------
latest_single_results = {
    "provider": [],
    "claim": []
}

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('i.html')

# -------------------------
# Single Analysis Endpoints
# -------------------------
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

        out = df.copy()
        out['prediction'] = ['Fraud' if pred==1 else 'No Fraud']
        out['fraud_probability'] = [prob]
        out['risk_level'] = [rl]
        out['potential_saving'] = [saving]

        latest_single_results["provider"] = out.to_dict(orient='records')

        return jsonify({
            "prediction": str("Fraud" if pred==1 else "No Fraud"),
            "fraud_probability": float(round(prob,4)),
            "risk_level": str(rl),
            "potential_saving": float(round(saving,2)),
            "input": data
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
        saving = potential_saving_calc(prob, data.get('DeductibleAmtPaid',0)*2)

        out = df.copy()
        out['prediction'] = ['Fraud' if pred==1 else 'No Fraud']
        out['fraud_probability'] = [prob]
        out['risk_level'] = [rl]
        out['potential_saving'] = [saving]

        latest_single_results["claim"] = out.to_dict(orient='records')

        return jsonify({
            "prediction": str("Fraud" if pred==1 else "No Fraud"),
            "fraud_probability": float(round(prob,4)),
            "risk_level": str(rl),
            "potential_saving": float(round(saving,2)),
            "input": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}),500

# -------------------------
# Batch Upload Endpoints
# -------------------------
@app.route('/upload_provider_batch', methods=['POST'])
def upload_provider_batch():
    try:
        file = request.files['file']
        if not (file and allowed_file(file.filename)):
            return jsonify({'error':'Invalid file format'}),400

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        df = pd.read_csv(path).fillna(0)

        df_features = df[PROVIDER_FEATURES]
        preds, probs = model_predict_df(df_features, provider_model)

        df['prediction'] = np.where(preds==1,'Fraud','No Fraud')
        df['fraud_probability'] = probs
        df['risk_level'] = [risk_level_from_prob(p) for p in probs]
        df['potential_saving'] = [potential_saving_calc(p,a) for p,a in zip(probs, df.get('InscClaimAmtReimbursed',[0]*len(df)))]

        report_file = f"provider_batch_{uuid.uuid4().hex}.csv"
        report_path = os.path.join(REPORT_FOLDER, report_file)
        df.to_csv(report_path,index=False)
        df.to_csv(os.path.join(REPORT_FOLDER,'master_provider_batch.csv'),index=False)

        first_row = df.iloc[0]

        return jsonify({
            'message':'Provider batch processed',
            'report_path': f"/reports/{report_file}",
            'total_records': int(len(df)),
            'summary_preview': df.head(10).to_dict(orient='records'),
            'prediction': str(first_row['prediction']),
            'fraud_probability': float(round(first_row['fraud_probability'],4)),
            'risk_level': str(first_row['risk_level']),
            'potential_saving': float(round(first_row['potential_saving'],2))
        })
    except Exception as e:
        return jsonify({'error': str(e)}),500

@app.route('/upload_claim_batch', methods=['POST'])
def upload_claim_batch():
    try:
        file = request.files['file']
        if not (file and allowed_file(file.filename)):
            return jsonify({'error':'Invalid file format'}),400

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        df = pd.read_csv(path).fillna(0)

        df_features = df[CLAIM_FEATURES]
        preds, probs = model_predict_df(df_features, claim_model)

        df['prediction'] = np.where(preds==1,'Fraud','No Fraud')
        df['fraud_probability'] = probs
        df['risk_level'] = [risk_level_from_prob(p) for p in probs]
        df['potential_saving'] = [potential_saving_calc(p,a) for p,a in zip(probs, df.get('DeductibleAmtPaid',[0]*len(df)))]

        report_file = f"claim_batch_{uuid.uuid4().hex}.csv"
        report_path = os.path.join(REPORT_FOLDER, report_file)
        df.to_csv(report_path,index=False)
        df.to_csv(os.path.join(REPORT_FOLDER,'master_claim_batch.csv'),index=False)

        first_row = df.iloc[0]

        return jsonify({
            'message':'Claim batch processed',
            'report_path': f"/reports/{report_file}",
            'total_records': int(len(df)),
            'summary_preview': df.head(10).to_dict(orient='records'),
            'prediction': str(first_row['prediction']),
            'fraud_probability': float(round(first_row['fraud_probability'],4)),
            'risk_level': str(first_row['risk_level']),
            'potential_saving': float(round(first_row['potential_saving'],2))
        })
    except Exception as e:
        return jsonify({'error': str(e)}),500

# -------------------------
# Batch AI Summary
# -------------------------
@app.route('/provider_batch_ai_summary', methods=['POST'])
def provider_batch_ai_summary():
    try:
        data = request.get_json()
        report_file = data.get('report_file')
        if not report_file:
            return jsonify({'error':'No report file provided'}),400

        path = os.path.join(REPORT_FOLDER, report_file)
        if not os.path.exists(path):
            return jsonify({'error':'Report file not found'}),404

        df = pd.read_csv(path)
        avg_prob = df['fraud_probability'].mean()
        total_records = len(df)
        fraud_count = int((df['prediction']=='Fraud').sum())
        saving_sum = df['potential_saving'].sum()

        stats = {
            "total_records": int(total_records),
            "fraud_count": int(fraud_count),
            "fraud_percentage": float(round(fraud_count/total_records*100,2)) if total_records>0 else 0,
            "fraud_probability": float(round(avg_prob,4)),
            "risk_level": risk_level_from_prob(avg_prob),
            "potential_saving": float(round(saving_sum,2))
        }

        ai_summary = generate_ai_summary(stats, "Provider Batch")

        return jsonify({
            "ai_summary": ai_summary,
            **stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}),500

@app.route('/claim_batch_ai_summary', methods=['POST'])
def claim_batch_ai_summary():
    try:
        data = request.get_json()
        report_file = data.get('report_file')
        if not report_file:
            return jsonify({'error':'No report file provided'}),400

        path = os.path.join(REPORT_FOLDER, report_file)
        if not os.path.exists(path):
            return jsonify({'error':'Report file not found'}),404

        df = pd.read_csv(path)
        avg_prob = df['fraud_probability'].mean()
        total_records = len(df)
        fraud_count = int((df['prediction']=='Fraud').sum())
        saving_sum = df['potential_saving'].sum()

        stats = {
            "total_records": int(total_records),
            "fraud_count": int(fraud_count),
            "fraud_percentage": float(round(fraud_count/total_records*100,2)) if total_records>0 else 0,
            "fraud_probability": float(round(avg_prob,4)),
            "risk_level": risk_level_from_prob(avg_prob),
            "potential_saving": float(round(saving_sum,2))
        }

        ai_summary = generate_ai_summary(stats, "Claim Batch")

        return jsonify({
            "ai_summary": ai_summary,
            **stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}),500
@app.route('/provider_single_ai_summary', methods=['POST'])
def provider_single_ai_summary():
    try:
        result_list = latest_single_results.get('provider', [])
        if not result_list or not isinstance(result_list, list):
            return jsonify({"error": "No analytics data found. Please run single analysis first."}), 400

        result = result_list[-1]
        prediction = str(result.get('prediction', 'Unknown')).lower()
        fraud_prob = float(result.get('fraud_probability', 0))
        potential_saving = result.get('potential_saving', "Not calculated")

        # Risk level logic
        if fraud_prob >= 0.75:
            risk_level = "High"
        elif fraud_prob >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Generate AI summary
        ai_summary = (
            f"Provider Analysis Summary:\n"
            f"- Prediction: {prediction.capitalize()}\n"
            f"- Fraud Probability: {fraud_prob*100:.2f}%\n"
            f"- Risk Level: {risk_level}\n"
            f"- Potential Saving: {potential_saving}\n"
            f"- Recommendation: "
            f"{'Immediate investigation recommended due to high fraud likelihood.' if risk_level == 'High' else 'No immediate action required. Continue monitoring provider behavior.'}"
        )

        return jsonify({
            "prediction": prediction.capitalize(),
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
            "potential_saving": potential_saving,
            "ai_summary": ai_summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/claim_single_ai_summary', methods=['POST'])
def claim_single_ai_summary():
    try:
        result_list = latest_single_results.get('claim', [])
        if not result_list or not isinstance(result_list, list):
            return jsonify({"error": "No analytics data found. Please run single analysis first."}), 400

        result = result_list[-1]
        prediction = str(result.get('prediction', 'Unknown')).lower()
        fraud_prob = float(result.get('fraud_probability', 0))
        potential_saving = result.get('potential_saving', "Not calculated")

        if fraud_prob >= 0.75:
            risk_level = "High"
        elif fraud_prob >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        ai_summary = (
            f"Claim Analysis Summary:\n"
            f"- Prediction: {prediction.capitalize()}\n"
            f"- Fraud Probability: {fraud_prob*100:.2f}%\n"
            f"- Risk Level: {risk_level}\n"
            f"- Potential Saving: {potential_saving}\n"
            f"- Recommendation: "
            f"{'Flag this claim for review due to high fraud probability.' if risk_level == 'High' else 'Claim appears legitimate. Routine checks sufficient.'}"
        )

        return jsonify({
            "prediction": prediction.capitalize(),
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
            "potential_saving": potential_saving,
            "ai_summary": ai_summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# -------------------------
# Dashboard Route
# -------------------------
@app.route('/dashboard_data/<dashboard_type>')
def dashboard_data_filtered(dashboard_type):
    try:
        dashboard_type = dashboard_type.lower()
        mode = request.args.get('mode','batch')  # 'single' or 'batch'

        # Load DataFrame based on mode
        if mode == 'single':
            df = pd.DataFrame(latest_single_results.get(dashboard_type, []))
        else:
            if dashboard_type == 'provider':
                file_path = os.path.join(REPORT_FOLDER,'master_provider_batch.csv')
            else:
                file_path = os.path.join(REPORT_FOLDER,'master_claim_batch.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                df = pd.DataFrame()

        # If no data, return zero counts
        if df.empty:
            return jsonify({"fraud_count":0,"no_fraud_count":0,"records":[]})

        # Ensure fraud_probability is numeric
        df['fraud_probability'] = pd.to_numeric(df['fraud_probability'], errors='coerce').fillna(0)

        # Normalize prediction column for consistent counting
        df['prediction'] = df['prediction'].astype(str).str.strip().str.lower()

        # Count fraud and no-fraud correctly
        fraud_count = int((df['prediction'] == 'fraud').sum())
        no_fraud_count = int((df['prediction'] == 'no fraud').sum())

        # Prepare records for frontend
        recs = []
        for idx, row in df.reset_index().head(200).iterrows():
            recs.append({
                "id": int(row.get('index', idx)),
                "fraud_probability": float(row.get('fraud_probability', 0))
            })

        return jsonify({
            "fraud_count": fraud_count,
            "no_fraud_count": no_fraud_count,
            "records": recs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/dashboard')
def default_dashboard():
    return render_template('dashboard.html', dashboard_type='provider')

# -------------------------
# Static Reports
# -------------------------
@app.route('/reports/<path:filename>')
def reports(filename):
    return send_from_directory(REPORT_FOLDER, filename)

# -------------------------
# Run app
# -------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
