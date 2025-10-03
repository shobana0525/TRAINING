from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
xgb_model = joblib.load('xgb_fraud_detection_model.joblib')

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = xgb_model.predict(input_df)[0]
        probability = xgb_model.predict_proba(input_df)[0][1]
        result = 'Fraud' if prediction == 1 else 'No Fraud'

        return jsonify({
            'prediction': result,
            'fraud_probability': round(float(probability), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
