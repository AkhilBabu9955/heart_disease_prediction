import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Load the trained model
model_path = "C:\\Users\\akhil\\OneDrive\\Documents\\flask_projects\\heart_disease_prediction\\logistic_regression_model.pkl"
scaler_path = "C:\\Users\\akhil\\OneDrive\\Documents\\flask_projects\\heart_disease_prediction\\scaler.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Define Flask app
app = Flask(__name__)

# Define features
FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = {feature: float(request.form[feature]) for feature in FEATURES}
        df = pd.DataFrame([data], columns=FEATURES)
        
        # Standardize features using the trained scaler
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(df_scaled)[0]
        result_text = "yes Cardiovascular Disease Detected" if prediction == 1 else "No Cardiovascular Disease "
    except Exception as e:
        result_text = f"Error: {str(e)}"

    return render_template("result.html", prediction=result_text)

if __name__ == "__main__":
    app.run(debug=True)
