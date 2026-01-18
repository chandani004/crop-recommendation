from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ---------------- LOAD MODEL & SCALER ----------------

model = joblib.load("crop_model.pkl")

try:
    scaler = joblib.load("standardscaler.pkl")
except FileNotFoundError:
    scaler = joblib.load("minmaxscaler.pkl")

# ---------------- CROP LABELS ----------------

crop_labels = {
    0: "rice",
    1: "maize",
    2: "chickpea",
    3: "kidney beans",
    4: "pigeon peas",
    5: "moth beans",
    6: "mung beans",
    7: "black gram",
    8: "lentil",
    9: "banana",
    10: "mango",
    11: "grapes",
    12: "apple",
    13: "orange",
    14: "papaya",
    15: "coconut",
    16: "cotton",
    17: "jute",
    18: "coffee"
}

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        nitrogen = float(request.form["nitrogen"])
        phosphorus = float(request.form["phosphorus"])
        potassium = float(request.form["potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        soil_ph = float(request.form["soil_ph"])
        rainfall = float(request.form["rainfall"])
    except (ValueError, KeyError):
        return render_template("index.html", result="Invalid input")

    features = np.array([[nitrogen, phosphorus, potassium,
                          temperature, humidity, soil_ph, rainfall]])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    crop_name = crop_labels.get(int(prediction), "Unknown Crop")

    return render_template("index.html", result=crop_name)

# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run()

