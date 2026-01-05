from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load('crop_model.pkl')

# Try loading StandardScaler first, fallback to MinMaxScaler
try:
    scaler = joblib.load('standardscaler.pkl')
except FileNotFoundError:
    scaler = joblib.load('minmaxscaler.pkl')

# Map prediction indices to crop names
crop_labels = {
    0: "rice", 1: "maize", 2: "chickpea", 3: "kidney beans", 4: "pigeon peas",
    5: "moth beans", 6: "mung beans", 7: "black gram", 8: "lentil", 9: "banana",
    10: "mango", 11: "grapes", 12: "apple", 13: "orange", 14: "papaya",
    15: "coconut", 16: "cotton", 17: "jute", 18: "coffee"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        nitrogen = float(request.form.get('nitrogen', 0))
        phosphorus = float(request.form.get('phosphorus', 0))
        potassium = float(request.form.get('potassium', 0))
        temperature = float(request.form.get('temperature', 0))
        humidity = float(request.form.get('humidity', 0))
        soil_ph = float(request.form.get('soil_ph', 0))
        rainfall = float(request.form.get('rainfall', 0))
    except ValueError:
        return render_template('index.html', result=None)

    features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, soil_ph, rainfall]])
    scaled = scaler.transform(features)
    pred_idx = int(model.predict(scaled)[0])
    crop_name = crop_labels.get(pred_idx, "Unknown Crop")

    return render_template('index.html', result=crop_name)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def home():
    # For HEAD requests, Flask will automatically skip the body
    return render_template("index.html")
