from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# -------------------------------
# 1️⃣ Initialize Flask
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 2️⃣ Load trained model
# -------------------------------
model_path = os.path.join("model", "model.pk1")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Graph paths
accuracy_graph = "static/accuracy_graph.png"
feature_graph = "static/feature_importance.png"

# -------------------------------
# 3️⃣ Home route
# -------------------------------
@app.route('/')
def home():
    return render_template("index.html")  # Your input form

# -------------------------------
# 4️⃣ Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Input validation
        if any(val < 0 for val in [N,P,K,temperature,humidity,ph,rainfall]):
            return "All inputs must be non-negative numbers!"

        # Prepare input for model
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Predict crop
        predicted_crop = model.predict(input_features)[0]

        # Explain why crop is recommended
        explanation = {
            "N": N,
            "P": P,
            "K": K,
            "Temperature": f"{temperature} °C",
            "Humidity": f"{humidity} %",
            "pH": ph,
            "Rainfall": f"{rainfall} mm",
            "Reason": f"Based on these soil nutrients and climatic conditions, the AI model predicts {predicted_crop} as the most suitable crop."
        }

        return render_template(
            "result.html",
            crop=predicted_crop,
            explanation=explanation,
            accuracy_graph=accuracy_graph,
            feature_graph=feature_graph
        )
    
    except ValueError:
        return "Please enter valid numeric values for all fields."

# -------------------------------
# 5️⃣ Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)