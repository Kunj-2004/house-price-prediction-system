from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ Load model & scaler (same as FastAPI code)
model = joblib.load("Model/house_price_model.pkl")
scaler = joblib.load("Model/scaler.pkl")


# 🏠 Home page
@app.route("/")
def index():
    return render_template("index.html")


# 🔮 Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # ✅ Create DataFrame EXACTLY like training
    new_data = pd.DataFrame([{
        'Avg. Area Income': data['Avg. Area Income'],
        'Avg. Area House Age': data['Avg. Area House Age'],
        'Avg. Area Number of Rooms': data['Avg. Area Number of Rooms'],
        'Avg. Area Number of Bedrooms': data['Avg. Area Number of Bedrooms'],
        'Area Population': data['Area Population']
    }])

    # ✅ Apply same scaler
    new_data_scaled = scaler.transform(new_data)

    # ✅ Predict
    predicted_price = model.predict(new_data_scaled)[0]

    rupees = int(predicted_price) * 84

    return jsonify({
        #"Predicted Price ($)": f"${int(predicted_price)}",
        "predicted_price": round(float(rupees), 2)
        #"Predicted Price (₹)": f"₹{rupees:,}"
    })


if __name__ == "__main__":
    app.run(debug=True)
