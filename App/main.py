from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model & scaler
model = joblib.load("..\Model\house_price_model.pkl")
scaler = joblib.load("..\Model\scaler.pkl")

# Define input schema
class HouseInput(BaseModel):
    Avg_Area_Income: float|int
    Avg_Area_House_Age: float|int
    Avg_Area_Number_of_Rooms: float|int
    Avg_Area_Number_of_Bedrooms: float|int
    Area_Population: float|int

@app.post("/predict")
def predict_price(data: HouseInput):

    # Create DataFrame EXACTLY like training
    new_data = pd.DataFrame([{
        'Avg. Area Income': data.Avg_Area_Income,
        'Avg. Area House Age': data.Avg_Area_House_Age,
        'Avg. Area Number of Rooms': data.Avg_Area_Number_of_Rooms,
        'Avg. Area Number of Bedrooms': data.Avg_Area_Number_of_Bedrooms,
        'Area Population': data.Area_Population
    }])

    # Apply SAME preprocessing
    new_data_scaled = scaler.transform(new_data)

    # Predict
    predicted_price = model.predict(new_data_scaled)[0]

    rupees = int(predicted_price) * 84  # convert USD to INR

    return {
        "🏠 Predicted House Price:": f"${int(predicted_price)}",
        "🏠 Predicted House Price in Rupees: ": f"₹{rupees:,}"
    }
