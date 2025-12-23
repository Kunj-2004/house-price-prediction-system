# House Price Prediction System

This project predicts house prices using a Machine Learning model
and exposes the prediction via a FastAPI REST API.

## Tech Stack
- Python
- Scikit-learn
- FastAPI
- Uvicorn

## Model
- Linear Regression (best performing after comparison)
- Features scaled using StandardScaler

## How to Run
```bash
pip install -r requirements.txt
cd App
python -m uvicorn main:app --reload


API Endpoint: POST /predict

Sample Input : 

{
  "Avg_Area_Income": 75000,
  "Avg_Area_House_Age": 5,
  "Avg_Area_Number_of_Rooms": 7,
  "Avg_Area_Number_of_Bedrooms": 4,
  "Area_Population": 30000
}

Sample Output:

{
    "🏠 Predicted House Price:": "$554075",
    "🏠 Predicted House Price in Rupees: ": "₹46,542,300"
}
