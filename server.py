from flask import Flask, jsonify, request
from joblib import load
import pandas as pd
from flask_cors import CORS
from main import score_to_grade

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})  # specify the origin


# Load the model and the scaler when the server starts
model = load('model.joblib')
scaler = load('scaler.joblib')

@app.route('/')
def home():
    return "Server is running"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the request
    data = request.json

    # Convert the input values to a DataFrame
    df = pd.DataFrame(data, index=[0])


    # Normalize the input values
    df_scaled = scaler.transform(df[['mid1', 'mid2', 'quiz']])

    # Use the model to make a prediction
    prediction = model.predict(df_scaled)

    # Convert the prediction to a grade
    grade = score_to_grade(prediction[0])

    # Return the prediction and the grade
    return jsonify({'final_score': int(prediction[0]), 'final_grade': grade})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
