from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
from src.pipeline.predict_pipeline import PredictPipeline, CustomData



app = Flask(__name__)

# Load model and preprocessor
model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

model = pickle.load(open(model_path, "rb"))
preprocessor = pickle.load(open(preprocessor_path, "rb"))

@app.route('/')
def index():
    return render_template('index.html')  # Form input

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            return render_template('home.html')

        df=CustomData(
            Hours_Studied=int(request.form.get('Hours_Studied')),
            Attendance=int(request.form.get('Attendance')),
            Parental_Involvement=request.form.get('Parental_Involvement'),
            Access_to_Resources=request.form.get('Access_to_Resources'),
            Extracurricular_Activities=request.form.get('Extracurricular_Activities'),
            Sleep_Hours=int(request.form.get('Sleep_Hours')),
            Previous_Scores=int(request.form.get('Previous_Scores')),
            Motivation_Level=request.form.get('Motivation_Level'),
            Internet_Access=request.form.get('Internet_Access'),
            Tutoring_Sessions=int(request.form.get('Tutoring_Sessions')),
            Family_Income=request.form.get('Family_Income'),
            Teacher_Quality=request.form.get('Teacher_Quality'),
            School_Type=request.form.get('School_Type'),
            Peer_Influence=request.form.get('Peer_Influence'),
            Physical_Activity=int(request.form.get('Physical_Activities')),
            Learning_Disabilities=request.form.get('Learning_Disabilities'),
            Parental_Education_Level=request.form.get('Parental_Education_Level'),
            Distance_from_Home=request.form.get('Distance_from_Home'),
            Gender=request.form.get('Gender')
        )
        pred_df = df.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        results = np.round(results, 2)  # Round the predictions to 2 decimal places
        return render_template('index.html', prediction_text=f'Predicted Exam Score: {results[0]}')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)  # Set debug=True for development
