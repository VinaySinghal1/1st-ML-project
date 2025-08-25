from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    try:
        # Get form data safely
        gender = request.form.get('gender', '')
        race_ethnicity = request.form.get('race_ethnicity', '')
        parental_level_of_education = request.form.get('parental_level_of_education', '')
        lunch = request.form.get('lunch', '')
        test_preparation_course = request.form.get('test_preparation_course', '')
        
        # Validate numeric inputs
        try:
            reading_score = float(request.form.get('reading_score', ''))
            writing_score = float(request.form.get('writing_score', ''))
            
            if not (0 <= reading_score <= 100 and 0 <= writing_score <= 100):
                return render_template('home.html', error="Scores must be between 0 and 100")
                
        except ValueError:
            return render_template('home.html', error="Invalid score values")

        # Create CustomData instance
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        
        # Check if prediction was successful
        if isinstance(results, list) and len(results) > 0:
            return render_template('home.html', result=results[0])
        else:
            return render_template('home.html', error="No prediction results received")
            
    except Exception as e:
        return render_template('home.html', error=f"An error occurred: {str(e)}")
if __name__ == "__main__":
    app.run(host="0.0.0.0")