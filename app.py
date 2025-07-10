# app.py

from flask import Flask, render_template, request
import os
from Fertilizer_Pred.pipeline.prediction_pipeline import FertilizerPredictor

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            input_data = {
                'temperature': request.form['temperature'],
                'humidity': request.form['humidity'],
                'soil_type': request.form['soil_type'],
                'nitrogen': request.form['nitrogen'],
                'potassium': request.form['potassium'],
                'phosphorous': request.form['phosphorous'],
                'ph': request.form['ph'],
                'rainfall': request.form['rainfall']
            }

            predictor = FertilizerPredictor()
            prediction = predictor.predict(input_data)
            prediction = predictor.predict(input_data)


            return render_template('results.html', prediction=prediction)

        except Exception as e:
            print('The Exception message is:', e)
            return 'Something went wrong. Please check your input values.'

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
