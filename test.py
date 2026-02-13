import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open('rainfall.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

@app.route('/') # route to display the home page
def home():
    return render_template('index.html')  # rendering the home page


@app.route('/predict', methods=["POST", "GET"])
def predict():
    input_feature = [x for x in request.form.values()]

    features_values = np.array(input_feature, dtype=float).reshape(1, -1)

    names = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
             'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
             'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday',
             'WindGustDir', 'WindDir9am', 'WindDir3pm', 'year', 'month', 'day']

    data = pandas.DataFrame(features_values, columns=names)

    data = scale.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        return render_template("chance.html")
    else:
        return render_template("noChance.html")


if __name__ == "__main__":
    app.run(debug=True)