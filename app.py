from flask import Flask, request, jsonify
import os
import numpy as np
import yaml
import joblib

app = Flask(__name__)


def model_prediction(data):
    model = joblib.load('./src/artifacts/1/c4152475dba54ad6b949e39753ae70f9/artifacts/model/model.pkl')
    prediction = model.predict(data)

    return prediction

@app.route('/predict', methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.json:
                dict_req = dict(request.json)
                response = model_prediction(dict_req)

                print("Prediction: ", response)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)        


