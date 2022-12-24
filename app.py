import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("C:\\Users\\sefa\\Desktop\\FlaskDeployment\\FlaskDeployment\\DataGlacier_4.week\\model.plk", "rb"))


@flask_app.route("/")
def index():
    return render_template("homem.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("homem.html", result_of_prediction="Diabetic {}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True, port=5421)
