"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
import pickle
from pathlib import Path
import logging
import numpy as np
import json
from comet_ml import API
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib

import ift6758

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

app = Flask(__name__)

# Placeholder for the model download status
model = None
current_model = None
COMET_API_KEY = os.environ.get("COMET_API_KEY")
model_downloaded = False


# @app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    global COMET_API_KEY

    # Ensure the log file exists
    Path(LOG_FILE).touch()

    # basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    with open('COMET_API_KEY', 'r') as f:
       COMET_API_KEY = f.read()

    # TODO: any other initialization before the first request (e.g. load default model)
    # TODO: Load your default model here using joblib or any other method
    # Replace the following line with your model loading logic
    # api = API(str(COMET_API_KEY))
    api = API(api_key="cX0b8GkNwZ3M1Bzj4d2oeqFmd")
    # api = API(api_key=str(COMET_API_KEY))
    api.download_registry_model("nhl-analytics-milestone-2", "logisticregressiondistancetogoal",
                                "1.1.0", output_path="comet_models/", expand=True)
    model = pickle.load(open('comet_models/LogisticRegressionDistanceToGoal.pkl', 'rb'))
    # model = joblib.load("models/logistic_regression_distance_to_goal.pkl")
    app.logger.info('\nDefault model downloaded from Comet!\n')


before_first_request()

# @app.route("/", methods=["GET"])
# def index():
#     return jsonify({"message": "Welcome to the Flask app!"})


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    with open(LOG_FILE, "r") as log_file:
        logs_data = log_file.read().splitlines()

    return jsonify({"logs": logs_data})


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    global model, current_model, model_downloaded, COMET_API_KEY

    with open('COMET_API_KEY', 'r') as f:
        COMET_API_KEY = f.read()

    model_name = ""
    current_model = json['model']
    app.logger.info(model)

    if json['model'] == 'logisticregressiondistancetogoal':
        model_name = 'LogisticRegressionDistanceToGoal.pkl'
    elif json['model'] == 'logisticregressionshootingangle':
        model_name = 'LogisticRegressionShootingAngle.pkl'
    elif json['model'] == 'logisticregressiondistancetogoal_shootingangle':
        model_name = 'LogisticRegressionDistanceToGoal_ShootingAngle.pkl'

    # TODO: check to see if the model you are querying for is already downloaded
    if model_downloaded:
        app.logger.info("\nModel already downloaded. Loading the existing model...\n")
        return jsonify({"\nstatus": "Model already downloaded\n"})

    current_model = json['model']

    if os.path.isfile(f"comet_models/{model_name}"):
        model = pickle.load(open(f"comet_models/{model_name}", 'rb'))
        app.logger.info(model_name)
        app.logger.info("\nModel present!\n")
    else:
        app.logger.info("\nModel not downloaded yet, downloading it now...\n")
        api = API(str(COMET_API_KEY))
        api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="comet_models/",
                                    expand=True)
        model = pickle.load(open(f"comet_models/{model_name}", 'rb'))
        model_downloaded = True

    response = f'\n{model} has been downloaded successfully!\n'
    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # global current_model

    X = pd.DataFrame.from_dict(json) # pd.read_json(json.dumps(json), orient='records')

    if current_model == 'logisticregressiondistancetogoal':
        X = X[['DistanceToGoal']].to_numpy()
        # X[['DistanceToGoal']] = StandardScaler().fit_transform(X[['DistanceToGoal']])
        # X = X.to_numpy()
    elif current_model == 'logisticregressionshootingangle':
        X = X[['ShootingAngle']].to_numpy()
        # X[['ShootingAngle']] = StandardScaler().fit_transform(X[['ShootingAngle']])
        # X = X.to_numpy()
    elif current_model == 'logisticregressiondistancetogoal_shootingangle':
        X = X[['DistanceToGoal', 'ShootingAngle']].to_numpy()
        # X[['DistanceToGoal', 'ShootingAngle']] = StandardScaler().fit_transform(X[['DistanceToGoal', 'ShootingAngle']])
        # X = X.to_numpy()

    predictions = pd.Series(model.predict_proba(X)[:, 1])

    return predictions.to_json()
