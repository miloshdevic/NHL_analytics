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

from comet_ml import API
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib

import ift6758

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

app = Flask(__name__)

# MODIFY NAME OF MODEL TO BE DOWNLOADED HERE
model = None  # "logistic_regression_distance_to_goal.pkl"
current_model = None
COMET_API_KEY = os.environ.get("COMET_API_KEY")


# Placeholder for the loaded model
# loaded_model = None

# Placeholder for the model download status
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
    api = API(str(COMET_API_KEY))
    api.download_registry_model("nhl-analytics-milestone-2", "logisticregressiondistancetogoal",
                                "1.1.0", output_path="comet_models/", expand=True)
    model = pickle.load(open('comet_models/LogisticRegressionDistanceToGoal.pkl', 'rb'))
    # model = joblib.load("models/logistic_regression_distance_to_goal.pkl")
    app.logger.info('Default model downloaded from Comet!')


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

    model = "logistic_regression_distance_to_goal.pkl"
    app.logger.info(model)

    # TODO: check to see if the model you are querying for is already downloaded
    if model_downloaded:
        app.logger.info("Model already downloaded. Loading the existing model...")
        return jsonify({"status": "Model already downloaded"})

    current_model = json['model']

    if os.path.isfile(f"models/{model}"):
        loaded_model = pickle.load(open(f"models/{model}", 'rb'))
        app.logger.info(model)
        app.logger.info("Model present!")
    else:
        app.logger.info("Model not downloaded yet, downloading it now...")
        api = API(str(COMET_API_KEY))
        api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="./", expand=True)
        loaded_model = pickle.load(open(f"models/{model}", 'rb'))

    response = f'{model} has been downloaded successfully!'
    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!

    # TODO: if yes, load that model and write to the log about the model change.
    # eg: app.logger.info(<LOG STRING>)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    # try:
    #     # Example: Using requests to download the model file
    #     # TODO: Modify the model_url based on your model registry setup
    #     model_url = f"https://comet-ml/models/{json['workspace']}/{json['model']}/{json['version']}/download"
    #     response = request.get(model_url)
    #     response.raise_for_status()
    #
    #     # Save the downloaded model
    #     with open("downloaded_model.pkl", "wb") as model_file:
    #         model_file.write(response.content)
    #
    #     # Load the downloaded model
    #     loaded_model = joblib.load("downloaded_model.pkl")
    #
    #     # Update the model download status
    #     model_downloaded = True
    #
    #     app.logger.info("Model downloaded and loaded successfully.")
    #     return jsonify({"status": "Model downloaded and loaded successfully"})
    #
    # except Exception as e:
    #     # Log the failure and keep the currently loaded model
    #     app.logger.error(f"Failed to download model. Error: {str(e)}")
    #     return jsonify({"error": f"Failed to download model. Error: {str(e)}"})

    # raise NotImplementedError("TODO: implement this endpoint")
    #
    # response = None
    #
    # app.logger.info(response)
    # return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data

    global current_model

    json = request.get_json()
    app.logger.info(json)

    # TODO:
    # if loaded_model is None:
    #     return jsonify({"error": "Model not loaded. Please load or download a model first."})

    X = json['X_logreg']

    # TODO: Preprocess input data if needed (convert to DataFrame, etc.)
    input_data = pd.DataFrame(X)  # pd.DataFrame.from_dict(json, orient="index").transpose()
    # response = pd.Series(model.predict_proba(input_data)[::, 1])

    # TODO: Perform predictions using the loaded model
    # Example:
    predictions = model.predict_proba(input_data)

    response = {"predictions": predictions}  # Update with actual predictions

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!
