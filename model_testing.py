import pandas as pd
import tensorflow as tf
from tensorflow import keras
from comet_ml import API
from utils.model_utils import *


def get_test_data_baseline_models():
    df_reg_season = pd.read_csv("baseline_model_test_data_reg_season.csv")  # regular season data
    df_playoff = pd.read_csv("baseline_model_test_data_playoffs.csv")  # playoffs data
    return df_reg_season, df_playoff


def get_test_data_advanced_models():
    df_reg_season = pd.read_csv("advanced_models_test_data_reg_season.csv")  # regular season data
    df_playoff = pd.read_csv("advanced_models_test_data_playoffs.csv")  # playoffs data
    return df_reg_season, df_playoff


def predict_logreg(df):
    df = df.copy()

    # TODO: complete this function to test the baseline models
    pass


def predict_xgboost(df):
    df = df.copy()

    # TODO: complete this function to test the XGBoost models
    pass


def predict_decision_tree(df):
    df = df.copy()

    # TODO: complete this function to test the decision tree model
    pass


def predict_neural_network(df):
    df = df.copy()

    # preprocess the test data in the same way as the training data
    X_test, y_test = preprocess_neural_network(df)

    # Make predictions
    model = tf.keras.models.load_model("models/neural_network.keras")
    prediction = model.predict(X_test)

    return model, prediction


if __name__ == '__main__':
    # TODO: THE SECTION BELOW NEEDS TO BE MODIFIED ACCORDINGLY
    # I got this from the website in the instructions google doc
    #####################################################################

    api = API()

    # TODO: INSERT NAMES
    workspace_name = "nhl-analytics-milestone-2"
    project_name = ""
    experiment_name = ""
    model_name = ""

    experiment = api.get(f"{workspace_name}/{project_name}/{experiment_name}",
                         output_path="./", expand=True)

    # Download an Experiment Model:
    experiment.download_model(f"{model_name}", output_path="./", expand=True)

    # Download a Registry Model:
    api.download_registry_model(f"{workspace_name}", f"{model_name}", "1.0.0",
                                output_path="./", expand=True)

    #####################################################################

    # get test data for baseline models
    df_rs_bm, df_playoffs_bm = get_test_data_advanced_models()

    # get test data for advanced models
    df_rs_am, df_playoffs_am = get_test_data_advanced_models()

    # test baseline models:
    # TODO: insert here code to call the function to run the baseline models

    # test XGBoost models:
    # TODO: insert here code to call the function to run the XGBoost models

    # test the neural network:
    model_nn_rs, prediction_nn_rs = predict_neural_network(df_rs_am)  # regular season
    model_nn_pl, prediction_nn_pl = predict_neural_network(df_playoffs_am)  # playoffs
