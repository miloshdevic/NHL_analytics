import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import comet_ml
from comet_ml import API
from utils.model_utils import *
from utils.plot_utils import *


def get_test_data_baseline_models():
    df_reg_season = pd.read_csv("baseline_model_test_data_reg_season.csv")  # regular season data
    df_playoff = pd.read_csv("baseline_model_test_data_playoffs.csv")  # playoffs data
    return df_reg_season, df_playoff


def get_test_data_advanced_models():
    df_reg_season = pd.read_csv("advanced_models_test_data_reg_season.csv")  # regular season data
    df_playoff = pd.read_csv("advanced_models_test_data_playoffs.csv")  # playoffs data
    return df_reg_season, df_playoff


def download_model(api_key, workspace_name, model_name, version):
    api = API(api_key=f"{api_key}")

    # Download a Registry Model:
    api.download_registry_model(f"{workspace_name}", f"{model_name}", f"{version}",
                                output_path="comet_models/", expand=True)


def predict_logreg(df):
    df = df.copy()

    # TODO: complete this function to test the baseline models
    pass


def predict_xgboost(df):
    df = df.copy()

    # Load the model from Comet Registry
    xgb_model = pickle.load(open("comet_models/xgboost_3rd.pkl", "rb"))

    # preprocess the test dataset
    # dropna
    df.dropna(inplace=True)
    # selecting the same features as in the training dataset
    X_test = df[
        ['DistanceLastEvent', 'TimeLastEvent', 'LastEvent_XCoord', 'isEmptyNet', 'LastEvent_YCoord', 'DistanceToGoal']]
    y_test = df['isGoal']

    # predictions
    prediction = xgb_model.predict(X_test)
    return prediction, y_test


def predict_decision_tree(df):
    df = df.copy()

    # TODO: complete this function to test the decision tree model
    pass


def predict_neural_network(df):
    df = df.copy()

    # preprocess the test data in the same way as the training data
    X_test, y_test = preprocess_neural_network_rfc(df)

    # Make predictions
    model = tf.keras.models.load_model("comet_models/neural_network_rfc_ft.keras")
    prediction = model.predict(X_test)

    return prediction, y_test


if __name__ == '__main__':
    api_key = "cX0b8GkNwZ3M1Bzj4d2oeqFmd"  # insert your API key
    workspace_name = "nhl-analytics-milestone-2"

    # get test data for baseline models
    df_rs_bm, df_playoffs_bm = get_test_data_advanced_models()

    # get test data for advanced models
    df_rs_am, df_playoffs_am = get_test_data_advanced_models()

    # to have all the predictions in one figure (1 for regular season and 1 for playoffs)
    all_predictions_rs = []
    all_y_true_rs = []
    all_predictions_pl = []
    all_y_true_pl = []

    # BASELINE MODELS
    ###############################################################

    # download logistic regression models programmatically using the Python API
    # TODO: specify name and version
    # download_model(api_key, workspace_name, model_name="", version="")

    # test baseline models:
    # TODO: insert here code to call the function to run the baseline models

    # add prediction and y_true to the lists
    # TODO

    ###############################################################

    # ADVANCED MODELS
    ###############################################################

    # download xgboost model programmatically using the Python API
    download_model(api_key, workspace_name, model_name="xgboost_1st", version="1.2.0")

    # test XGBoost models:
    # model_xgb_rs, prediction_xgb_rs = predict_xgboost(df_rs_am)  # regular season
    # model_xgb_pl, prediction_xgb_pl = predict_xgboost(df_playoffs_am)  # playoffs

    # add prediction and y_true to the lists
    # TODO

    ###############################################################

    # OTHER MODEL
    ###############################################################

    # download neural network model
    download_model(api_key, workspace_name, model_name="first-neural-network", version="1.8.0")

    # test the neural network:
    prediction_nn_rs, y_true_nn_rs = predict_neural_network(df_rs_am)  # regular season
    prediction_nn_pl, y_true_nn_pl = predict_neural_network(df_playoffs_am)  # playoffs

    # add prediction and y_true to the lists
    all_predictions_rs.append(prediction_nn_rs)
    all_predictions_pl.append(prediction_nn_pl)
    all_y_true_rs.append(y_true_nn_rs)
    all_y_true_pl.append(y_true_nn_pl)

    # plot curves for regular season
    # TODO: these curves will be removed bc we need to have the curves of all the models in one figure
    # will be removed before submission
    plot_roc_curve_nn(prediction_nn_rs, y_true_nn_rs)  # plot the ROC curves
    # make the probability predictions 1D
    prediction_nn_rs = prediction_nn_rs.flatten()
    shot_prob_model_percentile_nn(prediction_nn_rs, y_true_nn_rs)  # plot goal percentile curves
    plot_cumulative_sum_nn(prediction_nn_rs, y_true_nn_rs)  # plot cumulative goals
    plot_calibration_curve_nn(prediction_nn_rs, y_true_nn_rs)  # plot calibration curves

    # plot curves for playoffs
    # TODO: these curves will be removed bc we need to have the curves of all the models in one figure
    # will be removed before submission
    plot_roc_curve_nn(prediction_nn_pl, y_true_nn_pl)  # plot the ROC curves
    # make the probability predictions 1D
    prediction_nn_pl = prediction_nn_pl.flatten()
    shot_prob_model_percentile_nn(prediction_nn_rs, y_true_nn_rs)  # plot goal percentile curves
    plot_cumulative_sum_nn(prediction_nn_pl, y_true_nn_pl)  # plot cumulative goals
    plot_calibration_curve_nn(prediction_nn_pl, y_true_nn_pl)  # plot calibration curves

    ###############################################################

    # PLOT ONE FIGURE FOR ALL THE MODELS
    ###############################################################

    # TODO: plot one ROC figure with all the curves of each model (same for the other 3 figures)

    ###############################################################
