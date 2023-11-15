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


def predict_logreg(df, feature):
    df = df.copy()
	
    # Load the model from Comet Registry
    lr_model = pickle.load(open("comet_models/logistic_regression_"+feature[0]+".pkl", "rb"))
    
    # test the baseline models
    df.dropna(inplace=True)
    X_test = df[feature].to_numpy()
    y_test = df[["isGoal"]].to_numpy()
    
    # predictions
    prediction = xgb_model.predict_proba(X_test)[:,1]
    return prediction, y_test


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
    prediction = xgb_model.predict_proba(X_test)[:,1]
    return prediction, y_test



def predict_neural_network(df):
    df = df.copy()

    # preprocess the test data in the same way as the training data
    X_test, y_test = preprocess_neural_network_rfc(df)

    # Make predictions
    model = tf.keras.models.load_model("comet_models/neural_network_rfc_final.keras")
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
    # specify name and version
    download_model(api_key, workspace_name, model_name="logisticregressiondistancetogoal", version="1.1.0")

    # add prediction and y_true to the lists
    # feature_list = [["DistanceToGoal"], ["ShootingAngle"], ["DistanceToGoal","ShootingAngle"]]
    # for feature in feature_list:
    	# model_lr_rs, prediction_lr_rs = predict_logreg(df_rs_am, feature)  # regular season
    	# model_lr_pl, prediction_lr_pl = predict_logreg(df_rs_pl, feature)  # playoffs
    	# all_predictions_rs.append(prediction_lr_rs)
    	# all_predictions_pl.append(all_predictions_pl)

    ###############################################################

    # ADVANCED MODELS
    ###############################################################

    # download xgboost model programmatically using the Python API
    download_model(api_key, workspace_name, model_name="xgboost_2", version="1.2.0")

    # test XGBoost models:
    # model_xgb_rs, prediction_xgb_rs = predict_xgboost(df_rs_am)  # regular season
    # model_xgb_pl, prediction_xgb_pl = predict_xgboost(df_playoffs_am)  # playoffs

    # add prediction and y_true to the lists
    # all_predictions_rs.append(prediction_xgb_rs)
    # all_predictions_pl.append(prediction_xgb_pl)

    ###############################################################

    # OTHER MODEL
    ###############################################################

    # download neural network model
    download_model(api_key, workspace_name, model_name="first-neural-network", version="1.10.0")

    # test the neural network:
    prediction_nn_rs, y_true_nn_rs = predict_neural_network(df_rs_am)  # regular season
    prediction_nn_pl, y_true_nn_pl = predict_neural_network(df_playoffs_am)  # playoffs

    # add prediction and y_true to the lists
    all_predictions_rs.append(prediction_nn_rs)
    all_predictions_pl.append(prediction_nn_pl)
    all_y_true_rs.append(y_true_nn_rs)
    all_y_true_pl.append(y_true_nn_pl)


    ###############################################################

    # PLOT ONE FIGURE FOR ALL THE MODELS
    ###############################################################

    # plot one ROC figure with all the curves of each model (same for the other 3 figures)

    ###############################################################
    labels = ['LRDistance', 'LRShooting', 'LRDistance_ShootingAngle', 'nn', 'XGBoost']
    linestyles = ['-', '-', '-', '-', '-']
    
    # For regular season data, plot 5 models together
    # ROC curve
    # plot_roc_curve(all_predictions_rs, all_y_true_rs, linestyles, labels)
    
    # goal_rate figure
    # percentile, percentile_pred, y_valid_df = shot_prob_model_percentile(all_predictions_rs[0], all_y_true_rs[0])
    # plot_goal_rate(all_predictions_rs, all_y_true_rs, labels)
    
    # cumulative proportion figure
    # plot_cumulative_sum(all_predictions_rs, all_y_true_rs, labels)
    
    # calibration figure
    # plot_calibration(all_predictions_rs, all_y_true_rs, labels)
    
    ###############################################################
    
    # For playoffs data, plot 5 models together
    # ROC curve
    # plot_roc_curve(all_predictions_pl, all_y_true_pl, linestyles, labels)
    
    # goal_rate figure
    # percentile, percentile_pred, y_valid_df = shot_prob_model_percentile(all_predictions_pl[0], all_y_true_pl[0])
    # plot_goal_rate(all_predictions_pl, all_y_true_pl, labels)
    
    # cumulative proportion figure
    # plot_cumulative_sum(all_predictions_pl, all_y_true_pl, labels)
    
    # calibration figure
    # plot_calibration(all_predictions_pl, all_y_true_pl, labels)
    
    
