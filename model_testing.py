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

    
