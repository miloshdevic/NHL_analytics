import pickle
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
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

    if len(feature) == 2:
        lr_model = pickle.load(open("comet_models/LogisticRegression" + feature[0] + '_' + feature[1] + ".pkl", "rb"))
    else:
        lr_model = pickle.load(open("comet_models/LogisticRegression" + feature[0] + ".pkl", "rb"))

    # Load the model from Comet Registry
    # lr_model = pickle.load(open("comet_models/LogisticRegression" + feature[0] + ".pkl", "rb"))

    # test the baseline models
    df.dropna(inplace=True)
    X_test = df[feature].to_numpy()
    y_test = df[["isGoal"]].to_numpy()

    # predictions
    prediction = lr_model.predict_proba(X_test)[:, 1]

    # UNCOMMENT THIS SECTION TO GET THE METRICS FOR THIS MODEL'S PREDICTIONS
    # pred = lr_model.predict(X_test)
    #
    # # Evaluate the model
    # print("\nLOGISTIC REGRESSION MODEL METRICS:")
    # test_accuracy = accuracy_score(y_test, pred)
    # print(f'Test accuracy: {test_accuracy * 100:.2f}%')
    #
    # # confusion matrix
    # ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
    # plt.show()
    # print(classification_report(y_test, pred))

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
        ['DistanceLastEvent', 'TimeLastEvent', 'isEmptyNet', 'Period', 'LastEvent_YCoord', 'DistanceToGoal']]
    y_test = df['isGoal']

    # predictions
    prediction = xgb_model.predict_proba(X_test)[:, 1]

    # UNCOMMENT THIS SECTION TO GET THE METRICS FOR THIS MODEL'S PREDICTIONS
    # pred = xgb_model.predict(X_test)
    #
    # print("\n\nXGBOOST MODEL METRICS:")
    #
    # # Evaluate the model
    # test_accuracy = accuracy_score(y_test, pred)
    # print(f'Test accuracy: {test_accuracy * 100:.2f}%')
    #
    # # confusion matrix
    # ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
    # plt.show()
    # print(classification_report(y_test, pred))

    return prediction, y_test


def predict_neural_network(df):
    df = df.copy()

    # preprocess the test data in the same way as the training data
    X_test, y_test = preprocess_neural_network_rfc(df)

    # Make predictions
    nn_model = tf.keras.models.load_model("comet_models/neural_network_rfc_final.keras")
    prediction = nn_model.predict(X_test)

    # UNCOMMENT THIS SECTION TO GET THE METRICS FOR THIS MODEL'S PREDICTIONS
    # print("\n\nNEURAL NETWORK MODEL METRICS:")
    #
    # # Evaluate the model
    # test_loss, test_accuracy = nn_model.evaluate(X_test, y_test)
    # print(f'Test accuracy: {test_accuracy * 100:.2f}%')
    #
    # preds = np.round(nn_model.predict(X_test), 0)
    #
    # # confusion matrix
    # ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot()
    # plt.show()
    # print(classification_report(y_test, preds))

    return prediction, y_test
