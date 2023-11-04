import os
import comet_ml
import pandas as pd
import math
import random
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score
from utils.model_utils import *


def baseline_model(data, comet: bool):
    models = {}
    feature_list = [["distance_to_goal"], ["shooting_angle"], ["distance_to_goal","shooting_angle"]]
    experiments = {}
    for feature in feature_list:
        if comet:
            experiment = comet_ml.Experiment(
                # my_key=os.environ.get('COMET_API_KEY')
                # api_key=my_key,???doesn't work with the env file I created
                api_key="uY4cSO6Xov0q2zrVACUZfYdkt",
                project_name="logistic_regression" + "_".join(feature),
                workspace="nhl-analytics-milestone-2"
            )
            experiment.set_name("logistic_regression" + "_".join(feature))
            experiments["logistic_regression_" + "_".join(feature)] = experiment
        
        X_train, y_train, X_val, y_val = get_train_validation(data, feature, "isGoal", 0.2, balanced=True)
        lr_model = LogisticRegression(random_state=42)
        # train model
        lr_model.fit(X_train, y_train)
        pickle.dump(lr_model, open("./models/logistic_regression" + "_".join(feature) + ".pkl", "wb"))
        # score model (training set)
        score_training = lr_model.score(X_train, y_train)
        # score model (validation set)
        score_validation = lr_model.score(X_val, y_val)
    
        # Class predictions and probabilities 
        val_preds = lr_model.predict(X_val)
        score_prob = lr_model.predict_proba(X_val)[:, 1]
        f1 = f1_score(y_val, val_preds, average="macro")
        models["logistic_regression" + "_".join(feature)] = {"model": lr_model, "val_preds": val_preds, "score_prob": score_prob,
                                                       "f1": f1}
        print(score_training, score_validation, f1)
        if comet:
            experiment.log_model("logistic_regression" + "_".join(feature),
                                 "models/logistic_regression" + "_".join(feature) + ".pkl")
            experiment.log_metric("train_score", score_training)
            experiment.log_metric("validation_score", score_validation)
            experiment.log_metric("f1_score", f1)
            experiment.log_confusion_matrix(y_val.astype('int32'), val_preds.astype('int32'))
            experiment.end()
        
    return (X_train, y_train, X_val, y_val), models, experiments


data = pd.read_csv('tidy_data.csv', index_col=False)
data.dropna(inplace=True)
# df.reset_index(drop=True)
print(data.isna().sum())
print(baseline_model(data, comet=True))