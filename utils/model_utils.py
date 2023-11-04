import pandas as pd
import numpy as np
import random
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler



# train_test_split(distance_goal_data[:,0], distance_goal_data[:,1], test_size=0.2, random_state=42)
def get_train_validation(df: pd.DataFrame, data_features: list, labels_features: list, val_ratio: float,
                         balanced: bool = True, sampling='under'):
    """
    Get train and validation dataset. You can choose the size of each dataset and the column for labels and data.

    Args:
    	df: Data frame for the model
    	data_features: List of columns to be use for the inputs.
    	labels_features: List of columns to be use for labels
    	val_ratio: Size of the validation dataset
    	balanced: Booléan to tell the function to balanced data or not
    	sampling: Sampling method to use (over for oversampling, under for undersampling)
    Returns:
    	x_train, y_train, x_val, y_val
    """
    train, val = train_test_split(df, test_size=val_ratio, random_state=42)

    x, y = split_data_and_labels(train, data_features, labels_features)
    if balanced:
        if sampling == 'under':
            x, y = RandomUnderSampler(random_state=42).fit_resample(x, y)
        elif sampling == 'over':
            x, y = RandomOverSampler(random_state=42).fit_resample(x, y)

    x_val, y_val = split_data_and_labels(val, data_features, labels_features)
    return x, y, x_val, y_val

def split_data_and_labels(data, data_features, labels_features):
    x = data[data_features].to_numpy().reshape(-1, len(data_features))
    y = data[labels_features].to_numpy().reshape(-1, 1)
    return x, y

def verify_label_distribution(y):
    """
    To check the label distribution of y, verify if the resampling is finished.

    Args:
    	df: Data frame for the model
    	data_features: List of columns to be use for the inputs.
    	labels_features: List of columns to be use for labels
    	val_ratio: Size of the validation dataset
    	balanced: Booléan to tell the function to balanced data or not
    	sampling: Sampling method to use (over for oversampling, under for undersampling)
    Returns:
    	x_train, y_train, x_val, y_val
    """
    non_goal_ratio = len(y[y == 0])/len(y)
    goal_ratio = 1 - non_goal_ratio
    return "The ratio of labels that is non goal", non_goal_ratio, "The ratio of labels that is goal", goal_ratio