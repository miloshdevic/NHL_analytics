import pandas as pd
import numpy as np
import random
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


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
    non_goal_ratio = len(y[y == 0]) / len(y)
    goal_ratio = 1 - non_goal_ratio
    return "The ratio of labels that is non goal", non_goal_ratio, "The ratio of labels that is goal", goal_ratio


def preprocess_neural_network_all_ft(df: pd.DataFrame):
    df.dropna(inplace=True)

    # Define features and target variable
    data = pd.get_dummies(df, columns=['ShotType'], prefix='ShotType')
    data = pd.get_dummies(data, columns=['LastEvent'], prefix='LastEvent')

    y = data['isGoal'].to_numpy()

    data = data.drop(['isGoal', 'GameID'], axis=1)

    # Make all non-numerical values numerical
    data['Rebound'] = data['Rebound'].astype(int)

    data['ShotType_Backhand'] = data['ShotType_Backhand'].astype(int)
    data['ShotType_Snap Shot'] = data['ShotType_Snap Shot'].astype(int)
    data['ShotType_Slap Shot'] = data['ShotType_Slap Shot'].astype(int)
    data['ShotType_Deflected'] = data['ShotType_Deflected'].astype(int)
    data['ShotType_Wrap-around'] = data['ShotType_Wrap-around'].astype(int)
    data['ShotType_Wrist Shot'] = data['ShotType_Wrist Shot'].astype(int)
    data['ShotType_Tip-In'] = data['ShotType_Tip-In'].astype(int)

    data['LastEvent_SHOT'] = data['LastEvent_SHOT'].astype(int)
    data['LastEvent_FACEOFF'] = data['LastEvent_FACEOFF'].astype(int)
    data['LastEvent_GOAL'] = data['LastEvent_GOAL'].astype(int)
    data['LastEvent_HIT'] = data['LastEvent_HIT'].astype(int)
    data['LastEvent_PENALTY'] = data['LastEvent_PENALTY'].astype(int)
    data['LastEvent_TAKEAWAY'] = data['LastEvent_TAKEAWAY'].astype(int)
    data['LastEvent_GIVEAWAY'] = data['LastEvent_GIVEAWAY'].astype(int)
    data['LastEvent_MISSED_SHOT'] = data['LastEvent_MISSED_SHOT'].astype(int)
    data['LastEvent_BLOCKED_SHOT'] = data['LastEvent_BLOCKED_SHOT'].astype(int)

    X = data

    # Data scaling
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)  # Standardize the features

    # Create a ColumnTransformer that applies the StandardScaler to the specified columns
    columns_to_scale = ['DistanceToGoal', 'ShootingAngle', 'Speed', 'TimeLastEvent', 'DistanceLastEvent', 'GameTime']
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num', StandardScaler(), columns_to_scale)
        ],
        remainder='passthrough'  # This includes the other columns as is
    )

    X_res_scaled = preprocessor.fit_transform(X)  # Standardize the features

    return X_res_scaled, y


def preprocess_neural_network_rfc(df: pd.DataFrame):
    df.dropna(inplace=True)
    y = df['isGoal'].to_numpy()

    data = pd.get_dummies(df, columns=['LastEvent'], prefix='LastEvent')

    # Make all non-numerical values numerical
    data['LastEvent_FACEOFF'] = data['LastEvent_FACEOFF'].astype(int)

    data = data[['TimeLastEvent', 'Speed', 'DistanceLastEvent', 'LastEvent_XCoord', 'LastEvent_YCoord',
               'LastEvent_FACEOFF', 'DistanceToGoal', 'XCoord', 'Period', 'GameTime']]

    X = data

    # Create a ColumnTransformer that applies the StandardScaler to the specified columns
    columns_to_scale = ['TimeLastEvent', 'Speed', 'DistanceLastEvent', 'LastEvent_XCoord', 'LastEvent_YCoord',
                        'DistanceToGoal', 'XCoord', 'GameTime']
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num', StandardScaler(), columns_to_scale)
        ],
        remainder='passthrough'  # This includes the other columns as is
    )

    X_res_scaled = preprocessor.fit_transform(X)  # Standardize the features

    return X_res_scaled, y


def preprocess_neural_network_corr(df: pd.DataFrame):
    df.dropna(inplace=True)
    y = df['isGoal'].to_numpy()

    data = pd.get_dummies(df, columns=['LastEvent'], prefix='LastEvent')

    data = data[['Speed', 'AngleChange', 'Rebound', 'isEmptyNet', 'LastEvent_GIVEAWAY', 'LastEvent_SHOT',
                 'LastEvent_FACEOFF', 'LastEvent_HIT', 'isEmptyNet', 'DistanceToGoal']]

    # Make all non-numerical values numerical
    data['Rebound'] = data['Rebound'].astype(int)

    data['LastEvent_GIVEAWAY'] = data['LastEvent_GIVEAWAY'].astype(int)
    data['LastEvent_SHOT'] = data['LastEvent_SHOT'].astype(int)
    data['LastEvent_FACEOFF'] = data['LastEvent_FACEOFF'].astype(int)
    data['LastEvent_HIT'] = data['LastEvent_HIT'].astype(int)

    X = data

    # Create a ColumnTransformer that applies the StandardScaler to the specified columns
    columns_to_scale = ['Speed', 'AngleChange', 'Rebound', 'DistanceToGoal']
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num', StandardScaler(), columns_to_scale)
        ],
        remainder='passthrough'  # This includes the other columns as is
    )

    X_res_scaled = preprocessor.fit_transform(X)  # Standardize the features

    return X_res_scaled, y


def balance_data(X, y):
    # Balancing the dataset
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res, y_res
