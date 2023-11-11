import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def preprocess(df: pd.DataFrame):
    df.dropna(inplace=True)

    # Define features and target variable
    data = pd.get_dummies(df, columns=['ShotType'], prefix='ShotType')
    data = pd.get_dummies(data, columns=['LastEvent'], prefix='LastEvent')

    y = data['isGoal']
    data = data.drop(['Unnamed: 0', 'isGoal', 'GameTime'],
                     axis=1)  # , 'Period', 'XCoord', 'YCoord', 'LastEvent_XCoord',
    # 'LastEvent_YCoord', 'TimeLastEvent', 'LastEvent', 'DistanceLastEvent'], axis=1)

    # , 'GameTime', 'Period', 'XCoord', 'YCoord',
    #                 'LastEvent', 'LastEvent_XCoord', 'LastEvent_YCoord', 'TimeLastEvent', 'DistanceLastEvent',
    #                 'AngleChange', 'Speed'

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
    columns_to_scale = ['DistanceToGoal', 'ShootingAngle', 'Speed', 'TimeLastEvent', 'DistanceLastEvent']
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

