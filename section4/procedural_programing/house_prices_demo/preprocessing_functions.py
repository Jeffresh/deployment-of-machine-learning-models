import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib

# Individiual pre-processing and training functions


def load_data(df_path):
    '''
    Function loads data for training
    '''

    return pd.read_csv(df_path)


def divide_train_test(df, target):
    '''
    Function divides data set in train and test 
    '''
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),
                                                        df[target],
                                                        test_size=0.2,
                                                        random_state=0)

    return X_train, X_test, y_train, y_test


def extract_cabin_letter(df, var):
    return df['cabin'].str[0]


def add_missing_indicator(df, var):
    return np.where(df[var].isnull(), 1, 0)


def impute_na(df, var, replacement='Missing'):
    return df[var].fillna(replacement)


def remove_rare_labels(df, var, frequent_labels, replacement='Rare'):
    return np.where(df[var].isin(frequent_labels), df[var], replacement)


def encode_categorical(df, var):
    df = df.copy()
    df = pd.concat(
        [df, pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
    df.drop(labels=[var], axis=1, inplace=True)

    return df


def check_dummy_variables(df, dummy_list):
    missing_vars = [var for var in dummy_list if var not in df.columns]

    if len(missing_vars) == 0:
        print('All dummies were added')
    else:
        for var in missing_vars:
            df[var] = 0

    return df


def train_scaler(df, ouput_path):
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, ouput_path)
    return scaler
