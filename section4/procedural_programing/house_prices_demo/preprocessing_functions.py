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

