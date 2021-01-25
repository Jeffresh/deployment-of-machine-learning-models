from os import replace

import numpy as np

from .preprocessing_functions import *

from .config import *

import warnings

warnings.simplefilter(action='ignore')

# Training step - important to perpetuate the model

# Load data
data = load_data(PATH_TO_DATASET)

# divide data et
X_train, X_test, y_train, y_test = divide_train_test(data, TARGET)

# impute categorical variables

for var in CATEGORICAL_TO_IMPUTE:
    X_train[var] = impute_na(X_train, var)

# impute numerical variable

for var in NUMERICAL_TO_IMPUTE:

    X_train[var] = impute_na(
        X_train, var, replacement=LOTFRONTAGE_MODE if var == 'LotFrontage' else 'Missing')

# capture elapsed time

X_train[YEAR_VARIABLE] = elapsed_years(
    X_train, YEAR_VARIABLE, ref_var='YrSold')

# log transform numerical variables

for var in NUMERICAL_LOG:
    X_train[var] = log_transform(X_train, var)

# Group rare labels

for var in CATEGORICAL_ENCODE:
    X_train[var] = remove_rare_labels(X_train, var, FREQUENT_LABELS[var])

# encode categorical variables

for var in CATEGORICAL_ENCODE:
    X_train[var] = encode_categorical(X_train, var), ENCODING_MAPPINGS[var]

# train scaler and save

scaler = train_scaler(X_train[FEATURES], OUTPUT_SCALER_PATH)

# scale train set

X_train = scaler.transform(X_train[FEATURES])

# train model and save

train_model(X_train, np.log(y_train), OUTPUT_MODEL_PATH)


print('Finished training')
