from math import sqrt
from section4.procedural_programing.house_prices_demo.train import X_test
from .preprocessing_functions import *
from .config import *

# ======= Scoring pipeline


def predict(data):

    # impute NA

    for var in CATEGORICAL_TO_IMPUTE:
        data[var] = impute_na(data, var, replacement='Missing')

    for var in NUMERICAL_TO_IMPUTE:
        data[var] = impute_na(
            data, var, replacement=LOTFRONTAGE_MODE if var == 'LotFrontage' else 'Missing')

    # capture elapsed time

    data[YEAR_VARIABLE] = elapsed_years(data, YEAR_VARIABLE)

    # log transform numerical variables

    for var in NUMERICAL_LOG:
        data[var] = log_transform(data, var)

    # Group rare labels
    for var in CATEGORICAL_ENCODE:
        data[var] = remove_rare_labels(data, var, FREQUENT_LABELS[var])

    # encode variables

    for var in CATEGORICAL_ENCODE:
        data[var] = encode_categorical(data, var, ENCODING_MAPPINGS)

    # scale variables

    data = scale_features(data[FEATURES], OUTPUT_SCALER_PATH)

    # make predictions

    predictions = predict(data, OUTPUT_MODEL_PATH)

    return predictions

# ================================

# small test that scripts are working ok


if __name__ == '__main__':
    from math import sqrt
    import numpy as np

    from sklearn.metrics import mean_squared_error, r2_score

    import warnings
    warnings.simplefilter(action='ignore')

    # load data
    data = pd.load_data(PATH_TO_DATASET)
    X_train, X_test, y_train, y_test = divide_train_test(data, TARGET)

    pred = predict(X_test)

    # determine mse and rmse

    print('test mse: {}'.format(int(
        mean_squared_error(y_test, np.exp(pred))
    )))

    print('test rmse: {}'.format(int(
        sqrt(mean_squared_error(y_test, np.exp(pred)))
    )))

    print('test r2: {}'.format(
        r2_score(y_test, np.exp(pred))
    ))

    print()
