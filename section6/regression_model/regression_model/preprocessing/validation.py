from regression_model.config import config

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs fro unprocessable values."""

    validated_data = input_data.copy()

    # check for numerical varibles with NA not seen during training

    if input_data[config.NUMERICAL_NA_NOT_ALLOWED].isnull.any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.NUMERICAL_NA_NOT_ALLOWED
        )

    # check for categorical variables with NA not seen during training

    if input_data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna()

    # check for values <- 0 for the log trasformed variables

    if (input_data[config.NUMERICALS_LOG_VARS] <= 0).any().any():
        vars_with_neg_values = config.NUMERICALS_LOG_VARS[(
            input_data[config.NUMERICALS_LOG_VARS] <= 0).any()]

        validated_data = validated_data[validated_data[vars_with_neg_values] > 0]

    return validated_data
