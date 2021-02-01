import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CategorialImputer':
        """Fit statement to accomodate the sklearn pipeline"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.Dataframe:
        """Apply the transforms to the dataframe"""

        X = X.copy()

        X[self.variables] = X[self.variables].fillna('Missing')

        return X
