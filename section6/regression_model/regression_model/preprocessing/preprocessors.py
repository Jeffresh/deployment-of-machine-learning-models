import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CategoricalImputer':
        """Fit statement to accomodate the sklearn pipeline"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.Dataframe:
        """Apply the transforms to the dataframe"""

        X = X.copy()

        X[self.variables] = X[self.variables].fillna('Missing')

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical data missing value imputer"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.Dataframe, y: pd.Series = None) -> 'NumericalImputer':
        # persist mdoe dictionary

        self.imputer_dict_ = {}

        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]

        return self

    def transform(self, X: pd.Dataframe) -> pd.DataFrame:
        X = X.copy()

        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)

        return X


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """Temporal variable calculator"""

    def __init__(self, variables=None, reference_variable: int = None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variable = reference_variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'TemporalVariableEstimator':
        # we need this to fit the sklearn pipelines
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol: float = 0.05, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.tol = tol

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'RareLabelCategoricalEncoder':

        self.encoder_dict_ = {}

        for feature in self.variables:
            prob = X[feature].value_counts / len(feature)

            self.encoder_dict_[feature] = prob[prob >= self.tol].index.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.encoder_dict_[feature]), X[feature], 'Rare')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """ String to numbers categorical encoder"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CategoricalEncoder':
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns + ['target'])

        # persist trasnforming dictionary

        self.encoder_dict_ = {}

        for var in self.variables:
            labels_ordered = temp.groupby(
                var)['target'].mean().sort_values(ascending=True).index

            self.encoder_dict_[var] = {k: i for i,
                                       k in enumerate(labels_ordered, 0)}

        return self

    def transform(self, X: pd.DataFrame) -> pd.Dataframe:
        X = X.copy()

        for feature in self.variables:
            X[feature].map(self.encoder_dict_[feature])

        # check if transformer introduces NaN

        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull.any()

            vars_ = {key: value for value, key in null_counts.items()
                     if value is True}

            raise ValueError(
                f'Categorical encoder has introduced NaN when '
                f'transforming categorical variables: {vars_.keys()}'
            )

        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer"""

    def __init__(self, variables=None) -> None:
        if not isisntance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'LogTransformer':
        # to accomodate the pipeline

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # check taht the values are non-negative for log transform

        if not(X[self.variables] > 0).all().all():
            vars_ = self.variables[(X[self.variables] <= 0).any()]

            raise ValueError(
                f'Variables contain zero or negative values,'
                f"cant't apply log for vars: {vars_}"
            )

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None) -> None:
        self.variables = variables_to_drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DropUnecessaryFeatures':
        # to accomodate pipeline

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X.drop(columns=self.variables_to_drop, axis=1, inplace=True)

        return X
