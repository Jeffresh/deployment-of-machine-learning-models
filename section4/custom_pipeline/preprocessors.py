import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


class Pipeline:

    def __init__(self, target, categorical_to_impute, year_variable,
                 numerical_to_impute, numerical_log, categorical_encode,
                 features, test_size=0.1, random_state=0,
                 rare_percentage=0.01, ref_variable='YrSold'):

        # data sets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # engineering parameters (bo te learn from data)
        self.imputing_dict = {}
        self.frequent_categori_dict = {}
        self.encoding_dict = {}

        # models
        self.scaler = MinMaxScaler()
        self.model = Lasso(alpha=0.005, random_state=random_state)

        # groups of variables to engineer
        self.target = target
        self.year_variable = year_variable
        self.categorical_to_impute = categorical_to_impute
        self.numerical_to_impute = numerical_to_impute
        self.numerical_log = numerical_log
        self.categorical_encode = categorical_encode
        self.features = features

        # more parameters
        self.test_size = test_size
        self.random_state = random_state
        self.percentage = rare_percentage
        self.rev_variable = ref_variable

    # ======================== functions to learn parameters from train set

    def find_imputation_replacements(self):
        ''' find value to be used from imputattion'''
        pass

    def find_frequent_categories(self):
        ''' find list of frequent categories in categorical variables'''
        pass
