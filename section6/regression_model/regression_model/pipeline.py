from sklearn.pipeline import Pipeline

import preprocessors as pp

CATEGORICAL_VARS = ['MSZoning',
                    'Nieghborhood',
                    'RoofStyle',
                    'MasVnrType',
                    'BsmtQual',
                    'BsmtExposure',
                    'HeatingQC',
                    'CentralAir',
                    'KitchenQual',
                    'FireplaceQu',
                    'GarageType',
                    'GarafeFinish',
                    'PavedDrive',
                    ]

PIPELINE_NAME = 'lasso_regression'

price_pipe = Pipeline(
    [Pipeline(
        ('categorical_imputer',
         pp.CategoricalImputer(variables=CATEGORICAL_VARS)),
    )]
)
