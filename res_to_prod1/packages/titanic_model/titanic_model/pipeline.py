from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import titanic_model.preprocessors as pp
from titanic_model import config

survived_pipe = Pipeline(
    [
        ('categorical_imputer', pp.CategoricalImputer(variables=config.CATEGORICAL_NA_VARS)),
        ('numerical_NA_imputer', pp.NumericalImputer(variables=config.NUMERICAL_NA_VARS)),
        # ('scaler', MinMaxScaler()),
        # ('linear_regression_model', LogisticRegression())
    ])

