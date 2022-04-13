from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import titanic_model.preprocessors as pp
from titanic_model import config

survived_pipe = Pipeline(
    [
        ('categorical_imputer', pp.CategoricalImputer(variables=config.CATEGORICAL_NA_VARS)),
        ('numerical_NA_imputer', pp.NumericalImputer(variables=config.NUMERICAL_NA_VARS)),
        ('dsdsds', pp.NumericalImputer(variables=config.NUMERICAL_NA_VARS)),
        ('rare_labels', pp.RareLabels(variables=config.CATEGORICAL_RARE, target=config.TARGET)),
        ('rare', pp.RareLabels(variables=config.CATEGORICAL_RARE, target=config.TARGET, rare_perc=config.RARE_PERC)),
        ('encoding', pp.EncodingCategories(variables=config.CATEGORICAL_ENCODE, mappings=config.ENCODING_MAPPINGS)),
        # ("drop_attributes", pp.DropAttributes(variables=config.DROP_ATTRIBUTES)),
        # ("drop_attributes2", pp.DropAttributes(variables=config.DROP_ATTRIBUTES)),

        ('scaler', MinMaxScaler()),
        ('linear_regression_model', LogisticRegression())
    ])
