from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import titanic_model.preprocessors as pp
from titanic_model import config

survived_pipe = Pipeline(
    [
        ('categorical_NA_imputer', pp.CategoricalImputer(variables=config.CATEGORICAL_NA_VARS)),
        ('numerical_NA_imputer', pp.NumericalImputer(variables=config.NUMERICAL_NA_VARS)),
        ("family_size", pp.MergeAttributes(variables=config.FAMILY_SIZE_VARS, output=config.OUTPUT)),
        ('encoding', pp.EncodingCategories(variables=config.CATEGORICAL_ENCODE, mappings=config.ENCODING_MAPPINGS)),
        ("extract_title", pp.ExtractTitles(variables=config.TITLE_EXTRACTION, titles=config.TITLES_ENCODING)),
        ("drop_atr", pp.DropAttributes(variables=config.DROP_ATTRIBUTES)),
        ("show", pp.Na_control(variables=config.ALL)),
        ("to_int", pp.ToInteger(variables=config.TO_INTEGER)),
        ('scaler', MinMaxScaler()),
        ('linear_regression_model', LogisticRegression())
    ])



