from sklearn.model_selection import train_test_split

from titanic_model.config import config
from titanic_model import preprocessors as pp
from titanic_model.data_management import load_dataset
from titanic_model.pipeline import survived_pipe


def test_pipeline_drops_unnecessary_features():
    test_data = load_dataset('train.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        test_data, test_data[config.TARGET], test_size=0.1, random_state=0)
    assert len(config.FEATURES) != len(X_train.columns)

    X_transformed, _ = survived_pipe._fit(X_train, y_train)

    for column in config.DROP_ATTRIBUTES:
        for x in X_transformed:
            assert column != x


def test_pipeline_not_na_values():
    test_data = load_dataset('train.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        test_data, test_data[config.TARGET], test_size=0.1, random_state=0)

    X_transformed, _ = survived_pipe._fit(X_train, y_train)

    for x in X_transformed:
        for v in x:
            assert v >= 0
            assert v != None

