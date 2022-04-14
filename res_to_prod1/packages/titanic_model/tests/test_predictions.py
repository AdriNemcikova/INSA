import math
from titanic_model.data_management import load_dataset
from titanic_model.predict import make_prediction


def test_make_prediction():
    test_data = load_dataset("train.csv")
    X_test = test_data[0:5]
    y_test = test_data["Survived"][0:5]
    result = make_prediction(X_test, y_test)

    assert result is not None # ci sa nieco predikovalo

    print(result)