import math
import numpy as np
from titanic_model.data_management import load_dataset
from titanic_model.predict import make_prediction


def test_make_single_prediction():
    test_data = load_dataset("test.csv")
    single_test = test_data[0:1].to_json(orient='records')
    result = make_prediction(single_test)

    assert result is not None
    assert isinstance(result.get('predictions')[0], (int, np.integer))

    print('predictions: ', result.get('predictions'))


def test_make_multiple_predictions():
    test_data = load_dataset('test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient='records')

    subject = make_prediction(input_data=multiple_test_json)

    assert subject is not None
    assert len(subject.get('predictions')) != original_data_length
