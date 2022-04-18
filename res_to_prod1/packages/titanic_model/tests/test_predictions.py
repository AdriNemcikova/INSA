import math
import numpy as np
from titanic_model.data_management import load_dataset
from titanic_model.predict import make_prediction
from titanic_model.config import config


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


def test_prediction_quality():
    # Given
    train_data = load_dataset('train.csv')
    input_df = train_data.drop(config.TARGET, axis=1)
    output_df = train_data[config.TARGET]

    multiple_test_json = input_df.to_json(orient='records')
    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], (int, np.integer))

    TPN = 0
    for i in range(0, len(subject.get('predictions'))):
        if subject.get('predictions')[i] == output_df[i]:
            TPN += 1
    print("Correctly predicted cases: ", TPN)
    print("Total number of cases: ", len(train_data))

    min_limit = TPN * 1.3
    max_limit = TPN * 0.7

    assert len(output_df) > max_limit
    assert len(output_df) < min_limit

    accuracy = (TPN / len(train_data)) * 100
    print('Accuracy of the binary classification = {:0.3f} %'.format(accuracy))
