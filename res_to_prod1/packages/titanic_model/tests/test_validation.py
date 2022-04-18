from titanic_model.data_management import load_dataset
from titanic_model.validation import validate_inputs


def test_validation_na():
    test_data = load_dataset('train.csv')
    test_data = test_data[:10]

    assert len(test_data) == 10
    assert "Pclass" in test_data.columns
    assert test_data["Pclass"].isnull().sum() == 0

    test_data['Pclass'][0] = None
    assert test_data["Pclass"].isnull().sum() == 1

    validated_data = validate_inputs(test_data)
    assert validated_data["Pclass"].isnull().sum() == 0

    validated_data = validate_inputs(test_data)
    assert validated_data["Pclass"].isnull().sum() == 0
