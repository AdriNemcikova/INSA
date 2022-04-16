import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from titanic_model import config, data_management,validation

_survived_pipe = data_management.load_pipeline()


def make_prediction(input_data): # Xtest_data, ytest_data
    data = pd.read_json(input_data)
    validated_data = validation.validate_inputs(data)
    # Xtest = input_data
    # ytest = ytest_data
    prediction = _survived_pipe.predict(validated_data[config.FEATURES])
    output = prediction
    response = {"predictions": output}
    return response
