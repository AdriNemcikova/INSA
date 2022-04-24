import pandas as pd
import logging

from titanic_model import data_management,validation
from titanic_model.config import config
from titanic_model import __version__ as _version

_logger = logging.getLogger(__name__)

_survived_pipe = data_management.load_pipeline()


def make_prediction(input_data):
    data = pd.read_json(input_data)
    validated_data = validation.validate_inputs(data)
    prediction = _survived_pipe.predict(validated_data[config.FEATURES])
    output = prediction
    response = {"predictions": output}

    _logger.info(
        f"Making prediction with model version: {_version}"
        f"Inputs:\n {validated_data}"
        f"Predictions: \n {response}"
    )

    return response
