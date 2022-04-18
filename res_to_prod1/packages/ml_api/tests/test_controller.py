import json
from titanic_model.config import config as model_config
from titanic_model.data_management import load_dataset


def test_health_endpoint_returns_200(flask_test_client):
    response = flask_test_client.get('/health')
    assert response.status_code == 200


def test_prediction_endpoint_returns_prediction(flask_test_client):
    test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    post_json = test_data.to_json(orient='records')
    response = flask_test_client.post('/v1/predict', json=post_json)

    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    assert len(prediction) != len(test_data)
