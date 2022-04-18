from titanic_model.config import config


def test_config_allowed_features():
    assert ["Survived", "PassengerId"] not in config.FEATURES
