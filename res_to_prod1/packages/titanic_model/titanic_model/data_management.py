import joblib
import pandas as pd
from titanic_model import config


def load_dataset(file_name):
    _data = pd.read_csv(config.DATASET_DIR / file_name)
    return _data


def save_pipeline(pipeline_to_save):
    save_name = config.MODEL_NAME
    save_path = config.TRAINED_MODEL_DIR / save_name
    joblib.dump(pipeline_to_save, save_path)
    print("model saved")


def load_pipeline():
    load_file_name = config.MODEL_NAME
    file_path = config.TRAINED_MODEL_DIR / load_file_name
    pipeline = joblib.load(file_path)
    return pipeline
