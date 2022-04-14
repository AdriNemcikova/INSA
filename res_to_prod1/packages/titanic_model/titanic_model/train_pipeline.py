import pathlib
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from titanic_model import pipeline, config, predict


def save_pipeline(pipeline_to_save):
    save_name = config.MODEL_NAME
    save_path = config.TRAINED_MODEL_DIR / save_name
    joblib.dump(pipeline_to_save, save_path)
    print("model saved")


def run_training():
    print('training started')
    data = pd.read_csv(config.DATASET_DIR / config.TRAINING_DATA_FILE, encoding='utf-8')
    X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES], data[config.TARGET], test_size = 0.1, random_state=0)

    pipeline.survived_pipe.fit(X_train[config.FEATURES], y_train)
    save_pipeline(pipeline.survived_pipe)
    print('training ended')

    # predict.make_prediction(X_test, y_test)


if __name__ == '__main__':
    run_training()
