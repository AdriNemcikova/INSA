import pathlib
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from titanic_model import pipeline, config


def save_pipeline(pipeline_to_save):
    save_name = config.MODEL_NAME
    save_path = config.TRAINED_MODEL_DIR / save_name
    joblib.dump(pipeline_to_save, save_path)
    print("model saved")

def run_training():
    print('training started')
    data = pd.read_csv(config.DATASET_DIR / config.TRAINING_DATA_FILE, encoding='utf-8')
    X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES], data[config.TARGET], test_size = 0.1, random_state=0)

    pipeline.survived_pipe.fit(X_train[config.FEATURES], X_test)

    print('training ended')
    print(X_train.head())
    # print(X_test.head())

    save_pipeline(pipeline.survived_pipe)


if __name__ == '__main__':
    run_training()
