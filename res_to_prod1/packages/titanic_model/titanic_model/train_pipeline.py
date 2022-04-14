import pathlib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from titanic_model import pipeline, config, predict, data_management


def run_training():
    print('training started')
    data = data_management.load_dataset(config.TRAINING_DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES], data[config.TARGET], test_size=0.1,
                                                        random_state=0)

    pipeline.survived_pipe.fit(X_train[config.FEATURES], y_train)
    data_management.save_pipeline(pipeline.survived_pipe)
    print('training ended')


if __name__ == '__main__':
    run_training()
