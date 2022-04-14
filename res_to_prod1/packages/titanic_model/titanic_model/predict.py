# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.metrics import confusion_matrix
#
#
# from titanic_model import config
#
# def load_pipeline():
#     load_file_name = config.MODEL_NAME
#     file_path = config.TRAINED_MODEL_DIR / load_file_name
#     pipeline = joblib.load(file_path)
#     return pipeline
#
#
# _survived_pipe = load_pipeline()
#
#
# def make_prediction(Xtest_data, ytest_data):
#     Xtest = Xtest_data
#     ytest = ytest_data
#     prediction = _survived_pipe.predict(Xtest[config.FEATURES])
#     TN, FP, FN, TP = confusion_matrix(ytest, prediction).ravel()
#
#     print('True Positive(TP)  = ', TP)
#     print('False Positive(FP) = ', FP)
#     print('True Negative(TN)  = ', TN)
#     print('False Negative(FN) = ', FN)
#
#     accuracy = (TP + TN) / (TP + FP + TN + FN)
#
#     print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))