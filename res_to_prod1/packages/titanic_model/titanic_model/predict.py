import joblib
from sklearn.metrics import confusion_matrix
from titanic_model import config, data_management

_survived_pipe = data_management.load_pipeline()


def make_prediction(Xtest_data, ytest_data):
    Xtest = Xtest_data
    ytest = ytest_data
    prediction = _survived_pipe.predict(Xtest[config.FEATURES])
    output = prediction
    response = {"prediction": output
                }
    TN, FP, FN, TP = confusion_matrix(ytest, prediction).ravel()

    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)

    accuracy = (TP + TN) / (TP + FP + TN + FN)

    print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))

    return response