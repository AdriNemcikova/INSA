import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'
MODEL_NAME = "survived_model.pkl"

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'

TARGET = 'Survived'

FEATURES = ['Pclass', 'Sex', 'SibSp', 'Age', 'Cabin', 'Embarked']

CATEGORICAL_NA_VARS = ['Cabin', 'Embarked']

NUMERICAL_NA_VARS = ['Age']