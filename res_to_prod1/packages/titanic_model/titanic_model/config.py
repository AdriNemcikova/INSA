import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'
MODEL_NAME = "survived_model.pkl"

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'

TARGET = 'Survived'

FEATURES = [ 'Survived', 'Pclass', 'Sex', 'SibSp', 'Age', 'Cabin', 'Embarked']

CATEGORICAL_NA_VARS = ['Cabin', 'Embarked']

CATEGORICAL_RARE = ['Cabin', 'Embarked']

NUMERICAL_NA_VARS = ['Age']

RARE_PERC = 0.01

CATEGORICAL_ENCODE = ['Sex', 'Cabin', 'Embarked']

ENCODING_MAPPINGS = {'Sex': {'male': 0, 'female': 1},
                     'Cabin': {'Missing': 0, 'Rare': 1},
                     'Embarked': {'S': 0, 'Q': 1, 'C': 2, 'Rare': 3}}

DROP_ATTRIBUTES = ['Survived']