import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'
MODEL_NAME = "survived_model.pkl"

TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
TESTING_DATA_FILE = DATASET_DIR / 'test.csv'

TARGET = 'Survived'

FEATURES = [
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
CATEGORICAL_NA_VARS = ['Cabin', 'Embarked']
NUMERICAL_NA_VARS = ['Age', 'Embarked']

FAMILY_SIZE_VARS = ["SibSp", "Parch"]
OUTPUT = ["Number_of_relatives", "Alone_y_n"]

CATEGORICAL_ENCODE = ['Sex', 'Embarked']
ENCODING_MAPPINGS = {'Sex': {'male': 0, 'female': 1},
                     'Embarked': {'S': 0, 'Q': 1, 'C': 2}}

TITLE_EXTRACTION = ["Name"]
TITLES_ENCODING = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

DROP_ATTRIBUTES = ['Name', 'Ticket', 'Cabin']

TO_INTEGER = ["Age", "Fare", "Embarked", "Alone_y_n", "Title"]

ALL = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Number_of_relatives', 'Alone_y_n', 'Title']

NUMERICALS_VARS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

VARS_WITH_NA = ["Age", "Cabin", "Embarked"]

NA_NOT_ALLOWED = [feature for feature in FEATURES if feature not in VARS_WITH_NA]

