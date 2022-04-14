import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import re

'''
    Doplnenie namiesto chybajucich hodnot numerickych atributov MODE hodnotu
'''
class NumericalImputer(BaseEstimator, TransformerMixin):
    # konstruktor
    # pride bud list alebo premennu premeni na list
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.imputer_dict_ = {}

    # doplni do vsetkych atributov mode
    def fit(self, X, y=None):
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    # transformujem zvolene data a vyplnim prazdne hodnoty
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        print("doplnenie veku: ", X)
        return X


'''
    Doplnenie Missing do chybajucich zaznamov kategorickych atributov
'''
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')
        print("doplnenie missing: ", X)
        return X


'''
    Zistenie velkosti rodiny  pomocou atributov SibSp a Parch
'''
class MergeAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, output=None):
        if not isinstance(variables, list) or len(variables) != 2:
            raise ValueError(
                f"Variables is not list of two attributes"
            )
        else:
            self.variables = variables

        if not isinstance(output, list) or len(output) != 2:
            raise ValueError(
                f"Output variables is not list of two attributes"
            )
        else:
            self.output = output

    def fit(self, X, y=None):
        return self

    # vytvorim novy stplec kde dam sucet dvoch atributov a transformujem na 0 - cestuje s niekym, 1 - sam
    def transform(self, X):
        X = X.copy()
        sum_column = X[self.variables[0]] + X[self.variables[1]]
        X[self.output[0]] = sum_column
        X.loc[X[self.output[0]] > 0, self.output[1]] = 0
        X.loc[X[self.output[0]] == 0, self.output[1]] = 1
        print("doplnenie rodiny: \n",X)
        return X


'''
    Namapovanie kategorickych vars na int
'''
class EncodingCategories(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, mappings=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].map(self.mappings[var])
        print("after encoding: \n", X)
        return X


'''
    Extrahovanie titulu od mena
'''
class ExtractTitles(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, titles = None):
        if not isinstance(variables, list) or len(variables) != 1:
            raise ValueError(
                f"Variables is not list of one attribute"
            )
        else:
            self.variables = variables
        self.titles = titles

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        X = X.copy()
        X['Title'] = X.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        X['Title'] = X['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady',
                                                'Capt', 'Don'], 'Others')
        X['Title'] = X['Title'].replace('Mlle', 'Miss')
        X['Title'] = X['Title'].replace('Ms', 'Miss')
        X['Title'] = X['Title'].replace('Mme', 'Mrs')
        X['Title'] = X['Title'].map(self.titles)
        X['Title'] = X['Title'].fillna(0)
        print("titles encoding \n", X)
        return X

'''
    Dropnutie nepotrebnych atributov
'''
class DropAttributes(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X.drop(feature, inplace=True, axis=1)
        print("after drop: \n", X)
        return X

'''
    Kontrola chybajucich hodnot
'''
class Na_control(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(0)
        return X


'''
    Zmena na int datovy typ
'''
class ToInteger(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].astype(int)
        print("after to int: \n", X)
        return X




