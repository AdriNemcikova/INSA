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
        return X


'''
    Odstranenie rare labels
'''


class RareLabels(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, frequentLabels=None, target=None, rare_perc=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.frequentLabels = frequentLabels
        self.target = target
        self.rare_perc = rare_perc
        self.rare_dict_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        tmp = None
        for var in self.variables:
            tmp = X.groupby(var)[self.target].count() / len(X)

        self.rare_dict_ = tmp[tmp > self.rare_perc].index
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.rare_dict_), X[var], 'Rare')
        return X


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
        return X


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
        print(X)
        return X

# '''
#     Zmena na int datovy typ
# '''
# class ToInteger(BaseEstimator, TransformerMixin):
#     def __init__(self, variables=None):
#         if not isinstance(variables, list):
#             self.variables = [variables]
#         else:
#             self.variables = variables
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X = X.copy()
#         for feature in self.variables:
#             X[feature] = X[feature].astype(int)
#         return X
#
#
# '''
#     Zistenie ci cestujuci boli sami alebo s rodinou
#     Parch + SibSp = Relatives
# '''
# class MergeAttributes(BaseEstimator, TransformerMixin):
#     def __init__(self, variables=None, output=None):
#         if not isinstance(variables, list) or len(variables) != 2:
#             raise ValueError(
#                 f"Variables is not list of two attributes"
#             )
#         else:
#             self.variables = variables
#
#         if not isinstance(output, list) or len(output) != 2:
#             raise ValueError(
#                 f"Output variables is not list of two attributes"
#             )
#         else:
#             self.output = output
#
#     def fit(self, X, y=None):
#
#         return self
#
#     # vytvorim novy stplec kde dam sucet dvoch atributov a transformujem na 0 - cestuje s niekym, 1 - sam
#     def transform(self, X):
#         X = X.copy()
#         sum_column = X[self.variables[0]] + X[self.variables[1]]
#         X[self.output[0]] = sum_column
#
#         X.loc[X[self.output[0]] > 0, self.output[1]] = 0
#         X.loc[X[self.output[0]] == 0, self.output[1]] = 1
#         return X
#
#
# '''
#     Extrakcia pismena z nazvu kabiny a zmena na numericky typ v novom stlpci Deck
# '''
# class ExtractDeck(BaseEstimator, TransformerMixin):
#
#     def __init__(self, variable=None, output=None, map=None):
#         if not isinstance(variable, str) and isinstance(output, str):
#             raise ValueError(
#                 f"Variable is not string of one attribute"
#             )
#         else:
#             self.variable = variable
#             self.map = map
#             self.output = output
#
#     def fit(self, X, y=None):
#
#         return self
#
#     def transform(self, X):
#         X = X.copy()
#         X[self.variable] = X[self.variable].fillna("U0")
#         X[self.output] = X[self.variable].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
#         X[self.output] = X[self.output].map(self.map)
#         X[self.output] = X[self.output].fillna(0)
#         return X
#
#
# '''
#     Extrakcia titulov od mien a ich namapovanie na int
# '''
# class ExtractTitles(BaseEstimator, TransformerMixin):
#
#     def __init__(self, variables=None):
#         if not isinstance(variables, list) or len(variables) != 1:
#             raise ValueError(
#                 f"Variables is not list of one attribute"
#             )
#         else:
#             self.variables = variables
#
#     def fit(self, X, y=None):
#
#         return self
#
#     def transform(self, X):
#         X = X.copy()
#         titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
#
#         X['Title'] = X.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#         X['Title'] = X['Title'].replace(
#             ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
#         X['Title'] = X['Title'].replace('Mlle', 'Miss')
#         X['Title'] = X['Title'].replace('Ms', 'Miss')
#         X['Title'] = X['Title'].replace('Mme', 'Mrs')
#         X['Title'] = X['Title'].map(titles)
#         X['Title'] = X['Title'].fillna(0)
#         return X
#
#
# '''
#     Zmena na binarny atribut
# '''
# class MapToInteger(BaseEstimator, TransformerMixin):
#
#     def __init__(self, variable=None, map=None):
#         if not isinstance(variable, str):
#             self.variable = [variable]
#         else:
#             self.variable = variable
#             self.map = map
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X = X.copy()
#         X[self.variable] = X[self.variable].map(self.map)
#         return X
#
#
# '''
#     Mazanie atributov
# '''
# class DropAttributes(BaseEstimator, TransformerMixin):
#
#     def __init__(self, variables=None):
#         if not isinstance(variables, list):
#             self.variables = [variables]
#         else:
#             self.variables = variables
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X = X.copy()
#         for feature in self.variables:
#             X.drop(feature, inplace=True, axis=1)
#         return X
#
#
# '''
#     Vytvorenie skupin podla veku cestujuich
# '''
# class CreateGroups(BaseEstimator, TransformerMixin):
#
#     def __init__(self, variables=None):
#         if not isinstance(variables, list):
#             self.variables = [variables]
#         else:
#             self.variables = variables
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X = X.copy()
#         for feature in self.variables:
#             X[feature] = X[feature].astype(int)
#             X.loc[X[feature] <= 11, feature] = 0
#             X.loc[(X[feature] > 11) & (X[feature] <= 18), feature] = 1
#             X.loc[(X[feature] > 18) & (X[feature] <= 22), feature] = 2
#             X.loc[(X[feature] > 22) & (X[feature] <= 27), feature] = 3
#             X.loc[(X[feature] > 27) & (X[feature] <= 33), feature] = 4
#             X.loc[(X[feature] > 33) & (X[feature] <= 40), feature] = 5
#             X.loc[(X[feature] > 40) & (X[feature] <= 66), feature] = 6
#             X.loc[X[feature] > 66, feature] = 6
#         return X
#
#
# '''
#     Vytvorenie skupin cestujuich podla atributu Fare
# '''
# class CreateGroups2(BaseEstimator, TransformerMixin):
#
#     def __init__(self, variables=None):
#         if not isinstance(variables, list):
#             self.variables = [variables]
#         else:
#             self.variables = variables
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X = X.copy()
#         for feature in self.variables:
#             X.loc[X[feature] <= 7.91, feature] = 0
#             X.loc[(X[feature] > 7.91) & (X[feature] <= 14.454), feature] = 1
#             X.loc[(X[feature] > 14.454) & (X[feature] <= 31), feature] = 2
#             X.loc[(X[feature] > 31) & (X[feature] <= 99), feature] = 3
#             X.loc[(X[feature] > 99) & (X[feature] <= 250), feature] = 4
#             X.loc[X[feature] > 250, feature] = 5
#             X[feature] = X[feature].astype(int)
#         return X
#
#
# '''
#     Vytvorenie noveho atributu = cena / osoba
# '''
# class CreateNew(BaseEstimator, TransformerMixin):
#
#     def __init__(self, variables=None):
#         if not isinstance(variables, list):
#             self.variables = [variables]
#         else:
#             self.variables = variables
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X = X.copy()
#         X['Fare_Per_Person'] = X['Fare'] / (X['relatives'] + 1)
#         X['Fare_Per_Person'] = X['Fare_Per_Person'].astype(int)
#         return X
