# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:08:42 2020

@author: Rudra Guin
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_validate


def make_ratings(ratings):
    ratings['genre2'] = ratings['genre2'].fillna("Unknown")
    ratings['genre3'] = ratings['genre3'].fillna("Unknown")

    ratings['gender'] = ratings['gender'].astype('category')
    ratings['genre1'] = ratings['genre1'].astype('category')
    ratings['genre2'] = ratings['genre2'].astype('category')
    ratings['genre3'] = ratings['genre3'].astype('category')

    gender = pd.get_dummies(ratings['gender'])
    gender = gender.drop(columns=['F'])
    
    genre1 = pd.get_dummies(ratings['genre1'])
    genre1 = genre1.drop(columns=['War'])

    genre2 = pd.get_dummies(ratings['genre2'])
    genre2 = genre2.drop(columns=['Unknown'])

    genre3 = pd.get_dummies(ratings['genre3'])
    genre3 = genre3.drop(columns=['Unknown'])

    ratings = ratings.drop(columns=['gender', 'genre1', 'genre2', 'genre3'])

    ratings = ratings.merge(gender, left_index=True, right_index=True)
    ratings = ratings.merge(genre1, left_index=True, right_index=True)
    ratings = ratings.merge(genre2, left_index=True, right_index=True)
    ratings = ratings.merge(genre3, left_index=True, right_index=True)

    del gender
    del genre1
    del genre2
    del genre3
    
    return ratings


def make_X_and_Y(ratings):
    y = ratings['rating']
    ratings = ratings.drop(columns=['userID', 'movieID', 'year', 'rating', 'name'])
    
    return ratings, y


def fit_NN_model(ratings):
    X, y = make_X_and_Y(ratings)
    
    return MLPRegressor().fit(X, y)


def fit_RegTree_model(ratings):
    X, y = make_X_and_Y(ratings)
    
    return DecisionTreeRegressor(max_depth=8).fit(X, y)


def fit_RandForReg_model(ratings):
    X, y = make_X_and_Y(ratings)
    
    return RandomForestRegressor(max_depth=8).fit(X, y)


def fit_GradBoostReg(ratings):
    X, y = make_X_and_Y(ratings)
    
    return GradientBoostingRegressor("""loss='lad'""", learning_rate=0.01).fit(X, y)


def cross_val(model, ratings, k):
    X, y = make_X_and_Y(ratings)
    
    scores = cross_validate(model, X, y, cv = k, scoring = 'neg_mean_absolute_error', return_train_score = False)
    
    return np.mean(-1 * scores['test_score'])


if __name__ == "__main__":
    movies = pd.read_csv("data/movies.tsv", sep='\t')
    users = pd.read_csv("data/users.csv")
    ratings = pd.read_csv("data/ratings.csv")

    ratings = ratings.merge(movies, how='left', left_on=['movieID'], right_on=['movieID'])
    ratings = ratings.merge(users, how='left', left_on=['userID'], right_on=['userID'])

    ratings = make_ratings(ratings)
    
    mod = fit_NN_model(ratings)
    
    cross_val_test_score = cross_val(mod, ratings, 10)
    print("\nMAD prediction error for Neural Network (10-fold cross-validation) = {}".format(cross_val_test_score))
    
    mod = fit_RegTree_model(ratings)
    
    cross_val_test_score = cross_val(mod, ratings, 10)
    print("\nMAD prediction error for Regression Tree (10-fold cross-validation) = {}".format(cross_val_test_score))
    
    mod = fit_RegTree_model(ratings)
    
    cross_val_test_score = cross_val(mod, ratings, 10)
    print("\nMAD prediction error for Random Forest (10-fold cross-validation) = {}".format(cross_val_test_score))
    
    mod = fit_RegTree_model(ratings)
    
    cross_val_test_score = cross_val(mod, ratings, 10)
    print("\nMAD prediction error for Gradient Booster Regressor (10-fold cross-validation) = {}".format(cross_val_test_score))
