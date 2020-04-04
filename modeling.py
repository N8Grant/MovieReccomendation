# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:08:42 2020

@author: Rudra Guin
"""

import pandas as pd

movies = pd.read_csv("movies.tsv", sep='\t')
users = pd.read_csv("users.csv")

movies = movies["movieID"].unique()
users = users["userID"].unique()

movie_matrix = pd.DataFrame(index = users, columns = movies)

ratings = pd.read_csv("ratings.csv")

for i in range(len(ratings)):
    u, m = ratings.loc[i, "userID"], ratings.loc[i, "movieID"]
    movie_matrix.set_value(u, m, ratings.loc[i, "rating"])
    del u
    del m
    
del movies
del ratings
del users
del i