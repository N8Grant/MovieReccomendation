import pandas as pd
import numpy as np
import os

class MovieData:
    def __init__(self):
        current_dir  = os.getcwd()
        self.movies_df = pd.read_csv(current_dir + '/data/movies.tsv', sep='\t')   
        ratings_df = pd.read_csv(current_dir + '/data/ratings.csv')
        R_df = ratings_df.pivot(index = 'userID', columns ='movieID', values = 'rating').fillna(0)
        R = R_df.to_numpy()
        user_ratings_mean = np.mean(R, axis = 1)
        self.R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    def get_R(self):
        return self.R_demeaned
    def get_movies(self):
        return self.movies_df

if __name__ == "__main__": 
    md = MovieData()
    print(np.min(md.get_R()))