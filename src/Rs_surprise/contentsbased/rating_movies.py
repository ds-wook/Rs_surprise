import numpy as np
import pandas as pd

def load_data(path):
    movies = pd.read_csv(path + "/movies.csv")
    ratings = pd.read_csv(path + "/ratings.csv")
    ratings = ratings[["userId", "movieId", "rating"]]
    return movies, ratings

def rating_pred_matrix(P, Q, rating_matrix):
    pred_matrix = np.dot(P, Q.T)
    
    return pd.DataFrame(data=pred_matrix, index = rating_matrix.index, columns = rating_matrix.columns)

def get_unseen_movies(ratings_matrix, userId):
    user_rating = ratings_matrix.loc[userId, :]
    
    already_seen = user_rating[user_rating > 0].index.tolist()

    movies_list = ratings_matrix.columns.tolist()

    unseen_list = [movie for movie in movies_list if movie not in already_seen]

    return unseen_list

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n = 10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending = False)[:top_n]
    recomm_movies = pd.DataFrame(data = recomm_movies.values, index = recomm_movies.index, columns = ["pred_score"])
    return recomm_movies
