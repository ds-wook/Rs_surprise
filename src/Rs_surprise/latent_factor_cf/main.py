import numpy as np
import pandas as pd
import argparse

from matrix_factorization import matrix_factorization
from rating_movies import load_data, rating_pred_matrix, recomm_movie_by_userid, get_unseen_movies

if __name__ == "__main__":
    path = "/content/drive/My Drive/recommend_system/ml-latest-small"
    # path = "./data/ml-latest-small"
    movies, ratings = load_data(path)
    rating_matrix = ratings.groupby(["userId", "movieId"])["rating"].sum().unstack()

    rating_movies = pd.merge(ratings, movies, on = "movieId")

    rating_matrix = rating_movies.groupby(["userId", "title"])["rating"].sum().unstack()
    
    parser = argparse.ArgumentParser(description="Matrix Factorization")
    parser.add_argument("--steps", default=200, type = int, help = "steps")
    parser.add_argument("--learning_rate", default = 0.01, type = float, help = "learning rate")
    parser.add_argument("--r_lambda", default = 0.01, type = float, help = "r2 lambda")
    parser.add_argument("--userId", default = 1, type = int, help = "Input userId")
    parser.add_argument("--top_n", default = 10, type = int, help = "top number")
    args = parser.parse_args()

    P, Q = matrix_factorization(rating_matrix.values, K = 50, steps = args.steps, learning_rate = args.learning_rate, r_lambda = args.r_lambda)
    
    rating_pred_matrix = rating_pred_matrix(P, Q, rating_matrix)
    unseen_list = get_unseen_movies(ratings_matrix = rating_matrix, userId = args.userId)
    recomm_movies = recomm_movie_by_userid(pred_df = rating_pred_matrix, userId = args.userId, unseen_list = unseen_list, top_n = args.top_n)
    print(recomm_movies)