import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path):
    credits_df = pd.read_csv(path + "/tmdb_5000_credits.csv")
    credits_df.rename(columns = {"movie_id" : "id"}, inplace = True)
    movies_df = pd.read_csv(path + "/tmdb_5000_movies.csv")
    movies_df = movies_df.merge(credits_df, on = ['id', 'title'])
    return movies_df

def dtype_change(movies_df):
    # string -> data structure type
    movies_df["genres"] = movies_df["genres"].apply(eval)
    movies_df["keywords"] = movies_df["keywords"].apply(eval)

def change_column(movies_df):
    movies_df["genres"] = movies_df["genres"].apply(lambda x : [dic["name"] for dic in x])
    movies_df["keywords"] = movies_df["keywords"].apply(lambda x : [dic["name"] for dic in x])
    movies_df["genres_literal"] = movies_df["genres"].apply(lambda x : " ".join(x))

def countvector_preprocessing(movies_df):
    count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
    genre_mat = count_vect.fit_transform(movies_df.genres_literal)
    return genre_mat

def tfidf_preprocessing(movies_df):
    tfidf = TfidfVectorizer(stop_words="english")
    overview_mat = tfidf.fit_transform(movies_df["overview"])
    return overview_mat

def sorted_index(matrix):
    similarity = cosine_similarity(matrix, matrix)
    sim_sorted_ind = similarity.argsort()[:, ::-1]
    return sim_sorted_ind



