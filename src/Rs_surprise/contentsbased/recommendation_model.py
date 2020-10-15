# %%
import numpy as np
import pandas as pd
from genre_preprocessing import load_data

path = "./tmdb_5000"
movies_df = load_data(path)

# 특정 영화와 장르별 유사도가 높은 영화를 반환하는 함수 생성
def find_sim_movie(df, sorted_ind, title_name, top_n):
    title_movie = df[df["title"] == title_name]

    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, : top_n]
    similar_indexes = similar_indexes.flatten()
    
    return df.iloc[similar_indexes]

# %% [markdown]
'''
### 평가 횟수에 대한 가중치가 부여된 평점(Weighted Rating) 계산
$$ Weighted Rating(WR) = (\frac {v} {v + m} * R) + (\frac {m} {v + m} * C) $$
+ v : 개별 영화에 평점을 투표한 횟수
+ m :  평점에 부여하기 위한 최소 투표 횟수
+ R : 개별 영화에 대한 평균 평점
+ C : 전체 영화에 대한 평균 평점
'''
# %%
m = movies_df["vote_count"].quantile(0.6)
C = movies_df["vote_average"].mean()
# %%
def weighted_voted_average(record, m = m, C = C):
    v = record["vote_count"]
    R = record["vote_average"]
    return (v / (v + m) * R)+ (m / (v + m) * C)

def movies_weighted_weighted_vote(movies_df):
    movies_df["weighted_vote"] = movies_df.apply(weighted_voted_average, axis = 1)

def find_sim_movie_vote(df, sorted_ind, title_name, top_n):
    title_movie = df[df["title"] == title_name]

    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, : (top_n * 2)]
    similar_indexes = similar_indexes.flatten()
    # 기준 영화 index 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]

    return df.iloc[similar_indexes].sort_values("weighted_vote", ascending = False)[:top_n]
