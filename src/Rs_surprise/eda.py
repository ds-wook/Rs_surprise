# %% [markdown]
# 라이브러리 호출

import pandas as pd
from surprise.accuracy import rmse
from bayes_opt.bayesian_optimization import BayesianOptimization
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from typing import List, Any, Tuple

# %%

ratings = pd.read_csv('../../ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25, random_state=0)


# %%


svd_model = SVD(random_state=0)
svd_model.fit(trainset)
predictions = svd_model.test(testset)
rmse(predictions)

# %%


def bayesian_svd(
        n_epochs: float,
        n_factors: float,
        lr_all: float,
        reg_all: float) -> float:
    model = SVD(
            n_epochs=int(round(n_epochs)),
            n_factors=int(round(n_factors)),
            lr_all=max(min(lr_all, 1), 0),
            reg_all=max(min(reg_all, 1), 0)
    )
    result = cross_validate(model, data, measures=['RMSE'])
    score = result['test_rmse'].mean()
    return -1.0 * score


# %%

svd_params = {
    'n_epochs': (20, 60),
    'n_factors': (50, 200),
    'lr_all': (0.005, 0.01),
    'reg_all': (0.02, 0.1)
    }

svd_bo =\
    BayesianOptimization(bayesian_svd, svd_params, verbose=2, random_state=0)
svd_bo.maximize(init_points=5, n_iter=5, acq='ei', xi=0.01)


# %%


svd_bo.max


# %%

svd_tun = SVD(
        n_epochs=int(round(svd_bo.max['params']['n_epochs'])),
        n_factors=int(round(svd_bo.max['params']['n_factors'])),
        lr_all=max(min(svd_bo.max['params']['lr_all'], 1), 0),
        reg_all=max(min(svd_bo.max['params']['reg_all'], 1), 0),
        random_state=0
    )
svd_tun.fit(trainset)
predictions = svd_tun.test(testset)
rmse(predictions)


# %%

movies = pd.read_csv('../../ml-latest-small/movies.csv')
movie_ids = ratings[ratings['userId']==9]['movieId']
if movie_ids[movie_ids == 9].count() == 0:
    print('사용자 아이디 9는 영화 아이디 42의 평점 없음')
print(movies[movies['movieId'] == 42])


# %%


def get_unseen_surprise(
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        user_id: int) -> List[str]:
    seen_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    total_movies = movies['movieId'].tolist()

    unseen_movies =\
        [movie for movie in total_movies if movie not in seen_movies]

    print(f'평점 매긴 영화수: {len(seen_movies)}, 추천 대상 영화수: {len(unseen_movies)}\
 전체 영화수:{len(total_movies)}')
    return unseen_movies


def recomm_movie_by_surprise(
        algo: Any,
        user_id: int,
        unseen_movies: List[str],
        top_n: int = 10) -> List[Tuple[int, str, float]]:
    movies = pd.read_csv('../../ml-latest-small/movies.csv')
    predictions = [algo.predict(str(user_id), str(movie_id))
                   for movie_id in unseen_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:top_n]
    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_rating = [pred.est for pred in top_predictions]
    top_movie_titles = movies[movies['movieId'].isin(top_movie_ids)]['title']
    top_movie_preds =\
        [(iid, title, rating)
         for iid, title, rating
         in zip(top_movie_ids, top_movie_titles, top_movie_rating)]
    return top_movie_preds

# %%


unseen_movies = get_unseen_surprise(ratings, movies, 9)
top_movie_preds = recomm_movie_by_surprise(svd_tun, 9, unseen_movies)

for top_movie in top_movie_preds:
    print(f'{top_movie[1]} : {top_movie[2]}')

