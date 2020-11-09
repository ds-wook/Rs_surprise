from surprise import SVD


def algorithm(algorithm, n_epochs, n_factors, random_state):
    algo = SVD(n_epochs=n_epochs, n_factors=n_factors, random_state=random_state) if algorithm == "SVD" else None
    return algo


def train(algo, trainset):
    algo.fit(trainset)


def get_unseen_surprise(ratings, movies, userId):
    seen_movies = ratings[ratings["userId"] == userId]["movieId"].tolist()
    total_movies = movies["movieId"].tolist()

    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]

    print(f"seen movies : {len(seen_movies)}, recommend movie : {len(unseen_movies)}, total movies : {len(total_movies)}\n")

    return unseen_movies


def recomm_movie_by_surprise(movies, algo, userId, unseen_movies, top_n):
    predictions =\
        [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:top_n]

    # top_n으로 추출된 영화 정보 추출, 영화 아읻, 추천 예상 평점, 제목 추출
    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_rating = [pred.est for pred in top_predictions]
    top_movie_titles = movies[movies["movieId"].isin(top_movie_ids)]["title"]
    top_movie_preds =\
        [(id, title, rating)
         for id, title, rating
         in zip(top_movie_ids, top_movie_titles, top_movie_rating)]

    return top_movie_preds
