import argparse

from surprise_data import load_data, data_auto_folds
from surprise_model import algorithm,train, get_unseen_surprise, recomm_movie_by_surprise


if __name__ == "__main__":
    path = "./ml-latest-small"
    ratings, movies = load_data(path)

    parser = argparse.ArgumentParser(description="Python Surprise Recommend System")

    parser.add_argument("--line_format", type = str, default = "user item rating timestamp", help = "line format")
    parser.add_argument("--ratings_file", type = str, default="./ml-latest-small/ratings_noh.csv", help = "non header file")
    parser.add_argument("--rating_scale", type = float, nargs = '+', help = "rating scale", default=[0.5, 5]) # 리스트 형 인수를 넘길때 입력 예시 0.5 5
    parser.add_argument("--userId", type = int, help = "User Id", default=10)
    parser.add_argument("--top_n", type = int, default=10, help = "top recommend")
    parser.add_argument("--test_size", type = float, default=0.25, help = "test_size")
    parser.add_argument("--n_epochs", type = int, default=20, help = "epochs")
    parser.add_argument("--n_factors", type = int, default=50)
    parser.add_argument("--random_state", type = int, default=0)
    parser.add_argument("--algorithm", type = str, default="SVD")
    args = parser.parse_args()
    
    
    trainset = data_auto_folds(ratings_file=args.ratings_file, line_format=args.line_format, rating_scale=tuple(args.rating_scale))

    algo = algorithm(algorithm=args.algorithm, n_epochs=args.n_epochs, n_factors=args.n_factors, random_state=args.random_state)

    train(algo, trainset)

    unseen_movies = get_unseen_surprise(ratings=ratings, movies=movies, userId=args.userId)

    top_movie_preds = recomm_movie_by_surprise(movies=movies, algo=algo, userId=args.userId, unseen_movies=unseen_movies, top_n=args.top_n)

    print(f"## Top-{args.top_n} Recommend Movie List ##\n")
    
    for top_movie in top_movie_preds:
        print(f"{top_movie[1]} : {top_movie[2]:.3f}")
    