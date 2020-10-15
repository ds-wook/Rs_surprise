import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.dataset import DatasetAutoFolds

def load_data(path):
    # 데이터 로드
    ratings = pd.read_csv(path + "/ratings.csv")
    movies = pd.read_csv(path  + "/movies.csv")
    return ratings, movies

def reader_data(path, line_format, rating_scale):
    # 데이터 분리
    reader = Reader(line_format=line_format, sep = ",", rating_scale = rating_scale)
    data = Dataset.load_from_file(path, reader = reader)
    return data

def data_split(data, test_size, random_state):
    # 데이터 분리
    trainset, testset = train_test_split(data, test_size = test_size, random_state = random_state)
    return trainset, testset

def data_auto_folds(ratings_file, line_format, rating_scale):
    # 데이터 자동 fold
    reader = Reader(line_format=line_format, sep = ",", rating_scale = rating_scale)
    data_folds = DatasetAutoFolds(ratings_file=ratings_file, reader = reader)
    trainset = data_folds.build_full_trainset() # 전체를 학습데이터로 설정
    return trainset