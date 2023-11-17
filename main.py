# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the MovieLens dataset
# Download the dataset from https://grouplens.org/datasets/movielens/
# Use the 'movies.csv' and 'ratings.csv' files
movies = pd.read_csv('/home/fkadwell/Downloads/movielenslg/ml-latest/movies.csv')
ratings = pd.read_csv('/home/fkadwell/Downloads/movielenslg/ml-latest/ratings.csv')

# Merge the movies and ratings dataframes
movie_ratings = pd.merge(ratings, movies, on='movieId')

print(movie_ratings)
