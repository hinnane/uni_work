import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def extract_ratings_from_filenames(directory):
    ratings = []
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            rating = int(file.split('_')[1].split('.')[0])
            ratings.append(rating)
    return ratings

def mapper(file):
    with open(file, 'r') as f:
        words = f.read().split()
    return words

def reducer(files):
    unique_words = set()  # To store unique words across all files
    matrix = []

    for file in files:
        words = mapper(file)
        unique_words.update(words)  # Update unique words set
        matrix.append(words)

    return matrix, list(unique_words)

def generate_word_index_map(unique_words):
    word_index_map = {}
    for idx, word in enumerate(unique_words):
        word_index_map[word] = idx
    return word_index_map

def generate_matrix(matrix, word_index_map):
    result_matrix = []
    for words in matrix:
        row = [0] * len(word_index_map)  # Initialize row with zeros
        for word in words:
            if word in word_index_map:
                idx = word_index_map[word]
                row[idx] += 1  # Increment count of word in row
        result_matrix.append(row)
    return result_matrix

if __name__ == "__main__":
    # Define the directory path (input directory)
    directory_path = "train"

    # Check if the specified directory exists
    if not os.path.isdir(directory_path):
        print("Error: Directory does not exist.")
        sys.exit(1)

    # Extract ratings from file names
    training_targets = extract_ratings_from_filenames(directory_path)

    # Generate list of files in the directory
    files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.txt')]

    # Generate matrix and unique words list
    matrix, unique_words = reducer(files)

    # Generate word-index map
    word_index_map = generate_word_index_map(unique_words)

    # Generate matrix with N rows and D columns
    training_data = generate_matrix(matrix, word_index_map)

    # Print the result vector and matrix
    print("Train vector:", training_targets)
    print("Train Matrix:")
    for row in training_data:
        print(row)
