import numpy as np
import os
from task1 import extract_ratings_from_filenames 
from task1 import reducer 
from task1 import generate_word_index_map 
from task1 import generate_matrix 

def get_train_data():
    directory_path = "train"

    # Extract ratings from file names
    training_targets = extract_ratings_from_filenames(directory_path)

    files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.txt')]

    # Generate matrix and unique words list
    matrix, unique_words = reducer(files)

    # Generate word-index map
    word_index_map = generate_word_index_map(unique_words)

    # Generate matrix with N rows and D columns
    training_data = generate_matrix(matrix, word_index_map)

    return training_data, training_targets, matrix, unique_words

def compute_correlation(X, y):
    # Number of samples (N) and number of features (D)
    N, D = X.shape
    
    # Compute the mean vector of all features
    m = np.mean(X, axis=0)
    
    # Center the feature vectors
    X_centered = X - m
    
    # Compute the L2 norm of each centered feature vector
    X_norms = np.linalg.norm(X_centered, axis=0, keepdims=True)
    
    # Normalize the centered feature vectors
    X_normalized = X_centered / X_norms
    
    # Normalize the review scores
    y_norm = np.linalg.norm(y)
    y_normalized = y / y_norm
    
    # Compute the correlation vector
    r = np.sum(y_normalized[:, np.newaxis] * X_normalized, axis=0)
    
    return r

def main():
    training_data, training_targets, matrix, unique_words = get_train_data()
    correlation_vector = compute_correlation(np.array(training_data), np.array(training_targets))
    print(correlation_vector)

if __name__ == "__main__":
    main()