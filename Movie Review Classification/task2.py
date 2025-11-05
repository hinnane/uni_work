import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from task1 import extract_ratings_from_filenames 
from task1 import reducer 
from task1 import generate_word_index_map 
from task1 import generate_matrix 

def get_test_data():
    directory_path = "test"

    # Extract ratings from file names
    test_targets = extract_ratings_from_filenames(directory_path)

    files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.txt')]

    # Generate matrix and unique words list
    matrix, unique_words = reducer(files)

    # Generate word-index map
    word_index_map = generate_word_index_map(unique_words)

    # Generate matrix with N rows and D columns
    test_data = generate_matrix(matrix, word_index_map)

    return test_data, test_targets, matrix, unique_words

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

def main():
    # Assuming you have identified different unique words for training and test sets
    training_data, training_targets, training_matrix, training_unique_words = get_train_data()
    test_data, test_targets, test_matrix, test_unique_words = get_test_data()

    # Create a union of all unique words
    all_unique_words = set(training_unique_words).union(set(test_unique_words))

    # Generate word-index map for combined vocabulary
    word_index_map = generate_word_index_map(all_unique_words)

    # Reconstruct matrices with combined vocabulary
    combined_training_data = generate_matrix(training_matrix, word_index_map)
    combined_test_data = generate_matrix(test_matrix, word_index_map)

    # Convert targets to binary labels
    training_labels = np.array([1 if rating > 5 else 0 for rating in training_targets])  # Convert to numpy array
    test_labels = np.array([1 if rating > 5 else 0 for rating in test_targets])

    # Apply Min-Max normalization
    scaler = MinMaxScaler()

    # Fit and transform training data
    normalized_train = scaler.fit_transform(combined_training_data)

    # Transform test data using fitted scaler
    normalized_test = scaler.transform(combined_test_data)

    # Define the parameter grid to search
    param_grid = {'max_depth': [5, 10, 15], 'n_estimators': [50, 100, 200]}

    # Create a Random Forest Classifier
    rfc = RandomForestClassifier()

    # Perform Grid Search Cross Validation
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3)
    grid_search.fit(normalized_train, training_labels)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Apply the best model on test data
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(normalized_train, training_labels)
    predictions = best_model.predict(normalized_test)

    #Calculate prediction accuracy
    accuracy = accuracy_score(test_labels, predictions)

    # Print predictions and accuracy
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()