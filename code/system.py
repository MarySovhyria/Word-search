"""Dummy classification system.

Dummy solution the COM2004/3004 assignment.

REWRITE THE FUNCTIONS BELOW AND REWRITE THIS DOCSTRING

version: v1.0
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from utils.utils import Puzzle

# number of dimensions for the feature vectors.
N_DIMENSIONS = 20


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce dimensionality from 400 features to N_DIMENSIONS (20) using PCA.
    
    During training, fits PCA and stores the components and mean in the model.
    During testing, uses stored components to transform test data.
    Includes feature standardization for robustness to noise.
    
    Args:
        data (np.ndarray): Feature vectors to reduce (N_samples, 400).
        model (dict): Model dictionary to store/retrieve PCA parameters.
    
    Returns:
        np.ndarray: Reduced feature vectors (N_samples, 20).
    """
    
    # Testing phase
    if "pca_components" in model:
        
        mean = np.array(model["pca_mean"])
        std = np.array(model["pca_std"])
        components = np.array(model["pca_components"])
        
        #Standartization for consistent scaling
        standardized = (data - mean) / (std + 1e-8)
        
        # Project onto principal components
        reduced_data = standardized @ components.T
    else:
        # Training phase
        from sklearn.decomposition import PCA
        
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
       #Standartization for consistent scaling
        standardized = (data - mean) / (std + 1e-8)
        
        pca = PCA(n_components=N_DIMENSIONS)
        reduced_data = pca.fit_transform(standardized)
        
        model["pca_mean"] = mean.tolist()
        model["pca_std"] = std.tolist()
        model["pca_components"] = pca.components_.tolist()
        model["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Train KNN classifier with PCA dimensionality reduction.

    Reduces dimensionality using PCA and stores training data for KNN.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data including training vectors and labels.
    """
    model = {}
    
    # Reduce dimensions
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    
    #Reduced training data and labels for KNN
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    model["labels_train"] = labels_train.tolist()
    
    return model


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Classify test squares using distance-weighted K-Nearest Neighbors.

    Args:
        fvectors_test (np.ndarray): Reduced feature vectors to classify (N_samples, 20).
        model (dict): Dictionary containing trained model data including
                     'fvectors_train' and 'labels_train'.

    Returns:
        List[str]: A list of predicted labels, one per input feature vector.
    """
    #Training data from model 
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])
    
    # High K used to reduce effect of noise in low-quality images
    # Best K was determined by practice
    K = 15  
    predictions = []
    
    # Classify each test sample
    for test_vec in fvectors_test:
        # Distances to all training points 
        # Manhattan distance was used because it is less prone to outliers
        differences = fvectors_train - test_vec
        absolute_differences = np.abs(differences)
        distances = np.sum(absolute_differences, axis=1)
        
        k_nearest_indices = np.argsort(distances)[:K]
        k_nearest_labels = labels_train[k_nearest_indices]
        k_nearest_distances = distances[k_nearest_indices]
        
        # Distance-weighted voting: closer neighbors count more
        # small epsilon is used to avoid division by zero
        epsilon = 1e-10
        weights = 1.0 / (k_nearest_distances + epsilon)
        
        # Weighted votes for each label
        votes = {}
        for i in range(K):
            label = k_nearest_labels[i]
            weight = weights[i]
            if label in votes:
                votes[label] += weight
            else:
                votes[label] = weight
        
        # Predict label 
        max_votes = 0
        predicted_label = None
        for label, vote_count in votes.items():
            if vote_count > max_votes:
                max_votes = vote_count
                predicted_label = label
        predictions.append(predicted_label)
    
    return predictions


def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Search for words in the puzzle grid in all 8 directions.

    Searches for each word horizontally (left-right, right-left), 
    vertically (top-bottom, bottom-top), and diagonally (4 directions).
    Returns the start and end position for each word found.

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training (unused here).

    Returns:
        list[tuple]: A list of four-element tuples (start_row, start_col, end_row, end_col)
                     indicating the position of each word.
    """
    rows, cols = labels.shape
    positions = []
    
    directions = [
        (0, 1),   # right
        (0, -1),  # left
        (1, 0),   # down
        (-1, 0),  # up
        (1, 1),   # down-right diagonal
        (1, -1),  # down-left diagonal
        (-1, 1),  # up-right diagonal
        (-1, -1)  # up-left diagonal
    ]
    
    def search_from_position(word, start_row, start_col, direction, max_mismatches=0):
        """Search for a word starting from a position in a given direction.
        
        Args:
            word: The word to search for
            start_row: Starting row position
            start_col: Starting column position
            direction: (dr, dc) tuple for search direction
            max_mismatches: Allow up to this many letter mismatches (0 = exact match)
        
        Returns:
            tuple or None: (start_row, start_col, end_row, end_col, mismatches) if word found, else None
        """
        dr, dc = direction
        word = word.upper()  
        
        # Check if word fits in the direction
        word_length = len(word)
        end_row = start_row + dr * (word_length - 1)
        end_col = start_col + dc * (word_length - 1)
        
        if end_row < 0 or end_row >= rows or end_col < 0 or end_col >= cols:
            return None
        
        mismatches = 0
        for i in range(len(word)):
            r = start_row + dr * i
            c = start_col + dc * i
            if labels[r, c] != word[i]:
                mismatches += 1
                if mismatches > max_mismatches:
                    return None
        return (start_row, start_col, end_row, end_col, mismatches)
    
    # Search for each word
    for word in words:
        best_match = None
        #Number large enough so it will always be > than other values
        best_mismatches = 1000 
        
        # Allow a few mismatches for long words
        # Helps with low quality images
        if len(word) > 9:
            max_allowed_mismatches = 3
        elif len(word) > 4:
            max_allowed_mismatches = 2
        else:
            max_allowed_mismatches = 1
        
       # Try all positions and directions
        for start_row in range(rows):
            for start_col in range(cols):
                for direction in directions:
                    result = search_from_position(word, start_row, start_col, direction, max_allowed_mismatches)
                    
                    if result is not None:
                        start_r, start_c, end_r, end_c, mismatches = result
                        
                        #Track best match
                        if mismatches < best_mismatches:
                            best_match = (start_r, start_c, end_r, end_c)
                            best_mismatches = mismatches
                            
        if best_match is not None:
            positions.append(best_match)
        else:
            positions.append((0, 0, 0, 0))
    
    return positions
