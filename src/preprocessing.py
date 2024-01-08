import numpy as np
import cv2
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from variables import *


def load_data():
    """
    Load and preprocess image data for training and testing.

    Returns:
        X (np.array): Array of preprocessed images.
        y (np.array): Array of corresponding labels.
    """
    X = []
    y = []
    image_size = 150

    # Load training and testing data
    for data_type in ['Training', 'Testing']:
        for label in LABELS:
            folder_path = os.path.join('../data', data_type, label)
            for filename in tqdm(os.listdir(folder_path)):
                img = cv2.imread(os.path.join(folder_path, filename))
                img = cv2.resize(img, (image_size, image_size))
                X.append(img)
                y.append(label)

    return np.array(X), np.array(y)


def split_data(X, y):
    """
    Shuffles the data and splits it into training and test sets.

    Parameters:
        X (array-like): The input features.
        y (array-like): The target variable.
    Returns:
        tuple: A tuple containing the training and test sets for X and y.
    """
    X, y = shuffle(X, y, random_state=69)

    return train_test_split(X, y, test_size=0.1)


def one_hot(y_train, y_test):
    """
    Converts labels to one-hot encoded vectors.

    Parameters:
        y_train (List[str]): The training labels.
        y_test (List[str]): The testing labels.
    Returns:
        Tuple[np.ndarray, np.ndarray]: The one-hot encoded vectors for the training and testing labels.
    """
    # Convert labels to their corresponding indices in LABELS list
    y_train_indices = [LABELS.index(i) for i in y_train]
    y_test_indices = [LABELS.index(i) for i in y_test]

    # Convert indices to one-hot encoded vectors
    y_train_encoded = tf.keras.utils.to_categorical(y_train_indices)
    y_test_encoded = tf.keras.utils.to_categorical(y_test_indices)

    return y_train_encoded, y_test_encoded


def main():
    """
    This function loads the data, splits it into training and testing sets,
    and performs one-hot encoding on the target labels.

    Returns:
        X_train (array-like): Training data features
        X_test (array-like): Testing data features
        y_train (array-like): One-hot encoded training target labels
        y_test (array-like): One-hot encoded testing target labels
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    y_train, y_test = one_hot(y_train, y_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = main()
    print('finished')

