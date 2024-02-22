import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from settings import *


def load_data(data_sets):
    """
    Load and preprocess image data for training and testing.

    Parameters
    ----------
    data_sets : list
        List of data sets to load. E.g. ['Training', 'Testing']

    Returns
    -------
        X (np.array): Array of preprocessed images.
        y (np.array): Array of corresponding labels.
    """

    X = []
    y = []

    # Load training and testing data
    for data_type in data_sets:
        for label in LABELS:
            folder_path = os.path.join(PROJECT_DIR, 'data', data_type, label)
            for filename in tqdm(sorted(os.listdir(folder_path))):
                img = cv2.imread(os.path.join(folder_path, filename))
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
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

    if IMG_LIMIT:
        X = X[:IMG_LIMIT]
        y = y[:IMG_LIMIT]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    print('Number of training images:', len(X_train))
    print('Number of validation images:', len(X_val))
    print('Number of testing images:', len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def one_hot(y_train, y_val, y_test):
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
    y_val_indices = [LABELS.index(i) for i in y_val]
    y_test_indices = [LABELS.index(i) for i in y_test]

    # Convert indices to one-hot encoded vectors
    y_train_encoded = tf.keras.utils.to_categorical(y_train_indices)
    y_val_encoded = tf.keras.utils.to_categorical(y_val_indices)
    y_test_encoded = tf.keras.utils.to_categorical(y_test_indices)

    return y_train_encoded, y_val_encoded, y_test_encoded


def main():
    """
    This function loads the data, splits it into training, validation, and testing sets,
    and performs one-hot encoding on the target labels.

    Returns:
        X_train (array-like): Training data features
        X_val (array-like): Validation data features
        X_test (array-like): Testing data features
        y_train (array-like): One-hot encoded training target labels
        y_val (array-like): One-hot encoded validation target labels
        y_test (array-like): One-hot encoded testing target labels
    """

    print('Starting preprocessing...')

    X, y = load_data(['Training', 'Testing'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    y_train, y_val, y_test = one_hot(y_train, y_val, y_test)

    print('Preprocessing done !', end='\n\n')

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':

    X_train, X_val, X_test, y_train, y_val, y_test = main()

    print('finished')

