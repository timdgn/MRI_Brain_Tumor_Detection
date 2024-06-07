import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import imutils
from pathlib import Path

from constants import *


def load_data(data_sets, plot=True):
    """
    Load and preprocess image data for training and testing.

    Parameters:
        data_sets (list): List of data sets to load. E.g. ['Training', 'Testing']
        plot (bool, optional): Whether to plot the distribution of the images. Defaults to True.

    Returns
        X (np.array): Array of preprocessed images.
        y (np.array): Array of corresponding labels.
    """

    X = []
    y = []

    # Load training and testing data
    for data_type in data_sets:
        for label in LABELS:
            folder_path = os.path.join(PROJECT_DIR, 'data', 'processed', data_type, label)
            for filename in tqdm(sorted(os.listdir(folder_path))):
                img = cv2.imread(os.path.join(folder_path, filename))
                X.append(img)
                y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Plot label distribution
    if plot:
        label_counts = [np.sum(y == label) for label in LABELS]
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=LABELS, y=label_counts, hue=LABELS, palette="viridis", legend=False)

        # Add the numbers above the bars
        for i, count in enumerate(label_counts):
            ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

        plt.title('Labels distribution of the images', size=18, fontweight='bold')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.tight_layout()

        # Make the bars thinner
        for patch in ax.patches:
            current_width = patch.get_width()
            diff = current_width - 0.5
            patch.set_width(0.5)
            patch.set_x(patch.get_x() + diff * .5)

        output_dir = os.path.join(PROJECT_DIR, 'plots', 'labels_distribution')
        filename = 'labels_distribution.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300)

        plt.show()
        plt.close()

    return X, y


def split_data(X, y):
    """
    Shuffles the data and splits it into training, validation, and test sets.

    Parameters:
        X (array-like): The input features.
        y (array-like): The target variable.

    Returns:
        tuple: A tuple containing the training, validation, and test sets for X and y.
    """

    X, y = shuffle(X, y)

    if IMG_LIMIT:
        X = X[:IMG_LIMIT]
        y = y[:IMG_LIMIT]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    print(f'Total number of images: {len(X)}')
    print(f'Number of training images: {len(X_train)}')
    print(f'Number of validation images: {len(X_val)}')
    print(f'Number of testing images: {len(X_test)}')

    return X_train, X_val, X_test, y_train, y_val, y_test


def one_hot(y_train, y_val, y_test):
    """
    Converts labels to one-hot encoded vectors.

    Parameters:
        y_train (List[str]): The training labels.
        y_val (List[str]): The validation labels.
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


def preprocessing():
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

    print('\nStarting preprocessing...')

    X, y = load_data(['Training', 'Testing'], plot=False)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    y_train, y_val, y_test = one_hot(y_train, y_val, y_test)

    print('Preprocessing done !', end='\n\n')

    return X_train, X_val, X_test, y_train, y_val, y_test


def crop_img(img):
    """
    It takes an input image, processes it to identify the main object or region of interest,
    and then crops the image based on its extreme points to include only that region.

    Parameters:
        img: input image to be cropped

    Returns:
        new_img: cropped image based on extreme points
    """

    # Convert image to grayscale and apply gaussian blur to smooths out the image and reduces noise
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find the contours in the thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Add pixels to each extreme point and crop the image using the extreme points
    ADD_PIXELS = 0
    new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
                  extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()

    return new_img


def process_images(input_dir, output_dir):
    """
    Process images from the input directory and save the processed images to the output directory.

    Parameters:
        input_dir (str): The input directory containing the original images.
        output_dir (str): The output directory where the processed images will be saved.

    Returns:
        None
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for element in tqdm(input_dir.iterdir()):

        # This line ensures that 'element' is not a file
        if element.is_dir():

            save_path = output_dir / element.name
            for img_path in element.iterdir():

                image = cv2.imread(str(img_path))
                new_img = crop_img(image)
                new_img = cv2.resize(new_img, (IMAGE_SIZE, IMAGE_SIZE))

                save_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path / img_path.name), new_img)


if __name__ == "__main__":

    for dataset in ['Training', 'Testing']:
        process_images(os.path.join(PROJECT_DIR, 'data', 'raw', dataset),
                       os.path.join(PROJECT_DIR, 'data', 'processed', dataset))

    print('Finished')
