import numpy as np

from constants import *
from training import load_last_model
from preprocessing import load_data


def inference(model, img):
    """
    Generate the predicted labels for the given test data using the provided model.

    Parameters:
        model (object): The trained model object used for prediction.
        img (array-like): The test data to be used for prediction.

    Returns:
        tuple: A tuple containing the predicted labels and the ground truth labels as arrays.
    """

    y_pred = model.predict(img)
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred[0]


if __name__ == "__main__":
    X, y = load_data(['Testing'])
    img = X[37]
    img = np.expand_dims(img, axis=0)
    model = load_last_model()
    pred_number = inference(model, img)
    pred_label = LABELS[pred_number]

    print(pred_label)
