from training import *
import preprocessing


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

    X, y = preprocessing.load_data(['Training', 'Testing'])
    # img = X[1].tolist()
    # img = np.array(img)
    # img = np.expand_dims(img, axis=0)
    model = load_last_model()
    prediction = inference(model, X[1])

    print(prediction)
