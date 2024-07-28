from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2

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


def get_heatmap_grad_cam(model, img_array, layer_name='top_conv'):
    """
    Generates a heatmap of the gradient-weighted class activation mapping (CAM) for a given image using a pre-trained model.

    Parameters:
        model (tf.keras.Model): The pre-trained model used for prediction.
        img_array (numpy.ndarray): The input image as a numpy array.
        layer_name (str, optional): The name of the layer to compute the CAM for. Defaults to 'top_conv'.

    Returns:
        numpy.ndarray: The heatmap of the CAM, where each pixel represents the importance of the corresponding feature map.
    """

    # Create a new model that outputs both the output of the specified layer and the model's output
    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])

    # Compute the feature maps from the specified layer and the loss for the predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    # Computes the feature maps for the input image and the gradients of the loss w.r.t. the feature maps
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Computes the mean of the gradients over the width and height dimensions
    guided_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Initialize the CAM heatmap and loop over each channel in the guided gradients and feature maps,
    # accumulating the weighted sum.
    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(guided_grads):
        cam += w * output[:, :, i]

    # Apply ReLU and normalize
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    return heatmap


def get_image_grad_cam(img, heatmap, alpha=0.4):
    """
    Generates a superimposed image of the original image and the heatmap.

    Parameters:
        img (numpy.ndarray): The original image.
        heatmap (numpy.ndarray): The heatmap to be superimposed.
        alpha (float, optional): The alpha value for blending the image and the heatmap. Defaults to 0.4.

    Returns:
        str: The file path of the saved superimposed image.

    """

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = img * alpha + heatmap * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    now = datetime.now()
    filename = f'Grad-CAM_{now.strftime("%m-%d-%H-%M-%S-%f")}.png'
    file_path = os.path.join(CURRENT_DIR, filename)

    cv2.imwrite(file_path, superimposed_img)

    return file_path


if __name__ == "__main__":

    X, _ = load_data(['Testing'], plot=False)
    model = load_last_model()
    model.summary()

    for i in range(500, 503):
        img = X[i]
        img_expanded = np.expand_dims(img, axis=0)

        pred_number = inference(model, img_expanded)
        pred_label = LABELS[pred_number]

        heatmaps = get_heatmap_grad_cam(model, img_expanded)
        plot_file = get_image_grad_cam(img, heatmaps)

        print(pred_label)
