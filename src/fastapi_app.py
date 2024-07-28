import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from inference import inference, get_heatmap_grad_cam, get_image_grad_cam
from training import load_last_model
from constants import LABELS


class UserInput(BaseModel):
    image: list


# Load the model
model = load_last_model()

app = FastAPI()


@app.post("/mri_app")
def get_results(input: UserInput):
    """
    This function takes a UserInput object as input and returns the classification result of the brain MRI.

    Parameters:
        input (UserInput): A UserInput object with one attribute: img.

    Returns:
        dict: A dictionary containing the classification result and the path to the heatmap image.
    """

    img = np.array(input.image)
    img_expanded = np.expand_dims(img, axis=0)
    pred_number = inference(model, img_expanded)
    pred_class = LABELS[pred_number]

    heatmap = get_heatmap_grad_cam(model, img_expanded)
    grad_cam_image = get_image_grad_cam(img, heatmap)

    return {'prediction': pred_class, 'grad_cam_image': grad_cam_image}
