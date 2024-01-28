from fastapi import FastAPI
from pydantic import BaseModel
from inference import inference
from training import load_last_model
from settings import LABELS
import numpy as np


class UserInput(BaseModel):
    image: list


# Load the model
model = load_last_model()

app = FastAPI()


@app.post("/mri_app")
def get_results(input: UserInput):
    """
    This function takes a UserInput object as input and returns the classification result of the brain MRI.

    Parameters
    ----------
    input (UserInput)
        A UserInput object with one attribute: img.

    Returns
    -------
    str
        A string that gives the classification result of the brain MRI.
    """
    img = np.array(input.image)
    img = np.expand_dims(img, axis=0)
    pred_number = inference(model, img)
    pred_class = LABELS[pred_number]

    return pred_class
