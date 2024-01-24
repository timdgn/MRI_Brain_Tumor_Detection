import streamlit as st
import json
import requests
import preprocessing
import numpy as np


st.title("Brain Tumor Detection 🧠🩻")
st.write("#")

# Showing random images
X, y = preprocessing.load_data()
np.random.seed(69)
numbers_list = np.random.choice(len(X), size=3, replace=True)
st.image(X[numbers_list], width=150, caption=numbers_list)
st.write("")

# Selecting the image number
number = st.selectbox("Select the number of the brain MRI image you want to diagnose :", numbers_list)
st.write("")

# Selecting the image
img = X[number]
true_label = y[number]

# converting the inputs into a json format
inputs = {"image": img}

# when the user clicks on button it will fetch the API
if st.button('Click to diagnose 👨‍⚕️'):
    response = requests.post(url="http://fastapi_container:8000/myapp",  # todo change url to "http://fastapi_container:8000/myapp"
                             data=json.dumps(inputs))
    if response.status_code == 200:
        st.write(f"The image you selected is a **{true_label}** brain MRI image")
        st.subheader(response.text[1:-1])
    else:
        st.subheader(response.text)



