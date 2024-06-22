import os
import time
import requests
import json
import numpy as np
import streamlit as st

from preprocessing import load_data
from constants import PROJECT_DIR, TRANSLATION


def display_title():
    """
    Display the title for the Brain Tumor Detection app using Streamlit.
    """

    st.title('Brain Tumor Detection 🧠')
    st.write('### with Deep Learning')
    st.write('#')
    st.write("Now you can play a little game ! 👇 ")
    st.write("Each image shows one of the following: Glioma, meningioma, pituitary tumor, or no tumor if the patient is lucky !")
    st.write("#")
    st.write("1️⃣ - Here are some MRI images")


def load_and_select_random_numbers():
    """
    Load and select random numbers from the data and store them in the session state.
    """

    if 'numbers_list' not in st.session_state:
        st.session_state.X, st.session_state.y = load_data(['Testing'], plot=False)
        st.session_state.numbers_list = np.random.choice(len(st.session_state.X), size=8, replace=False)


def display_random_images():
    """
    Function to display random images based on the session state numbers list.
    """

    st.image(st.session_state.X[st.session_state.numbers_list], width=150, caption=st.session_state.numbers_list)
    st.write('')


def display_text_and_form():
    """
    A function to display text and a form, and return the chosen number and submit button.
    """

    st.write("2️⃣ - Try to **visually** find a tumor (if there is one 🔍 )")
    st.write("3️⃣ - Select the image number here 👇 and click the button to find out if you're better than my AI 🚀")
    st.write('')
    with st.form('my_form'):
        chosen_number = st.selectbox('Choose a number', st.session_state.numbers_list, label_visibility='collapsed')
        submit_button = st.form_submit_button(label='Diagnose 👨‍⚕️')
    st.write('')
    return chosen_number, submit_button


def display_progress_bar(inputs):
    """
    A function to display a progress bar while analyzing an image,
    and then send the inputs to a specified URL and return the response.
    """

    progress_text = "AI image analysis in progress..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)

        if percent_complete == 90:
            response = requests.post(url='http://127.0.0.1:8000/mri_app',
                                     data=json.dumps(inputs))

    time.sleep(1)
    my_bar.empty()

    return response


def diagnose_tumor(chosen_number, submit_button):
    if submit_button:
        img = st.session_state.X[chosen_number]
        true_label = st.session_state.y[chosen_number]
        inputs = {'image': img.tolist()}
        response = display_progress_bar(inputs)

        if response.status_code == 200:
            pred_label = response.text[1:-1]
            if true_label == pred_label:
                st.write(f"The image {chosen_number} has been identified by the AI as **{TRANSLATION[pred_label]}**,"
                         f" which is the right diagnosis ✅")
            else:
                st.write(f"The image {chosen_number} has been identified by the AI as \"**{TRANSLATION[pred_label]}**\", "
                         f"but the true diagnosis is \"**{TRANSLATION[true_label]}**\"...")
        else:
            st.subheader(response.text)


def display_infos():
    """
    Function to retrieve the latest model version from the specified directory.
    """

    models_dir = os.path.join(PROJECT_DIR, 'models')
    files = os.listdir(models_dir)
    ext = 'h5'
    last_model_filename = sorted([f for f in files if f.endswith(ext)])[-1]

    st.write("#")
    st.write("")
    st.write("#")
    st.caption(f'By Timmothy Dangeon, PharmD & Machine Learning Engineer')
    st.caption('Linkedin : linkedin.com/in/timdangeon')
    st.caption('Github : github.com/timdgn')
    st.caption(f'Model version : {last_model_filename[7:-len(ext) - 1]}')


def main():
    """
    This function contains the main logic for the Brain Tumor Detection application.
    It displays a title and introductory text, selects random numbers, shows random images,
    creates a form for user interaction, processes user input, and displays the diagnostic results.
    """

    display_title()
    load_and_select_random_numbers()
    display_random_images()
    chosen_number, submit_button = display_text_and_form()
    diagnose_tumor(chosen_number, submit_button)
    display_infos()


if __name__ == '__main__':
    main()
