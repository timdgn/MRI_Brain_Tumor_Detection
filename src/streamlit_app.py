import os
import time
import requests
import json
import numpy as np
import streamlit as st

from preprocessing import load_data
from settings import PROJECT_DIR, TRANSLATION


def progress_bar_inference(inputs):
    """
    A function to display a progress bar while analyzing an image,
    and then send the inputs to a specified URL and return the response.
    """

    progress_text = "Analyse de l'image par IA en cours..."
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


def show_infos():
    """
    Function to retrieve the latest model version from the specified directory.
    """

    models_dir = os.path.join(PROJECT_DIR, 'models')
    files = os.listdir(models_dir)
    ext = 'h5'
    last_model_filename = sorted([f for f in files if f.endswith(ext)])[-1]

    st.write("#")
    st.write("#")
    st.write("#")

    st.caption(f'By Timmothy Dangeon, PharmD & Machine Learning Engineer')
    st.caption('Linkedin : linkedin.com/in/timdangeon')
    st.caption('Github : github.com/timdgn')
    st.caption(f'Model version : {last_model_filename[7:-len(ext)-1]}')


def main():
    """
    This function contains the main logic for the Brain Tumor Detection application.
    It displays a title and introductory text, selects random numbers, shows random images,
    creates a form for user interaction, processes user input, and displays the diagnostic results.
    """

    # Text
    st.title('Brain Tumor Detection 🧠')
    st.write('### with a ✨ Deep Learning ✨ algorithm')
    st.write('#')
    st.write("Tu vas pouvoir jouer à un petit jeu ! 👇 ")
    st.write("Chaque image présente l'un de ces éléments : Un glioblastome, un méningiome, une tumeur pituitaire ou bien aucune tumeur si le patient est chanceux !")
    st.write("#")
    st.write("1️⃣ - Voici des images d'IRM")

    # Selecting random numbers
    if 'numbers_list' not in st.session_state:
        st.session_state.X, st.session_state.y = load_data(['Testing'])
        st.session_state.numbers_list = np.random.choice(len(st.session_state.X), size=8, replace=False)

    # Showing random images
    st.image(st.session_state.X[st.session_state.numbers_list], width=150, caption=st.session_state.numbers_list)
    st.write('')

    # Text
    st.write("2️⃣ - Essaye de trouver visuellement une tumeur (s'il y en a une 🔍 )")
    st.write("3️⃣ - Sélectionne le numéro de l'image ici 👇 et clique sur le bouton pour découvrir si tu es meilleur(e) que mon IA 🚀")
    st.write('')

    # Creating the form
    with st.form('my_form'):
        chosen_number = st.selectbox('Choisis un numéro', st.session_state.numbers_list, label_visibility='collapsed')
        submit_button = st.form_submit_button(label='Diagnostic 👨‍⚕️')
    st.write('')

    if submit_button:

        # Selecting the image
        img = st.session_state.X[chosen_number]
        true_label = st.session_state.y[chosen_number]

        # converting the inputs into a json format
        inputs = {'image': img.tolist()}

        response = progress_bar_inference(inputs)

        if response.status_code == 200:

            pred_label = response.text[1:-1]
            if true_label == pred_label:
                st.write(f"L'image {chosen_number} a été identifiée par l'IA comme **{TRANSLATION[pred_label]}**,"
                         f" ce qui est le bon diagnostic ✅")

            else:
                st.write(f"L'image {chosen_number} a été identifiée par l'IA comme \"**{TRANSLATION[pred_label]}**\", "
                         f"mais le vrai diagnostic est \"**{TRANSLATION[true_label]}**\"...")
        else:
            st.subheader(response.text)

    show_infos()


if __name__ == '__main__':

    main()
