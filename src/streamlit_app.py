import streamlit as st
import json
import requests
import preprocessing
import numpy as np


st.title('# Brain Tumor Detection 🧠')
st.write('### with a ✨ Deep Learning ✨ algorithm')
st.write('#')

st.write("Tu vas pouvoir jouer à un petit jeu ! 👇 ")
st.write("Chaque image présente l'un de ces éléments : Un glioblastome, un méningiome, une tumeur pituitaire ou bien aucune tumeur si le patient est chanceux !")
st.write("#")
st.write("1️⃣ - Choisis une image")

# Selecting random numbers
if 'numbers_list' not in st.session_state:
    X, y = preprocessing.load_data()
    np.random.seed()
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.numbers_list = np.random.choice(len(X), size=8, replace=True)

# Showing random images
st.image(st.session_state.X[st.session_state.numbers_list], width=150, caption=st.session_state.y[st.session_state.numbers_list])
st.write('')

st.write("2️⃣ - Essaye de trouver visuellement une tumeur (s'il y en a une 🔍 )")
st.write("3️⃣ - Sélectionne le numéro de l'image ici 👇 et clique sur le bouton pour découvrir si tu es meilleur(e) que mon intelligence artificielle 🚀")
st.write('')

# Creating the form
with st.form('my_form'):
    chosen_number = st.selectbox('Choisis un numéro', st.session_state.numbers_list, label_visibility='collapsed')
    submit_button = st.form_submit_button(label='Diagnostic 👨‍⚕️')

if submit_button:
    st.session_state.chosen_number = chosen_number
    st.write(st.session_state.y[st.session_state.chosen_number])
    st.write(st.session_state.chosen_number)

    # Selecting the image
    img = st.session_state.X[st.session_state.chosen_number]
    true_label = st.session_state.y[st.session_state.chosen_number]

    # converting the inputs into a json format
    inputs = {'image': img.tolist()}

    trad = {'no_tumor': 'aucune tumeur',
            'glioma_tumor': 'glioblastome',
            'meningioma_tumor': 'méningiome',
            'pituitary_tumor': 'tumeur pituitaire'}

    response = requests.post(url='http://127.0.0.1:8000/mri_app',
                             data=json.dumps(inputs))
    if response.status_code == 200:
        resp = response.text[1:-1]
        if true_label == resp:
            st.write(f"Le modèle de deep learning a analysé l'image,"
                     f" et l'a identifiée comme **{trad[resp]}**,"
                     f" ce qui est le bon label ✅")
        else:
            st.write(f"Le modèle de deep learning a analysé l'image,"
                     f" et l'a identifiée comme **{trad[resp]}**,"
                     f" ce qui n'est **PAS** le bon label (**{trad[true_label]}**) ❌")
    else:
        st.subheader(response.text)
