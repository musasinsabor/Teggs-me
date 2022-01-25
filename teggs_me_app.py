# importing the libraries
import streamlit as st
from PIL import Image
from model.pre_trained_model_prediction import PretrainedObjectDetectionModelPrediction
import os

# importing theme image
directory = os.path.dirname(os.path.realpath(__file__))
my_default_image = "eggs_in_the_nest.jpeg"
my_img = os.path.join(directory, my_default_image)


def user_ver(user: str, password: str):
    """ Callback function during adding a new project. """
    # display a warning if the user entered an existing name
    if user == "admin" and password == "admin":
        st.session_state.logged_user = user
    else:
        return "You can use 'admin' as user and password"


def classifier_load():
    classifier = PretrainedObjectDetectionModelPrediction()
    st.session_state.classifier = classifier


def main():
    """Simple Login App"""

    st.title("Teggs-me: Eggs in Nest Image Classification App")
    st.write('Enjoy your day :sunglasses: and wait for **all the eggs** to be in the nest.')
    if 'logged_user' not in st.session_state:
        with st.container():
            st.subheader("Login")
            username = st.text_input("User Name")
            password = st.text_input("Password", type='password')
            st.button("Login", on_click=user_ver, args=(username, password))

    else:
        if 'classifier' not in st.session_state:
            with st.container():
                st.subheader("Let's start classifying")
                if st.button("Load the classifier", on_click=classifier_load):
                    st.subheader("Add Your Image to classify")

                    uploaded_file = st.file_uploader(" ", type=['png', 'jpg', 'jpeg'])
                    if uploaded_file is None:
                        st.write("Please upload an Image to Classify")
                    else:
                        st.success("Your image is gonna start being classified")
                        u_img = Image.open(uploaded_file)
                        show = st.image(u_img, use_column_width=True)
                        show.image(u_img, 'Uploaded Image', use_column_width=True)


if __name__ == '__main__':
    main()
