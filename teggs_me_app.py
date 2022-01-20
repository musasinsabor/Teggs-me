# importing the libraries
import streamlit as st
from PIL import Image
from model.pre_trained_model_prediction import PretrainedObjectDetectionModelPrediction
import os
import pandas as pd
import hashlib
import sqlite3

# importing theme image
directory = os.path.dirname(os.path.realpath(__file__))
my_default_image = "eggs_in_the_nest.jpeg"
my_img = os.path.join(directory, my_default_image)

# loading the classifier
classifier = PretrainedObjectDetectionModelPrediction()

# If classifier is already initialized, don't do anything
if 'classifier' not in st.session_state:
    st.session_state.classifier = classifier


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


conn = sqlite3.connect('data.db')
c = conn.cursor()


# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


def main():
    """Simple Login App"""

    st.title("Teggs-me: Eggs in Nest Image Classification App")
    st.write('Enjoy your day :sunglasses: and wait for **all the eggs** to be in the nest.')
    menu = ["Instructions", "Login", "SignUp"]
    choice = st.sidebar.selectbox("App options", menu)

    if choice == "Instructions":
        st.subheader("Instructions")
        to_do = "With this app you can notice how eggs are in your nest. The first thing you will have to do is **Login** with your username and password. Later, you can start to classify your nest images. So, please choose in the sidebar the login option and add your information"
        st.write(to_do)
        image = Image.open(my_img)
        st.image(image, use_column_width=True)
    elif choice == "Login":
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:
                st.sidebar.success("Logged In as {}".format(username))

                task = st.selectbox("Task", ["Classifier", "About the app", "Profiles"])
                if task == "Classifier":
                    st.subheader("Add Your Image")

                    # Disabling warning
                    st.set_option('deprecation.showfileUploaderEncoding', False)
                    # Choose your own image
                    uploaded_file = st.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

                    if uploaded_file is None:
                        st.write("Please upload an Image to Classify")
                    else:
                        st.success("Your image is gonna start being classified")
                        u_img = Image.open(uploaded_file)
                        show = st.image(u_img, use_column_width=True)
                        show.image(u_img, 'Uploaded Image', use_column_width=True)
                        with st.spinner('Model Classifying ...'):
                            prediction = classifier.plot_image_classified(uploaded_file)
                            st.success('Done!')
                            st.header("Predicted class:")
                            st.write(prediction)
                elif task == "About the app":
                    st.subheader("About the app")
                    st.write('')
                elif task == "Profiles":
                    st.subheader("User Profiles")
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result, columns=["Username", "Password"])
                    st.dataframe(clean_db)
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")


if __name__ == '__main__':
    main()
