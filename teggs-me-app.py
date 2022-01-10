# Imports
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# The following are specifications about the app
st.title('Teggs-me')
st.write('Enjoy your day :sunglasses: and wait for **all the eggs** to be in the nest.')



# currently, we need a image to be uploaded to know the state of the nest.
uploaded_file = st.file_uploader("Choose a recent file of your nest")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

number = st.number_input('If you are a small farmer, insert your number excluding the code of your country. Else, insert 0',
                         min_value=0,
                         max_value=12345678910,
                         step=1)
if number == 0:
    st.write("This is not an app for you, see you later!")
else:
    st.write('The current number is ', number)