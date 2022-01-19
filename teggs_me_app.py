# importing the libraries
import streamlit as st
from PIL import Image
from model.pre_trained_model_prediction import PretrainedObjectDetectionModelPrediction
import os

directory = os.path.dirname(os.path.realpath(__file__))
my_default_image = "eggs_in_the_nest.jpeg"
my_img = os.path.join(directory, my_default_image)

# loading the classifier
classifier = PretrainedObjectDetectionModelPrediction()
# Designing the interface
st.title("Teggs-me: Eggs in Nest Image Classification App")

st.write('Enjoy your day :sunglasses: and wait for **all the eggs** to be in the nest.')

image = Image.open(my_img)
show = st.image(image, use_column_width=True)

st.sidebar.title("Upload Image")

# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

if uploaded_file is None:
    st.sidebar.write("Please upload an Image to Classify")
else:
    st.sidebar.write("Your image is gonna start being classified")
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)
    with st.spinner('Pretrained Classifying ...'):
        prediction = classifier.image_classification_classifier_pretrained(uploaded_file)
        st.success('Done!')
        st.sidebar.header("Predicted class:")
        st.sidebar.write(prediction)
