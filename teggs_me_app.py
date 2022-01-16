# importing the libraries
import streamlit as st
from PIL import Image
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from transformers import ViTFeatureExtractor, ViTForImageClassification
import os

directory = os.path.dirname(os.path.realpath(__file__))
my_default_image = "eggs_in_the_nest.jpeg"
my_img = os.path.join(directory, my_default_image)

# loading the classifier

pretrained_path = 'facebook/detr-resnet-101-dc5'
classifier = VisionClassifierInference(
    feature_extractor=ViTFeatureExtractor.from_pretrained(pretrained_path),
    model=ViTForImageClassification.from_pretrained(pretrained_path),
    )


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

if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)

# For newline
model = st.sidebar.selectbox(
     'Select a model',
     options=['pre_trained_model', 'my_model'])

if st.sidebar.button("Click Here to Classify"):

    if uploaded_file is None:

        st.sidebar.write("Please upload an Image to Classify")

    else:

        if model == "pretrained_model":
            with st.spinner('Pretrained Classifying ...'):

                prediction = classifier.predict(img_path=uploaded_file)
                st.success('Done!')

            st.sidebar.header("Predicted class:", prediction)

            # Formatted probability value to 3 decimal places
            probability = "{:.3f}".format(float(prediction * 100))

            # Classify cat being present in the picture if prediction > 0.5

            if prediction > 0.5:

                st.sidebar.write("It's a 'Cat' picture.", '\n')

                st.sidebar.write('**Probability: **', probability, '%')

            else:
                st.sidebar.write(" It's a 'Non-Cat' picture ", '\n')

                st.sidebar.write('**Probability: **', probability, '%')
