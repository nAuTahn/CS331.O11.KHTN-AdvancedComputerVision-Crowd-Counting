import streamlit as st
from PIL import Image
import numpy as np
import base64

import plotly.graph_objects as go

from util import test

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('background.png')

# set title
st.title('Crowd Counting')
st.write('Anh Nguyen Tuan - Hoang Ha Van')
# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])


# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    mcnn, gloss, vgg19, res50 = test(image)

    # write classification
    st.write("### Estimated number - MCNN     : {}".format(int(mcnn)))
    st.write("### Estimated number - VGG19    : {}".format(int(vgg19)))    
    st.write("### Estimated number - GLoss    : {}".format(int(gloss)))
    st.write("### Estimated number - ResNet-50: {}".format(int(res50)))