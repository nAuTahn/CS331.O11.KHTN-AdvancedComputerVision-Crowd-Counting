import streamlit as st
from PIL import Image
import numpy as np

from util import test

#set_background('./bgs/bg5.png')

# set title
st.title('Crowd Counting')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])


# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    enumber = test(image)

    # write classification
    st.write("### Estimated number: {}".format(int(enumber)))