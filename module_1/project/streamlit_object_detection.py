import numpy as np
from PIL import Image
import streamlit as st
import utils_image as ut_image


def run():
    st.title("Object Detection for Images")
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")
        image = Image.open(file)
        image = np.array(image)
        detections = ut_image.process_image(image)
        prc_image = ut_image.annotate_image(image, detections)
        st.image(prc_image, caption="Processed Image")


run()
