import numpy as np
from PIL import Image
import streamlit as st
from module_1.project.utils.utils_object_detection import process_image, annotate_image


def run():
    st.title("Object Detection for Images")
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")
        image = Image.open(file)
        image = np.array(image)
        detections = process_image(image)
        prc_image = annotate_image(image, detections)
        st.image(prc_image, caption="Processed Image")


run()
