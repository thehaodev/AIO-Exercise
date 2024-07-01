import cv2
import numpy as np
from PIL import Image
import streamlit as st

MODEL = "D:/AI_VIETNAM/CODE_EXERCISE/AIO-Exercise/module_1/project/model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "D:/AI_VIETNAM/CODE_EXERCISE/AIO-Exercise/module_1/project/model/MobileNetSSD_deploy.prototxt.txt"


def process_image(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()

    return detections


def annotate_image(image: Image, detections, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), 70, 2)

    return image


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
