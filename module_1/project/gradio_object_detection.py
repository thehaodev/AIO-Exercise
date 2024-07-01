import numpy as np
import gradio as gr
from PIL import Image
from module_1.project.utils_image import process_image, annotate_image


def process(input_img):
    if input_img is not None:
        # transfrom ndarray to Image
        img = Image.fromarray(input_img, 'RGB')
        img = np.array(img)
        detections = process_image(img)
        prc_image = annotate_image(img, detections)
        return prc_image
    else:
        return print("Do it again")


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # OBJECT DETECTION FOR IMAGES
    """)
    gr.Interface(process, gr.Image(), "image")

demo.launch()
