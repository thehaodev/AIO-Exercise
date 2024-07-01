## Table of Contents

- Introduction
- Installation
- Usage example
## Introduction
This repository build app using Gradio and Streamlit. Both have three ap with same feature but different UI. 

Three app main feature are: Objec-Detection, Simple ChatBot with HuggingFace and WordCorrection
## Installation
### 1. Clone the repository  
```
git clone https://github.com/thehaodev/AIO-Exercise.git
```
### 2. Create and activate a virtual environment 
```
py -m venv .venv
.venv\Scripts\activate
```
### 3. Install the required dependencies 
```
pip install -r requirements.txt
```
## Usage example
### 1. Gradio
With gradio, only Chatbot will need to run locally. The other two have deploy and domain show in Demo
```
cd module_1/project
python gradio_chatbot.py
```
This is the example result: 
![GRADIO_CHATBOT](https://github.com/thehaodev/AIO-Exercise/assets/112054658/b552e680-103a-45e5-9167-b86bd669c29d)
### 2. Streamlit
With Streamlit all three app have deploy on streamlit server and domain show in Demo. If you want to run locally try this.
```
cd module_1/project
streamlit run streamlit_chatobt
```
## Demo
### 1. Gradio
Here is working live demo for two app: ObjectDetection and WordCorrection

https://huggingface.co/spaces/oldHao/WORD_CORRECTION/

https://huggingface.co/spaces/oldHao/OBJECT_DETECTION
### 2. Streamlit
Here is working live demo for three app: ObjectDetection and WordCorrection

https://aio-exercise-chatbot.streamlit.app/

https://aio-exercise-object-detection.streamlit.app/

https://aio-exercise-word-correction.streamlit.app/

#### Note: If the app sleep (cause of inactivity). Just hit the restar button and wait couple of minutes .
