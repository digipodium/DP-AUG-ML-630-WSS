import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image


def load_model():
    try:
        model = tf.keras.models.load_model('cifar_model.h5')
        return model
    except Exception as e:
        st.error(e)

def predict_class(model, image, shape=(32, 32)):
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    image = np.expand_dims(image, axis=0)
    # st.write(image.shape)
    image = tf.image.resize(image, shape)
    output = model.predict(image)
    return class_names[output.argmax()]
    

st.set_page_config(
    page_title="Object Detection App",
    page_icon="📸",
    layout="wide",
)

with st.spinner("Loading Model Into Memory..."):
    model = load_model()


st.title("Object Detection App")
st.write("This is a simple object detection web app to detect objects in images.")
with st.form("Form1"):
    file = st.file_uploader("Upload an image", type=["jpg", "png"])
    submit = st.form_submit_button("Submit")

c1 , c2 = st.columns(2)
if file:
    c1.image(file)
    image = Image.open(file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    result = predict_class(model, image)
    c2.markdown(f'''# Prediction
    {result}
    ''')
