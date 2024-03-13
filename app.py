import streamlit as st
from style_transfer_model import model
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub

st.title("welcome! Please chose from below your preferred action!")

tabs1,tabs2=st.tabs(["Transfer style","About Us"])

with tabs1:
    content_image=st.file_uploader("Upload the image you want the style change!")
    style_image=st.file_uploader("Upload the image you want the style")

    if (content_image is not None) and (style_image is not None):
        content_image=Image.open(content_image)
        style_image=Image.open(style_image)
        content_image=np.array(content_image)
        style_image=np.array(style_image)
        style_image=style_image.astype(np.float32)[np.newaxis, ...]/255.
        content_image=content_image.astype(np.float32)[np.newaxis, ...]/255.
        style_image=tf.image.resize(style_image,(256,256))
        with st.spinner("Please wait()"):    
            stylized_output=model(tf.constant(content_image),tf.constant(style_image))
        stylized_output=np.array(stylized_output[0])
        st.image(stylized_output)

