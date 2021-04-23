import streamlit as st
import numpy as np
import pandas as pd
from skimage.io import imread
import plotly.express as px


def playground():
    st.title("Playground")
    # Set the sidebar inputs
    st.sidebar.title("Inputs")
    model_type = st.sidebar.radio("Model", ["Deforestation", "Land scenes"])
    input_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "png", "tif"],
        accept_multiple_files=False,
        help="Test our model on an image of your liking! "
        "But remember that this should only work for satellite imagery, "
        "ideally around 256x256 size.",
    )
    # Set the model
    if model_type == "Deforestation":
        model = lambda 
    if input_file is not None:
        # Load and display the uploaded image
        with st.spinner("Loading image..."):
            img = imread(input_file)
            fig = px.imshow(img)
            st.plotly_chart(fig)



def overview():
    st.title("Overview")
    # Set the sidebar inputs
    st.sidebar.title("Inputs")
    model_type = st.sidebar.radio("Model", ["Deforestation", "Land scenes"])
    dataset_name = st.sidebar.radio("Dataset", ["Amazon", "Oil palm"])
    sample_name = st.sidebar.selectbox(
        "Sample",
        ["train_0", "train_1", "train_2", "train_3", "train_4", "train_5"],
    )
