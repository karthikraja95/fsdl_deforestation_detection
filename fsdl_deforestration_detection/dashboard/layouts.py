import streamlit as st
import sys
import numpy as np
import pandas as pd
from skimage.io import imread
import plotly.express as px

sys.path.append("../data/")
from data_utils import TAGS, DEFORESTATION_TAGS

import time

N_LABELS_PER_ROW = 4


def playground():
    st.title("Playground")
    # Set the sidebar inputs
    st.sidebar.title("Inputs")
    model_type = st.sidebar.radio("Model", ["Deforestation", "Land scenes"])
    input_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "tif"],
        accept_multiple_files=False,
        help="Test our model on an image of your liking! "
        "But remember that this should only work for satellite imagery, "
        "ideally around 256x256 size.",
    )
    # Set the model
    # NOTE Using random data while the models aren't usable
    if model_type == "Deforestation":
        model = lambda x: np.random.rand(1)
    else:
        model = lambda x: np.random.rand(17)
    if input_file is not None:
        # Load and display the uploaded image
        with st.spinner("Loading image..."):
            img = imread(input_file)
            fig = px.imshow(img)
            st.plotly_chart(fig)
        # Run the model on the image
        with st.spinner("Running the model on the image..."):
            time.sleep(1)
            output = model(img)
            output = output.round()
            st.subheader("Model output:")
            if model_type == "Deforestation":
                if output == 1:
                    st.warning("deforestation")
                else:
                    st.success("no deforestation detected")
            else:
                output_labels = [TAGS[idx] for idx in np.where(output == 1)[0]]
                n_pred_labels = len(output_labels)
                n_rows = n_pred_labels // N_LABELS_PER_ROW
                n_final_cols = n_pred_labels % N_LABELS_PER_ROW
                row, idx = 0, 0
                while row <= n_rows:
                    n_cols = N_LABELS_PER_ROW
                    if row == n_rows:
                        if n_final_cols == 0:
                            break
                        else:
                            n_cols = n_final_cols
                    cols = st.beta_columns(n_cols)
                    for i in range(n_cols):
                        with cols[i]:
                            if output_labels[idx] in DEFORESTATION_TAGS:
                                st.warning(output_labels[idx])
                            else:
                                st.success(output_labels[idx])
                        idx += 1
                    row += 1


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
