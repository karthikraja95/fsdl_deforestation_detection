import streamlit as st
import sys
import numpy as np
import pandas as pd
from skimage.io import imread
import plotly.express as px

sys.path.append("../data/")
from data_utils import TAGS, DEFORESTATION_TAGS
from dashboard_utils import run_model
from session_state import session_state

import time

N_LABELS_PER_ROW = 4


def playground():
    st.title("Playground")
    # Initial instructions
    init_info = st.empty()
    init_info.info(
        "ℹ️ Upload an image on the sidebar to run the model on it!\n"
        "Just be sure that it's satellite imagery, otherwise you're "
        "just going to get random outputs 🤷‍♂️"
    )
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
            # Check if it's a different image than the one before
            if input_file.name != session_state.image_name:
                session_state.image_name = input_file.name
                # Reset buttons
                session_state.user_feedback_positive = False
                session_state.user_feedback_negative = False
            fig = px.imshow(img)
            st.plotly_chart(fig)
            init_info.empty()
        # Run the model on the image
        with st.spinner("Running the model on the image..."):
            output = run_model(model, img)
            st.subheader("Model output:")
            if model_type == "Deforestation":
                if output == 1:
                    st.error("deforestation")
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
                                st.error(output_labels[idx])
                            else:
                                st.success(output_labels[idx])
                        idx += 1
                    row += 1
            # User feedback / data flywheel
            st.write("Did the model output match what you expected?")
            feedback_cols = st.beta_columns(2)
            with feedback_cols[0]:
                positive_btn = st.button("✅")
            with feedback_cols[1]:
                negative_btn = st.button("❌")
            if (
                positive_btn or session_state.user_feedback_positive
            ) and not negative_btn:
                session_state.user_feedback_positive = True
                session_state.user_feedback_negative = False
                st.info(
                    "Thank you for your feedback! This can help us "
                    "improve our models 🙌"
                )
                # TODO Upload the data to our database
                if st.button("Delete my image and feedback data"):
                    st.info(
                        "Alright, we deleted it. Just know that we had "
                        "high expectations that you could help us improve "
                        "deforestation detection models. We thought we "
                        "were friends 🙁"
                    )
            elif (
                negative_btn or session_state.user_feedback_negative
            ) and not positive_btn:
                session_state.user_feedback_positive = False
                session_state.user_feedback_negative = True
                st.info(
                    "Thank you for your feedback! This can help us "
                    "improve our models 🙌\n"
                    "It would be even better if you could tell us "
                    "what makes you think the model failed. Mind "
                    "leaving a comment bellow?"
                )
                # TODO Upload the data to our database
                user_comment = st.empty()
                user_comment_txt = user_comment.text_input(
                    label="Leave a comment on why the model failed.",
                    max_chars=280,
                )
                # TODO Update the data with the user comment
                if st.button("Delete my image and feedback data"):
                    st.info(
                        "Alright, we deleted it. Just know that we had "
                        "high expectations that you could help us improve "
                        "deforestation detection models. We thought we "
                        "were friends 🙁"
                    )
            with st.beta_expander("Peek inside the black box"):
                explain_cols = st.beta_columns(2)
                with explain_cols[0]:
                    st.subheader("Model structure")
                    # TODO Add a graph of the model structure and add references to ResNet
                with explain_cols[1]:
                    st.subheader("Output interpretation")
                    # TODO Add the result of applying SHAP to the model in the current sample


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
