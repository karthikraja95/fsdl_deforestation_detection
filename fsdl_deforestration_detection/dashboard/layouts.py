import streamlit as st
import sys
import numpy as np
import pandas as pd
from fastai.vision.all import load_learner
import torch
from skimage.io import imread
import plotly.express as px

sys.path.append("../data/")
from data_utils import DATA_PATH, IMG_PATH
from dashboard_utils import (
    set_paths,
    run_model,
    show_model_output,
    show_labels,
    get_performance_metrics,
    load_data,
    get_gauge_plot,
    get_number_plot,
    get_hist_plot,
    get_pixel_dist_plot,
)
from session_state import session_state


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
        model = load_learner("../modeling/resnet50-128.pkl")
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
        output = run_model(model, img)
        st.subheader("Model output:")
        show_model_output(model_type, output)
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
        # Model interpretation
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
    # Initial instructions
    init_info = st.empty()
    # Set the sidebar inputs
    st.sidebar.title("Inputs")
    model_type = st.sidebar.radio("Model", ["Deforestation", "Land scenes"])
    dataset_name = st.sidebar.radio("Dataset", ["Amazon", "Oil palm"])
    chosen_set = None
    if dataset_name == "Amazon":
        chosen_set = st.sidebar.radio("Set", ["Train", "Validation"])
        init_info.info(
            "ℹ️ You've selected the "
            "[Amazon dataset](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/), "
            "which is the one in which our models were trained on. As such, you can look at performance "
            "on either the train or validation set."
        )
        img_path, _, _, _ = set_paths(dataset_name, model_type)
    else:
        init_info.info(
            "ℹ️ You've selected the "
            "[oil palm dataset](https://www.kaggle.com/c/widsdatathon2019/), "
            "which can be seen as a test dataset, i.e. it wasn't used during training. "
            "While it should be somewhat similar to the Amazon dataset, it can be interesting "
            "to compare results on potentially out-of-domain data."
        )
        img_path, _, _, _ = set_paths(dataset_name, model_type)
    sample_name = st.sidebar.selectbox(
        "Sample",
        # TODO Get the real file names
        ["train_0", "train_1", "train_2", "train_3", "train_4", "train_5"],
    )
    # Set the model
    # NOTE Using random data while the models aren't usable
    if model_type == "Deforestation":
        model = lambda x: np.random.rand(1)
    else:
        model = load_learner("../modeling/resnet50-128.pkl")
    # Load all the data (or some samples) from the selected database
    imgs, labels = load_data(dataset_name, model_type, chosen_set)
    # Show some performance metrics
    # TODO Use all the set data to get the correct performance metrics
    st.header("Performance")
    metrics_cols = st.beta_columns(2)
    with st.spinner("Getting performance results..."):
        # NOTE Dummy data
        # acc = 0.956407
        # fbeta = 0.926633
        pred, acc, fbeta = get_performance_metrics(model, imgs, labels)
        acc, fbeta = 100 * acc, 100 * fbeta
        with metrics_cols[0]:
            fig = get_gauge_plot(acc, title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        with metrics_cols[1]:
            fig = get_gauge_plot(fbeta, title="F2")
            st.plotly_chart(fig, use_container_width=True)
    # Show number of samples
    # TODO Consider replacing this value with all the images
    # associated with the current set, even if some plots
    # rely on smaller subsets
    fig = get_number_plot(len(imgs), title="Samples")
    st.plotly_chart(fig, use_container_width=True)
    # Show label analysis
    st.header("Label analysis")
    labels_cols = st.beta_columns(2)
    with labels_cols[0]:
        fig = get_hist_plot(
            labels,
            "labels",
            dataset_name,
            model_type,
            title="Labels distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    with labels_cols[1]:
        fig = get_hist_plot(
            pred,
            "predictions",
            dataset_name,
            model_type,
            title="Predicted labels distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    # Show imagery analysis
    # TODO Cache the imagery plots
    st.header("Imagery analysis")
    st.subheader("Image size")
    img_size_cols = st.beta_columns(3)
    with img_size_cols[0]:
        fig = get_number_plot(imgs.shape[2], title="Height")
        st.plotly_chart(fig, use_container_width=True)
    with img_size_cols[1]:
        fig = get_number_plot(imgs.shape[3], title="Width")
        st.plotly_chart(fig, use_container_width=True)
    with img_size_cols[2]:
        fig = get_number_plot(imgs.shape[1], title="Channels")
        st.plotly_chart(fig, use_container_width=True)
    fig = get_pixel_dist_plot(imgs)
    st.plotly_chart(fig, use_container_width=True)
    # TODO Show sample analysis
    # TODO Cache the model inference results
    st.header("Sample analysis")
    # Load and display the uploaded image
    with st.spinner("Loading image..."):
        # TODO Adjust the sample loading to the appropriate path according to the chosen dataset and set
        # TODO Load image from Google Cloud Storage bucket
        img = imread(f"{DATA_PATH}{img_path}{sample_name}")
        fig = px.imshow(img)
        st.plotly_chart(fig)
    # Run the model on the image
    output = run_model(model, img)
    sample_analysis_cols = st.beta_columns(2)
    with sample_analysis_cols[0]:
        st.subheader("Model output:")
        show_model_output(model_type, output, n_labels_per_row=2)
    with sample_analysis_cols[1]:
        st.subheader("Real labels:")
        show_labels(dataset_name, sample_name, n_labels_per_row=2)
    # Model interpretation
    with st.beta_expander("Peek inside the black box"):
        explain_cols = st.beta_columns(2)
        with explain_cols[0]:
            st.subheader("Model structure")
            # TODO Add a graph of the model structure and add references to ResNet
        with explain_cols[1]:
            st.subheader("Output interpretation")
            # TODO Add the result of applying SHAP to the model in the current sample