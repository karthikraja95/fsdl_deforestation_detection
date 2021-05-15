import streamlit as st
import sys
from fastai.vision.all import load_learner
import torch
from skimage.io import imread
import plotly.express as px
from datetime import datetime

sys.path.append("fsdl_deforestation_detection/data/")
sys.path.append("fsdl_deforestation_detection/dashboard/")
from data_utils import DATA_PATH
from dashboard_utils import (
    set_paths,
    load_image,
    load_image_names,
    load_data,
    run_model,
    show_model_output,
    show_labels,
    get_performance_metrics,
    get_gauge_plot,
    get_number_plot,
    get_hist_plot,
    get_pixel_dist_plot,
    gen_user_id,
    gen_image_id,
    upload_user_data,
    upload_user_comment,
    delete_user_data,
)
from session_state import session_state

# Get a user ID
session_state.user_id = gen_user_id()


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
    # NOTE For the sake of time, we're just going to use the `Land scenes` model
    # model_type = st.sidebar.radio("Model", ["Deforestation", "Land scenes"])
    model_type = "Land scenes"
    input_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "tif"],
        accept_multiple_files=False,
        help="Test our model on an image of your liking! "
        "But remember that this should only work for satellite imagery, "
        "ideally around 256x256 size.",
    )
    st.sidebar.markdown(
        "Made by [André Ferreira](https://andrecnf.com/) and [Karthik Bhaskar](https://www.kbhaskar.com/)."
    )
    # Set the model
    model = load_learner(
        "fsdl_deforestation_detection/modeling/resnet50-128.pkl"
    )
    # Speed up model inference by deactivating gradients
    model.model.eval()
    torch.no_grad()
    if input_file is not None:
        # Load and display the uploaded image
        with st.spinner("Loading image..."):
            img = imread(input_file)
            # Check if it's a different image than the one before
            if input_file.name != session_state.image_name:
                session_state.image_name = input_file.name
                session_state.image_id = gen_image_id()
                session_state.ts = datetime.now()
                session_state.user_data_uploaded = False
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
        st.info(
            "ℹ️ Green labels represent categories that we don't associate with deforestation "
            "risk (e.g. natural occurences or old structures), while red labels can serve as "
            "a potential deforestation signal (e.g. new constructions, empty patches in forests)."
        )
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
                "ℹ️ Thank you for your feedback! This can help us "
                "improve our models 🙌"
            )
            if session_state.user_data_uploaded is False:
                upload_user_data(
                    session_state.user_id,
                    session_state.ts,
                    session_state.image_id,
                    img,
                    output[1],
                    session_state.user_feedback_positive,
                )
                session_state.user_data_uploaded = True
            if st.button("Delete my image and feedback data"):
                st.info(
                    "ℹ️ Alright, we deleted it. Just know that we had "
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
                "ℹ️ Thank you for your feedback! This can help us "
                "improve our models 🙌\n"
                "It would be even better if you could tell us "
                "what makes you think the model failed. Mind "
                "leaving a comment bellow?"
            )
            if session_state.user_data_uploaded is False:
                upload_user_data(
                    session_state.user_id,
                    session_state.ts,
                    session_state.image_id,
                    img,
                    output[1],
                    session_state.user_feedback_positive,
                )
                session_state.user_data_uploaded = True
            user_comment = st.empty()
            user_comment_txt = user_comment.text_input(
                label="Leave a comment on why the model failed.",
                max_chars=280,
            )
            if len(user_comment_txt) > 0:
                upload_user_comment(
                    session_state.user_id,
                    session_state.image_id,
                    user_comment_txt,
                )
            if st.button("Delete my image and feedback data"):
                st.info(
                    "ℹ️ Alright, we deleted it. Just know that we had "
                    "high expectations that you could help us improve "
                    "deforestation detection models. We thought we "
                    "were friends 🙁"
                )
                delete_user_data(session_state.user_id, session_state.image_id)
        # Model interpretation
        with st.beta_expander("Peek inside the black box"):
            explain_cols = st.beta_columns(2)
            with explain_cols[0]:
                st.subheader("Model structure")
                st.info(
                    "ℹ️ Our model is largely based on the [ResNet](https://paperswithcode.com/method/resnet) "
                    "archirtecture, using a ResNet50 from [FastAI](https://docs.fast.ai/). "
                    "Bellow you can see the model's layer definition."
                )
                st.text(model.model)
            with explain_cols[1]:
                st.subheader("Output interpretation")
                # TODO Add the result of applying SHAP to the model in the current sample
                st.info(
                    "ℹ️ Given some difficulties with using [SHAP](https://github.com/slundberg/shap) "
                    "with [FastAI](https://docs.fast.ai/), we haven't implemented this yet. "
                    "Would you like to give it a try?"
                )


def overview():
    st.title("Overview")
    # Initial instructions
    init_info = st.empty()
    # Set the sidebar inputs
    st.sidebar.title("Inputs")
    # NOTE For the sake of time, we're just going to use the `Land scenes` model
    # model_type = st.sidebar.radio("Model", ["Deforestation", "Land scenes"])
    model_type = "Land scenes"
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
        (
            bucket_name,
            img_path,
            labels_table,
            img_name_col,
            label_col,
        ) = set_paths(dataset_name, model_type)
    else:
        init_info.info(
            "ℹ️ You've selected the "
            "[oil palm dataset](https://www.kaggle.com/c/widsdatathon2019/), "
            "which can be seen as a test dataset, i.e. it wasn't used during training. "
            "While it should be somewhat similar to the Amazon dataset, it can be interesting "
            "to compare results on potentially out-of-domain data."
        )
        (
            bucket_name,
            img_path,
            labels_table,
            img_name_col,
            label_col,
        ) = set_paths(dataset_name, model_type)
    # Set the model
    model = load_learner(
        "fsdl_deforestation_detection/modeling/resnet50-128.pkl"
    )
    # Speed up model inference by deactivating gradients
    model.model.eval()
    torch.no_grad()
    img_names = load_image_names(model, chosen_set, bucket_name, labels_table)
    sample_name = st.sidebar.selectbox(
        "Sample",
        img_names,
    )
    st.sidebar.markdown(
        "Made by [André Ferreira](https://andrecnf.com/) and [Karthik Bhaskar](https://www.kbhaskar.com/)."
    )
    # Load all the data (or some samples) from the selected database
    n_samples = 250
    imgs, labels = load_data(
        dataset_name,
        model_type,
        bucket_name,
        img_path,
        img_names,
        labels_table,
        img_name_col,
        n_samples=n_samples,
    )
    # Show some performance metrics
    # TODO Use all the set data to get the correct performance metrics
    st.header("Performance")
    metrics_cols = st.beta_columns(2)
    with st.spinner("Getting performance results..."):
        if dataset_name == "Amazon":
            # NOTE This are the metrics obtained for the validation set,
            # when training the model in Colab; ideally, this should still
            # be calculated dynamically in here, but it's proving to be
            # slow and impractical in the approach that we were taking
            acc = 0.956407
            fbeta = 0.926633
            # pred, acc, fbeta = get_performance_metrics(
            #     model, imgs, labels, dataset_name
            # )
            pred, _, _ = get_performance_metrics(
                model, imgs, labels, dataset_name
            )
        else:
            pred, acc, fbeta = get_performance_metrics(
                model, imgs, labels, dataset_name
            )
        acc, fbeta = 100 * acc, 100 * fbeta
        with metrics_cols[0]:
            fig = get_gauge_plot(acc, title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        with metrics_cols[1]:
            fig = get_gauge_plot(fbeta, title="F2")
            st.plotly_chart(fig, use_container_width=True)
        if dataset_name == "Amazon":
            st.info(
                "ℹ️ These are the validation metrics [obtained when training the model](https://colab.research.google.com/github/karthikraja95/fsdl_deforestation_detection/blob/master/fsdl_deforestation_detection/experimental/FSDL_Final_Model.ipynb)."
            )
        else:
            st.info(
                "ℹ️ Showing performance metrics here by mapping the original labels to a "
                "binary, deforestation label. This should be somewhat relatable to the "
                "presence of oil palm plantations, which is the label in this dataset."
            )
    # Show number of samples
    fig = get_number_plot(len(img_names), title="Samples")
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
    st.info(
        f"ℹ️ Using only a subset of {n_samples} samples, so as to make this plot practically fast."
    )
    # Show imagery analysis
    st.header("Imagery analysis")
    st.subheader("Image size")
    img_size_cols = st.beta_columns(3)
    with img_size_cols[0]:
        fig = get_number_plot(imgs.shape[1], title="Height")
        st.plotly_chart(fig, use_container_width=True)
    with img_size_cols[1]:
        fig = get_number_plot(imgs.shape[2], title="Width")
        st.plotly_chart(fig, use_container_width=True)
    with img_size_cols[2]:
        fig = get_number_plot(imgs.shape[3], title="Channels")
        st.plotly_chart(fig, use_container_width=True)
    fig = get_pixel_dist_plot(imgs)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"ℹ️ Using only a subset of {n_samples} samples, so as to make this plot practically fast."
    )
    # Show sample analysis
    st.header("Sample analysis")
    # Load and display the uploaded image
    with st.spinner("Loading image..."):
        img = load_image(bucket_name, img_path, sample_name)
        fig = px.imshow(img)
        st.plotly_chart(fig)
    # Run the model on the image
    output = run_model(model, img)
    st.subheader("Model output:")
    show_model_output(model_type, output)
    st.subheader("Real labels:")
    show_labels(
        dataset_name, sample_name, labels_table, img_name_col, label_col
    )
    st.info(
        "ℹ️ Green labels represent categories that we don't associate with deforestation "
        "risk (e.g. natural occurences or old structures), while red labels can serve as "
        "a potential deforestation signal (e.g. new constructions, empty patches in forests)."
    )
    # Model interpretation
    with st.beta_expander("Peek inside the black box"):
        explain_cols = st.beta_columns(2)
        with explain_cols[0]:
            st.subheader("Model structure")
            st.info(
                "ℹ️ Our model is largely based on the [ResNet](https://paperswithcode.com/method/resnet) "
                "archirtecture, using a ResNet50 from [FastAI](https://docs.fast.ai/). "
                "Bellow you can see the model's layer definition."
            )
            st.text(model.model)
        with explain_cols[1]:
            st.subheader("Output interpretation")
            # TODO Add the result of applying SHAP to the model in the current sample
            st.info(
                "ℹ️ Given some difficulties with using [SHAP](https://github.com/slundberg/shap) "
                "with [FastAI](https://docs.fast.ai/), we haven't implemented this yet. "
                "Would you like to give it a try?"
            )
