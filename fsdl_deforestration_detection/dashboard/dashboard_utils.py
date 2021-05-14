import streamlit as st
import sys
from glob import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from fastai.vision.all import Learner, Normalize, imagenet_stats
from fastai.metrics import accuracy_multi, FBetaMulti, FBeta
import torch
import plotly.express as px
import plotly.graph_objects as go
import colorlover as cl

sys.path.append("../data/")
from data_utils import (
    DATA_PATH,
    TAGS,
    DEFORESTATION_TAGS,
    encode_tags,
    add_deforestation_label,
)

PERF_COLORS = cl.scales["8"]["div"]["RdYlGn"]


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def hash_fastai_model(model):
    """Just hash the model based on its number of outputs. We don't expect
    to keep changing the model, only want to hash different outputs for
    the binary and the multilabel models."""
    return model.n_out


def set_paths(dataset_name, model_type):
    if dataset_name == "Amazon":
        img_path = "planet/planet/train-jpg/"
        labels_path = "planet/planet/train_classes.csv"
        img_name_col = "image_name"
        if model_type == "Deforestation":
            label_col = "deforestation"
        else:
            label_col = TAGS
    else:
        img_path = "widsdatathon2019/leaderboard_holdout_data/"
        labels_path = "widsdatathon2019/holdout.csv"
        img_name_col = "image_id"
        label_col = "has_oilpalm"
    return img_path, labels_path, img_name_col, label_col


def load_image_names(model, chosen_set, img_path, labels_path):
    if chosen_set is None:
        img_names = sorted(glob(f"{DATA_PATH}{img_path}*.jpg"))
        img_names = [
            file_path.split("/")[-1].split(".")[0] for file_path in img_names
        ]
    else:
        if chosen_set == "Train":
            split_idx = model.dls.splits[0]
        else:
            split_idx = model.dls.splits[1]
        # Load the original dataframe and extract the image names from there
        labels_df = pd.read_csv(f"{DATA_PATH}{labels_path}")
        img_names = list(labels_df.iloc[split_idx, 0])
    return img_names


@st.cache
def load_data(
    dataset_name,
    model_type,
    img_path,
    img_names,
    labels_path,
    img_name_col,
    n_samples=100,
):
    # Load and preprocess the images
    imgs = np.empty(shape=(0, 0, 0, 0))
    count = 0
    for i in range(n_samples):
        img = imread(f"{DATA_PATH}{img_path}{img_names[i]}.jpg")
        img = np.expand_dims(img, axis=0)
        if count == 0:
            imgs = img
        else:
            imgs = np.concatenate((imgs, img))
        count += 1
    imgs = Normalize.from_stats(*imagenet_stats)(imgs)
    imgs = torch.from_numpy(imgs)
    # imgs = imgs.permute((0, 3, 1, 2))  # These 2 lines are only needed if running inference
    # imgs = imgs.float()                # directly through the PyTorch model
    # Load and preprocess the labels
    labels_df = pd.read_csv(f"{DATA_PATH}{labels_path}")
    labels_df.sort_values(img_name_col, inplace=True)
    if dataset_name == "Amazon":
        labels_df = encode_tags(labels_df, drop_tags_col=True)
        if model_type == "Deforestation":
            labels_df = add_deforestation_label(labels_df)
            labels = labels_df.iloc[:n_samples, -1].values
        else:
            labels = labels_df.iloc[:n_samples, 1:].values
    else:
        labels = labels_df.iloc[:n_samples, 1].values
    labels = torch.from_numpy(labels)
    return imgs, labels


@st.cache(hash_funcs={Learner: hash_fastai_model})
def run_model(model, img):
    output = model.predict(img[:, :, :3])
    return output


def show_amazon_labels(labels, n_labels_per_row=4):
    labels = [TAGS[idx] for idx in np.where(labels == 1)[0]]
    n_labels = len(labels)
    if n_labels < n_labels_per_row:
        n_rows = 1
        n_final_cols = n_labels
        row = 1
    else:
        n_rows = n_labels // n_labels_per_row
        n_final_cols = n_labels % n_labels_per_row
        row = 0
    idx = 0
    while row <= n_rows:
        n_cols = n_labels_per_row
        if row == n_rows:
            if n_final_cols == 0:
                break
            else:
                n_cols = n_final_cols
        cols = st.beta_columns(n_cols)
        for i in range(n_cols):
            with cols[i]:
                if labels[idx] in DEFORESTATION_TAGS:
                    st.error(labels[idx])
                else:
                    st.success(labels[idx])
            idx += 1
        row += 1


def show_model_output(model_type, output, n_labels_per_row=4):
    if model_type == "Deforestation":
        if output == 1:
            st.error("deforestation")
        else:
            st.success("no deforestation detected")
    else:
        if not any(output[1]):
            pred_proba = torch.sigmoid(output[2])
            top_pred = torch.topk(pred_proba, k=3).indices
            output[1][top_pred] = True
            st.info(
                "ℹ️ None of the model's ativations are high enough to assign "
                "any label with some confidence. Still, here are the top 3 "
                "predictions, with the respective probabilities given by the "
                f"model: {pred_proba[top_pred]}."
            )
        output = output[1]
        show_amazon_labels(output, n_labels_per_row)


def show_labels(
    dataset_name,
    sample_name,
    labels_path,
    img_name_col,
    label_col,
    n_labels_per_row=4,
):
    # Load the labels
    labels_df = pd.read_csv(f"{DATA_PATH}{labels_path}")
    if dataset_name == "Amazon":
        # TODO Remove this once we're loading from BigQuery
        labels_df = encode_tags(labels_df, drop_tags_col=True)
    labels_df[img_name_col] = labels_df[img_name_col].str.replace(".jpg", "")
    labels = labels_df.loc[
        labels_df[img_name_col] == sample_name, label_col
    ].values[0]
    # Show the labels in the dashboard
    if dataset_name == "Oil palm":
        if labels == 1:
            st.error("oil palm")
        else:
            st.success("no oil palm detected")
    else:
        show_amazon_labels(labels, n_labels_per_row)


@st.cache(hash_funcs={Learner: hash_fastai_model})
def get_performance_metrics(model, imgs, labels, dataset_name):
    # pred_logits = model.model(imgs)  # This is the PyTorch way to do it, which is faster but
    # doesn't apply all the preprocessing exactly like in FastAI
    pred_logits = list()
    for i in range(imgs.shape[0]):
        img_pred = model.predict(imgs[i])[2]
        pred_logits.append(img_pred)
    pred_logits = torch.stack(pred_logits)
    pred_proba = torch.sigmoid(pred_logits)
    pred = (pred_proba > 0.6).int()  # 0.6 seems to be threshold used by FastAI
    if dataset_name == "Oil palm":
        # Extract a deforestation label based on relevant tags
        deforestation_tags_idx = [
            TAGS.index(deforestation_tag)
            for deforestation_tag in DEFORESTATION_TAGS
        ]
        pred_logits = pred_logits[:, deforestation_tags_idx]
        pred_logits = torch.sum(pred_logits, dim=1)
        pred = pred[:, deforestation_tags_idx]
        pred = (torch.sum(pred, dim=1) > 0).int()
        acc = float(torch.mean((pred == labels).float()))
        fbeta = FBeta(beta=2)(preds=pred, targs=labels)
    else:
        acc = float(accuracy_multi(inp=pred_logits, targ=labels, thresh=0.6))
        fbeta = FBetaMulti(beta=2, average="samples", thresh=0.6)(
            preds=pred, targs=labels
        )
    return pred, acc, fbeta


def get_gauge_plot(value, title):
    max_value = 100
    color = PERF_COLORS[int(max((value / max_value) * len(PERF_COLORS) - 1, 0))]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(suffix="%"),
            domain=dict(x=[0, 1], y=[0, 1]),
            gauge=dict(
                axis=dict(range=[0, max_value]),
                bar=dict(thickness=1, color=color),
            ),
            title=dict(text=title),
        )
    )
    fig.update_layout(margin=dict(l=25, r=40, b=0, t=0, pad=0), height=380)
    return fig


def get_number_plot(value, title):
    fig = go.Figure(
        go.Indicator(mode="number", value=value, title=dict(text=title))
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), height=150)
    return fig


def get_hist_plot(values, values_type, dataset_name, model_type, title):
    if dataset_name == "Oil palm" and values_type == "labels":
        tags = ["oil palm"]
    elif model_type == "Land scenes" and dataset_name == "Amazon":
        tags = TAGS
    else:
        tags = ["deforestation"]
    fig = px.histogram(values, title=title)
    fig.update_layout(bargap=0.15)
    for i, tag in enumerate(tags):
        fig.data[i].name = tag
        fig.data[i].hovertemplate = fig.data[i].hovertemplate.replace(
            str(i), tag
        )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50, pad=0))
    return fig


def get_pixel_dist_plot(imgs):
    # Flatten the image arrays per channel
    imgs_flat = np.empty(
        (imgs.shape[3], imgs.shape[0] * imgs.shape[1] * imgs.shape[2])
    )
    for i in range(imgs.shape[3]):
        imgs_flat[i, :] = imgs[:, :, :, i].reshape((-1)).numpy()
    # Get the histogram data
    pixel_min = int(np.min(imgs_flat))
    pixel_max = int(np.max(imgs_flat))
    bins = [i for i in range(pixel_min, pixel_max, 1)]
    bin_centers = list()
    for i in range(len(bins) - 1):
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
    y = np.empty((imgs_flat.shape[0], len(bins) - 1))
    for i in range(imgs_flat.shape[0]):
        y[i, :], _ = np.histogram(imgs_flat[i], bins)
    pixels_df = pd.DataFrame(
        dict(pixel_value=bin_centers, blue=y[2], red=y[0], green=y[1])
    )
    pixels_df.set_index("pixel_value", inplace=True)
    # Make the plot
    fig = px.line(pixels_df, title="Distribution of pixel values per channel")
    fig.update_layout(
        yaxis_title="count", margin=dict(l=0, r=0, b=0, t=50, pad=0), height=300
    )
    return fig