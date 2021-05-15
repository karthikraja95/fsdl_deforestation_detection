import streamlit as st
import sys
import os
import json
import numpy as np
import pandas as pd
from skimage.io import imread
from PIL import Image
from fastai.vision.all import Learner, Normalize, imagenet_stats
from fastai.metrics import accuracy_multi, FBetaMulti, FBeta
import torch
import plotly.express as px
import plotly.graph_objects as go
import colorlover as cl
from google.cloud import storage, bigquery
import io

sys.path.append("fsdl_deforestation_detection/data/")
from data_utils import (
    DATA_PATH,
    TAGS,
    DEFORESTATION_TAGS,
    encode_tags,
    add_deforestation_label,
)

PERF_COLORS = cl.scales["8"]["div"]["RdYlGn"]

# Setup Google Cloud authentication
with open("google_credentials.json", "w") as fp:
    json.dump(st.secrets["google_credentials"], fp)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"
storage_client = storage.Client()
bq_client = bigquery.Client()


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
        bucket_name = "planet_amazon"
        img_path = "train-jpg/"
        labels_table = "planet_labels"
        img_name_col = "image_name"
        if model_type == "Deforestation":
            label_col = "deforestation"
        else:
            label_col = TAGS
    else:
        bucket_name = "wids_oil_palm"
        img_path = "leaderboard_holdout_data/"
        labels_table = "wids_oil_palm_labels"
        img_name_col = "image_id"
        label_col = "has_oilpalm"
    return bucket_name, img_path, labels_table, img_name_col, label_col


def load_image(bucket_name, img_path, image_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(f"{img_path}{image_name}.jpg")
    blob = blob.download_as_bytes()
    blob = io.BytesIO(blob)
    image = imread(blob)
    return image


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_labels_df(labels_table):
    labels_df = pd.read_gbq(
        f"SELECT * FROM `fsdl-305310.deforestation_data.{labels_table}`"
    )
    return labels_df


def load_image_names(model, chosen_set, bucket_name, labels_table):
    if chosen_set is None:
        img_names = storage_client.list_blobs(bucket_name)
        img_names = sorted(
            [
                f.name.split("/")[-1].split(".")[0]
                for f in img_names
                if ".jpg" in f.name
            ]
        )
    else:
        if chosen_set == "Train":
            split_idx = model.dls.splits[0]
        else:
            split_idx = model.dls.splits[1]
        # Load the original dataframe and extract the image names from there
        labels_df = load_labels_df(labels_table)
        img_names = sorted(list(labels_df.iloc[split_idx, 0]))
    return img_names


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data(
    dataset_name,
    model_type,
    bucket_name,
    img_path,
    img_names,
    labels_table,
    img_name_col,
    n_samples=100,
):
    # Load and preprocess the images
    imgs = np.empty(shape=(0, 0, 0, 0))
    count, i = 0, 0
    while count < n_samples:
        try:
            img = load_image(bucket_name, img_path, img_names[i])
            img = np.expand_dims(img, axis=0)
            if count == 0:
                imgs = img
            else:
                imgs = np.concatenate((imgs, img))
            count += 1
        except AttributeError:
            st.warning(
                f"Couldn't load image {img_names[count]} from {bucket_name}/{img_path}. Make sure that it was properly uploaded."
            )
        i += 1
    imgs = Normalize.from_stats(*imagenet_stats)(imgs)
    imgs = torch.from_numpy(imgs)
    # imgs = imgs.permute((0, 3, 1, 2))  # These 2 lines are only needed if running inference
    # imgs = imgs.float()                # directly through the PyTorch model
    # Load and preprocess the labels
    labels_df = load_labels_df(labels_table)
    labels_df.sort_values(img_name_col, inplace=True)
    if dataset_name == "Amazon":
        if model_type == "Deforestation":
            labels_df = add_deforestation_label(labels_df)
            labels = labels_df.iloc[:n_samples, -1].values
        else:
            labels = labels_df.iloc[:n_samples, 1:].values
    else:
        labels = labels_df.iloc[:n_samples, 1].values
    labels = torch.from_numpy(labels)
    return imgs, labels


@st.cache(
    hash_funcs={Learner: hash_fastai_model},
    suppress_st_warning=True,
    allow_output_mutation=True,
)
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
    labels_table,
    img_name_col,
    label_col,
    n_labels_per_row=4,
):
    # Load the labels
    labels_df = load_labels_df(labels_table)
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
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), height=160)
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


def gen_user_id():
    # Load the latest user ID
    user_data_df = pd.read_gbq(
        "SELECT * FROM `fsdl-305310.user_data.playground_uploads`"
    )
    if len(user_data_df) == 0:
        last_id = 0
    else:
        last_id = user_data_df.user_id.max()
    # Get a new, unique user ID
    user_id = last_id + 1
    return user_id


def gen_image_id():
    # Load the latest image ID
    user_data_df = pd.read_gbq(
        "SELECT * FROM `fsdl-305310.user_data.playground_uploads`"
    )
    if len(user_data_df) == 0:
        last_id = 0
    else:
        last_id = user_data_df.image_id.max()
    # Get a new, unique image ID
    image_id = last_id + 1
    return image_id


def upload_user_data(
    user_id, ts, image_id, image, output, user_feedback_positive
):
    # Upload the user's input image
    bucket = storage_client.bucket("playground_images")
    blob = bucket.blob(f"{image_id}.jpg")
    f = io.BytesIO()
    pil_img = Image.fromarray(image)
    pil_img.save(f, "jpeg")
    pil_img.close()
    blob.upload_from_string(f.getvalue(), content_type="image/jpeg")
    # Upload a row with the feedback
    user_data = dict(
        user_id=user_id,
        ts=ts,
        image_id=image_id,
        user_feedback_positive=user_feedback_positive,
        user_comment="",
        output_agriculture=output[0],
        output_artisinal_mine=output[1],
        output_bare_ground=output[2],
        output_blooming=output[3],
        output_blow_down=output[4],
        output_clear=output[5],
        output_cloudy=output[6],
        output_conventional_mine=output[7],
        output_cultivation=output[8],
        output_habitation=output[9],
        output_haze=output[10],
        output_partly_cloudy=output[11],
        output_primary=output[12],
        output_road=output[13],
        output_selective_logging=output[14],
        output_slash_burn=output[15],
        output_water=output[16],
    )
    user_df = pd.Series(user_data).to_frame().transpose()
    user_df.user_id = user_df.user_id.astype(int)
    user_df.image_id = user_df.image_id.astype(int)
    user_df.user_feedback_positive = user_df.user_feedback_positive.astype(bool)
    user_df.user_comment = user_df.user_comment.astype(str)
    user_df.output_agriculture = user_df.output_agriculture.astype(bool)
    user_df.output_artisinal_mine = user_df.output_artisinal_mine.astype(bool)
    user_df.output_bare_ground = user_df.output_bare_ground.astype(bool)
    user_df.output_blooming = user_df.output_blooming.astype(bool)
    user_df.output_blow_down = user_df.output_blow_down.astype(bool)
    user_df.output_clear = user_df.output_clear.astype(bool)
    user_df.output_cloudy = user_df.output_cloudy.astype(bool)
    user_df.output_conventional_mine = user_df.output_conventional_mine.astype(
        bool
    )
    user_df.output_cultivation = user_df.output_cultivation.astype(bool)
    user_df.output_habitation = user_df.output_habitation.astype(bool)
    user_df.output_haze = user_df.output_haze.astype(bool)
    user_df.output_partly_cloudy = user_df.output_partly_cloudy.astype(bool)
    user_df.output_primary = user_df.output_primary.astype(bool)
    user_df.output_road = user_df.output_road.astype(bool)
    user_df.output_selective_logging = user_df.output_selective_logging.astype(
        bool
    )
    user_df.output_slash_burn = user_df.output_slash_burn.astype(bool)
    user_df.output_water = user_df.output_water.astype(bool)
    user_df.to_gbq("user_data.playground_uploads", if_exists="append")


def upload_user_comment(user_id, image_id, user_comment):
    dml_statement = (
        "UPDATE user_data.playground_uploads "
        f"SET user_comment = '{user_comment}' "
        f"WHERE (user_id = {user_id} AND image_id = {image_id})"
    )
    query_job = bq_client.query(dml_statement)


def delete_user_data(user_id, image_id):
    # Delete the image
    bucket = storage_client.bucket("playground_images")
    blob = bucket.blob(f"{image_id}.jpg")
    blob.delete()
    # Delete the row from BigQuery
    dml_statement = (
        "DELETE user_data.playground_uploads "
        f"WHERE (user_id = {user_id} AND image_id = {image_id})"
    )
    query_job = bq_client.query(dml_statement)