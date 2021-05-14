import os
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.io import imread
from typing import List, Tuple

# NOTE This is just for the Planet Amazon dataset;
# we need to refactor this when we use other datasets
DATA_PATH = "/Users/andrecnf/Documents/datasets/fsdl/"
PLANET_PATH = "planet-understanding-the-amazon-from-space/"
IMG_PATH = "train-jpg/"
TIFF_PATH = "train-tif-v2/"
LABELS_PATH = "train_v2.csv/train_v2.csv"
TAGS = [
    "agriculture",
    "artisinal_mine",
    "bare_ground",
    "blooming",
    "blow_down",
    "clear",
    "cloudy",
    "conventional_mine",
    "cultivation",
    "habitation",
    "haze",
    "partly_cloudy",
    "primary",
    "road",
    "selective_logging",
    "slash_burn",
    "water",
]
DEFORESTATION_TAGS = [
    "agriculture",
    "artisinal_mine",
    "conventional_mine",
    "cultivation",
    "road",
    "selective_logging",
    "slash_burn",
]


def decompress_tar_7z(fn: str, input_dir: str, output_dir: str):
    """Decompress the image directories, which are compressed in
    .tar.7z files.

    Args:
        fn (str): Name of the compressed file.
        input_dir (str): Path where the compressed file is in.
        output_dir (str): Path to where we want to decompress
        the file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    subprocess.run(
        [
            "7z",
            "x",
            "-so",
            f"{input_dir}{fn}",
            "|",
            "tar",
            "xf",
            "-",
            "-C",
            output_dir,
        ]
    )


def decode_img(file_name, label):
    image_string = tf.io.read_file(file_name)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_decoded = tf.cast(image_decoded, tf.float32)
    image_decoded = tf.image.resize(image_decoded, [256, 256])
    return image_decoded, label


def get_amazon_sample(
    df: pd.DataFrame, load_tiff: bool = False
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Generator that iterates through the labels and gets us the image (JPG or TIFF)
    and the label.

    Args:
        df (pd.DataFrame): Dataframe containing the image file names and their
        associated labels.
        load_tiff (bool, optional): Indicates whether to load the image in the TIFF (True)
        or the JPG (False) format. Defaults to False.

    Yields:
        Iterator[Tuple[np.ndarray, np.ndarray]]: Returns the current image data and the
        tags (i.e. the labels as in the original data).
    """
    for row in df.itertuples():
        if load_tiff:
            img_data = imread(
                f"{DATA_PATH}{PLANET_PATH}{TIFF_PATH}{row[1]}.tif"
            )
        else:
            img_data = imread(f"{DATA_PATH}{PLANET_PATH}{IMG_PATH}{row[1]}.jpg")
        yield img_data, np.array(row[2:])


def encode_tags(
    df: pd.DataFrame,
    tags_col_name: str = "tags",
    tags: List = TAGS,
    drop_tags_col: bool = False,
) -> pd.DataFrame:
    """Convert the tags (or labels) in a dataframe into a one hot encoded representation.

    Args:
        df (pd.DataFrame): Original dataframe that contains the uncleaned tags.
        tags_col_name (str, optional): Name of the column that contains the tags.
        Defaults to 'tags'.
        tags (List, optional): Names of the possible tags. Defaults to TAGS.
        drop_tags_col (bool, optional): If set to True, the original `tags` column is
        removed. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe with the tags one hot encoded in new columns.
    """
    for label in tags:
        df[label] = df[tags_col_name].apply(
            lambda x: 1 if label in x.split(" ") else 0
        )
    if drop_tags_col:
        df.drop(columns=tags_col_name, inplace=True)
    return df


def add_deforestation_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a deforestation label to the dataframe, based on other tags.

    Args:
        df (pd.DataFrame): Dataframe with one hot encoded tags.

    Returns:
        pd.DataFrame: Dataframe with the new deforestation column.
    """
    df["deforestation"] = 0
    deforestation_idx = df.query(
        " == 1 | ".join(DEFORESTATION_TAGS) + " == 1"
    ).index
    df.loc[deforestation_idx, "deforestation"] = 1
    df["deforestation"] = df["deforestation"].astype("uint8")
    return df