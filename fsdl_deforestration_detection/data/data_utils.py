import os
import subprocess
import numpy as np
import pandas as pd
from skimage.io import imread
from typing import List, Tuple

# NOTE This is just for the Planet Amazon dataset;
# we need to refactor this when we use other datasets
DATA_PATH = "/home/andreferreira/data/"
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
        Iterator[Tuple[int, np.ndarray, np.ndarray]]: Returns the current iteration's index,
        image data and the tags (i.e. the labels as in the original data).
    """
    for row in df.itertuples():
        if load_tiff:
            img_data = imread(f"{DATA_PATH}{TIFF_PATH}{row[1]}.tif")
        else:
            img_data = imread(f"{DATA_PATH}{IMG_PATH}{row[1]}.jpg")
        yield row[0], img_data, np.array(row[2:])


def encode_tags(
    df: pd.DataFrame, tags_col_name: str = "tags", tags: List = TAGS
) -> pd.DataFrame:
    """Convert the tags (or labels) in a dataframe into a one hot encoded representation.

    Args:
        df (pd.DataFrame): Original dataframe that contains the uncleaned tags.
        tags_col_name (str, optional): Name of the column that contains the tags.
        Defaults to 'tags'.
        tags (List, optional): Names of the possible tags. Defaults to TAGS.

    Returns:
        pd.DataFrame: Dataframe with the tags one hot encoded in new columns.
    """
    for label in tags:
        df[label] = df[tags_col_name].apply(
            lambda x: 1 if label in x.split(" ") else 0
        )
    return df


def add_deforestation_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a deforestation label to the dataframe, based on other tags.

    Args:
        df (pd.DataFrame): Dataframe with one hot encoded tags.

    Returns:
        pd.DataFrame: Dataframe with the new deforestation column.
    """
    df["deforestation"] = (
        (df["agriculture"] == 1)
        | (df["artisinal_mine"] == 1)
        | (df["conventional_mine"] == 1)
        | (df["cultivation"] == 1)
        | (df["road"] == 1)
        | (df["selective_logging"] == 1)
        | (df["slash_burn"] == 1)
    )
    df["deforestation"] = df["deforestation"].astype("uint8")
    return df