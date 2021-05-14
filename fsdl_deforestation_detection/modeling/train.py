import argparse
import sys
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.metrics import FBetaScore
from tensorflow.keras import optimizers, callbacks, Model
from tensorflow.data.experimental import AUTOTUNE
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
import wandb
from wandb.keras import WandbCallback

sys.path.append(
    "fsdl_deforestation_detection/fsdl_deforestation_detection/data/"
)
sys.path.append(
    "fsdl_deforestation_detection/fsdl_deforestation_detection/modeling/"
)
import data_utils
import model_utils
from models import ResNet, func_resnet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained",
    action="store_true",
    help="use a ResNet pretrained on BigEarthNet",
)
parser.add_argument(
    "--learning_rate",
    default=0.03,
    help="initial learning rate",
)
parser.add_argument(
    "--epochs",
    default=[50, 50],
    nargs="+",
    help="number of epochs to train on",
)
parser.add_argument(
    "--batch_size",
    default=32,
    help="number of samples per batch",
)
parser.add_argument(
    "--random_seed",
    default=42,
    help="random seed for reproducible results",
)
parser.add_argument(
    "--task",
    default=["orig_labels", "deforestation"],
    nargs="+",
    help="random seed for reproducible results",
)
args = parser.parse_args()

# Handle arguments
if not isinstance(args.task, list):
    args.task = list(args.task)
if not isinstance(args.epochs, list):
    args.epochs = list(args.epochs)
args.epochs = [int(epochs) for epochs in args.epochs]
if args.pretrained:
    assert (
        len(args.epochs) == len(args.task) * 2,
        "When training a pretrained model on a pipeline of tasks, "
        "the number of epochs should be specified twice for each task (for each task, "
        "initially we train just the final layer, then we fine-tune all the weights). "
        "As such, we expect len(args.epochs) == len(args.task) * 2. "
        f"Currently specified {len(args.epochs)} epoch sets and {len(args.task)} tasks.",
    )
else:
    assert (
        len(args.epochs) == (len(args.task) * 2) - 1,
        "When training a model from scratch on a pipeline of tasks, "
        "the number of epochs should be specified once for an initial task and twice "
        "for the remaining ones (for each task except the first, initially we train just "
        "the final layer, then we fine-tune all the weights). "
        "As such, we expect len(args.epochs) == len(args.task) * 2 - 1. "
        f"Currently specified {len(args.epochs)} epoch sets and {len(args.task)} tasks.",
    )
# Set the random seed for reproducibility
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)
opt = optimizers.Adam(learning_rate=args.learning_rate)
loss = "binary_crossentropy"
count = 0
for task in tqdm(args.task, desc="Tasks"):
    # Load the data
    labels_df = pd.read_csv(data_utils.DATA_PATH + data_utils.LABELS_PATH)
    labels_df = data_utils.encode_tags(labels_df, drop_tags_col=True)
    if task == "deforestation":
        labels_df = data_utils.add_deforestation_label(labels_df)
        labels_df = labels_df[["image_name", "deforestation"]]
    # Specify the dataframe so that the generator has no required arguments
    def data_gen():
        for i in data_utils.get_amazon_sample(labels_df):
            yield i

    # Create the dataset
    if task == "deforestation":
        labels_shape = 1
    else:
        labels_shape = len(data_utils.TAGS)
    dataset = tf.data.Dataset.from_generator(
        data_gen,
        output_signature=(
            tf.TensorSpec(shape=([256, 256, 3]), dtype=tf.float16),
            tf.TensorSpec(shape=(labels_shape), dtype=tf.uint8),
        ),
    )
    # Split into training, validation and test sets
    n_samples = len(labels_df)
    train_set = dataset.take(int(0.9 * n_samples))
    test_set = dataset.skip(int(0.9 * n_samples))
    train_set = train_set.skip(int(0.1 * n_samples))
    val_set = train_set.take(int(0.1 * n_samples))
    train_set = (
        train_set
        #  .cache()
        .shuffle(buffer_size=1000)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    val_set = (
        val_set
        # .cache()
        .shuffle(buffer_size=1000)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    test_set = (
        test_set
        # .cache()
        .shuffle(buffer_size=1000)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    if args.pretrained:
        pretrain_dataset = "bigearthnet"
    else:
        pretrain_dataset = None
    # Load the model
    # model = ResNet(pretrain_dataset=pretrain_dataset, pooling="max", task=task)
    model = func_resnet(
        pretrain_dataset=pretrain_dataset, pooling="max", task=task
    )
    if task == "orig_labels":
        n_outputs = 17
    elif task == "deforestation":
        n_outputs = 1
    else:
        raise Exception(
            f'ERROR: Unrecognized task "{task}". Please select one of "orig_labels" or "deforestation".'
        )
    model_metrics = [
        "accuracy",
        FBetaScore(num_classes=n_outputs, average="macro", beta=2.0),
    ]
    wandb.init(
        project="fsdl_deforestation_detection",
        entity="fsdl-andre-karthik",
        tags="mvp",
        reinit=True,
        config={**vars(args), **dict(current_task=task)},
    )
    # The epoch on which to start the full model training
    initial_epoch = 0
    if args.pretrained or count > 0:
        # Train initially the final layer
        model.layers[-2].trainable = False
        if count > 0:
            outputs = layers.Dense(n_outputs, activation="sigmoid")(
                model.layers[-2].output
            )
            model = Model(input=model.input, output=[outputs])
        model.compile(optimizer=opt, loss=loss, metrics=model_metrics)
        model.fit(
            train_set,
            validation_data=val_set,
            epochs=args.epochs[count],
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_loss", min_delta=1e-4, patience=9
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=5,
                    factor=0.1,
                    min_lr=1e-7,
                ),
                TqdmCallback(),
                WandbCallback(data_type="image", predictions=5),
                model_utils.ClearMemory(),
            ],
        )
        initial_epoch = args.epochs[count]
        count += 1
    # Train all the model's weights
    model.layers[-2].trainable = True
    model.compile(optimizer=opt, loss=loss, metrics=model_metrics)
    model.fit(
        train_set,
        validation_data=val_set,
        epochs=args.epochs[count],
        verbose=0,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=9
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                min_delta=1e-4,
                patience=5,
                factor=0.1,
                min_lr=1e-7,
            ),
            TqdmCallback(),
            WandbCallback(data_type="image", predictions=5),
            model_utils.ClearMemory(),
        ],
        initial_epoch=initial_epoch,
    )
    # Test the model
    test_scores = model.evaluate(test_set)
    wandb.log(
        dict(
            test_loss=test_scores[0],
            test_accuracy=test_scores[1],
            test_f2=test_scores[2],
        )
    )
    count += 1