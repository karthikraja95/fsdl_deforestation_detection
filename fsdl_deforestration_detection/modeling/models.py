import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow_hub as hub
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers.experimental.preprocessing import Normalization

data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip(),
        layers.experimental.preprocessing.RandomContrast(0.1),
        layers.experimental.preprocessing.RandomRotation(2 * math.pi),
        layers.experimental.preprocessing.RandomCrop(224, 224),
    ]
)


class ResNet(tf.keras.Model):
    def __init__(
        self,
        pretrain_dataset: str = None,
        pooling: str = "max",
        task: str = "orig_labels",
    ):
        """A ResNet50 model that can be pretrained or trained from scratch,
        adapting to each relevant variation in this project. This is the
        model class version.

        WARNING: Although this seems more clean, it doesn't work well with
        Weights and Biases' callback. Please use func_resnet instead.

        Args:
            pretrain_dataset (str, optional): The dataset in which the model
            is pretrained. If left unspecified, the model starts with random
            weights. Available options are "imagenet" and "bigearthnet".
            Defaults to None.
            pooling (str, optional): The type of global pooling to perform
            after the ResNet layers. Available options are "max" and "avg".
            Defaults to "max".
            task (str, optional): The task on which the model will be trained
            or fine-tuned on. Available options are "orig_labels" (original
            labels from the Kaggle challenge) and "deforestation".
            Defaults to "orig_labels".

        Raises:
            Exception: [description]
        """
        super(ResNet, self).__init__()
        self.pretrain_dataset = pretrain_dataset
        self.pooling = pooling
        self.task = task
        if self.task == "orig_labels":
            self.n_outputs = 17
        elif self.task == "deforestation":
            self.n_outputs = 1
        else:
            raise Exception(
                f'ERROR: Unrecognized task "{task}". Please select one of "orig_labels" or "deforestation".'
            )
        if pretrain_dataset == "bigearthnet":
            self.core = hub.KerasLayer(
                "https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1"
            )
            # TensorFlow Hub modules require data in a [0, 1] range
            # stats estimated from subset of data in `02_eda_amazon_planet` notebook
            self.preprocess_input = Normalization(
                mean=[79.67114306, 87.08461826, 76.46177919],
                variance=[1857.54070494, 1382.94249315, 1266.69265399],
            )
        else:
            self.core = ResNet50(
                include_top=False,
                weights=pretrain_dataset,
                pooling=self.pooling,
            )
            # Using TensorFlow's ResNet-specific preprocessing
            self.preprocess_input = preprocess_input
        self.classifier = layers.Dense(self.n_outputs, activation="sigmoid")

    def call(self, inputs):
        x = self.preprocess_input(inputs)
        x = data_augmentation(x)
        x = self.core(x)
        return self.classifier(x)


def func_resnet(
    pretrain_dataset: str = None,
    pooling: str = "max",
    task: str = "orig_labels",
):
    """A ResNet50 model that can be pretrained or trained from scratch,
    adapting to each relevant variation in this project. This is the
    functional API version.

    Works well with Weights and Biases' callback.

    Args:
        pretrain_dataset (str, optional): The dataset in which the model
        is pretrained. If left unspecified, the model starts with random
        weights. Available options are "imagenet" and "bigearthnet".
        Defaults to None.
        pooling (str, optional): The type of global pooling to perform
        after the ResNet layers. Available options are "max" and "avg".
        Defaults to "max".
        task (str, optional): The task on which the model will be trained
        or fine-tuned on. Available options are "orig_labels" (original
        labels from the Kaggle challenge) and "deforestation".
        Defaults to "orig_labels".

    Raises:
        Exception: [description]
    """
    inputs = layers.Input(shape=(256, 256, 3))
    if task == "orig_labels":
        n_outputs = 17
    elif task == "deforestation":
        n_outputs = 1
    else:
        raise Exception(
            f'ERROR: Unrecognized task "{task}". Please select one of "orig_labels" or "deforestation".'
        )
    if pretrain_dataset == "bigearthnet":
        # TensorFlow Hub modules require data in a [0, 1] range
        # stats estimated from subset of data in `02_eda_amazon_planet` notebook
        x = Normalization(
            mean=[79.67114306, 87.08461826, 76.46177919],
            variance=[1857.54070494, 1382.94249315, 1266.69265399],
        )(inputs)
        x = data_augmentation(x)
        x = hub.KerasLayer(
            "https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1"
        )(x)
    else:
        # Using TensorFlow's ResNet-specific preprocessing
        x = preprocess_input(x)
        x = data_augmentation(x)
        x = ResNet50(
            include_top=False,
            weights=pretrain_dataset,
            pooling=pooling,
        )(x)
    outputs = layers.Dense(n_outputs, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
