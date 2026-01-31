"""Main loops for training and evaluation."""

import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from optax import adamw
from optax.losses import softmax_cross_entropy_with_integer_labels

from paella.train.data import Batch

type ImageClassificationModel = nnx.Sequential
type State = tuple[ImageClassificationModel, nnx.Optimizer]


@dataclass
class ImagesProperties:
    """Properties of the image dataset."""

    width: int
    height: int
    channels: int
    number_of_classes: int


def _image_classifier(
    backbone: tp.Callable, props: ImagesProperties, rngs: nnx.Rngs
) -> ImageClassificationModel:
    x = jax.ShapeDtypeStruct(
        (1, props.width, props.height, props.channels), jnp.float32
    )
    out_features = jax.eval_shape(backbone, x).shape[1]
    _head = nnx.Linear(
        in_features=out_features, out_features=props.number_of_classes, rngs=rngs
    )
    return nnx.Sequential(backbone, _head)


class CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="VALID", rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(32, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(4, 4), padding="VALID", rngs=rngs)
        self.batch_norm2 = nnx.BatchNorm(64, rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(64 * 5 * 5, 128, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.025, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x)))))
        x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))

        x = x.reshape(x.shape[0], -1)  # flatten

        x = nnx.relu(self.dropout2(self.linear1(x)))
        return x


def init_state(img_properties: ImagesProperties, *, rngs: nnx.Rngs) -> State:
    _backbone = CNN(rngs=rngs)

    _backbone.eval()

    model = _image_classifier(backbone=_backbone, props=img_properties, rngs=rngs)

    # optimizer = nnx.Optimizer(
    #     model, instantiate(conf.gradient_descent.optimizer), wrt=nnx.Param
    # )
    learning_rate = 0.005
    momentum = 0.9

    optimizer = nnx.Optimizer(model, adamw(learning_rate, momentum), wrt=nnx.Param)

    return (model, optimizer)


# StateConf = builds(init_state)


def cross_entropy_loss(fun: tp.Callable, batch: Batch) -> tuple[jax.Array, jax.Array]:
    r"""Compute the cross entropy error of a model on a batch.

    Args:
        fun: Callable function that compute the logits for `n` classes.
        batch: Batch of data.

    Returns:
       tuple of jax.Array

    """
    # Compute the logits for the batch of images
    logits = fun(batch.image)

    # Compte the cross-entropy loss
    loss = softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch.label
    ).mean()

    # Return both the loss and the logits.
    return loss, logits


def single_train_step(state: State, metrics: nnx.MultiMetric, batch: Batch) -> None:
    """Train for a single step.

    Args:
        model: ImageClassificationModel The model to train.
        batch:: Batch
        optimizer (nnx.Optimizer): The optimizer of the model.
        metrics (nnx.MultiMetric): The metrics of the model.

    """
    model, optimizer = state
    grad_fn = nnx.value_and_grad(cross_entropy_loss, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch.label)
    optimizer.update(model, grads)


def single_eval_step(
    model: ImageClassificationModel,
    metrics: nnx.MultiMetric,
    batch: Batch,
) -> None:
    """Evaluate the model on the batch."""
    loss, logits = cross_entropy_loss(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch.label)


@nnx.jit
def pred_step(model: ImageClassificationModel, images: jax.Array) -> jax.Array:
    """Make predictions on images."""
    logits = model(images)
    return logits.argmax(axis=1)
