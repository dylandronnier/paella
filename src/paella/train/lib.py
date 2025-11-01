"""Main loops for training and evaluation."""

import typing as tp
from dataclasses import dataclass

import grain
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from datasets import Dataset
from flax import nnx
from flax.typing import Sharding
from optax.losses import softmax_cross_entropy_with_integer_labels


def ckpt_save_args(
    model_state: tp.Optional[nnx.GraphState] = None,
    optimizer_state: tp.Optional[nnx.GraphState] = None,
    data_iterator: tp.Optional[tp.Any] = None,
    key_state: tp.Optional[jax.Array] = None,
) -> ocp.args.Composite:
    """Opiniated checkpoint saver."""
    args_dict: dict[str, ocp.args.CheckpointArgs] = dict()
    if model_state:
        args_dict["model"] = ocp.args.StandardSave(item=model_state)

    if optimizer_state:
        args_dict["optimizer"] = ocp.args.StandardSave(item=optimizer_state)

    if data_iterator:
        args_dict["training_set_iterator"] = grain.checkpoint.CheckpointSave(
            item=data_iterator
        )

    if key_state:
        args_dict["key"] = ocp.args.JaxRandomKeySave(item=key_state)

    return ocp.args.Composite(**args_dict)


def ckpt_restore_args(
    model_state: tp.Optional[nnx.GraphState] = None,
    optimizer_state: tp.Optional[nnx.GraphState] = None,
    data_iterator: tp.Optional[tp.Any] = None,
    key_state: tp.Optional[jax.Array] = None,
) -> ocp.args.Composite:
    """Opiniated checkpoint manager."""
    args_dict: dict[str, ocp.args.CheckpointArgs] = dict()
    if model_state:
        args_dict["model"] = ocp.args.StandardRestore(item=model_state)

    if optimizer_state:
        args_dict["optimizer"] = ocp.args.StandardRestore(item=optimizer_state)

    if data_iterator:
        args_dict["training_set_iterator"] = grain.checkpoint.CheckpointRestore(
            item=data_iterator
        )

    if key_state:
        args_dict["key"] = ocp.args.JaxRandomKeyRestore()

    return ocp.args.Composite(**args_dict)


type ImageClassificationModel = nnx.Sequential


@dataclass
class ImageDataSetProperties:
    """Properties of the image dataset."""

    width: int
    length: int
    channels: int
    number_of_classes: int


def image_classifier(
    backbone: tp.Callable, props: ImageDataSetProperties, rngs: nnx.Rngs
) -> ImageClassificationModel:
    x = jax.ShapeDtypeStruct(
        (1, props.width, props.length, props.channels), jnp.float32
    )
    out_features = jax.eval_shape(backbone, x).shape[1]
    _head = nnx.Linear(
        in_features=out_features, out_features=props.number_of_classes, rngs=rngs
    )
    return nnx.Sequential(backbone, _head)


@jax.tree_util.register_dataclass
@dataclass
class Batch:
    """Batch of image and labels."""

    label: jax.Array
    """Label array."""

    image: jax.Array
    """Image array."""


def process_image(sample: dict[str, tp.Any]) -> Batch:
    sample["image"] = (
        jnp.expand_dims(jnp.array(sample["image"]), -1).astype(jnp.float32) / 255.0
    )
    return Batch(**sample)


def prepare_dataset(
    dataset: Dataset,
    batch_size: int = 32,
    sharding: tp.Optional[Sharding] = None,
) -> grain.IterDataset:
    """Transform an Huggingace dataset into a Grain IterDataset.

    Args:
        dataset: HuggingFace dataset
        batch_size: int
        sharding: Optional Sharding argument

    Returns:
        IterDataset

    """
    train_dataset = (
        grain.MapDataset.source(dataset).shuffle(42).map(process_image).repeat(None)
    )
    iter_dataset = train_dataset.to_iter_dataset().batch(batch_size)
    if sharding:
        iter_dataset = grain.experimental.device_put(iter_dataset, device=sharding)
    return iter_dataset


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


@nnx.jit
def train_step(
    model: ImageClassificationModel,
    batch: Batch,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
) -> None:
    """Train for a single step.

    Args:
        model: ImageClassificationModel The model to train.
        batch:: Batch
        optimizer (nnx.Optimizer): The optimizer of the model.
        metrics (nnx.MultiMetric): The metrics of the model.

    """
    grad_fn = nnx.value_and_grad(cross_entropy_loss, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch.label)
    optimizer.update(model, grads)


@nnx.jit
def eval_step(
    model: ImageClassificationModel,
    batch: Batch,
    metrics: nnx.MultiMetric,
) -> None:
    """Evaluate the model on the batch."""
    loss, logits = cross_entropy_loss(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch.label)


@nnx.jit
def pred_step(model: ImageClassificationModel, images: jax.Array) -> jax.Array:
    """Make predictions on images."""
    logits = model(images)
    return logits.argmax(axis=1)


def train_loop(
    model: nnx.Sequential,
    train_dataset: grain.DatasetIterator,
    optimizer: nnx.Optimizer,
    *,
    metrics: nnx.MultiMetric,
    steps: int,
) -> None:
    """Train model network on a dataset.

    Args:
        model: ImageClassificationModel
            Model to train.
        train_dataset : Dataset
            Dataset on which the model will be train.
        optimizer: nnx.Optimizer
            Optimizer for minimizing the
        metrics: nnx.MultiMetric
            Saving metrics.
        batch_size: int
            Size of the batch for the training.

    """
    # Set the Module in train mode
    model.train()
    metrics.reset()
    for _ in range(steps):
        train_step(model, next(train_dataset), optimizer, metrics)


def eval_loop(
    model: ImageClassificationModel,
    dataset_eval: grain.DatasetIterator,
    *,
    metrics: nnx.MultiMetric,
    steps: int,
) -> None:
    """Evaluate the module on the dataset."""
    # Sets the Module to evaluation mode.
    model.eval()
    metrics.reset()
    for _ in range(steps):
        eval_step(model, next(dataset_eval), metrics)
