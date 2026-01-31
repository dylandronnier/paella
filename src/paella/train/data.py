import logging
import typing as tp
from dataclasses import dataclass

import grain
import jax
import jax.numpy as jnp
from datasets import Dataset, DatasetDict, load_dataset
from hydra_zen import builds


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
    *,
    batch_size: int = 32,
    step_size: tp.Optional[int] = None,
    sharding: tp.Optional[jax.sharding.Sharding] = None,
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

    if step_size:
        iter_dataset = iter_dataset.batch(step_size)

    if sharding:
        iter_dataset = grain.experimental.device_put(iter_dataset, device=sharding)
    return iter_dataset


def dataset_from_hf(
    hf_id: str,
    *,
    batch_size: int,
    step_size: tp.Optional[int] = None,
    image_column_name: str = "image",
) -> tuple[grain.IterDataset, grain.IterDataset]:
    # Load dataset
    hf_dataset = load_dataset(path=hf_id)

    # Ensure the datasets is a Dataset Dictionary
    if not (isinstance(hf_dataset, DatasetDict)):
        logging.error(msg="Dataset is not splitted in test and train.")
        raise Exception

    # Rename the image column with the proper name
    if image_column_name != "image":
        hf_dataset = hf_dataset.rename_column(image_column_name, "image")

    # Load the data in the GPU
    iter_dataset_train = prepare_dataset(
        hf_dataset["train"], batch_size=batch_size, step_size=step_size
    )
    iter_dataset_test = prepare_dataset(
        hf_dataset["test"], batch_size=batch_size, step_size=step_size
    )

    logging.info(msg=f"The dataset {hf_id} has been preprocessed. Ready to learn.")

    return iter_dataset_train, iter_dataset_test


DataConf = builds(dataset_from_hf, populate_full_signature=True)
