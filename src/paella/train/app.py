"""Training app using hydra."""

import logging
import sys

import grain
import orbax.checkpoint as ocp
from datasets import DatasetDict, load_dataset
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from hydra_zen import instantiate

from paella.train.lib import (
    eval_loop,
    image_classifier,
    prepare_dataset,
    train_loop,
)
from paella.utils import compute_mean_std
from paella.utils.config import GlobalConfig


def app(conf: GlobalConfig) -> None:
    """Train a model on a dataset.

    Args:
        conf (Config): Configuration of the experiment.

    """
    # Load dataset
    hf_dataset = load_dataset(path=conf.dataset.hf_id)

    # Ensure the datasets is a Dataset Dictionary
    if not (isinstance(hf_dataset, DatasetDict)):
        logging.error(msg="Dataset is not splitted in test and train.")
        return

    # Rename the image column with the proper name
    if conf.dataset.image_column_name != "image":
        hf_dataset = hf_dataset.rename_column(conf.dataset.image_column_name, "image")

    # Load the data in the GPU
    iter_dataset_train = prepare_dataset(
        hf_dataset["train"], batch_size=conf.gradient_descent.batch_size
    )

    iter_dataset_test = prepare_dataset(
        hf_dataset["test"], batch_size=conf.gradient_descent.batch_size
    )

    # Compute mean & std of the dataset per channel
    mean, std = compute_mean_std(iter_dataset_train, axis=(0, 1, 2))

    # Transformation of data
    # transform = augmax.Chain(
    #     augmax.Normalize(mean=mean, std=std),
    #     augmax.ByteToFloat(),
    # )

    # color_range = transform(
    #     inputs=jnp.array([0.0, 255.0]).reshape((1, -1, 1)), rng=jrand.PRNGKey(0)
    # )
    # print(color_range)
    # jit_transform = jit(partial(transform, rng=jrand.PRNGKey(conf.seed)))
    # hf_dataset = hf_dataset.map(
    #     lambda ex: {"image": jit_transform(inputs=jnp.expand_dims(ex["image"], -1))},
    #     batched=True,
    #     batch_size=16,
    # )

    # Recognize the gpu
    # hf_dataset = hf_dataset.cast_column(
    #     "image",
    #     feature=Array3D(
    #         shape=(
    #             conf.dataset.images_width,
    #             conf.dataset.images_height,
    #             conf.dataset.channels,
    #         ),
    #         dtype="float32",
    #     ),
    # )

    logging.info(
        msg=f"The dataset {conf.dataset.hf_id} has been preprocessed. "
        + "It is ready for the learning task."
    )

    rngs = nnx.Rngs(conf.seed)
    # Initialize the model
    mod = image_classifier()
    # Save model architecture in AIM
    # run["model"] = {"nb_parameters": number_of_parameters(mod)}

    # Train and evaluate
    # Log configuration parameters
    # run["hparams"] = conf.gradient_descent

    # Init the training state
    if conf.gradient_descent.early_stop is not None:
        early_stop = instantiate(conf.gradient_descent.early_stop)
    else:
        early_stop = EarlyStopping(patience=sys.maxsize)

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    mngr_options = ocp.CheckpointManagerOptions(
        best_fn=lambda m: m["eval"]["accuracy"],
        best_mode="max",
        preservation_policy=ocp.checkpoint_managers.LatestN(2),
    )
    regisry = ocp.DefaultCheckpointHandlerRegistry()
    regisry.add(item="models", args=ocp.args.StandardSave)
    regisry.add(item="models", args=ocp.args.StandardRestore)
    regisry.add(item="data", args=grain.checkpoint.CheckpointSave)
    regisry.add(item="data", args=grain.checkpoint.CheckpointRestore)
    # regisry.add(item="key", args=ocp.args.JaxRandomKeySave)
    # regisry.add(item="key", args=ocp.args.JaxRandomKeyRestore)

    mngr = ocp.CheckpointManager(
        directory="data/training", options=mngr_options, handler_registry=regisry
    )

    optimizer = nnx.Optimizer(
        mod, instantiate(conf.gradient_descent.optimizer), wrt=nnx.Param
    )

    for epoch in range(1, conf.gradient_descent.epochs + 1):
        # Training loop
        train_loop(mod, iter_dataset_train, optimizer, metrics)

        train_metrics = metrics.compute()

        # Log training metrics
        # log_and_track_metrics(metrics, subset="Train", run=run, epoch=epoch)

        # Reset metrics for test set
        metrics.reset()

        # Evaluation loop
        eval_loop(mod, iter_dataset_test, metrics)

        eval_metrics = metrics.compute()

        # Log test metrics
        # log_and_track_metrics(metrics, subset="Validation", run=run, epoch=epoch)
        mngr.save(
            step=epoch,
            metrics={"train": train_metrics, "eval": eval_metrics},
        )

        early_stop = early_stop.update(eval_metrics["loss"])

        if early_stop.should_stop:
            logging.warning(
                "No improvments of the evaluation loss during"
                + f" the last {early_stop.patience} epochs."
            )
            logging.warning(f"Could not reach epoch {conf.gradient_descent.epochs}.")
            break

        metrics.reset()  # reset metrics for next training epoch

        # if epoch % conf.tracking.frequency == 0:
        #     try:
        #         fig = pretty_print(
        #             **next(
        #                 hf_dataset["test"].iter(
        #                     batch_size=conf.tracking.sample_prediction_size
        #                 )
        #             ),
        #             model=mod,
        #         )
        #         # run.track(Figure(fig), epoch=epoch, name="Sample prediction")
        #     except:
        #         logging.warning("Fail to save the sample prediction.")

    logging.info(f"Best metric is equal to {early_stop.best_metric}")


if __name__ == "__main__":
    app()
