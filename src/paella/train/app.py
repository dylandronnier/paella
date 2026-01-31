"""Training app using hydra."""

# import logging
import sys
import typing as tp
from dataclasses import dataclass

import grain
import jax
import toolz
import tqdm
from absl import logging
from etils.epath import Path
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from hydra_zen import builds, instantiate, store, zen
from orbax.checkpoint import v1 as ocp

from paella.train.data import DataConf
from paella.train.lib import (
    ImagesProperties,
    init_state,
    single_eval_step,
    single_train_step,
)

training = ocp.training

EarlyStoppingConf = builds(EarlyStopping)

logging.set_verbosity(logging.ERROR)


@jax.tree_util.register_dataclass
@dataclass
class TrainingState:
    model: nnx.State
    optimizer: nnx.State


@store(name="train", dataset=DataConf)
def app(
    dataset: tuple[grain.IterDataset, grain.IterDataset],
    ckpt_directory: Path,
    earlystopconf: tp.Optional[EarlyStoppingConf] = None,
    epochs: int = 20,
    seed: int = 42,
) -> None:
    """Train a model on a dataset.

    Args:
        conf (Config): Configuration of the experiment.

    """
    iter_dataset_train, iter_dataset_test = dataset

    # Compute mean & std of the dataset per channel
    # mean, std = compute_mean_std(iter_dataset_train, axis=(0, 1, 2))

    rngs = nnx.Rngs(seed)
    # Initialize the model

    img_props = ImagesProperties(width=28, height=28, channels=1, number_of_classes=10)

    (model, optimizer) = init_state(img_properties=img_props, rngs=rngs)
    # Save model architecture in AIM
    # run["model"] = {"nb_parameters": number_of_parameters(mod)}

    # Train and evaluate
    # Log configuration parameters
    # run["hparams"] = conf.gradient_descent

    # Init the training state
    if earlystopconf is not None:
        early_stop = instantiate(earlystopconf)
    else:
        early_stop = EarlyStopping(patience=sys.maxsize)

    preservation_policy = training.preservation_policies.AnyPreservationPolicy(
        [
            training.preservation_policies.LatestN(2),
            training.preservation_policies.BestN(
                get_metric_fn=lambda m: m["validation"]["accuracy"],
                reverse=False,
                n=3,
            ),
        ]
    )

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss")
    )

    @nnx.jit(donate_argnums=0)
    @nnx.scan
    def train_loop(s, dat):
        st, met = s
        return s, single_train_step(st, met, dat)

    @nnx.jit(donate_argnums=0)
    @nnx.scan
    def eval_loop(s, dat):
        mod, met = s
        return s, single_eval_step(mod, met, dat)

    train_iterator = iter(iter_dataset_train)
    eval_iterator = iter(iter_dataset_test)

    model.train()
    with training.Checkpointer(
        ckpt_directory, preservation_policy=preservation_policy
    ) as ckptr:
        for epoch in (prog := tqdm.tqdm(range(1, epochs + 1))):
            # Training loop

            metrics.reset()
            train_loop(((model, optimizer), metrics), next(train_iterator))

            train_metrics = toolz.valmap(float, metrics.compute())

            # Log training metrics
            # log_and_track_metrics(metrics, subset="Train", run=run, epoch=epoch)

            # Reset metrics for test set
            # metrics.reset()

            # Evaluation loop

            model.eval()
            metrics.reset()
            eval_loop((model, metrics), next(eval_iterator))
            model.train()

            eval_metrics = toolz.valmap(float, metrics.compute())

            prog.set_postfix_str(f"Accuracy={eval_metrics['accuracy']:.4f}")

            # Log test metrics
            # log_and_track_metrics(metrics, subset="Validation", run=run, epoch=epoch)
            saved = ckptr.save_checkpointables_async(
                step=epoch,
                checkpointables=dict(
                    model=nnx.state(model),
                    optimizer=nnx.state(optimizer),
                    data_train=train_iterator,
                    data_test=eval_iterator,
                ),
                metrics=dict(training=train_metrics, validation=eval_metrics),
            )

            if saved:  # Will be True if the save_decision_policy decided to save.
                logging.info(f"  Saved checkpoint for step {epoch}...")

            early_stop = early_stop.update(eval_metrics["loss"])

            if early_stop.should_stop:
                logging.warning(
                    "No improvments of the evaluation loss during"
                    + f" the last {early_stop.patience} epochs."
                )
                logging.warning(f"Could not reach epoch {epochs}.")
                break

            # metrics.reset()  # reset metrics for next training epoch

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
    store.add_to_hydra_store()
    zen(app).hydra_main(config_name="train", version_base="1.1")
