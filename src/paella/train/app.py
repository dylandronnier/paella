"""Training app using hydra."""

# import logging
import sys
import typing as tp

import grain
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
from paella.utils import number_of_parameters

training = ocp.training

EarlyStoppingConf = builds(EarlyStopping)


@store(name="train", dataset=DataConf)
def app(
    dataset: tuple[grain.IterDataset, grain.IterDataset],
    ckpt_directory: Path,
    earlystopconf: tp.Optional[EarlyStoppingConf] = None,
    number_of_steps: int = 960,
    step_size: int = 20,
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
    print(number_of_parameters(model))
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

    train_iterator = iter(iter_dataset_train.batch(step_size))
    eval_iterator = iter(iter_dataset_test.batch(step_size))

    model.train()
    with training.Checkpointer(
        ckpt_directory, preservation_policy=preservation_policy
    ) as ckptr:
        for step in (prog := tqdm.trange(1, number_of_steps + 1, step_size)):
            # Training loop

            train_loop(((model, optimizer), metrics), next(train_iterator))

            if (step - 1) % (step_size * 7) == 6 * step_size:
                train_metrics = toolz.valmap(float, metrics.compute())
                metrics.reset()
                model.eval()
                for _ in range(3):
                    eval_loop((model, metrics), next(eval_iterator))

                eval_metrics = toolz.valmap(float, metrics.compute())
                metrics.reset()

                # prog.set_postfix_str(f"Accuracy={eval_metrics['accuracy']:.4f}")
                prog.set_postfix(
                    val_accuracy=f"{eval_metrics['accuracy']:.4f}",
                    train_accuracy=f"{train_metrics['accuracy']:.4f}",
                )

                # Log test metrics
                # log_and_track_metrics(metrics, subset="Validation", run=run, epoch=epoch)
                saved = ckptr.save_checkpointables_async(
                    step=step,
                    checkpointables=dict(
                        model=nnx.state(model),
                        optimizer=nnx.state(optimizer),
                        data_train=train_iterator,
                        data_test=eval_iterator,
                    ),
                    metrics=dict(training=train_metrics, validation=eval_metrics),
                )

                model.train()

                if saved:  # Will be True if the save_decision_policy decided to save.
                    logging.info(f"Saved checkpoint for step {step}...")

                early_stop = early_stop.update(eval_metrics["loss"])

                if early_stop.should_stop:
                    logging.warning(
                        "No improvments of the evaluation loss during"
                        + f" the last {early_stop.patience} epochs."
                    )
                    logging.warning(f"Could not reach epoch {step}.")

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
