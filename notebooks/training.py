import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="full",
    app_title="Training",
    layout_file="layouts/training.slides.json",
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import orbax.checkpoint as ocp
    from dataclasses import asdict
    import logging


    def best_fn(m: dict[str, dict[str, float]]) -> float:
        return m["eval"]["accuracy"]


    preservation_policy = ocp.checkpoint_managers.AnyPreservationPolicy(
        policies=[
            ocp.checkpoint_managers.BestN(get_metric_fn=best_fn, n=1),
            ocp.checkpoint_managers.LatestN(3),
        ]
    )

    mngr_options = ocp.CheckpointManagerOptions(
        best_fn=best_fn,
        best_mode="max",
        preservation_policy=preservation_policy,
    )

    mngr = ocp.CheckpointManager(
        directory="/tmp/training/",
        options=mngr_options,
    )
    return logging, mngr


@app.cell
def _():
    import datasets

    hf_dataset = datasets.load_dataset("mnist")
    return (hf_dataset,)


@app.cell
def _(hf_dataset):
    from paella.train.lib import prepare_dataset

    iter_train = iter(prepare_dataset(hf_dataset["train"], batch_size=32))
    iter_eval = iter(prepare_dataset(hf_dataset["test"], batch_size=32))
    return iter_eval, iter_train


@app.cell
def _():
    from flax import nnx
    from paella.models import CNNArchitecture, SimpleCNN
    from paella.train.lib import (
        ImageDataSetProperties,
        ImageClassificationModel,
        image_classifier,
    )

    cnn_architecture = CNNArchitecture(cnn_fliters=[32, 64], layers_sizes=[1, 1])


    def generate_model(rngs: nnx.Rngs) -> ImageClassificationModel:
        model_backbone = SimpleCNN(
            architecture=cnn_architecture,
            channels=1,
            rngs=rngs,
        )
        model_backbone.eval()

        return image_classifier(
            backbone=model_backbone,
            props=ImageDataSetProperties(
                channels=1, number_of_classes=10, length=28, width=28
            ),
            rngs=rngs,
        )
    return generate_model, nnx


@app.cell
def _(generate_model, iter_train, logging, mngr, nnx):
    from paella.train.lib import ckpt_restore_args
    import optax

    if mngr.latest_step():
        logging.info("Retrieving checkpoints")
        abstract_model = nnx.eval_shape(lambda: generate_model(nnx.Rngs(0)))
        model_graph, abstract_model_state = nnx.split(abstract_model)
        optimizer_graph, abstract_optimizer_state = nnx.split(
            nnx.eval_shape(
                lambda: nnx.Optimizer(
                    model=abstract_model,
                    tx=optax.sgd(learning_rate=1e-3),
                    wrt=nnx.Param,
                )
            )
        )
        restored_data = mngr.restore(
            mngr.latest_step(),
            args=ckpt_restore_args(
                model_state=abstract_model_state,
                data_iterator=iter_train,
                optimizer_state=abstract_optimizer_state,
            ),
        )
        model = nnx.merge(model_graph, restored_data.model)
        optimizer = nnx.merge(optimizer_graph, restored_data.optimizer)
        add_to_steps = mngr.latest_step()
        ds_iter = restored_data.training_set_iterator
    else:
        logging.info("no checkpoint, generating new model")
        model = generate_model(nnx.Rngs(42))
        optimizer = nnx.Optimizer(
            model=model, tx=optax.sgd(learning_rate=1e-3), wrt=nnx.Param
        )
        add_to_steps = 0
        ds_iter = iter_train
    return add_to_steps, ds_iter, model, optimizer


@app.cell
def _(iter_eval, model, nnx):
    print(nnx.tabulate(model, next(iter_eval).image, depth=1))
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now we can finally train our model.
    """)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=1, stop=10, step=1)
    slider
    return (slider,)


@app.cell
def _(
    add_to_steps,
    ds_iter,
    iter_eval,
    iter_train,
    logging,
    mngr,
    model,
    nnx,
    optimizer,
    slider,
):
    from paella.train.lib import ckpt_save_args, train_loop, eval_loop
    import jax

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )
    metrics_dict = dict()

    for j in range(1, slider.value + 1):
        logging.info(f"Starting step {j + add_to_steps}")
        train_loop(model, ds_iter, optimizer, metrics=metrics, steps=800)
        metrics_dict["train"] = metrics.compute()
        eval_loop(model, iter_eval, metrics=metrics, steps=200)
        metrics_dict["eval"] = metrics.compute()
        model_state = nnx.state(model)
        optimizer_state = nnx.state(optimizer)
        mngr.save(
            step=j + add_to_steps,
            metrics=jax.tree.map(float, metrics_dict),
            args=ckpt_save_args(
                model_state=model_state,
                optimizer_state=optimizer_state,
                data_iterator=iter_train,
            ),
        )

    mngr.wait_until_finished()
    mngr.close()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
