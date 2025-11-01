import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import datasets
    import grain
    import jax

    from paella.train.lib import prepare_dataset

    hf_dataset = datasets.load_dataset("mnist")
    iter_train = iter(prepare_dataset(hf_dataset["train"], batch_size=32))
    iter_eval = iter(prepare_dataset(hf_dataset["test"], batch_size=32))
    return (iter_train,)


@app.cell
def _(mo):
    mo.md(r"""
    In this app, we visualize our dataset.
    """)
    return


@app.cell
def _(iter_train):
    from plotly.express import imshow

    batch = next(iter_train)
    fig = imshow(batch.image[:, :, :, 0], facet_col=0, facet_col_wrap=4)
    for _i, _lab in enumerate(batch.label):
        fig.layout.annotations[_i]["text"] = f"true = {_lab}"
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)

    fig
    return


if __name__ == "__main__":
    app.run()
