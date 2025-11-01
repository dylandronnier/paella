from plotly.express import imshow
from plotly.graph_objects import Figure

from paella.train.lib import Batch, ImageClassificationModel, pred_step


def pretty_print(batch: Batch, model: ImageClassificationModel) -> Figure:
    """Display predictions using Plotly."""
    pred_label = pred_step(model, batch.image)
    fig = imshow(batch.image[..., 0], facet_col=0, facet_col_wrap=4)
    for i, (t, p) in enumerate(zip(batch.label, pred_label)):
        fig.layout.annotations[i]["text"] = f"pred = {p}, true = {t}"
    return fig
