from grain import IterDataset
from jax import Array
from jax.numpy import mean, std


def compute_mean_std(
    data: IterDataset, axis=None, sample_size: int = 1_024
) -> tuple[Array, Array]:
    batch = next(data)["image"]
    m = mean(batch, axis=axis)
    s = std(batch, axis=axis)
    return m, s
