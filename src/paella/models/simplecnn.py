from dataclasses import dataclass
from typing import Sequence

import jax
from flax import nnx

from paella.models._basic_cnn_block import BasicBlock


@dataclass
class CNNArchitecture:
    cnn_fliters: Sequence[int]
    layers_sizes: Sequence[int]


class _SimpleCNNBlock(nnx.Module):
    def __init__(
        self, in_features: int, out_features: int, nb_conv_layers: int, rngs: nnx.Rngs
    ) -> None:
        assert nb_conv_layers > 0
        self.cnn_layers = nnx.List()
        self.cnn_layers.append(
            BasicBlock(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(3, 3),
                rngs=rngs,
            )
        )
        for _ in range(nb_conv_layers):
            self.cnn_layers.append(
                BasicBlock(
                    in_features=out_features,
                    out_features=out_features,
                    kernel_size=(3, 3),
                    rngs=rngs,
                )
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        x = nnx.max_pool(inputs=x, window_shape=(2, 2), strides=(2, 2))
        return x


class SimpleCNN(nnx.Module):
    """A simple CNN model."""

    def __init__(
        self,
        architecture: CNNArchitecture,
        *,
        channels: int,
        rngs: nnx.Rngs,
    ):
        i = channels
        self._cnn = nnx.Sequential(
            *(
                _SimpleCNNBlock(
                    in_features=i, out_features=(i := f), nb_conv_layers=nb, rngs=rngs
                )
                for f, nb in zip(architecture.cnn_fliters, architecture.layers_sizes)
            )
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self._cnn(x)

        return x.reshape(x.shape[0], -1)  # flatten
