"""Construct the ConvNext Module from."""

import typing as tp
from dataclasses import dataclass

import jax
from flax import nnx
from flax.typing import Sharding
from jax.numpy import mean

from paella.models._basic_cnn_block import BasicBlock


class _ConvNeXTBlock(nnx.Module):
    """Residual Block."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strides: tuple[int, int],
        *,
        sharding: tp.Optional[Sharding] = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Construct a Residual Block.

        Args:
        ----
            in_features (int): Number of input channels.
            out_features (int): Number of output channels.
            strides (tuple[int, int]): Stride of the convolutions inside the block.
            sharding (Optional[Sharding]): Sharding of the model
            rngs (nnx.Rngs): Key for the random initialization of the paramters.

        """
        if sharding is None:
            default_initializer = nnx.initializers.lecun_normal()
            axis_name = None
        else:
            default_initializer = nnx.with_partitioning(
                nnx.initializers.lecun_normal(), sharding
            )
            axis_name = sharding[0]

        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_init=default_initializer,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

        self.ln = nnx.LayerNorm(
            epsilon=1e-5,
            num_features=out_features,
            rngs=rngs,
            axis_name=axis_name,
        )

        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_init=default_initializer,
            kernel_size=(1, 1),
            strides=strides,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_init=default_initializer,
            kernel_size=(1, 1),
            strides=strides,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

        if in_features != out_features or strides != (1, 1):
            self.proj = nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_init=default_initializer,
                kernel_size=(1, 1),
                strides=strides,
                rngs=rngs,
            )
            self.proj_norm = nnx.BatchNorm(
                epsilon=1e-5,
                momentum=0.9,
                num_features=out_features,
                rngs=rngs,
            )

    def __call__(self, x: jax.Array):
        """Run Residual Block.

        Args:
        ----
            x (tensor): Input tensor of shape [N, H, W, C].
            train (bool): Training mode.

        Returns:
        -------
            (tensor): Output shape of shape [N, H', W', features].

        """
        out = self.ln(self.conv1(x))
        out = nnx.gelu(self.conv2(x))
        out = self.conv3(out)

        # out = nnx.relu(self.bn2(self.conv2(out), use_running_average=train))

        if x.shape != out.shape:
            x = self.proj_norm(self.proj(x))

        return nnx.relu(out + x)


@dataclass
class ConvNextArchitecture:
    stage_sizes: list[int]
    num_filers: int = 64


class ConvNext(nnx.Module):
    """Residual Neural Network."""

    def __init__(
        self,
        architecture: ConvNextArchitecture,
        *,
        channels: int = 3,
        sharding: tp.Optional[Sharding] = None,
        rngs: nnx.Rngs,
    ) -> None:
        # Basic block for first layer
        self._basic = BasicBlock(
            in_features=channels,
            out_features=architecture.num_filers,
            kernel_size=(3, 3),
            sharding=sharding,
            rngs=rngs,
        )

        # Residual blocks
        self._resnetblocks = nnx.List([])

        for i, block_size in enumerate(architecture.stage_sizes):
            strides = (2, 2) if i > 0 else (1, 1)
            self._resnetblocks.append(
                _ConvNeXTBlock(
                    in_features=architecture.num_filers * 2 ** max(i - 1, 0),
                    out_features=architecture.num_filers * 2**i,
                    strides=strides,
                    sharding=sharding,
                    rngs=rngs,
                )
            )
            for _ in range(1, block_size):
                self._resnetblocks.append(
                    _ConvNeXTBlock(
                        in_features=architecture.num_filers * 2**i,
                        out_features=architecture.num_filers * 2**i,
                        strides=(1, 1),
                        sharding=sharding,
                        rngs=rngs,
                    )
                )

    def __call__(self, x, train: bool = False):
        # Apply first CNN layer followed by MaxPooling
        x = self._basic(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Apply the residual blocks
        for blk in self._resnetblocks:
            x = blk(x, train)

        # Return mean
        return mean(x, axis=(1, 2))
