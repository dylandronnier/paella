import jax
import jax.numpy as jnp
from flax import nnx


class _ResidualBlock(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs) -> None:
        """Create a residual block with two convolutions and normalization.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            rngs (flax.nnx.Rngs): A set of.

        """
        # Convolutional layers with layer normalization.
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )
        self.norm1 = nnx.LayerNorm(out_channels, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(out_channels, rngs=rngs)

        # Projection shortcut if dimensions change.
        self.shortcut = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            rngs=rngs,
        )

    # The forward pass through the residual block.
    def __call__(self, x: jax.Array) -> jax.Array:
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.gelu(x)

        return x + identity


def _pos_encoding(t: jax.Array, dim: int) -> jax.Array:
    """Apply sinusoidal positional encoding for time embedding.

    Args:
        t (jax.Array): The time embedding, representing the timestep.
        dim (int): The dimension of the output positional encoding.

    Returns:
        jax.Array: The sinusoidal positional embedding per timestep.

    """
    # Calculate half the embedding dimension.
    half_dim = dim // 2
    # Compute the logarithmic scaling factor for sinusoidal frequencies.
    emb = jnp.log(10000.0) / (half_dim - 1)
    # Generate a range of sinusoidal frequencies.
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    # Create the positional encoding by multiplying time embeddings with.
    emb = t[:, None] * emb[None, :]
    # Concatenate sine and cosine components for richer representation.
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb


def _upsample(x: jax.Array, target_size=int) -> jax.Array:
    return jax.image.resize(
        x, (x.shape[0], target_size, target_size, x.shape[3]), method="nearest"
    )


def _downsample(x: jax.Array, target_size=int) -> jax.Array:
    return nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")


class UNet(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: int,
        time_emb_dim: int = 128,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the U-Net architecture with time embedding."""
        self.features = features

        # Time embedding layers for diffusion timestep conditioning.
        self.time_mlp_1 = nnx.Linear(
            in_features=time_emb_dim, out_features=time_emb_dim, rngs=rngs
        )
        self.time_mlp_2 = nnx.Linear(
            in_features=time_emb_dim, out_features=time_emb_dim, rngs=rngs
        )

        # Time projection layers for different scales.
        self.time_proj1 = nnx.Linear(
            in_features=time_emb_dim, out_features=features, rngs=rngs
        )
        self.time_proj2 = nnx.Linear(
            in_features=time_emb_dim, out_features=features * 2, rngs=rngs
        )
        self.time_proj3 = nnx.Linear(
            in_features=time_emb_dim, out_features=features * 4, rngs=rngs
        )
        self.time_proj4 = nnx.Linear(
            in_features=time_emb_dim, out_features=features * 8, rngs=rngs
        )

        # The encoder path.
        self.down_conv1 = _ResidualBlock(in_channels, features, rngs=rngs)
        self.down_conv2 = _ResidualBlock(features, features * 2, rngs=rngs)
        self.down_conv3 = _ResidualBlock(features * 2, features * 4, rngs=rngs)
        self.down_conv4 = _ResidualBlock(features * 4, features * 8, rngs=rngs)

        # Multi-head self-attention blocks.
        self.attention1 = nnx.MultiHeadAttention(
            num_heads=1, in_features=features * 4, rngs=rngs
        )
        self.attention2 = nnx.MultiHeadAttention(
            num_heads=1, in_features=features * 8, rngs=rngs
        )

        # The bridge connecting the encoder and the decoder.
        self.bridge_down = _ResidualBlock(features * 8, features * 16, rngs=rngs)
        self.bridge_attention = nnx.MultiHeadAttention(
            num_heads=1, in_features=features * 16, rngs=rngs
        )
        self.bridge_up = _ResidualBlock(features * 16, features * 16, rngs=rngs)

        # Decoder path with skip connections.
        self.up_conv4 = _ResidualBlock(features * 24, features * 8, rngs=rngs)
        self.up_conv3 = _ResidualBlock(features * 12, features * 4, rngs=rngs)
        self.up_conv2 = _ResidualBlock(features * 6, features * 2, rngs=rngs)
        self.up_conv1 = _ResidualBlock(features * 3, features, rngs=rngs)

        # Output layers.
        self.final_norm = nnx.LayerNorm(features, rngs=rngs)
        self.final_conv = nnx.Conv(
            in_features=features,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Perform the forward pass through the U-Net using time embeddings."""
        # Time embedding and projection.
        t_emb = _pos_encoding(t, 128)  # Sinusoidal positional encoding for time.
        t_emb = self.time_mlp_1(t_emb)  # Project and activate the time embedding
        t_emb = nnx.gelu(t_emb)  # Activation function: `flax.nnx.gelu` (GeLU).
        t_emb = self.time_mlp_2(t_emb)

        # Project time embeddings for each scale.
        # Project to the correct dimensions for each encoder block.
        t_emb1 = self.time_proj1(t_emb)[:, None, None, :]
        t_emb2 = self.time_proj2(t_emb)[:, None, None, :]
        t_emb3 = self.time_proj3(t_emb)[:, None, None, :]
        t_emb4 = self.time_proj4(t_emb)[:, None, None, :]

        # The encoder path with time injection.
        d1 = self.down_conv1(x)
        t_emb1 = jnp.broadcast_to(
            t_emb1, d1.shape
        )  # Broadcast the time embedding to match feature map shape.
        d1 = d1 + t_emb1  # Add the time embedding to the feature map.

        d2 = self.down_conv2(_downsample(d1))
        t_emb2 = jnp.broadcast_to(t_emb2, d2.shape)
        d2 = d2 + t_emb2

        d3 = self.down_conv3(_downsample(d2))
        d3 = self.attention1(d3)  # Apply self-attention.
        t_emb3 = jnp.broadcast_to(t_emb3, d3.shape)
        d3 = d3 + t_emb3

        d4 = self.down_conv4(_downsample(d3))

        d4 = self.attention2(d4)
        t_emb4 = jnp.broadcast_to(t_emb4, d4.shape)
        d4 = d4 + t_emb4

        # The bridge.
        b = _downsample(d4)
        b = self.bridge_down(b)
        b = self.bridge_attention(b)
        b = self.bridge_up(b)

        # The decoder path with skip connections.
        u4 = self.up_conv4(jnp.concatenate([_upsample(b, d4.shape[1]), d4], axis=-1))
        u3 = self.up_conv3(jnp.concatenate([_upsample(u4, d3.shape[1]), d3], axis=-1))
        u2 = self.up_conv2(jnp.concatenate([_upsample(u3, d2.shape[1]), d2], axis=-1))
        u1 = self.up_conv1(jnp.concatenate([_upsample(u2, d1.shape[1]), d1], axis=-1))

        # Final layers.
        x = self.final_norm(u1)
        x = nnx.gelu(x)
        return self.final_conv(x)
