from flax import nnx
from jax.tree_util import tree_leaves
from numpy import prod


def number_of_parameters(mod: nnx.Module) -> int:
    """Compute the number of parameters in the model."""
    params = nnx.state(mod, nnx.Param)
    return sum(prod(p.shape) for p in tree_leaves(params))
