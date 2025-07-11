import jax
from typing import Callable, Any, Tuple, Iterable,Optional
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling, lecun_uniform, he_uniform, constant, zeros_init, orthogonal
import math
from jax._src import core
from jax._src import dtypes
from jax._src.typing import Array, ArrayLike
from flax.linen.initializers import Initializer

KeyArray = Array
DTypeLikeInexact = Any  


def _lecun_uniform(fan_in: float, dtype: DTypeLikeInexact = jnp.float_):
    scale = jnp.sqrt(1.0 / fan_in)
    def init(key: KeyArray, shape: core.Shape, dtype: DTypeLikeInexact = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        scale_array = jnp.array(scale, dtype)
        uniform = 2.0 * scale_array * jax.random.uniform(key, shape, dtype)  # [0, 2 * scale)
        return uniform - scale_array  # [-scale, scale)
    return init


def sparse_init(sparsity=0.9, dtype=jnp.float32) -> Initializer:
    assert 0.0 <= sparsity <= 1.0

    def init(key: KeyArray, shape: core.Shape, dtype: DTypeLikeInexact = dtype) -> Array:
        if len(shape) == 2:
            fan_in, fan_out = shape
            is_conv = False
        elif len(shape) == 4:  # Probably a conv layer
            height, width, in_channels, out_channels = shape
            fan_in = height * width * in_channels
            is_conv = True
        else:
            raise ValueError("Unknown weight shape")
        num_zeros = int(math.ceil(sparsity * fan_in))

        initializer = _lecun_uniform(fan_in)

        # Initialize weights
        key, init_key = jax.random.split(key)
        weights = initializer(init_key, shape, dtype)

        # Generate masks to set random weights to zero
        def generate_mask(key):
            perm = jax.random.permutation(key, fan_in)
            zero_indices = perm[:num_zeros]
            mask = jnp.ones(fan_in, dtype=dtype)
            return mask.at[zero_indices].set(0)

        out_size = out_channels if is_conv else fan_out
        mask_keys = jax.random.split(key, out_size)
        masks = jax.vmap(generate_mask)(mask_keys).transpose()  # shape is now (fan_in, out_size)

        # Apply masks to weights
        return weights * masks.reshape(shape)

    return init


def simple_sparse_init(
        sparsity=0.9,
        initializer=jax.nn.initializers.lecun_uniform(),
        dtype=jnp.float32,
    ) -> Initializer:
    assert 0.0 <= sparsity <= 1.0

    def init(key: KeyArray, shape: core.Shape, dtype: DTypeLikeInexact = dtype) -> Array:
        # Initialize weights
        key, init_key = jax.random.split(key)
        weights = initializer(init_key, shape, dtype)

        # Generate mask to set random weights to zero
        size = weights.size
        num_zeros = round(sparsity * size)
        mask = jnp.ones_like(weights).reshape(-1)
        zero_indices = jax.random.permutation(key, size)[:num_zeros]
        mask = mask.at[zero_indices].set(0.0)

        # Apply mask to weights
        return weights * mask.reshape(shape)

    return init
