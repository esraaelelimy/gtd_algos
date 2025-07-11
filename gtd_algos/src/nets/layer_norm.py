import flax.linen as nn


def layer_norm(x):
    all_axes_except_first = tuple(range(1, len(x.shape)))
    return nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5, reduction_axes=all_axes_except_first, use_fast_variance=False)(x)
