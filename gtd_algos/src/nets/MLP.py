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
from .layer_norm import layer_norm
from .sparse_init import sparse_init

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    hiddens: Iterable[int]
    act: Callable = nn.relu
    kernel_init: Callable = orthogonal(scale = jnp.sqrt(2))
    bias_init: Callable = constant(0.0)
    pre_act_norm: bool = False
    @nn.compact
    def __call__(self, x):
      for size in self.hiddens[:-1]:
          x = nn.Dense(size,kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
          if self.pre_act_norm:
              x = layer_norm(x)
          x = self.act(x)
      return nn.Dense(self.hiddens[-1],kernel_init=self.kernel_init, bias_init=constant(0.0))(x)
