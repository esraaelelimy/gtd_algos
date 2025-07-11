from functools import reduce

from jax import tree_map, vmap
import jax.numpy as jnp


def add(*args):
    return reduce(lambda a, b: tree_map(jnp.add, a, b), args)


def subtract(*args):
    return reduce(lambda a, b: tree_map(jnp.subtract, a, b), args)


def scale(scalar, arg):
    return tree_map(lambda x: scalar * x, arg)


def vmap_scale(vector, arg):
    f = vmap(scale, in_axes=[0, 0])
    return f(vector, arg)


def neg(arg):
    return scale(-1.0, arg)


def zeros(arg):
    return scale(0.0, arg)
