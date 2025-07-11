import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce


def l2_norm(x):
    return jnp.sqrt(tree_reduce(
        jnp.add,
        tree_map(lambda x: jnp.sum(x**2), x)),
    )


def l1_norm(x):
    return tree_reduce(
        jnp.add,
        tree_map(lambda x: jnp.sum(jnp.abs(x)), x),
    )


def l1_normalize(x):
    norm = l1_norm(x)
    return tree_map(lambda x: x / norm, x)


def l2_normalize(x):
    norm = l2_norm(x)
    return tree_map(lambda x: x / norm, x)
