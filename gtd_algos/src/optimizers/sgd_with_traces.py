from functools import partial

import jax
from jax import tree_map
import jax.numpy as jnp
import optax


def sgd_with_traces(lr, gamma, lambd):
    assert lr > 0.0
    assert 0.0 <= gamma <= 1.0
    assert 0.0 <= lambd <= 1.0

    def init(params):
        return tree_map(jnp.zeros_like, params)

    @partial(jax.jit, static_argnames=['reset'])
    def update(grads, opt_state, params, td_error, reset):
        # Spike and decay traces
        z = opt_state
        z = tree_map(lambda x: gamma * lambd * x, z)
        z = tree_map(jnp.add, z, grads)

        # Compute update to parameters
        updates = tree_map(lambda x: lr * td_error * x, z)

        # Reset traces at end of episode or other condition
        if reset:
            z = init(params)

        opt_state = z
        return updates, opt_state

    return optax.GradientTransformationExtraArgs(init, update)
