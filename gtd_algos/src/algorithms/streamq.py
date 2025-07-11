### StreamQ implemtnation in JAX
from functools import partial
from typing import NamedTuple

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.configs.Config import Config
from gtd_algos.src.optimizers import sgd_with_traces, obgd_with_traces
from gtd_algos.src.nets.MLP import sparse_init
from gtd_algos.src.agents.value_networks import DenseQNetwork, MinAtarQNetwork


class AgentState(NamedTuple):
    agent_config: Config
    train_state: TrainState


def init_agent_state(agent_config: Config, action_dim: int, obs_shape: tuple, rng: jax.random.PRNGKey):
    # Create network
    net_kwargs = {
        'action_dim': action_dim,
        'layer_norm': agent_config.layer_norm,
        'activation': agent_config.activation,
        'kernel_init': sparse_init(sparsity=agent_config.sparse_init),
    }
    net_arch = agent_config.net_arch
    if net_arch == 'minatar':
        network = MinAtarQNetwork(**net_kwargs)
        init_x = jnp.zeros(obs_shape)
    elif net_arch == 'mlp':
        network = DenseQNetwork(**net_kwargs, hiddens=agent_config.mlp_layers)
        init_x = jnp.zeros(obs_shape)
    else:
        raise ValueError(f"unknown network architecture: {net_arch}")

    # Initialize network parameters
    rng, _rng = jax.random.split(rng)
    params = network.init(_rng, init_x)

    def params_sum(params):
        return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params)))
    print(f"Total number of params: {params_sum(params)}")

    # Create optimizer
    tx_cls = {
        'obgd': obgd_with_traces,
        'sgd': sgd_with_traces,
    }[agent_config.opt]
    opt_kwrgs = {
        'lr': agent_config.lr,
        'gamma': agent_config.gamma,
        'lambd': agent_config.lamda,
    }
    if agent_config.opt == 'obgd':
        opt_kwrgs['kappa'] = agent_config.kappa

    tx = tx_cls(**opt_kwrgs)

    # Create train state
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    return AgentState(agent_config, train_state), rng


@partial(jax.jit, static_argnames=['action_dim'])
def agent_step(agent_state: AgentState, obs: jnp.ndarray, action_dim: int, epsilon: float, rng: jax.random.PRNGKey):
    params = agent_state.train_state.params
    q = agent_state.train_state.apply_fn(params, obs)
    argmax = jnp.argmax(q)
    
    rng, rng_e = jax.random.split(rng)
    
    def random_action_fn(rng_in):
        rng_out, rng_a  = jax.random.split(rng_in)
        action = jax.random.randint(rng_a, shape=(), minval=0, maxval=action_dim)
        return action, rng_out
    
    def greedy_action_fn(rng_in):
        return argmax, rng_in
    
    action, rng_out = jax.lax.cond(jax.random.uniform(rng_e, minval=0.0,maxval=1.0) < epsilon,
                                   random_action_fn,
                                   greedy_action_fn,
                                   rng)
    is_nongreedy = (action != argmax)
    return action, is_nongreedy, rng_out


@partial(jax.jit, static_argnames=['terminated', 'truncated', 'is_nongreedy'])
def update_step(agent_state, transition, terminated, truncated, is_nongreedy):
    obs, action, next_obs, reward = transition

    config = agent_state.agent_config
    train_state = agent_state.train_state
    params = train_state.params
    opt_state = train_state.opt_state

    def get_q(params):
        q = train_state.apply_fn(params, obs)
        return q[action]

    q_taken, grads = jax.value_and_grad(get_q)(params)

    td_error = reward - q_taken
    if not terminated:
        next_q_vect = train_state.apply_fn(params, next_obs)
        td_error += config.gamma * jnp.max(next_q_vect, axis=-1)

    tx = train_state.tx
    updates, opt_state = tx.update(grads, opt_state, params, td_error, reset=(terminated or truncated or is_nongreedy))
    params = optax.apply_updates(params, updates)

    train_state = TrainState(
        step=train_state.step,
        apply_fn=train_state.apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
    )
    return AgentState(config, train_state)


StreamQAgent = Agent(init_agent_state, agent_step, update_step)
