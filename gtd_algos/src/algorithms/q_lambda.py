### Watkins Q(λ)
from functools import partial
from typing import NamedTuple

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import tree_map
from gtd_algos.src import tree
from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.configs.Config import Config
from gtd_algos.src.agents.value_networks import DenseQNetwork, MinAtarQNetwork
from gtd_algos.src.nets.MLP import sparse_init
from gtd_algos.src.algorithms.streamq import StreamQAgent

class AgentState(NamedTuple):
    agent_config: Config
    train_state: TrainState
    grad_q_trace: jnp.ndarray # the z trace


def init_agent_state(agent_config: Config, action_dim: int, obs_shape: tuple, rng: jax.random.PRNGKey):
    net_kwargs = {
        'action_dim': action_dim,
        'layer_norm': agent_config.layer_norm,
        'activation': agent_config.activation,
        'kernel_init': sparse_init(sparsity=agent_config.sparse_init),
    }
    net_arch = agent_config.net_arch
    if net_arch == 'mlp':
        net_kwargs['hiddens'] = agent_config.mlp_layers
    

    if net_arch == 'minatar':
        network = MinAtarQNetwork(**net_kwargs)
        init_x = jnp.zeros(obs_shape)
    elif net_arch == 'mlp':
        network = DenseQNetwork(**net_kwargs)
        init_x = jnp.zeros(obs_shape)
    else:
        raise ValueError(f"unknown network architecture: {net_arch}")
    
    rng, _rng = jax.random.split(rng)
    params = network.init(_rng, init_x)

    opt_kwrgs = {'learning_rate': agent_config.q_lr}
    tx = getattr(optax, agent_config.opt)(**opt_kwrgs)

    train_states = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx)


    def params_sum(params):
        return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params)))
    print(f"Total number of params: {params_sum(train_states.params)}")
    
    grad_q_trace = tree_map(jnp.zeros_like,train_states.params)
    return AgentState(agent_config, train_states, grad_q_trace), rng





def update_q_trace(e_tmins1,rho_t,gamma,lamda,grad):
    # e_{t} = rho_t*gamma*lamda*e_{t-1} + grad)
    e_t = tree_map(lambda x,y: (rho_t * gamma * lamda * x) + y, e_tmins1, grad)
    return e_t


def reset_trace(trace):
    return tree.zeros(trace)

# Updates according to Watkins Q(λ) 
@partial(jax.jit, static_argnames=['terminated', 'truncated', 'is_nongreedy'])
def update_step(agent_state, transition, terminated, truncated, is_nongreedy):
    obs, action, next_obs, reward = transition

    config = agent_state.agent_config
    train_state = agent_state.train_state
    q_params = train_state.params
    grad_q_trace_tm1 = agent_state.grad_q_trace

    def get_q(params):
        q = train_state.apply_fn(params, obs)
        return q[action]

    q_grads = jax.grad(get_q)(q_params)

    def get_td_error(params):
        q = train_state.apply_fn(params, obs)
        q_taken = q[action]
        next_q_vect = train_state.apply_fn(params, next_obs)
        td_error = (config.gamma * jnp.max(next_q_vect, axis=-1))*(1 - terminated) + reward - q_taken
        return td_error
        
    td_error = get_td_error(q_params)
    rho_t = 1.0 # rho here can either be 1 when a greedy action is taken or zero for non greedy action, and we simply just cut the traces if a non-greedy action is taken. i.e, when rho = 0.
    grad_q_trace_t = update_q_trace(grad_q_trace_tm1,rho_t, config.gamma, config.lamda, q_grads)
    
    # update q
    q_update = tree.scale(td_error, grad_q_trace_t)  # e * ∇δ
    q_train_state = train_state.apply_gradients(grads=tree.neg(q_update))  # Flip sign because Flax multiplies by -1
    
    
    if terminated or truncated or is_nongreedy:
        grad_q_trace_t = reset_trace(grad_q_trace_t)

    return AgentState(config, q_train_state, grad_q_trace_t)


QLambdaAgent = Agent(init_agent_state, StreamQAgent.step, update_step)
