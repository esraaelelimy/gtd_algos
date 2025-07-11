### QRC(λ)
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
    h_state: TrainState
    h_trace: jnp.ndarray      # the scalar trace for h (small z_t from the paper)
    grad_h_trace: jnp.ndarray # the trace of the gradient of h (z_t^{theta} from the paper)
    grad_q_trace: jnp.ndarray # the trace of the gradient of v (z_t^{w} from the paper)


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
    
    
    train_states = []
    lrs = [agent_config.q_lr, agent_config.h_lr_scale*agent_config.q_lr]
    # one network for q and one for h
    for net in range(2):
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

        
        tx = getattr(optax, agent_config.opt)(lrs[net])

        train_states.append( TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )
        )

    def params_sum(params):
        return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params)))
    print(f"Total number of params: {params_sum(train_states[0].params) + params_sum(train_states[1].params)}")
    # q_train_state, h_train_state = train_states
    grad_h_trace = tree_map(jnp.zeros_like,train_states[0].params)
    grad_v_trace = tree_map(jnp.zeros_like,train_states[1].params)
    h_trace = 0.0
    return AgentState(agent_config, *train_states,h_trace, grad_h_trace, grad_v_trace), rng





def update_q_trace(e_tmins1,rho_t,gamma,lamda,grad):
    # e_{t} = rho_t*gamma*lamda*e_{t-1} + grad)
    e_t = tree_map(lambda x,y: (rho_t * gamma * lamda * x) + y, e_tmins1, grad)
    return e_t


def reset_trace(trace):
    return tree.zeros(trace)

## Updates for QRC(λ) agent. Equations (26)-(28) in the paper
@partial(jax.jit, static_argnames=['terminated', 'truncated', 'is_nongreedy'])
def update_step(agent_state, transition, terminated, truncated, is_nongreedy):
    obs, action, next_obs, reward = transition

    config = agent_state.agent_config
    train_state = agent_state.train_state
    q_params = train_state.params
    h_params = agent_state.h_state.params
    h_tm1 = agent_state.h_trace
    grad_h_trace_tm1 = agent_state.grad_h_trace
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
        
    td_error, td_error_grad = jax.value_and_grad(get_td_error)(q_params)

    def get_h(params):
        h = agent_state.h_state.apply_fn(params, obs)
        return h[action]
    h_t, h_grads = jax.value_and_grad(get_h)(h_params)
    rho_t = 1.0 # rho here can either be 1 when a greedy action is taken or zero for non greedy action, and we simply just cut the traces if a non-greedy action is taken. i.e, when rho = 0.
    h_trace_t = (rho_t * config.gamma * config.lamda * h_tm1) + h_t
    grad_h_trace_t = update_q_trace(grad_h_trace_tm1,rho_t, config.gamma, config.lamda, h_grads)
    grad_q_trace_t = update_q_trace(grad_q_trace_tm1,rho_t, config.gamma, config.lamda, q_grads)
    
    # update q
    q_update = tree.scale(-h_trace_t, td_error_grad)  # GTD2 update: -trace(h) * ∇δ
    if config.gradient_correction:
        # TDC update: GTD2 + gradient correction
        q_update = tree.add(
            tree.scale(td_error, grad_q_trace_t),  # δ * trace(∇q)
            tree.scale(-h_t, q_grads),  # -h * ∇q
            q_update,
        )
    q_train_state = train_state.apply_gradients(grads=tree.neg(q_update))  # Flip sign because Flax multiplies by -1
    
    # update h
    delta_z_h = tree.scale(td_error, grad_h_trace_t)
    h_h_grad = tree.scale(-h_t, h_grads)
    beta_params = tree.scale(-config.reg_coeff, h_params)

    h_update = tree_map(lambda x , y , z: -( x + y + z), delta_z_h, h_h_grad, beta_params)
    
    h_train_state = agent_state.h_state.apply_gradients(grads=h_update)
    
    if terminated or truncated or is_nongreedy:
        h_trace_t = reset_trace(h_trace_t)
        grad_h_trace_t = reset_trace(grad_h_trace_t)
        grad_q_trace_t = reset_trace(grad_q_trace_t)

    return AgentState(config, q_train_state, h_train_state, h_trace_t, grad_h_trace_t, grad_q_trace_t)


QRCAgent = Agent(init_agent_state, StreamQAgent.step, update_step)
