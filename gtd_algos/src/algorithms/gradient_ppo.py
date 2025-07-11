### PPO with separate actor and critic networks
from flax import struct
from typing import NamedTuple, Any, Callable
import jax.numpy as jnp
from flax.training.train_state import TrainState
import jax
import optax 
import numpy as np
from functools import partial
import wandb 
PRNGKey = Any

from gtd_algos.src import tree
from gtd_algos.src.agents.ActorCritic import Actor, Critic
from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.configs.Config import Config
from gtd_algos.src.algorithms.ppo import PPOAgent, _calculate_gae, policy_loss, _create_minibaches


class AgentState(NamedTuple):
    agent_config: Config
    actor_network_state: TrainState
    critic_network_state: TrainState
    h_network_state: TrainState


def init_agent_state(agent_config: Config, action_dim:int,obs_shape:tuple,continous_action:bool, rng:PRNGKey):
    # create network
    actor_network = Actor(
        action_dim=action_dim,
        activation=agent_config.activation,
        d_actor=agent_config.d_actor_repr,
        cont=continous_action)

    critic_network = Critic(
        activation = agent_config.activation,
        d_critic = agent_config.d_critic_repr,
        )
    
    h_network = Critic(
        activation = agent_config.activation,
        d_critic = agent_config.d_critic_repr,
        )
    
    #input shape = (batch_size, obs_dim)
    init_x = jnp.zeros((1,*obs_shape))
    # parameters initialization
    actor_network_params = actor_network.init(rng, init_x)
    rng,_rng = jax.random.split(rng)
    critic_network_params = critic_network.init(_rng, init_x)
    rng,_rng = jax.random.split(rng)
    h_network_params = h_network.init(_rng, init_x)
    
    def params_sum(params):
            return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape),params)))
    print("Total Number of params: %d"%(params_sum(actor_network_params)+params_sum(critic_network_params)))
     
    # custom optimizer with gradient clipping
    def new_optimizer(lr):
        adam = optax.adam(lr, eps=1e-5)
        if agent_config.gradient_clipping:
            return optax.chain(
                optax.clip_by_global_norm(agent_config.max_grad_norm),
                adam,
            )
        else:
            return adam
    # train states 
    actor_network_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network_params,
        tx=new_optimizer(agent_config.actor_lr),
    )
    critic_network_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_network_params,
        tx=new_optimizer(agent_config.critic_lr),
    )
    h_network_state = TrainState.create(
        apply_fn=h_network.apply,
        params=h_network_params,
        tx=new_optimizer(agent_config.critic_lr * agent_config.h_lr_scale),
    )
    return AgentState(agent_config, actor_network_state, critic_network_state,h_network_state)

      



@jax.jit
def update_step(traj_batch, agent_state,last_val, rng):
    # calculate stale advantages for policy updates
    config = agent_state.agent_config
    stale_advantages, _ = _calculate_gae(traj_batch, last_val,config.gamma,config.gae_lambda)
    # Learning 
    update_state = (agent_state,traj_batch, stale_advantages,rng,config)
    update_state, aux_info = jax.lax.scan(_batch_update_tdrc, update_state, None, config.epochs)
    agent_state = update_state[0]
    return agent_state,aux_info,rng



def _calculate_gae_w_gradient(traj_batch,traj_val_grad,traj_val, last_val,last_val_grad,gamma,gae_lambda, rho):
    """Calculate advantages and their gradients (Algorithm 1 in the paper)
    GAE_t = delta_t + gamma * lambda * (1 - done_{t+1}) * GAE_{t+1}
    GAE_{traj_len+1} = 0

    grad GAE_t = grad delta_t + gamma * lambda * (1 - done_{t+1}) * grad GAE_{t+1}

    delta_t = reward_t + gamma * value_{t+1} * (1 - done_{t+1}) - value_t
    grad delta_t = reward_t + gamma * grad value_{t+1} * (1 - done_{t+1}) - grad value_t
    Args:
        traj_batch (list): A list of Transitions with shape (num_steps, _)
        last_val (float): Value of the last observation. This is the observation that follows the last observation in traj_batch.
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda
    Returns:
        advantages (jnp.ndarray): Advantages with shape (num_steps, _)
        targets (jnp.ndarray): Value targets with shape (num_steps, _)
    """
    ## Add rho_t to the calculation
    def _get_advantages(gae_and_next_value, transition):
            gae, gae_grad ,next_value,next_val_grad = gae_and_next_value
            transition,val_grad,value,rho = transition
            termination, done, reward = (
                transition.termination,
                transition.done,
                transition.reward,
            )
            
            
            delta = reward + gamma * next_value * (1 - done) - value
            #grad_delta= gamma * next_val_grad * (1 - termination) - val_grad
            grad_delta = jax.tree.map(lambda x,y: gamma*x*(1-done) - y,next_val_grad,val_grad )
            gae = rho * (
                delta
                + gamma * gae_lambda * (1 - done) * gae
            )
            #gae_grad = (grad_delta + gamma * gae_lambda * (1 - done) * gae_grad)
            gae_grad = jax.tree.map(lambda x,y: rho * (x + gamma*gae_lambda*x*(1 - done)*y),grad_delta,gae_grad )
            return (gae,gae_grad,value,val_grad), (gae,gae_grad)
    
    _, (advantages,advantages_grads) = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val),tree.zeros(last_val_grad),last_val,last_val_grad),
            (traj_batch,traj_val_grad,traj_val,rho),
            reverse=True,
        )
    return advantages, advantages_grads


@partial(jax.jit, static_argnums=(2,))
def evaluate(params, obs, apply_fn):
    return jnp.squeeze(
        apply_fn(params, jnp.expand_dims(obs, 0))
    )

evaluate_with_grad = jax.value_and_grad(evaluate)


# For each sequence in minibatch...
def updates_per_sequence(traj_batch,rho_t, agent_state):
    config = agent_state.agent_config
    critic_params = agent_state.critic_network_state.params
    h_params = agent_state.h_network_state.params
    
    # Vmapped grad fns
    value_grad_fn = jax.value_and_grad(evaluate)
    vmap_value_grad_fn = jax.vmap(value_grad_fn, in_axes=(None, 0, None), out_axes=0)

    obs_batch = traj_batch.obs
    traj_values, values_grads = vmap_value_grad_fn(critic_params, obs_batch, agent_state.critic_network_state.apply_fn)
    
    # split traj_values and values_grad and extract the last ones for bootstrapping
    seq_grads = tree.index.all_but_last(values_grads)
    advantages, advantages_grads = _calculate_gae_w_gradient(
        traj_batch=tree.index.all_but_last(traj_batch),
        traj_val_grad=seq_grads,
        traj_val=traj_values[:-1],
        last_val=traj_values[-1],
        last_val_grad=tree.index.last(values_grads),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        rho=rho_t[:-1],
    )
    hs, hs_grads = vmap_value_grad_fn(h_params, obs_batch[:-1], agent_state.h_network_state.apply_fn)

    if config.gradient_correction:
        critic_update = tree.subtract(
            tree.vmap_scale(advantages, seq_grads),  # δ * ∇v
            tree.vmap_scale(hs, tree.add(seq_grads, advantages_grads)),  # h * ∇(v + δ)
        )
    else:
        critic_update = tree.vmap_scale(-hs,advantages_grads)  # -h * ∇(δ)
        
        
    h_update = tree.subtract(
        tree.vmap_scale(advantages - hs, hs_grads),  # (δ - h) * ∇h
        tree.scale(config.reg_coeff, h_params),  # β * θ  (regularizer)
    )
    # Multiply by -1 because optax flips the sign:
    return tree.neg(critic_update), tree.neg(h_update)


# For each minibatch...
def _update_minibatch_tdrc(carry_in, batch_info):
    agent_state,config = carry_in
    traj_batch, stale_advantages = batch_info
    
    #### Policy Updates
    actor_train_state = agent_state.actor_network_state
    actor_grad_fn = jax.value_and_grad(policy_loss ,has_aux=True)
    # policy updates doesn't need sequences
    no_seq_traj = jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0]*x.shape[1],)+x.shape[2:]),traj_batch)
    stale_advantages = jnp.reshape(stale_advantages, -1)  # Flatten
    (_,aux_info), actor_grads = actor_grad_fn(
        actor_train_state.params,actor_train_state.apply_fn, no_seq_traj, stale_advantages,
        config.clip_eps,config.entropy_coef)
    
    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
    
    if config.is_correction:
        ## Get importance sampling ratio
        ## TODO: maybe add option to clipp rho if it is too large
        rho_t = jnp.exp(aux_info['log_ratio'])
        rho_t = rho_t.reshape(traj_batch.log_prob.shape)
    else:
        rho_t = jnp.ones_like(traj_batch.log_prob)
    #### Value and h Updates    
    critic_update, h_update = jax.vmap(updates_per_sequence,in_axes=(0,0,None), out_axes=0)(traj_batch,rho_t,agent_state)
    critic_network_state = _apply_mean_gradients(agent_state.critic_network_state, grads=critic_update, reduce_axis=(0, 1))
    h_network_update = _apply_mean_gradients(agent_state.h_network_state, grads=h_update, reduce_axis=(0, 1))

    agent_state = AgentState(config,actor_train_state,critic_network_state,h_network_update)
    return (agent_state,config), aux_info


def _apply_mean_gradients(train_state, grads, reduce_axis):
    mean_gradients = jax.tree_map(lambda x: jnp.mean(x, axis=reduce_axis), grads)
    return train_state.apply_gradients(grads=mean_gradients)


# For each epoch...
@jax.jit
def _batch_update_tdrc(update_data, unused):
    agent_state,traj_batch, stale_advantages,rng,config = update_data
    config = agent_state.agent_config
    batch = (traj_batch, stale_advantages)
    # Create minibatches
    minibatches,rng = _create_minibaches(config,batch, rng)
    # Loop through minibatches 
    carry_in = (agent_state,config)
    carry_out, aux_info = jax.lax.scan(_update_minibatch_tdrc, carry_in, minibatches)
    agent_state =  carry_out[0]
    update_data = (agent_state, traj_batch, stale_advantages, rng,config)
    return update_data, aux_info


GradientPPOAgent = Agent(init_agent_state, PPOAgent.step, update_step)
