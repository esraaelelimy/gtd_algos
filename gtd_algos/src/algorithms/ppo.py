### PPO with separate actor and critic networks
from flax import struct
from typing import NamedTuple, Any, Callable
import jax.numpy as jnp
from flax.training.train_state import TrainState
from gtd_algos.src.agents.ActorCritic import Actor,Critic
import jax
import optax 
import numpy as np
from functools import partial
import wandb 
PRNGKey = Any

from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.configs.Config import Config


class AgentState(NamedTuple):
    agent_config: Config
    actor_network_state: TrainState
    critic_network_state: TrainState

class Transition(NamedTuple):
    obs: jnp.ndarray    # o_t
    action: jnp.ndarray # a_t
    reward: jnp.ndarray # r[t+1]
    done: jnp.ndarray   # done[t+1] (termination or truncation)
    termination: jnp.ndarray # termination[t+1]
    value: jnp.ndarray  # v(o_t)
    log_prob: jnp.ndarray # log_prob Pi(a_t|o_t)
    info: jnp.ndarray


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
    
    
    #input shape = (batch_size, obs_dim)
    init_x = jnp.zeros((1,*obs_shape))
    # parameters initialization
    actor_network_params = actor_network.init(rng, init_x)
    rng,_rng = jax.random.split(rng)
    critic_network_params = critic_network.init(_rng, init_x)
    
    
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
    
    return AgentState(agent_config, actor_network_state, critic_network_state)

      
@jax.jit
def agent_step(agent_state: AgentState, obs: jnp.ndarray, rng: PRNGKey) :
    '''
    obs.shape = (batch_size,obs_dim)
    '''
    pi = agent_state.actor_network_state.apply_fn(agent_state.actor_network_state.params,obs)
    value = agent_state.critic_network_state.apply_fn(agent_state.critic_network_state.params,obs)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)  
    return action, value, log_prob    


@jax.jit
def update_step(traj_batch, agent_state, last_val, rng):
    update_data = (traj_batch, agent_state, last_val, rng)
    agent_state, aux_info,rng = _update_step_stale_target(update_data)
    return agent_state, aux_info,rng



@partial(jax.jit, static_argnums=(1,))
def critic_loss(critic_params, critic_fn,traj_batch, targets,clip_eps,vf_coef):
    # Re-run the network through the batch
    value = critic_fn(critic_params, traj_batch.obs)
    # Calculate value loss
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )
    value_loss =  vf_coef * value_loss 
    return value_loss
    


@partial(jax.jit, static_argnums=(1,))
def policy_loss(actor_params,actor_fn,traj_batch, gae,clip_eps,ent_coef):
    # Re-run the network through the batch
    pi = actor_fn(actor_params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)
    # Calculate actor loss
    log_ratio = log_prob - traj_batch.log_prob
    ratio = jnp.exp(log_ratio)
    approx_kl = ((ratio - 1) - log_ratio).mean()
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - clip_eps,
            1.0 + clip_eps,
        )
        * gae
    )
    
    # Calculate entropy
    entropy = pi.entropy().mean()
    
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    
    total_loss = loss_actor - ent_coef * entropy
    
    aux_info = {'log_ratio':(log_ratio),
                'approx_kl':approx_kl,
                'policy_loss':loss_actor,
                'entropy':entropy}
    return total_loss, aux_info



  
@jax.jit
def _update_minibatch(carry_in, batch_info):
    actor_train_state,critic_train_state,config = carry_in
    traj_batch, advantages, targets = batch_info
    actor_grad_fn = jax.value_and_grad(policy_loss ,has_aux=True)
    
    
    (_,aux_info), actor_grads = actor_grad_fn(
        actor_train_state.params,actor_train_state.apply_fn, traj_batch, advantages,
        config.clip_eps,config.entropy_coef)
    
    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
    
    
    critic_grad_fn = jax.value_and_grad(critic_loss)
    value_loss, critic_grads = critic_grad_fn(critic_train_state.params,critic_train_state.apply_fn, traj_batch, targets,config.clip_eps,config.vf_coef)
    
    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
    aux_info['value_loss'] = value_loss
    return (actor_train_state,critic_train_state,config), aux_info

@jax.jit
def _calculate_gae(traj_batch, last_val,gamma,gae_lambda):
    """Calculate advantages and value targets
    GAE_t = delta_t + gamma * lambda * (1 - done_{t+1}) * GAE_{t+1}
    GAE_{traj_len+1} = 0
    delta_t = reward_t + gamma * value_{t+1} * (1 - done_{t+1}) - value_t
    Args:
        traj_batch (list): A list of Transitions with shape (num_steps, _)
        last_val (float): Value of the last observation. This is the observation that follows the last observation in traj_batch.
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda
    Returns:
        advantages (jnp.ndarray): Advantages with shape (num_steps, _)
        targets (jnp.ndarray): Value targets with shape (num_steps, _)
    """
    def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            termination, done, value, reward = (
                transition.termination,
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + gamma * next_value * (1 - done) - value
            gae = (
                delta
                + gamma * gae_lambda * (1 - done) * gae
            )
            return (gae, value), gae
    _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
    return advantages, advantages + traj_batch.value  
  
@jax.jit
def _create_minibaches(config,batch, rng):
    """Create minibatches from a batch of trajectories """
    """ Logic:
    1. Divide the trajectories into number of sequences
    2. Shuffle the sequences 
    3. Divide the sequences into minibatches
    The output will have the shape of (num_minibatches, minibatch_size, seq_len,_)
    Note: when not using rnns, we don't need to have sequences (so seq_len is just 1)
    """
    number_sequences = config.rollout_steps // config.seq_len_in_minibatch 
    minibatch_size = config.rollout_steps//(config.seq_len_in_minibatch*config.num_mini_batch)
    # reshape the batch to (number_sequences, seq_len, _)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((number_sequences,config.seq_len_in_minibatch,)+x.shape[1:]), batch)
    # shuffle the sequences
    rng, _rng = jax.random.split(rng)
    permutation = jax.random.permutation(_rng, number_sequences)
    shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch)
    # reshape the shuffled batch to (num_minibatches, minibatch_size, seq_len,_)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((config.num_mini_batch,minibatch_size,)+x.shape[1:]), shuffled_batch)
    return batch,rng

@jax.jit
def _batch_update(update_state, unused):
    actor_train_state,critic_train_state, traj_batch, advantages, targets, rng,config = update_state
    batch = (traj_batch, advantages, targets)
    
    minibatches_info,rng = _create_minibaches(config,batch, rng)
    # Loop through minibatches 
    carry_in = (actor_train_state,critic_train_state,config)
    carry_out, aux_info = jax.lax.scan(_update_minibatch, carry_in, minibatches_info)
    actor_train_state, critic_train_state, _ = carry_out
    update_state = (actor_train_state,critic_train_state, traj_batch, advantages, targets, rng,config)
    return update_state, aux_info



@jax.jit
def _update_step_stale_target(update_data):
    traj_batch, agent_state, last_val, rng = update_data
    # calculate advantages and targets
    config = agent_state.agent_config
    advantages, targets = _calculate_gae(traj_batch, last_val,config.gamma,config.gae_lambda)
    # Learning 
    update_state = (agent_state.actor_network_state,agent_state.critic_network_state,traj_batch, advantages, targets, rng,config)
    update_state, aux_info = jax.lax.scan(_batch_update, update_state, None, config.epochs)
    agent_state = AgentState(config, actor_network_state=update_state[0], critic_network_state=update_state[1])
    return agent_state, aux_info,update_state[-2]


def wandb_ppo_logging(info_and_loss):
    metric, (losses) = info_and_loss
    return_values = metric["returned_episode_returns"]
    timesteps = metric["timestep"]
    wandb.log({'env_steps': np.mean(timesteps), 'undiscounted_return': np.mean(return_values)})
    wandb.log({'env_steps': np.mean(timesteps),
               #'value_loss': np.mean(np.mean(losses['value_loss'])),
               'policy_loss': np.mean(np.mean(losses['policy_loss'])),
               'entropy': np.mean(np.mean(losses['entropy'])),
               'log_ratio': np.mean(np.mean(losses['log_ratio'])),
               'approx_kl': np.mean(np.mean(losses['approx_kl']))})
    return


PPOAgent = Agent(init_agent_state, agent_step, update_step)
