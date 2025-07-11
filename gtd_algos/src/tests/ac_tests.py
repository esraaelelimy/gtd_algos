'''
Create Actor critic nets with custom activation function and kernel initializer
'''
from gtd_algos.src.agents.ActorCritic  import *
import flax.linen as nn
from flax.linen.initializers import variance_scaling, lecun_uniform, he_uniform, constant, zeros_init, orthogonal
import jax.numpy as jnp
import jax
from gtd_algos.src.nets.sparse_init import sparse_init


def create_ppo_ac():
    action_dim = 1
    activation = "tanh"
    d_actor = [64, 64]
    d_critic  = [64, 64]
    kernel_init = orthogonal(scale = jnp.sqrt(2))
    kernel_init_last = orthogonal(0.01)
    bias_init = constant(0.0)
    pre_act_norm = False
    cont = False
    
    
    actor_network = Actor(
        action_dim=action_dim,
        activation=activation,
        d_actor=d_actor,
        kernel_init=kernel_init,
        kernel_init_last = kernel_init_last,
        bias_init=bias_init,
        pre_act_norm=pre_act_norm,
        cont=cont)

    critic_network = Critic(
        activation = activation,
        d_critic = d_critic,
        kernel_init=kernel_init,
        kernel_init_last = kernel_init_last,
        bias_init = bias_init,
        pre_act_norm = pre_act_norm
        )
    
    
    #input shape = (batch_size, obs_dim)
    init_x = jnp.zeros((1,10))
    rng = jax.random.PRNGKey(0)
    actor_network_params = actor_network.init(rng, init_x)
    rng,_rng = jax.random.split(rng)
    critic_network_params = critic_network.init(_rng, init_x)
    
    return 


if __name__ == "__main__":
    create_ppo_ac()
    print("Actor critic nets created successfully")