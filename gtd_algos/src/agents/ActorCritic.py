import flax.linen as nn 
from typing import Optional, Tuple, Union, Any, Sequence, Dict, Callable,Iterable
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp
import numpy as np
import distrax
import functools
from gtd_algos.src.nets.MLP import MLP,sparse_init

#######################################
# Seperate Actor and Critic Networks (PPO Nets) #
#######################################

act_funcs = {'relu': nn.relu, 
               'tanh': nn.tanh, 
               'leaky_relu': nn.leaky_relu}

class Actor(nn.Module):
    """Actor Network: Gaussian Policy with independent Diagonal Covariance Matrix"""
    action_dim: Sequence[int]
    d_actor: Iterable[int] 
    activation: str = "tanh"
    kernel_init: Callable = orthogonal(scale = jnp.sqrt(2))
    kernel_init_last: Callable = orthogonal(0.01)
    bias_init: Callable = constant(0.0)
    pre_act_norm: bool = False
    cont: bool = False
    @nn.compact
    def __call__(self, x):
        
        activation = act_funcs[self.activation]
        actor_mean = MLP(hiddens=self.d_actor, act=activation, kernel_init=self.kernel_init,bias_init=self.bias_init,
                         pre_act_norm=self.pre_act_norm)(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=self.kernel_init_last, bias_init=constant(0.0))(actor_mean) 
        if self.cont:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)
        return pi

class Critic(nn.Module):
    """ Value Critic: V(s)"""
    d_critic: Iterable[int]
    activation: str = "tanh"
    kernel_init: Callable = orthogonal(scale = jnp.sqrt(2))
    kernel_init_last: Callable = orthogonal(1.0)
    bias_init: Callable = constant(0.0)
    pre_act_norm: bool = False
    @nn.compact
    def __call__(self, x):
        activation = act_funcs[self.activation]
        critic = MLP(hiddens=self.d_critic, act=activation, kernel_init=self.kernel_init,bias_init=self.bias_init,
                         pre_act_norm=self.pre_act_norm)(x)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=self.kernel_init_last, bias_init=constant(0.0))(critic)
        return jnp.squeeze(critic, axis=-1)



