'''
Create MLP with custom activation function and kernel initializer
'''
from gtd_algos.src.nets.MLP  import MLP
from gtd_algos.src.nets.sparse_init import sparse_init
import flax.linen as nn
from flax.linen.initializers import variance_scaling, lecun_uniform, he_uniform, constant, zeros_init, orthogonal
import jax.numpy as jnp
import jax


'''testing mlp'''
def create_custom_mlp():
    act_fun = nn.leaky_relu
    pre_act_norm = False
    hiddens = [64,64]
    kernel_init = orthogonal(scale = jnp.sqrt(2))
    bias_init = constant(0.0)
    
    mlp = MLP(hiddens=hiddens,act=act_fun,pre_act_norm=pre_act_norm,kernel_init=kernel_init,bias_init=bias_init)
    
    #input shape = (batch_size, obs_dim)
    init_x = jnp.zeros((1,1))
    rng = jax.random.PRNGKey(0)
    params = mlp.init(rng, init_x)
    out = mlp.apply(params,init_x)
    return 

'''testing sparse init'''
'''This is the network for stream drl'''
def create_custom_w_sparse_init_mlp():
    act_fun = nn.leaky_relu
    pre_act_norm = True
    hiddens = [64,64,1]
    kernel_init = sparse_init()
    bias_init = constant(0.0)
    
    mlp = MLP(hiddens=hiddens,act=act_fun,pre_act_norm=pre_act_norm,kernel_init=kernel_init,bias_init=bias_init)
    
    #input shape = (batch_size, obs_dim)
    init_x = jnp.zeros((1,1))
    rng = jax.random.PRNGKey(0)
    params = mlp.init(rng, init_x)
    out = mlp.apply(params,init_x)
    return


if __name__ == "__main__":
    create_custom_mlp()
    create_custom_w_sparse_init_mlp()
    