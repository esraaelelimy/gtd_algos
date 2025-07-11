import jax
import jax.numpy as jnp
import wandb

from gtd_algos.src.algorithms.ppo import PPOAgent, Transition, wandb_ppo_logging
from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.configs.ExpConfig import ExpConfig
from gtd_algos.src.envs.make_gym_envs import make_env
from gtd_algos.src.experiments.main import main


def exp_step(runner_state, env, i, agent):
    result = []
    def interacton_step(interaction_state,j):
        agent_state,last_obs, rng = interaction_state
        # agent step 
        rng, _rng = jax.random.split(rng)
        action, value, log_prob = agent.step(agent_state, jnp.expand_dims(last_obs,0), _rng)
        # env step
        if len(action.shape) > 1:
            action = action.squeeze(axis=0)
        else:
            action = action.item()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if terminated or truncated:
            result_dict = {
                    'timestep': i*runner_state[0].agent_config.rollout_steps+j,
                    'returned_episode_returns': info['episode']['r']
                    }
            result.append(result_dict)
            obs, info = env.reset()        
        # save transition
        transition = Transition(last_obs,action,reward,done,terminated, value.squeeze(),log_prob.squeeze(),None)
        interaction_state = (agent_state, obs, rng)
        return interaction_state, transition
    
    # Collect a trajectory
    rollout_steps = runner_state[0].agent_config.rollout_steps
    traj_batch = []
    for j in range(rollout_steps):
        runner_state, transition = interacton_step(runner_state,j)
        traj_batch.append(transition)
        
    obs_list, action_list, reward_list, done_list,terminated_list, value_list, log_prob_list,_ = zip(*traj_batch)
    traj_batch = Transition(
                        obs=jnp.stack(obs_list),
                        action=jnp.stack(action_list),
                        reward=jnp.stack(reward_list),
                        done=jnp.stack(done_list),
                        termination=jnp.stack(terminated_list),
                        value=jnp.stack(value_list),
                        log_prob=jnp.stack(log_prob_list),
                        info=None
                        )
    agent_state,last_obs, rng = runner_state
    
    # Calculate last value
    train_state = agent_state.critic_network_state
    last_value = train_state.apply_fn(train_state.params, jnp.expand_dims(last_obs,0))
    last_val = last_value.squeeze()
    
    # Update step
    agent_state, loss_info, rng = agent.update(traj_batch, agent_state, last_val, rng)
    
    ## Logging
    metric = {k: jnp.stack([dic[k] for dic in result]) for k in result[0]}
    jax.debug.callback(wandb_ppo_logging, (metric,loss_info))
    ## Update runner state
    runner_state = (agent_state, last_obs, rng)
    return runner_state, result


def experiment(config: ExpConfig, agent: Agent):
    agent_config = config.agent_config
    env_config = config.env_config
    rng = jax.random.PRNGKey(config.exp_seed)

    # Create and initialize the environment.
    env = make_env(env_config,agent_config.gamma)
    obs, _ = env.reset()
    # Initialize the agent 
    action_dim = None
    if env_config.continous_action:
        action_dim = env.action_space.shape[0]
    else:
        action_dim =  env.action_space.n
    
    rng, _rng = jax.random.split(rng)
    agent_state = agent.init_state(agent_config,action_dim,env.observation_space.shape,env_config.continous_action, _rng)
    
    runner_state = (agent_state, obs, rng)
    num_updates = agent_config.total_steps // agent_config.rollout_steps
    for i in range(num_updates):
        runner_state, result = exp_step(runner_state, env, i, agent)
    
    return {"runner_state": runner_state, "metrics": result}


def define_metrics():
    wandb.define_metric("env_steps")
    wandb.define_metric("undiscounted_return", step_metric="env_steps")
    wandb.define_metric("value_loss", step_metric="env_steps")
    wandb.define_metric("policy_loss", step_metric="env_steps")
    wandb.define_metric("minus_log_ratio", step_metric="env_steps")
    wandb.define_metric("approx_kl", step_metric="env_steps")
    wandb.define_metric("pi_std", step_metric="env_steps")


if __name__ == "__main__":
    main(
        experiment, PPOAgent, define_metrics,
        default_config_path='gtd_algos/exp_configs/mujoco_ppo.yaml',
    )
