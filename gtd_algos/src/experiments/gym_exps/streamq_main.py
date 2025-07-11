import itertools
from typing import Callable

import jax
import numpy as np
import wandb

from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.algorithms.streamq import StreamQAgent
from gtd_algos.src.configs.Config import Config
from gtd_algos.src.configs.ExpConfig import ExpConfig
from gtd_algos.src.envs.gym_envs_wrappers import StoreEpisodeReturnsAndLengths
from gtd_algos.src.envs.make_gym_envs import make_streaming_drl_env as make_env
from gtd_algos.src.experiments.main import main


def get_linear_epsilon_schedule(agent_config: Config) -> Callable[[int], float]:
    start_epsilon = agent_config.start_epsilon
    end_epsilon = agent_config.end_epsilon
    assert 0.0 <= end_epsilon <= start_epsilon <= 1.0
    anneal_time = agent_config.explore_frac * agent_config.total_steps
    assert anneal_time > 0.0

    def epsilon_schedule(t: int) -> float:
        frac_annealed = min(t / anneal_time, 1.0)
        return (1.0 - frac_annealed) * start_epsilon + frac_annealed * end_epsilon
    return epsilon_schedule


def experiment(config: ExpConfig, agent: Agent) -> StoreEpisodeReturnsAndLengths:
    agent_config = config.agent_config
    env_config = config.env_config
    rng = jax.random.PRNGKey(config.exp_seed)

    # Create and initialize the environment
    env = make_env(env_config, agent_config.gamma)
    obs, _ = env.reset(seed=env_config.env_seed)
    episodes = 0

    # Initialize the agent
    action_dim = int(env.action_space.n)
    agent_state, rng = agent.init_state(agent_config, action_dim, env.observation_space.shape, rng)
    epsilon_schedule = get_linear_epsilon_schedule(agent_config)

    for t in itertools.count():
        epsilon = epsilon_schedule(t)
        action, is_nongreedy, rng = agent.step(agent_state, obs, action_dim, epsilon, rng)
        action = action.item()
        is_nongreedy = is_nongreedy.item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        transition = (obs, action, next_obs, reward)
        
        agent_state = agent.update(agent_state, transition, terminated, truncated, is_nongreedy)

        if done:
            episodes += 1
            undisc_return = info['episode']['r'].item()
            wandb.log({
                'episodes': episodes,
                'env_steps': t,
                'undiscounted_return': undisc_return,
                'avg100_undiscounted_return': np.mean(env.get_wrapper_attr('return_queue')),
                'epsilon': epsilon,
            })
            if t >= agent_config.total_steps:
                return env
            next_obs, info = env.reset()

        obs = next_obs


def define_metrics():
    wandb.define_metric("episodes", step_metric="episodes")
    wandb.define_metric("env_steps", step_metric="episodes")
    wandb.define_metric("undiscounted_return", step_metric="env_steps")
    wandb.define_metric("avg100_undiscounted_return", step_metric="env_steps")
    wandb.define_metric("epsilon", step_metric="env_steps")


if __name__ == "__main__":
    main(
        experiment, StreamQAgent, define_metrics,
        default_config_path='gtd_algos/exp_configs/minatar_streamq_jax.yaml',
    )
