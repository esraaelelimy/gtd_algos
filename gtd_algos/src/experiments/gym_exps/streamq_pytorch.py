import numpy as np
import wandb
import torch

from gtd_algos.src.algorithms.streamq_pytorch import StreamQ
from gtd_algos.src.configs.ExpConfig import ExpConfig
from gtd_algos.src.envs.make_gym_envs import make_streaming_drl_env as make_env
from gtd_algos.src.experiments.gym_exps.streamq_main import define_metrics
from gtd_algos.src.experiments.main import main


def experiment(config: ExpConfig, _):
    agent_config = config.agent_config
    env_config = config.env_config

    seed = config.exp_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(env_config, agent_config.gamma)

    total_steps = agent_config.total_steps
    agent = StreamQ(
        n_channels=env.observation_space.shape[-1],
        n_actions=env.action_space.n,
        lr=agent_config.lr,
        gamma=agent_config.gamma,
        lamda=agent_config.lamda,
        epsilon_target=agent_config.end_epsilon,
        epsilon_start=agent_config.start_epsilon,
        exploration_fraction=agent_config.explore_frac,
        total_steps=total_steps,
        kappa_value=agent_config.kappa,
    )

    s, _ = env.reset(seed=seed)
    episode_num = 1
    for t in range(1, total_steps + 1):
        a, is_nongreedy = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime, terminated or truncated, is_nongreedy)
        s = s_prime
        if terminated or truncated:
            undisc_return = info['episode']['r'].item()
            wandb.log({
                'env_steps': t,
                'undiscounted_return': undisc_return,
                'avg100_undiscounted_return': np.mean(env.get_wrapper_attr('return_queue')),
                'epsilon': agent.epsilon,
            })
            terminated, truncated = False, False
            s, _ = env.reset()
            episode_num += 1

    return env


if __name__ == "__main__":
    main(
        experiment, None, define_metrics,
        default_config_path='gtd_algos/exp_configs/minatar_streamq_pytorch.yaml',
    )
