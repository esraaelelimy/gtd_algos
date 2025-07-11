import argparse
import time
from typing import Callable

import gymnasium as gym
import numpy as np
import jax
import wandb

from gtd_algos.src.algorithms.agent import Agent
from gtd_algos.src.configs.ExpConfig import ExpConfig
from gtd_algos.src.envs.gym_envs_wrappers import StoreEpisodeReturnsAndLengths


def main(
        experiment: Callable[[ExpConfig, Agent], gym.Env],
        agent: Agent,
        define_metrics: Callable[[None], None],
        default_config_path: str = None,
    ):
    # Reading config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=default_config_path)
    args = parser.parse_args()
    config = ExpConfig.from_yaml(args.config_file)
    ### wandb init
    wandb.init(config=config, project=config.wandb_project_name)
    define_metrics()
    ### start experiment
    start_time = time.time()
    env = jax.block_until_ready(experiment(config, agent))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time elapsed: {elapsed_time / 60:.2f} minutes')
    total_steps = config.agent_config.total_steps
    wandb.run.summary['SPS'] = int(total_steps / elapsed_time)
    wandb.finish()
