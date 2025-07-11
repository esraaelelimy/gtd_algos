import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import ClipAction, RescaleAction, RecordEpisodeStatistics, FlattenObservation, NormalizeObservation, NormalizeReward, TransformObservation, TransformReward

from gtd_algos.src.envs.gym_envs_wrappers import normalization, StoreEpisodeReturnsAndLengths

class EnvSeed(gym.Wrapper):
  def set_env_seed(self, seed: int):
    _, _ = self.env.reset(seed=seed)
    self.env.action_space.seed(seed)
    self.env.observation_space.seed(seed)
    
def make_env(env_config,gamma, **kwargs):
  """
  Make env for general tasks.
  """
  env = gym.make(env_config.env_name, **kwargs)
  # Episode statistics wrapper
  env = RecordEpisodeStatistics(env)
  env = StoreEpisodeReturnsAndLengths(env)

  if isinstance(env.action_space, spaces.Box): # Continuous action space
    env = ClipAction(RescaleAction(env, min_action=-1, max_action=1))
  if env_config.normalize_obs:
    env = NormalizeObservation(env)
  if env_config.normalize_reward:
    env = NormalizeReward(env, gamma=gamma)
    
  env = EnvSeed(env)
  env.set_env_seed(env_config.env_seed)
  return env


def make_streaming_drl_env(env_config, gamma, **kwargs):
    """Make env for general tasks with the wrappers from Streaming DRL paper."""
    env = gym.make(env_config.env_name, **kwargs)
    # Episode statistics wrapper
    env = RecordEpisodeStatistics(env)
    env = StoreEpisodeReturnsAndLengths(env)

    # Normalization must come after stats for correct results
    if env_config.normalize_reward:
        env = normalization.ScaleReward(env, gamma=gamma)
    if env_config.normalize_obs:
        env = normalization.NormalizeObservation(env)
        
    env = EnvSeed(env)
    env.set_env_seed(env_config.env_seed)
    return env
