import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics


class StoreEpisodeReturnsAndLengths(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env, RecordEpisodeStatistics)
        super().__init__(env)
        self.all_episode_returns = []
        self.all_episode_lengths = []

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.all_episode_returns.append(info['episode']['r'].item())
            self.all_episode_lengths.append(info['episode']['l'].item())
        return next_obs, reward, terminated, truncated, info
