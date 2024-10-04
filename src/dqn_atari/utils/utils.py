import gymnasium as gym
import numpy as np
import torch
from gymnasium import Env


def get_device() -> str:
    """Get the device (CPU, CUDA, or MPS) if available."""
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class FireResetEnv(gym.Wrapper):
    def __init__(self, env: Env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info["lives"] < self.lifes:
            observation, _, _, _, info = self.env.step(1)  # fire
        self.lifes = info["lives"]
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.lifes = info["lives"]
        observation, _, _, _, info = self.env.step(1)  # fire
        return observation, info


def wrap_atari(env: Env, eval_mode: bool = False) -> Env:
    """Wrap an Atari environment with common preprocessing steps."""
    if not eval_mode:
        env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))
    try:
        env = FireResetEnv(env)
    except AssertionError:
        pass
    env = gym.wrappers.AtariPreprocessing(
        env=env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=not eval_mode,
        grayscale_obs=True,
        scale_obs=True,
    )
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env
