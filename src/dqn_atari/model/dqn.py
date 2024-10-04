from __future__ import annotations

import logging

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm

from dqn_atari.buffer.buffer import ReplayBuffer
from dqn_atari.buffer.buffer import UniformReplayBuffer
from dqn_atari.buffer.buffer import get_buffer_class
from dqn_atari.model.policy import Policy
from dqn_atari.utils.utils import get_device
from dqn_atari.utils.utils import wrap_atari

logger = logging.getLogger(__name__)


class DQN:
    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        vectorization_mode: str = "sync",
        double_dqn: bool = False,
        dueling: bool = False,
        layers: list[int] = [64, 64],
        buffer_class: ReplayBuffer = UniformReplayBuffer,
        buffer_size: int = 100_000,
        buffer_kwargs: dict = {},
        gamma: float = 0.999,
        batch_size: int = 32,
        train_freq: int = 1,
        gradient_steps: int = 1,
        lr: float = 2e-5,
        target_update_freq: int = 10,
        tau: float = 1.0,
        epsilon_start: float = 0.1,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1e6,
        learning_starts: int = 1000,
        max_episode_length: int = 1e6,
        max_grad_norm: float = 10.0,
        force_cpu: bool = False,
    ) -> None:
        self.device = get_device() if not force_cpu else "cpu"
        logger.info(f"Using device: {self.device}.")

        # create the environment
        self.env_id = env_id
        self.num_envs = num_envs
        self.vectorization_mode = vectorization_mode
        is_atari = "ALE" in env_id or "NoFrameskip" in env_id
        if num_envs == 1:
            self.env = gym.make(env_id)
            if is_atari:
                self.env = wrap_atari(self.env)
            observation_shape = self.env.observation_space.shape
            n_actions = self.env.action_space.n
            logger.info(f"Initalized environment: {self.env_id}.")
        else:
            self.env = gym.make_vec(
                env_id,
                num_envs=num_envs,
                vectorization_mode=vectorization_mode,
                wrappers=[wrap_atari] if is_atari else None,
            )
            observation_shape = self.env.single_observation_space.shape
            n_actions = self.env.single_action_space.n
            logger.info(f"Initalized vectorized environment: {self.env}.")

        # create policy
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.layers = layers
        policy_kwargs = dict(
            observation_shape=observation_shape,
            n_actions=n_actions,
            dueling=dueling,
            layers=layers,
        )
        self.q_network = Policy(**policy_kwargs).to(self.device)
        self.target_network = Policy(**policy_kwargs).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # create replay buffer
        self.buffer_class = (
            get_buffer_class(buffer_class) if isinstance(buffer_class, str) else buffer_class
        )
        self.buffer_size = buffer_size
        self.buffer_kwargs = buffer_kwargs
        self.buffer: ReplayBuffer = self.buffer_class(
            buffer_size=buffer_size,
            state_shape=observation_shape,
            num_envs=num_envs,
            device=self.device,
            **buffer_kwargs,
        )

        # save hyperparams
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps if gradient_steps > 0 else num_envs
        self.lr = lr
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_starts = learning_starts
        self.max_episode_length = max_episode_length
        self.max_grad_norm = max_grad_norm
        self.force_cpu = force_cpu

        # training progress
        self.steps_trained = 0

    def train(self, training_steps: int, eval_every: int = 0, eval_runs: int = 20) -> None:
        """Train the DQN agent.

        Args:
            training_steps (int): The total number of training steps.
            eval_every (int, optional): Evaluate the agent every n steps. Defaults to 0 (disabled).
            eval_runs (int, optional): Number of evaluation runs. Defaults to 20.
        """
        # initialize the episode
        state, _ = self.env.reset()
        episode_length = np.zeros(self.num_envs) if self.num_envs > 1 else 0
        steps_since_eval = 0

        with tqdm(total=training_steps, desc="Training", unit="steps") as pbar:
            for step in range(1, training_steps + 1):
                # take a step in the environment and save the transition
                action = self.get_action(state, self.epsilon)
                next_state, reward, done, *_ = self.env.step(action)
                self.buffer.add((state, action, reward, next_state, done))

                # update current state
                state = next_state
                episode_length += 1

                # handle environment resets
                if self.num_envs == 1:
                    if done or (episode_length >= self.max_episode_length):
                        episode_length = 0
                        state, _ = self.env.reset()
                else:
                    for i in range(self.num_envs):
                        if done[i] or (episode_length[i] >= self.max_episode_length):
                            episode_length[i] = 0

                # update the Q-network using a batch of transitions
                if (
                    (step % self.train_freq == 0)
                    and (self.buffer.real_size >= self.learning_starts)
                    and (self.buffer.real_size >= self.batch_size)
                ):
                    for _ in range(self.gradient_steps):
                        batch, weights, indices = self.buffer.sample(self.batch_size)
                        td_errors = self._optimize_model(batch, weights)
                        self.buffer.update_priorities(indices, td_errors)

                # update the target network at specified intervals
                if self.double_dqn and (step % self.target_update_freq == 0):
                    self._update_target_model()

                # update training progress
                self.steps_trained += 1
                pbar.update(1)

                # evaluate
                if eval_every and (steps_since_eval := steps_since_eval + 1) >= eval_every:
                    avg_reward = self.evaluate(eval_runs)
                    logger.info(f"Step: {self.steps_trained} | Average reward: {avg_reward}.")
                    steps_since_eval = 0

    def get_action(self, state: np.ndarray, epsilon: float | None = None) -> np.ndarray:
        """Select an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state of the environment.
            epsilon (float, optional): The exploration rate. Defaults to None.

        Returns:
            np.ndarray: The selected action(s).
        """
        if isinstance(epsilon, float) and np.random.rand() < epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = torch.as_tensor(np.array(state), device=self.device)
            q_values = self.q_network(state)
            return q_values.argmax(dim=-1).cpu().numpy()

    @property
    def epsilon(self) -> float:
        """The current exploration rate.

        Epsilon decays linearly from epsilon_start to epsilon_end over epsilon_decay steps.
        """
        return max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end) * self.steps_trained / self.epsilon_decay,
        )

    def _optimize_model(self, batch: tuple, weights: Tensor | None) -> Tensor | None:
        """Optimize the Q-network using a batch of transitions."""
        states, actions, rewards, next_states, dones = batch

        # compute Q-values for current states
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1).long())

        # compute Q-values for next states
        with torch.no_grad():
            if self.double_dqn:
                # compute the actions that would be taken by the Q-network and
                # use the target network to compute Q-values for the states
                next_actions = self.q_network(next_states).argmax(dim=1)
                target_q_values = self.target_network(next_states)
                target_q_values = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # use the target network to compute Q-values for the next states
                target_q_values = self.target_network(next_states).max(dim=1).values

        # calculate loss
        target = rewards + self.gamma * target_q_values * (1 - dones)
        if weights is not None:
            td_errors = (q_values - target.unsqueeze(1)).squeeze(1)
            loss = (weights * td_errors).pow(2).mean()
        else:
            loss = F.mse_loss(q_values, target.unsqueeze(1))

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return td_errors.abs().detach().cpu() if weights is not None else None

    def _update_target_model(self) -> None:
        """Update the target network using polyak averaging."""
        for policy_param, target_param in zip(
            self.q_network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def evaluate(self, eval_runs: int = 20, gif_path_format: str | None = None) -> float:
        """Evaluate the agent in the environment.

        Args:
            eval_runs (int): Number of evaluation runs. Defaults to 20.
            gif_path_format (str): The path format to save GIFs. Defaults to None.
                E.g. "path/to/gif" will save GIFs as "path/to/gif_0.gif", "path/to/gif_1.gif", etc.

        Returns:
            float: The average reward over the evaluation runs.
        """
        eval_env = gym.make(self.env_id, render_mode="rgb_array")
        if "ALE" in self.env_id or "NoFrameskip" in self.env_id:
            eval_env = wrap_atari(eval_env, eval_mode=True)

        rewards = []
        for i in range(eval_runs):
            gif_path = f"{gif_path_format}_{i}.gif" if gif_path_format else None
            rewards.append(self._evaluation_run(eval_env, gif_path))

        return np.mean(rewards)

    def _evaluation_run(self, env: gym.Env, gif_path: str | None = None) -> int | float:
        """Run a single evaluation episode in the environment."""
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        done = False
        frames = []
        while not done and (episode_length <= self.max_episode_length):
            action = self.get_action(state)
            state, reward, done, *_ = env.step(action)
            episode_reward += reward
            episode_length += 1

            if gif_path:
                frames.append(env.unwrapped.render())

        if gif_path:
            imageio.mimsave(gif_path, frames, fps=25)

        return episode_reward

    def save(self, path: str, include_buffer: bool = False) -> None:
        """Save the model to a file.

        Args:
            path (str): The path to save the model.
            include_buffer (bool, optional): Save the replay buffer. Defaults to False.
        """
        checkpoint = {
            "dqn_kwargs": {
                "env_id": self.env_id,
                "num_envs": self.num_envs,
                "vectorization_mode": self.vectorization_mode,
                "double_dqn": self.double_dqn,
                "dueling": self.dueling,
                "layers": self.layers,
                "buffer_class": self.buffer.__class__.__name__,
                "buffer_size": self.buffer_size,
                "buffer_kwargs": self.buffer_kwargs,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "train_freq": self.train_freq,
                "gradient_steps": self.gradient_steps,
                "lr": self.lr,
                "target_update_freq": self.target_update_freq,
                "tau": self.tau,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "learning_starts": self.learning_starts,
                "max_episode_length": self.max_episode_length,
                "max_grad_norm": self.max_grad_norm,
                "force_cpu": self.force_cpu,
            },
            "dqn_state": {
                "q_network": self.q_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "buffer": self.buffer.state_dict() if include_buffer else None,
                "steps_trained": self.steps_trained,
            },
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load(path: str) -> DQN:
        """Load a model from a file.

        Args:
            path (str): The path to load the model.

        Returns:
            DQN: The loaded model.
        """
        checkpoint = torch.load(path, weights_only=False)
        dqn = DQN(**checkpoint["dqn_kwargs"])

        dqn.q_network.load_state_dict(checkpoint["dqn_state"]["q_network"])
        dqn.target_network.load_state_dict(checkpoint["dqn_state"]["q_network"])
        dqn.optimizer.load_state_dict(checkpoint["dqn_state"]["optimizer"])
        dqn.steps_trained = checkpoint["dqn_state"]["steps_trained"]

        if checkpoint["dqn_state"]["buffer"] is None:
            logger.warning("Replay buffer not saved. Initializing a new buffer.")
            return dqn

        dqn.buffer.load_state_dict(checkpoint["dqn_state"]["buffer"])
        return dqn
