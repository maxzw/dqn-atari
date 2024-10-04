from __future__ import annotations

import sys
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor

from dqn_atari.buffer.tree import SumTree


class ReplayBuffer(ABC):
    """Abstract base class for replay buffers."""

    def __init__(
        self, buffer_size: int, state_shape: tuple[int], num_envs: int = 1, device: str = "cpu"
    ) -> None:
        self.size = buffer_size
        self.state_shape = state_shape
        self.num_envs = num_envs
        self.device = device

        # set counters
        self.size = buffer_size
        self.count = 0
        self.real_size = 0

    def add(self, transition: tuple) -> None:
        """Add a transition to the buffer.

        Args:
            transition (tuple): The transition to add to the buffer. Is a tuple of (state, action,
                reward, next_state, done). If multiple environments are used, each element of the
                tuple should be an array with the first dimension corresponding to the number of
                environments.
        """
        state, action, reward, next_state, done = transition
        assert (
            state.shape == (self.num_envs, *self.state_shape)
            if self.num_envs > 1
            else self.state_shape
        ), f"State shape is {state.shape}, expected {(self.num_envs, *self.state_shape)}"

        if self.num_envs > 1:
            for i in range(self.num_envs):
                self._add_single_transition(
                    (state[i], action[i], reward[i], next_state[i], done[i])
                )
            return

        self._add_single_transition(transition)

    @abstractmethod
    def _add_single_transition(self, transition: tuple) -> None:
        """Add a single transition to the buffer.

        Args:
            transition (tuple): The transition to add to the buffer. Is a tuple of (state, action,
                reward, next_state, done).
        """
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple: A batch of transitions, the importance-sampling weights, and the indices of the
                sampled transitions in the tree.
        """
        pass

    def update_priorities(self, indices: list[int], priorities: Tensor) -> None:
        """Update the priorities of the sampled transitions. Is no-op for uniform replay buffer.

        Args:
            tree_idxs (list): The indices of the sampled transitions in the tree.
            priorities (Tensor): The updated priorities of the transitions (e.g. TD errors).
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        """Return the state of the buffer."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> ReplayBuffer:
        """Load the state of the buffer."""
        pass


class UniformReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples.

    The ReplayBuffer stores a fixed-size buffer of experience tuples, which can be sampled uniformly
    to train a reinforcement learning agent. The buffer stores transitions consisting of the current
    state, action, reward, next state, and done flag. The transitions can be sampled in batches to
    train the agent using mini-batch gradient descent.

    Args:
        buffer_size (int): The maximum number of transitions that the buffer can store.
        state_shape (tuple): The shape of the state space.
        num_envs (int): The number of environments used to collect transitions. Default is 1.
        device (str): The device to store the buffer on (e.g. "cpu" or "cuda"). Default is "cpu".
    """

    def __init__(
        self, buffer_size: int, state_shape: tuple[int], num_envs: int = 1, device: str = "cpu"
    ) -> None:
        super().__init__(buffer_size, state_shape, num_envs, device)

        # initialize buffer
        self.state = torch.empty(buffer_size, *state_shape, dtype=torch.float)
        self.action = torch.empty(buffer_size, dtype=torch.int)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, *state_shape, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

    def _add_single_transition(self, transition: tuple) -> None:
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(np.array(state))
        self.action[self.count] = torch.as_tensor(np.array(action))
        self.reward[self.count] = torch.as_tensor(np.array(reward))
        self.next_state[self.count] = torch.as_tensor(np.array(next_state))
        self.done[self.count] = torch.as_tensor(np.array(done))

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.real_size + 1, self.size)

    def sample(self, batch_size: int) -> tuple[tuple, None, None]:
        idx = torch.randint(0, self.real_size, (batch_size,))
        batch = (
            self.state[idx].to(self.device),
            self.action[idx].to(self.device),
            self.reward[idx].to(self.device),
            self.next_state[idx].to(self.device),
            self.done[idx].to(self.device),
        )
        return batch, None, None

    def state_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "count": self.count,
            "real_size": self.real_size,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> UniformReplayBuffer:
        self.state = state_dict["state"]
        self.action = state_dict["action"]
        self.reward = state_dict["reward"]
        self.next_state = state_dict["next_state"]
        self.done = state_dict["done"]
        self.count = state_dict["count"]
        self.real_size = state_dict["real_size"]


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay buffer.

    The Prioritized Experience Replay (PER) buffer is a variant of the standard replay buffer that
    prioritizes transitions based on their TD error. The transitions with the highest TD error are
    sampled more frequently, thus improving the data efficiency of the learning algorithm. The PER
    buffer is implemented using a SumTree data structure, which allows for efficient sampling and
    updating of priorities.

    Args:
        buffer_size (int): The maximum number of transitions that the buffer can store.
        state_shape (tuple): The shape of the state space.
        num_envs (int): The number of environments used to collect transitions.
        device (str): The device to store the buffer on (e.g. "cpu" or "cuda"). Default is "cpu".
        eps (float): A small constant that prevents zero probabilities. Default is 1e-6.
        alpha (float): Determines how much prioritization is used. Default is 0.6.
        beta (float): Determines the amount of importance-sampling correction. Default is 0.4.
    """

    def __init__(
        self,
        buffer_size: int,
        state_shape: tuple[int],
        num_envs: int = 1,
        device: str = "cpu",
        eps: float = 1e-6,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        super().__init__(buffer_size, state_shape, num_envs, device)
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps

        # initialize buffer
        self.state = torch.empty(buffer_size, *state_shape, dtype=torch.float)
        self.action = torch.empty(buffer_size, dtype=torch.int)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, *state_shape, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

    def _add_single_transition(self, transition: tuple) -> None:
        state, action, reward, next_state, done = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(np.array(state))
        self.action[self.count] = torch.as_tensor(np.array(action))
        self.reward[self.count] = torch.as_tensor(np.array(reward))
        self.next_state[self.count] = torch.as_tensor(np.array(next_state))
        self.done[self.count] = torch.as_tensor(np.array(done))

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.real_size + 1, self.size)

    def sample(self, batch_size: int) -> tuple[tuple, Tensor, list[int]]:
        assert self.real_size >= batch_size, "Buffer size is smaller than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = (torch.rand(1) * (b - a) + a).clip(0, self.tree.total).item()
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = torch.tensor(priority)
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()
        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
        )
        return batch, weights.to(self.device), tree_idxs

    def update_priorities(self, tree_idxs: list[int], priorities: Tensor) -> None:
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().numpy()
        for data_idx, priority in zip(tree_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def state_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "count": self.count,
            "real_size": self.real_size,
            "tree_nodes": self.tree.nodes,
            "tree_data": self.tree.data,
            "tree_count": self.tree.count,
            "tree_real_size": self.real_size,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> PrioritizedReplayBuffer:
        self.state = state_dict["state"]
        self.action = state_dict["action"]
        self.reward = state_dict["reward"]
        self.next_state = state_dict["next_state"]
        self.done = state_dict["done"]
        self.count = state_dict["count"]
        self.real_size = state_dict["real_size"]
        self.tree.nodes = state_dict["tree_nodes"]
        self.tree.data = state_dict["tree_data"]
        self.tree.count = state_dict["tree_count"]
        self.tree.real_size = state_dict["tree_real_size"]


def get_buffer_class(name: str) -> type[ReplayBuffer]:
    """Return the buffer class based on the name."""
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError:
        raise ValueError(f"Invalid buffer class: {name}")
