import torch
import torch.nn as nn
from torch import Tensor


class NatureCNN(nn.Module):
    """Convolutional neural network (CNN) from the Nature paper."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.cnn(x)


def _he_initialization(m: nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        m.bias.data.fill_(0.0)


def build_mlp(n_in: int, n_out: int, hidden_layers: list[int]) -> nn.Module:
    """Build a multi-layer perceptron (MLP) with optional layer normalization.

    Args:
        n_in (int): Number of input units.
        n_out (int): Number of output units.
        hidden_layers (list[int]): Number of units in each hidden layer.

    Returns:
        nn.Module: MLP model.
    """
    if len(hidden_layers) == 0:
        return nn.Sequential(nn.Linear(n_in, n_out))

    layers = [nn.Linear(n_in, hidden_layers[0]), nn.ReLU()]
    for i in range(1, len(hidden_layers)):
        layers.extend([nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.ReLU()])
    layers.append(nn.Linear(hidden_layers[-1], n_out))

    return nn.Sequential(*layers).apply(_he_initialization)


class Policy(nn.Module):
    """Deep Q-Network (DQN) policy network.

    Similar to Stable Baselines3, we separate the feature extractor from the action network. This
    allows us to use different architectures for the feature extractor such as a CNN for image-based
    observations and an MLP for vector-based observations.

    Args:
        observation_shape (list[int]): Shape of the observation space.
        n_actions (int): Number of actions in the environment.
        dueling (bool): Whether to use a dueling architecture. Default is False.
        layers (list[int]): Number of units in each hidden layer. Default is [64, 64].
    """

    def __init__(
        self,
        observation_shape: list[int],
        n_actions: int,
        dueling: bool = False,
        layers: list[int] = [64, 64],
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Identity()
        feature_dim = observation_shape[-1]

        if len(observation_shape) == 3:  # image (channels, height, width)
            obs = torch.zeros(1, *observation_shape)
            self.feature_extractor = NatureCNN(channels=obs.shape[1])
            feature_dim = self.feature_extractor(obs).shape[-1]

        self.action_net = build_mlp(feature_dim, n_actions, layers)
        if dueling:
            self.value_net = build_mlp(feature_dim, 1, layers)

    def forward(self, x: Tensor) -> Tensor:
        is_single_img: bool = isinstance(self.feature_extractor, NatureCNN) and len(x.shape) == 3

        features = self.feature_extractor(x.unsqueeze(0) if is_single_img else x)
        action_values = self.action_net(features)

        if hasattr(self, "value_net"):
            state_values = self.value_net(features)
            advantage = state_values + action_values - action_values.mean(dim=-1, keepdim=True)
            return advantage[0] if is_single_img else advantage

        return action_values[0] if is_single_img else action_values
