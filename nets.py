import torch
import torch.nn as nn

from utils import get_gaussian_distribution


def get_onehot_categorical_distribution(logits):
    dist = torch.distributions.OneHotCategorical(logits=logits)
    return dist


def get_dist(x, latent_state_type, latent_state_shape, min_std=0.1, max_std=2.0):

    if latent_state_type == 'deterministic':
        return x
    elif latent_state_type == 'gaussian':
        return x[:, :x.shape[1] // 2], torch.clip(x[:, x.shape[1] // 2:], min=min_std, max=max_std)
    elif latent_state_type == 'discrete':
        x = x.reshape(x.shape[0], *latent_state_shape)

        return x


class RepresentationNetwork(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 latent_state_type,
                 latent_state_shape,
                 epsilon=1e-6):
        super().__init__()

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.latent_state_type = latent_state_type
        self.latent_state_shape = latent_state_shape
        self.epsilon = epsilon

        if latent_state_type == 'deterministic':
            assert len(latent_state_shape) == 1

            out_channels = latent_state_shape[0]
        elif latent_state_type == "gaussian":
            assert len(latent_state_shape) == 1

            out_channels = 2 * latent_state_shape[0]
        elif latent_state_type == "discrete":
            assert len(latent_state_shape) == 2

            num_categorical_variables = latent_state_shape[0]
            num_categorical_values = latent_state_shape[1]

            out_channels = num_categorical_variables * num_categorical_values
        else:
            raise ValueError(f"Latent state type: {latent_state_type} can only be deterministic, gaussian, or discrete")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(in_features=num_linear_units, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=out_channels)
        )

    def forward(self, x):
        x = self.net(x)
        return get_dist(x, self.latent_state_type, self.latent_state_shape)

    def sample(self, x):
        if self.latent_state_type == 'deterministic':
            return self.forward(x)
        else:
            dist = self.forward(x)

            if self.latent_state_type == "gaussian":
                return get_gaussian_distribution(dist[0], dist[1]).sample().flatten(1)
            elif self.latent_state_type == "discrete":
                return get_onehot_categorical_distribution(dist).sample().flatten(1)
            else:
                raise ValueError(f"Latent state type: {self.latent_state_type} can only be deterministic, gaussian, or discrete")


class DynamicsNetwork(nn.Module):

    def __init__(self,
                 hidden_channels,
                 latent_state_type,
                 latent_state_shape,
                 epsilon=1e-16):
        super().__init__()

        self.latent_state_type = latent_state_type
        self.latent_state_shape = latent_state_shape
        self.epsilon = epsilon

        latent_state_dim = 1
        for x in latent_state_shape:
            latent_state_dim *= x

        self.reward_net = nn.Sequential(
            nn.Linear(in_features=latent_state_dim, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=1),
            nn.Flatten(0)
        )
        self.done_net = nn.Sequential(
            nn.Linear(in_features=latent_state_dim, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=1),
            nn.Flatten(0)
        )

        self.embedding_net = nn.Sequential(
            nn.Linear(in_features=latent_state_dim + 1, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=latent_state_dim)
        )

        if latent_state_type == "gaussian":
            assert len(latent_state_shape) == 1

            out_channels = 2 * latent_state_shape[0]
        elif latent_state_type == "discrete":
            assert len(latent_state_shape) == 2

            num_categorical_variables = latent_state_shape[0]
            num_categorical_values = latent_state_shape[1]

            out_channels = num_categorical_variables * num_categorical_values
        elif latent_state_type == "deterministic":
            assert len(latent_state_shape) == 1

            out_channels = latent_state_shape[0]
        else:
            raise ValueError(f"Latent state type: {latent_state_type} can only be deterministic, gaussian, or discrete")

        self.dynamics_net = nn.Sequential(
            nn.Linear(in_features=latent_state_dim + 1, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            nn.SiLU(),
            nn.Linear(in_features=hidden_channels, out_features=out_channels)
        )

    def forward(self, x):
        x = self.dynamics_net(x)
        return get_dist(x, self.latent_state_type, self.latent_state_shape)

    def predict(self, embedding):
        reward = self.reward_net(embedding)
        done = self.done_net(embedding)
        return reward, nn.Sigmoid()(done)

    def sample(self, x):
        if self.latent_state_type == 'deterministic':
            z = self.forward(x)
        else:
            dist = self.forward(x)

            if self.latent_state_type == "gaussian":
                z = get_gaussian_distribution(dist[0], dist[1]).sample().flatten(1)
            elif self.latent_state_type == "discrete":
                z = get_onehot_categorical_distribution(dist).sample().flatten(1)
            else:
                raise ValueError(f"Latent state type: {self.latent_state_type} can only be deterministic, gaussian, or discrete")

        embedding = self.embedding_net(x)
        reward, done = self.predict(embedding)
        return z, reward, done


class ReconstructionNetwork(nn.Module):

    def __init__(self, in_channels=128, num_channels=64):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.Unflatten(1, (in_channels, 1, 1)),
            # First deconvolution: 128x1x1 to 32x2x2
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # Second deconvolution: 32x2x2 to 32x5x5
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # Third deconvolution: 32x5x5 to 4x10x10
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=4, kernel_size=6, stride=2, padding=1),
        )

    def forward(self, x):
        return self.deconv_layers(x)
