import seaborn as sns
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# visualize the state
def render_state(state):
    n_channels = 4
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels + 2)]
    norm = colors.BoundaryNorm(bounds, n_channels + 1)
    _, ax = plt.subplots(1, 1)
    numerical_state = np.amax(state * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5
    ax.imshow(numerical_state, cmap=cmap, norm=norm, interpolation='none')
    plt.show()


# visualize the state and its reconstruction
def render_state_reconstruction_window(states, reconstructions, window_size):
    n_channels = 4
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels + 2)]
    norm = colors.BoundaryNorm(bounds, n_channels + 1)
    numerical_states = np.amax(states * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 3) + 0.5
    numerical_reconstructions = np.amax(reconstructions * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 3) + 0.5
    _, ax = plt.subplots(2, window_size)
    for i in range(window_size):
        ax[0, i].imshow(
            numerical_states[i], cmap=cmap, norm=norm, interpolation='none')
        ax[1, i].imshow(
            numerical_reconstructions[i], cmap=cmap, norm=norm, interpolation='none')
    plt.show()


def compute_KL_divergence_between_two_independent_gaussians(mu1, log_std1, mu2, log_std2, aggregate_method):
    """
    Compute the KL divergence between two independent gaussians
    """
    std1 = torch.exp(log_std1)
    std2 = torch.exp(log_std2)
    kl = log_std2 - log_std1 + (std1 ** 2 + (mu1 - mu2) ** 2) / (2 * std2 ** 2) - 0.5

    if aggregate_method == 'mean':
        normalized_kl = kl.mean(dim=-1)
    elif aggregate_method == 'sum':
        normalized_kl = kl.sum(dim=-1)
    else:
        raise ValueError('Aggregate method must be "mean" or "sum".')

    return normalized_kl


def compute_KL_divergence_between_two_independent_categoricals(logits1, logits2, aggregate_method, eps=1e-20):
    """
    Compute the KL divergence between two independent categoricals
    """
    p1 = torch.nn.functional.softmax(logits1, dim=-1) + eps
    p2 = torch.nn.functional.softmax(logits2, dim=-1) + eps
    kl = (p1 * (torch.log(p1) - torch.log(p2))).sum(dim=-1)

    if aggregate_method == 'mean':
        normalized_kl = kl.mean(dim=-1)
    elif aggregate_method == 'sum':
        normalized_kl = kl.sum(dim=-1)
    else:
        raise ValueError('Aggregate method must be "mean" or "sum".')

    return normalized_kl


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape, device=device)

    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, eps=1e-20):
    y = logits + sample_gumbel(logits.size(), eps)

    return torch.nn.functional.softmax(y / temperature, dim=-1)


def rsample_categorical(logits, temperature=1.0, eps=1e-20):
    y = gumbel_softmax_sample(logits, temperature, eps)
    _, k = y.max(dim=-1)
    y_hard = torch.zeros_like(logits, device=device)
    y_hard.scatter_(-1, k.view(k.shape[0], k.shape[1], 1), 1)
    y = (y_hard - y).detach() + y

    return y


def rsample_gaussian(mu, log_std, eps=1e-20):
    return mu + torch.randn_like(mu) * torch.exp(log_std)


def get_gaussian_distribution(mean, log_std, eps=1e-20):
    return torch.distributions.multivariate_normal.MultivariateNormal(mean, torch.diag_embed(torch.exp(log_std) + eps))
