import os

import numpy as np
import pandas
import pandas as pd
import fire

import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(blocks, data_save_location: str):
    block_counts = np.sum(blocks, axis=(1, 2))
    destroyed_block_counts = 30 - block_counts

    data = pd.DataFrame(destroyed_block_counts, columns=["No. Destroyed Blocks"])

    plt.figure(figsize=(8, 6))
    # plt.hist(data["No. Destroyed Blocks"], bins=np.arange(32) - 0.5, edgecolor='black', linewidth=1.1, color='skyblue', rwidth=1.0, density=True)
    sns.histplot(data, x="No. Destroyed Blocks", bins=np.arange(32) - 0.5, stat="density")

    xticks = range(0, 31, 5)  # Only multiples of 5
    plt.xticks(xticks, fontsize=14)  # Set the fontsize for xticks

    plt.xlabel("Number of Destroyed Blocks", fontsize=16)
    plt.ylabel("Fraction of Dataset", fontsize=16)

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{data_save_location}/distribution.png")
    plt.savefig(f"{data_save_location}/distribution.svg")
    plt.close()


def plot_heatmap_blocks(blocks, data_save_location: str):
    counts = np.sum(blocks, axis=0)

    df = pd.DataFrame(counts / blocks.shape[0])

    annotations = pd.DataFrame(counts / blocks.shape[0])
    annotations = annotations.round(decimals=3).astype(str)
    annotations[annotations == "0.0"] = "0"

    plt.figure(figsize=(8, 6))
    g = sns.heatmap(df, annot=annotations, fmt="s", cmap="viridis", annot_kws={"size": 12})
    g.set_xticks([])
    g.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{data_save_location}/blocks_lifetime.png")
    plt.savefig(f"{data_save_location}/blocks_lifetime.svg")
    plt.close()


def plot_heatmap_ball(ball, data_save_location: str):
    counts = np.sum(ball, axis=0)

    df = pd.DataFrame(counts / ball.shape[0])

    annotations = pd.DataFrame(counts / ball.shape[0])
    annotations = annotations.round(decimals=3).astype(str)
    annotations[annotations == "0.0"] = "0"

    plt.figure(figsize=(8, 6))
    g = sns.heatmap(df, annot=annotations, fmt="s", cmap="viridis", annot_kws={"size": 12})
    g.set_xticks([])
    g.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{data_save_location}/ball_locations.png")
    plt.savefig(f"{data_save_location}/ball_locations.svg")
    plt.close()


def plot_heatmap_paddle(paddle, data_save_location: str):
    counts = np.sum(paddle, axis=0)

    df = pd.DataFrame(counts / paddle.shape[0])

    annotations = pd.DataFrame(counts / paddle.shape[0])
    annotations = annotations.round(decimals=3).astype(str)
    annotations[annotations == "0.0"] = "0"

    plt.figure(figsize=(8, 6))
    g = sns.heatmap(df, annot=annotations, fmt="s", cmap="viridis", annot_kws={"size": 12})
    g.set_xticks([])
    g.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{data_save_location}/paddle_locations.png")
    plt.savefig(f"{data_save_location}/paddle_locations.svg")
    plt.close()


def plot_data(
        data_path: str = "data/dqn_training_seed_0/experience_data_episode_10000.npz"
    ):

    location, _, data_packet = data_path.rpartition('/')
    data_packet, _, _ = data_packet.rpartition('.')
    data_save_location = f"{location}/{data_packet}"
    os.makedirs(data_save_location, exist_ok=True)

    trajectory_data = np.load(data_path)
    states, actions, rewards, terminals = trajectory_data['states'], trajectory_data['actions'], trajectory_data['rewards'], trajectory_data['terminals']

    states = np.transpose(states, [0, 3, 1, 2])

    blocks = states[:, 3, :, :]
    ball = states[:, 1, :, :]
    paddle = states[:, 0, :, :]

    sns.set_context("talk")

    plot_distributions(blocks, data_save_location)
    plot_heatmap_blocks(blocks, data_save_location)
    plot_heatmap_ball(ball, data_save_location)
    plot_heatmap_paddle(paddle, data_save_location)

    with open(f"{data_save_location}/log.txt", "w") as f:
        unique_actions, counts_actions = np.unique(actions, return_counts=True)
        action_count_dict = dict(zip(unique_actions, counts_actions))

        f.write(f"Percentage of \"do nothing\":     {action_count_dict[0] / len(actions) * 100:.2f}%\n")
        f.write(f"Percentage of \"move left\":      {action_count_dict[1] / len(actions) * 100:.2f}%\n")
        f.write(f"Percentage of \"move right\":     {action_count_dict[2] / len(actions) * 100:.2f}%\n\n")

        unique_terminals, counts_terminals = np.unique(terminals, return_counts=True)
        terminals_count_dict = dict(zip(unique_terminals, counts_terminals))

        f.write(f"Percentage of \"not done\":       {terminals_count_dict[0] / len(actions) * 100:.2f}%\n")
        f.write(f"Percentage of \"done\":           {terminals_count_dict[1] / len(actions) * 100:.2f}%\n\n")


if __name__ == '__main__':
    fire.Fire(plot_data)
