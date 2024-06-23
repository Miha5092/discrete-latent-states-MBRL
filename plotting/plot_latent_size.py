import glob
import os
import fire
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def load_data_discrete(filter_str: str = None) -> pd.DataFrame:
    pattern = "data/model_evaluation/*_dim_*_size_*_seed_*"
    folder_list = glob.glob(pattern)
    trainings = [os.path.basename(folder) for folder in folder_list]

    if filter_str is not None:
        trainings = list(filter(lambda name: filter_str in name, trainings))

    extracting_pattern = re.compile(r'_dim_(\d+)_size_(\d+)_seed_(\d+)')
    dfs = []

    for training_name in trainings:
        with open(f"data/model_evaluation/{training_name}/episode_returns.npy", "rb") as episode_returns:
            match = extracting_pattern.search(training_name)

            if match:
                episode_returns = np.load(episode_returns)
                dim = int(match.group(1))
                size = int(match.group(2))
                seed = int(match.group(3))

                df = pd.DataFrame({
                    "no dim": dim,
                    "dim size": size,
                    "Latent Size": dim * size,
                    "seed": seed,
                    "sparsity": (dim * size - dim) / (dim * size),
                    "episode returns": episode_returns,
                    "Model Type": "Discrete"
                })

                dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).round(decimals=3)
    return result


def load_data_continuous(filter_str: str = None) -> pd.DataFrame:
    pattern = "data/model_evaluation/*_dim_*_seed_*"
    folder_list = glob.glob(pattern)
    trainings = [os.path.basename(folder) for folder in folder_list]

    if filter_str is not None:
        trainings = list(filter(lambda name: filter_str in name, trainings))

    extracting_pattern = re.compile(r'_dim_(\d+)_seed_(\d+)')
    dfs = []

    for training_name in trainings:
        with open(f"data/model_evaluation/{training_name}/episode_returns.npy", "rb") as episode_returns:
            match = extracting_pattern.search(training_name)

            if match:
                episode_returns = np.load(episode_returns)
                dim = int(match.group(1))
                seed = int(match.group(2))

                df = pd.DataFrame({
                    "Latent Size": dim,
                    "seed": seed,
                    "episode returns": episode_returns,
                    "Model Type": "Continuous"
                })

                dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).round(decimals=3)
    return result


def load_data(filter_str: str = None) -> pd.DataFrame:
    deterministic_data = load_data_continuous(filter_str=filter_str)
    discrete_data = load_data_discrete(filter_str=filter_str)

    data = pd.concat([deterministic_data, discrete_data], ignore_index=True)
    return data


def plot_latent_size(df: pd.DataFrame, output_name: str):
    sns.set_theme(style="white")
    sns.set_context("notebook")

    sns.lineplot(data=df, x="Latent Size", y="episode returns", hue="Model Type", errorbar="se")

    plt.tight_layout()
    plt.savefig(f"plotting/plots/latent_size/{output_name}.png")
    plt.savefig(f"plotting/plots/latent_size/{output_name}.svg")
    plt.close()


def analyze_sparcity(output_name: str, filter_str: str = None):
    sns.set_context(font_scale=200)

    data = load_data(filter_str)

    os.makedirs("plotting/plots/latent_size", exist_ok=True)

    plot_latent_size(data, output_name)


if __name__ == '__main__':
    fire.Fire(analyze_sparcity)
