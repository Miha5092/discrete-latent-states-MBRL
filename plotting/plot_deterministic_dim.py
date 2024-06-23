import glob
import os
import fire
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filter_str: str = None) -> pd.DataFrame:
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
                    "No. Dimensions": dim,
                    "seed": seed,
                    "Episode Returns": episode_returns
                })

                dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).round(decimals=3)
    return result


def plot_results(df: pd.DataFrame, output_name: str):
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook")

    sns.boxplot(data=df, x="No. Dimensions", y="Episode Returns",
                showmeans=True,
                meanprops={
                    "marker": "+",
                    "markeredgecolor": "red",
                    "markersize": "10"
                })

    plt.tight_layout()
    plt.savefig(f"plotting/plots/{output_name}.png")
    plt.close()


def analyze_sparcity(output_name: str, filter_str: str = None):
    sns.set_context(font_scale=200)

    data = load_data(filter_str)

    plot_results(data, output_name)


if __name__ == '__main__':
    fire.Fire(analyze_sparcity)
