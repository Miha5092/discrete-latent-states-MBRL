import glob
import os
import fire
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def compute_gini(prop_ones: float) -> float:
    prop_zeroes = 1 - prop_ones
    gini = 1 - (prop_ones ** 2 + prop_zeroes ** 2)
    return gini


def load_data(filter_str: str = None) -> pd.DataFrame:
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
                    "GINI": compute_gini(1 / size),
                    "episode returns": episode_returns,
                    "episode means": np.mean(episode_returns)
                })

                dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).round(decimals=3)
    return result


def plot_sparcity_(df: pd.DataFrame, output_name: str):
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook")

    # Create the catplot
    g = sns.catplot(
        data=df, x="no dim", y="episode returns",
        hue="no dim", kind="bar", col="sparsity",
        errorbar="se",
        palette="crest",
        legend=False,
        height=5,
        aspect=0.9
    )

    # Iterate over each axes to modify titles and labels
    for ax in g.axes.flat:
        sparsity = ax.get_title().split(' = ')[1]
        dim_size = df[df['sparsity'] == float(sparsity)]['dim size'].unique()[0]
        new_title = f"Sparsity: {sparsity}, Cat. Size: {dim_size}"
        ax.set_title(new_title, fontsize=15)
        ax.tick_params(axis='both', which='both', labelsize=15)

    # Set common labels
    g.set_xlabels("No. Categoricals", fontsize=15)
    g.set_ylabels("Episode Returns", fontsize=15)

    plt.tight_layout()
    plt.savefig(f"plotting/plots/sparsity/{output_name}_bar.png")
    plt.savefig(f"plotting/plots/sparsity/{output_name}_bar.svg")
    plt.close()


def plot_sparsity_heatmap(df: pd.DataFrame, output_name: str):
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook")

    df_mean = df[['no dim', 'dim size', 'episode returns']]
    df_mean = df.groupby(['no dim', 'dim size'], as_index=False).mean()

    color = sns.diverging_palette(20, 220, as_cmap=True)

    g = sns.heatmap(data=df_mean.pivot(index="no dim", columns="dim size", values="episode returns"), annot=True, fmt=".2f", cmap=color)

    g.set_xlabel("Size of the Categorical Distributions", fontsize=12)
    g.set_ylabel("Number of Categorical Distributions", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"plotting/plots/sparsity/{output_name}_heatmap.png")
    plt.savefig(f"plotting/plots/sparsity/{output_name}_heatmap.svg")
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_name: str):
    sns.set_theme(style="white")
    sns.set_context("notebook")

    trimed_data = df[["no dim", "sparsity", "episode returns", "Latent Size"]]

    corr = trimed_data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(20, 230, as_cmap=True)

    g = sns.heatmap(corr, cmap=cmap, mask=mask, annot=True)

    # g.set_xlabels(fontsize=15)
    # g.set_ylabels(fontsize=15)

    # g.set_axis_labels("No. Categoricals", "Episode Returns")

    plt.tight_layout()
    plt.savefig(f"plotting/plots/sparsity/{output_name}_corr.png")
    plt.savefig(f"plotting/plots/sparsity/{output_name}_corr.svg")
    plt.close()


def analyze_sparcity(output_name: str, filter_str: str = None):
    sns.set_context(font_scale=200)

    data = load_data(filter_str)

    os.makedirs("plotting/plots/sparsity", exist_ok=True)

    plot_correlation_matrix(data, output_name)
    plot_sparcity_(data, output_name)
    plot_sparsity_heatmap(data, output_name)


if __name__ == '__main__':
    fire.Fire(analyze_sparcity)
