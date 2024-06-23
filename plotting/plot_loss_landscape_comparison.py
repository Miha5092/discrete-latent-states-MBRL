import fire
import glob
import os
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


def load_continuous_data(filter_str: str = None) -> pd.DataFrame:
    pattern = "data/model_learning/*_dyn_*_rep_*_seed_*"
    folder_list = glob.glob(pattern)
    trainings = [os.path.basename(folder) for folder in folder_list]

    if filter_str is not None:
        trainings = list(filter(lambda name: filter_str in name, trainings))

    extracting_pattern = re.compile(r'_dyn_([0-9.]+)_rep_([0-9.]+)_seed_(\d+)')
    dfs = []

    for training_name in trainings:
        try:
            with open(f"data/model_learning/{training_name}/losses.npz", "rb") as losses:
                match = extracting_pattern.search(training_name)

                if match:
                    losses = np.load(losses)
                    dyn_value = float(match.group(1))
                    rep_value = float(match.group(2))
                    seed = match.group(3)

                    df = pd.DataFrame({
                        "dyn weight": dyn_value,
                        "rep weight": rep_value,
                        "seed": seed,
                        "total loss": losses['total_losses'],
                        "reconstruction loss": losses['image_prediction_losses'],
                        "reward loss": losses['reward_prediction_losses'],
                        "done loss": losses['continuity_prediction_losses'],
                        "dynamic loss": losses['dynamics_losses'],
                        "representation loss": losses['representation_losses'],
                        "epoch": np.arange(1, len(losses['total_losses']) + 1),
                        "Weights": f"Dyn: {dyn_value} Rep: {rep_value}"
                    })

                    dfs.append(df)
        except:
            print(f"Passing over {training_name} due to error loading data.")

    result = pd.concat(dfs, ignore_index=True)
    return result


def load_discrete_data(filter_str: str = None) -> pd.DataFrame:
    pattern = "data/model_learning/*_dyn_*_rep_*_clip_*_seed_*"
    folder_list = glob.glob(pattern)
    trainings = [os.path.basename(folder) for folder in folder_list]

    if filter_str is not None:
        trainings = list(filter(lambda name: filter_str in name, trainings))

    extracting_pattern = re.compile(r'_dyn_([0-9.]+)_rep_([0-9.]+)_clip_([0-9.]+)_seed_(\d+)')
    dfs = []

    for training_name in trainings:
        try:
            with open(f"data/model_learning/{training_name}/losses.npz", "rb") as losses:
                match = extracting_pattern.search(training_name)

                if match:
                    losses = np.load(losses)
                    dyn_value = float(match.group(1))
                    rep_value = float(match.group(2))
                    clip_value = float(match.group(3))
                    seed = match.group(4)

                    df = pd.DataFrame({
                        "dyn weight": dyn_value,
                        "rep weight": rep_value,
                        "clip value": clip_value,
                        "seed": seed,
                        "total loss": losses['total_losses'],
                        "reconstruction loss": losses['image_prediction_losses'],
                        "reward loss": losses['reward_prediction_losses'],
                        "done loss": losses['continuity_prediction_losses'],
                        "dynamic loss": losses['dynamics_losses'],
                        "representation loss": losses['representation_losses'],
                        "epoch": np.arange(1, len(losses['total_losses']) + 1),
                        "Weights": f"Dyn: {dyn_value} Rep: {rep_value}"
                    })

                    dfs.append(df)
        except:
            print(f"Passing over {training_name} due to error loading data.")

    result = pd.concat(dfs, ignore_index=True)
    return result


def all_continuous_data(weights: list[tuple]) -> [pd.DataFrame]:
    data_10k = load_continuous_data("deterministic_10")
    data_10k["Dataset"] = "10k"

    dfs = []

    for (dyn, rep) in weights:
        df = data_10k[(data_10k["dyn weight"] == dyn) & (data_10k["rep weight"] == rep)]
        dfs.append(df)

    data_10k = pd.concat(dfs, ignore_index=True)

    data_30k = load_discrete_data("deterministic_30_dyn")
    data_30k["Dataset"] = "30k"

    for (dyn, rep) in weights:
        df = data_30k[(data_30k["dyn weight"] == dyn) & (data_30k["rep weight"] == rep)]
        dfs.append(df)

    data_30k = pd.concat(dfs, ignore_index=True)

    return data_10k, data_30k


def all_discrete_data(weights: list[tuple]) -> [pd.DataFrame]:
    data_10k = load_discrete_data("discrete_dyn")
    data_10k["Dataset"] = "10k"

    dfs = []

    for (dyn, rep) in weights:
        df = data_10k[(data_10k["dyn weight"] == dyn) & (data_10k["rep weight"] == rep)]
        dfs.append(df)

    data_10k = pd.concat(dfs, ignore_index=True)

    data_30k = load_discrete_data("discrete_30")
    data_30k["Dataset"] = "30k"

    for (dyn, rep) in weights:
        df = data_30k[(data_30k["dyn weight"] == dyn) & (data_30k["rep weight"] == rep)]
        dfs.append(df)

    data_30k = pd.concat(dfs, ignore_index=True)

    return data_10k, data_30k


def plot_comparisons(plot_name: str, data_10k: pd.DataFrame, data_30k: pd.DataFrame, data_type: str) -> None:
    sns.set_theme(style="white")
    sns.set_context("notebook")
    sns.set_palette("colorblind")

    data = pd.concat([data_10k, data_30k], ignore_index=True)

    if data_type == "continuous":
        g = sns.FacetGrid(data, col="Weights", row="Dataset", margin_titles=True)
        g.map(sns.lineplot, "epoch", "total loss")
        g.add_legend()

        for ax in g.axes.flat:
            ax.set_title(ax.get_title(), fontsize=10)
            ax.tick_params(axis='both', which='both', labelsize=10)

        plt.tight_layout()

        plt.savefig(f"plotting/plots/landscape/{plot_name}_continuous.png")
        plt.close()
    elif data_type == "discrete":
        g = sns.FacetGrid(data, col="Weights", row="Dataset", hue="clip value", margin_titles=True)
        g.map(sns.lineplot, "epoch", "total loss")
        g.add_legend()

        for ax in g.axes.flat:
            ax.set_title(ax.get_title(), fontsize=10)
            ax.tick_params(axis='both', which='both', labelsize=10)

        plt.tight_layout()

        g.fig.subplots_adjust(bottom=0.18)
        sns.move_legend(g, "lower center", ncol=4, frameon=False)

        plt.savefig(f"plotting/plots/landscape/{plot_name}_discrete.png")
        plt.close()
    else:
        raise ValueError(f"Data type {data_type} is not supported.")


def plot(plot_name: str) -> None:
    # weights = [(0.8, 0.1), (0.8, 0.05), (1.0, 0.1), (0.9, 0.2)]
    weights = [(0.8, 0.05), (1.0, 0.1)]
    os.makedirs("plotting/plots/landscape", exist_ok=True)

    data_10k, data_30k = all_discrete_data(weights)
    plot_comparisons(plot_name, data_10k, data_30k, "discrete")

    data_10k, data_30k = all_continuous_data(weights)
    plot_comparisons(plot_name, data_10k, data_30k, "continuous")


if __name__ == "__main__":
    fire.Fire(plot)