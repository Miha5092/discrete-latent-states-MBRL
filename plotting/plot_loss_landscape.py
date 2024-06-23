import argparse
import fire
import glob
import os
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filter_str: str = None) -> pd.DataFrame:
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
                        "epoch": np.arange(1, len(losses['total_losses']) + 1)
                    })

                    dfs.append(df)
        except:
            print(f"Passing over {training_name} due to error loading data.")

    result = pd.concat(dfs, ignore_index=True)
    return result


def plot(data: pd.DataFrame, output_name: str, plt_type: str = "grid"):
    sns.set_theme(style="white")
    sns.set_context("notebook")

    if plt_type == "grid":
        g = sns.FacetGrid(data, col="dyn weight", row="rep weight", hue="clip value", margin_titles=True)
        g.map(sns.lineplot, "epoch", "total loss")
        g.add_legend()

        for ax in g.axes.flat:
            ax.set_title(ax.get_title(), fontsize=10)
            ax.tick_params(axis='both', which='both', labelsize=10)

        g.set_titles(col_template="Dyn Weight: {col_name}", row_template="Rep Weight: {row_name}", size=14)
    elif plt_type == "lines":
        g = sns.relplot(
            data=data, x="epoch", y="total loss", col="clip value", col_wrap=2,
            style="dyn weight", hue="dyn weight", kind="line"
        )
    else:
        raise NotImplementedError(f"Cannot graph using plot style: {plt_type}")

    os.makedirs("plotting/plots/landscape", exist_ok=True)

    plt.savefig(f"plotting/plots/landscape/{output_name}.png")
    plt.savefig(f"plotting/plots/landscape/{output_name}.svg")
    plt.close()


def plot_loss_landscape(output_name: str, filter: str = None, plt_type: str = "grid") -> None:
    data = load_data(filter)
    plot(data, output_name, plt_type)


if __name__ == "__main__":
    fire.Fire(plot_loss_landscape)
