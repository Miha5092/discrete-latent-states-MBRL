import glob
import os
import re

import fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filter_str: str = None) -> pd.DataFrame:
    pattern = "data/model_evaluation/optimal_*_*"

    folder_list = glob.glob(pattern)
    evaluation_names = [os.path.basename(folder) for folder in folder_list]

    if filter_str is not None:
        evaluation_names = list(filter(lambda name: filter_str in name, evaluation_names))

    extracting_pattern = re.compile(r"optimal_([^_]+)_.*_seed_(\d+)")

    dfs = []

    for evaluation_name in evaluation_names:
        try:
            match = extracting_pattern.search(evaluation_name)

            if match:
                rewards_array = np.load(f"data/model_evaluation/{evaluation_name}/episode_returns.npy")

                model_type = match.group(1)
                seed = int(match.group(2))

                df = pd.DataFrame({
                    "Model Type": model_type,
                    "Seed": seed,
                    "Rewards": rewards_array,
                })

                dfs.append(df)
        except:
            print(f"Passing over {evaluation_name} due to error loading data.")

    result = pd.concat(dfs, ignore_index=True)
    return result


def plot_comparison(data: pd.DataFrame, output_name: str, plot_type: str = "box") -> None:
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(10, 4))
    if plot_type == "box":
        sns.boxplot(data=data, y="Model Type", x="Rewards", hue="Model Type", legend=False, palette="viridis",
                        showmeans=True,
                        meanprops={
                            "marker": "+",
                            "markeredgecolor": "red",
                            "markersize": 10}
                        )
    elif plot_type == "violin":
        sns.violinplot(data=data, y="Model Type", x="Rewards", hue="Model Type",
                       legend=False, palette="viridis", inner="quart")
    elif plot_type == "hist":
        sns.histplot(data=data, x="Rewards", hue="Model Type", kde=True, element="step", stat="density", common_norm=False)

    plt.ylabel("Model Type", fontsize=12, labelpad=10)
    plt.xlabel("Episode returns", fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.savefig(f"plotting/plots/{output_name}_comparison.png")
    plt.close()


def analyze(output_name: str, filter_str: str = None, plot_type: str = "box") -> None:
    data = load_data(filter_str)

    plot_comparison(data, output_name, plot_type)


if __name__ == '__main__':
    fire.Fire(analyze)
