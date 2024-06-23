import glob
import os
import re

import fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filter_str: str = None) -> pd.DataFrame:
    generalization_pattern = "data/model_generalization/optimal_*_*"
    generalization_folder_list = glob.glob(generalization_pattern)
    generalization_folder_list = [os.path.basename(folder) for folder in generalization_folder_list]

    if filter_str is not None:
        generalization_folder_list = list(filter(lambda name: filter_str in name, generalization_folder_list))

    extracting_pattern = re.compile(r"optimal_([^_]+)_.*_seed_(\d+)")
    dfs = []

    for exp_folder in generalization_folder_list:
        try:
            match = extracting_pattern.search(exp_folder)

            if match:
                losses = np.load(f"data/model_generalization/{exp_folder}/losses.npz")
                rewards_array = np.load(f"data/model_evaluation/{exp_folder}/episode_returns.npy")

                model_type = match.group(1)
                seed = int(match.group(2))

                if model_type == "deterministic":
                    model_type = "Continuous"
                else:
                    model_type = "Discrete"

                df = pd.DataFrame({
                    "Model Type": model_type,
                    "Seed": seed,
                    "Generalization Total Loss": losses["generalization_losses"][0],
                    "Generalization Image Loss": losses["generalization_losses"][1],
                    "Generalization Reward Loss": losses["generalization_losses"][2],
                    "Generalization Continuity Loss": losses["generalization_losses"][3],
                    "Training Total Loss": losses["training_losses"][0],
                    "Training Image Loss": losses["training_losses"][1],
                    "Training Reward Loss": losses["training_losses"][2],
                    "Training Continuity Loss": losses["training_losses"][3],
                    "Delta Total Loss": losses["delta_losses"][0],
                    "Delta Image Loss": losses["delta_losses"][1],
                    "Delta Reward Loss": losses["delta_losses"][2],
                    "Delta Continuity Loss": losses["delta_losses"][3],
                    "Episode Returns": np.array([np.mean(rewards_array)]),
                })

                dfs.append(df)
        except:
            print(f"Passing over {exp_folder} due to error loading data.")

    result = pd.concat(dfs, ignore_index=True)
    return result


def plot_correlation_matrices(data: pd.DataFrame, output_name: str) -> None:
    sns.set_theme(style="white")
    sns.set_context("notebook")

    # Assuming 'Model Type' contains two distinct types you want to plot
    model_types = data['Model Type'].unique()

    fig, axes = plt.subplots(nrows=1, ncols=len(model_types), figsize=(12, 4))

    for ax, model_type in zip(axes, model_types):

        filtered_data = data[data['Model Type'] == model_type]

        filtered_data = filtered_data[[
            "Generalization Total Loss",
            "Generalization Image Loss",
            "Generalization Reward Loss",
            "Generalization Continuity Loss",
            # "Delta Image Loss",
            # "Delta Reward Loss",
            # "Delta Continuity Loss",
            # "Training Image Loss",
            # "Training Reward Loss",
            # "Training Continuity Loss",
            "Episode Returns"
        ]]

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(
            filtered_data.corr()[["Episode Returns"]].sort_values(by="Episode Returns", ascending=True).drop(["Episode Returns"], axis=0),
            vmin=-1, vmax=1, annot=True, cmap=cmap, ax=ax)

        ax.set_title(f"Losses Correlating with Rewards for {model_type} model", fontdict={'fontsize': 12}, pad=16)

    plt.tight_layout()
    plt.savefig(f"plotting/plots/generalization/{output_name}_correlation.png")
    plt.savefig(f"plotting/plots/generalization/{output_name}_correlation.svg")
    plt.close()


def plot_correlation_unified(data: pd.DataFrame, output_name: str) -> None:
    sns.set_theme(style="white")
    sns.set_context("notebook")

    # Assuming 'Model Type' contains two distinct types you want to plot
    model_types = data['Model Type'].unique()

    filtered_data = data[[
        "Generalization Total Loss",
        "Generalization Image Loss",
        "Generalization Reward Loss",
        "Generalization Continuity Loss",
        # "Delta Image Loss",
        # "Delta Reward Loss",
        # "Delta Continuity Loss",
        # "Training Image Loss",
        # "Training Reward Loss",
        # "Training Continuity Loss",
        "Episode Returns"
    ]]

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    heatmap = sns.heatmap(
        filtered_data.corr()[["Episode Returns"]].sort_values(by="Episode Returns", ascending=True).drop(["Episode Returns"], axis=0),
        vmin=-1, vmax=1, annot=True, cmap=cmap, annot_kws={"size": 14})

    # Increase the size of x and y labels
    heatmap.set_ylabel('Correlation Coefficient', fontsize=16)

    # Increase the size of the ticks
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)

    plt.tight_layout()
    plt.savefig(f"plotting/plots/generalization/{output_name}_correlation_unified.png")
    plt.savefig(f"plotting/plots/generalization/{output_name}_correlation_unified.svg")
    plt.close()


def plot_losses(data: pd.DataFrame, output_name: str) -> None:
    filtered_data = data[[
        "Model Type",
        "Seed",
        "Generalization Total Loss",
        "Generalization Image Loss",
        "Generalization Reward Loss",
        "Generalization Continuity Loss",

    ]]
    long_df = pd.melt(filtered_data, id_vars=["Model Type", "Seed"], var_name="Loss Type", value_name="Loss Value")

    df_mean = long_df.groupby(["Model Type", "Loss Type"], as_index=False).mean()
    df_count = long_df.groupby(["Model Type", "Loss Type"], as_index=False).count()
    df_std = long_df.groupby(["Model Type", "Loss Type"], as_index=False).std()

    # Calculate SEM
    df_sem = df_std.copy()
    df_sem['SEM'] = df_std['Loss Value'] / np.sqrt(df_count['Loss Value'])

    df_merged = pd.merge(df_mean, df_sem[['Model Type', 'Loss Type', 'SEM']], on=["Model Type", "Loss Type"])
    df_merged['Annotation'] = df_merged.apply(lambda row: f"{row['Loss Value']:.2f}\n±{row['SEM']:.2f}", axis=1)

    pivot_table = df_merged.pivot(index="Loss Type", columns="Model Type", values="Loss Value")
    annotations = df_merged.pivot(index="Loss Type", columns="Model Type", values="Annotation")

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    heatmap = sns.heatmap(data=pivot_table, annot=annotations, fmt="", cmap=cmap, annot_kws={"size": 14})

    # Increase the size of x and y labels
    heatmap.set_xlabel('Model Type', fontsize=16)
    heatmap.set_ylabel('Loss Type', fontsize=16)

    # Increase the size of the ticks
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)

    plt.tight_layout()
    plt.savefig(f"plotting/plots/generalization/{output_name}_losses.png")
    plt.savefig(f"plotting/plots/generalization/{output_name}_losses.svg")
    plt.close()


def analyze(output_name: str, filter_str: str = None) -> None:
    data = load_data(filter_str)

    os.makedirs("plotting/plots/generalization", exist_ok=True)

    plot_correlation_matrices(data, output_name)
    plot_correlation_unified(data, output_name)
    plot_losses(data, output_name)


if __name__ == '__main__':
    fire.Fire(analyze)
