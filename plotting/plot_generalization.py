import glob
import os
import re

import fire
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def load_data() -> pd.DataFrame:
    pattern = "data/model_generalization/optimal_*_full_*"
    folder_list = glob.glob(pattern)
    trainings = [os.path.basename(folder) for folder in folder_list]

    extracting_pattern = re.compile(r'optimal_([^_]+)_full_seed_(\d+)')
    dfs = []

    for training_name in trainings:
        try:
            with open(f"data/model_generalization/{training_name}/episode_rewards.npz", "rb") as rewards:
                match = extracting_pattern.search(training_name)

                if match:
                    rewards = np.load(rewards)
                    model_type = match.group(1)
                    seed = match.group(2)

                    if model_type == "deterministic":
                        model_type = "Continuous"
                    else:
                        model_type = "Discrete"

                    df = pd.DataFrame({
                        "Model Type": model_type,
                        "Seed": seed,
                        "Average Episode Rewards": rewards["actual_episode_returns"],
                        "Predicted Episode Rewards": rewards["predicted_episode_returns"],
                        "Average Prediction Difference": rewards["reward_difference"],
                        "Average Episode Length Difference": rewards["episode_length_difference"],
                    })

                    dfs.append(df)
        except:
            print(f"Passing over {training_name} due to error loading data.")

    result = pd.concat(dfs, ignore_index=True)
    return result


def plot_generalization(result: pd.DataFrame, output_name: str) -> None:
    result = result[[
        "Model Type",
        "Average Episode Rewards",
        "Predicted Episode Rewards",
        "Average Prediction Difference",
        "Average Episode Length Difference",
    ]]
    df_mean = result.groupby(["Model Type"], as_index=False).mean()
    df_mean.set_index('Model Type', inplace=True)

    print(df_mean)

    sns.heatmap(data=df_mean, annot=True, fmt=".2f", cmap="viridis")

    plt.tight_layout()
    plt.savefig(f"plotting/plots/{output_name}_losses.png")
    plt.close()


def main(output_name):
    data = load_data()
    plot_generalization(data, output_name)


if __name__ == "__main__":
    fire.Fire(main)
