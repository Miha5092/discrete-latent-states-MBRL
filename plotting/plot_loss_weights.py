import glob
import os
import re
import fire

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(model_type: str, filter_str: str = "discrete_30") -> pd.DataFrame:
    if model_type == "discrete":
        pattern = "data/model_evaluation/*_dyn_*_rep_*_clip_*"

        folder_list = glob.glob(pattern)
        evaluation_names = [os.path.basename(folder) for folder in folder_list]
        evaluation_names = list(filter(lambda name: filter_str in name, evaluation_names))

        extracting_pattern = re.compile(r'_dyn_([0-9.]+)_rep_([0-9.]+)_clip_([0-9.]+)')

        dfs = []
        for evaluation_name in evaluation_names:
            try:
                with open(f"data/model_evaluation/{evaluation_name}/episode_returns.npy", "rb") as rewards_array:
                    match = extracting_pattern.search(evaluation_name)

                    if match:
                        dyn_value = float(match.group(1))
                        rep_value = float(match.group(2))
                        clip_value = float(match.group(3))

                        rewards = np.load(rewards_array)

                        df = pd.DataFrame(rewards, columns=['reward'])
                        df['dyn_weight'] = dyn_value
                        df['Representation Weight'] = rep_value
                        df['clip'] = clip_value

                        dfs.append(df)
            except:
                print(f"Passing over {evaluation_name} due to error loading data.")

        results = pd.concat(dfs, ignore_index=True)

        return results
    else:
        pattern = "data/model_evaluation/*_dyn_*_rep_*"

        folder_list = glob.glob(pattern)
        evaluation_names = [str(folder).split("/")[-1] for folder in folder_list]
        evaluation_names = list(filter(lambda name: model_type in name, evaluation_names))

        extracting_pattern = re.compile(r'_dyn_([0-9.]+)_rep_([0-9.]+)')

        dfs = []

        for evaluation_name in evaluation_names:
            with open(f"data/model_evaluation/{evaluation_name}/episode_returns.npy", "rb") as rewards_array:
                match = extracting_pattern.search(evaluation_name)

                if match:
                    dyn_value = float(match.group(1))
                    rep_value = float(match.group(2))

                    rewards = np.load(rewards_array)

                    df = pd.DataFrame(rewards, columns=['reward'])
                    df['dyn_weight'] = dyn_value
                    df['Representation Weight'] = rep_value

                    dfs.append(df)

        results = pd.concat(dfs, ignore_index=True)

        return results


def plot_box(data: pd.DataFrame, model_type: str, plot_name: str):
    colors_hex = ['#8CBEB2', '#F2EBBF', '#F3B562', '#F06060', '#1E4F6A']
    custom_palette = sns.color_palette(colors_hex)

    if model_type == "discrete":
        g = sns.catplot(
            data=data, x="dyn_weight", y="reward", hue="Representation Weight",
            col="clip", col_wrap=2, kind="bar",
            errorbar="se",
            height=5,
            aspect=1.75,
            palette='colorblind',
            legend_out=False
        )

        g.fig.subplots_adjust(bottom=0.13)
        sns.move_legend(g, "lower center", ncol=4, frameon=False)

        plt.ylim(0, 6)

        g.set_xlabels("Dynamics Weight")
        g.set_ylabels("Reward")

        plt.savefig(f"plotting/plots/weights/{plot_name}.png")
        plt.close()
    else:
        plt.figure(figsize=(11, 8))

        g = sns.barplot(data=data, x="dyn_weight", y="reward", hue="Representation Weight",
                        errorbar="se",
                        palette='colorblind',
                        )

        sns.move_legend(g, "lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.25))

        plt.ylim(0, 20)

        plt.xlabel("Dynamics Weight")
        plt.ylabel("Reward")

        plt.tight_layout()

        plt.savefig(f"plotting/plots/weights/{plot_name}.png")
        plt.close()


def main(
        plot_name: str,
        model_type: str = "discrete",
        filter_str: str = "discrete_30"
):
    sns.set_context("paper", font_scale=1.5)

    data = load_data(model_type, filter_str)

    os.makedirs("plotting/plots/weights", exist_ok=True)

    plot_box(data, model_type, plot_name)


if __name__ == "__main__":
    fire.Fire(main)