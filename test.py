import glob

import torch

import learn_model
import evaluate_model


def run_grid():

    runs = [("deterministic", 0.8, 0.05, (128,), 10),
            ("deterministic", 0.8, 0.05, (128,), 30),
            ("deterministic", 1.0, 0.1, (128,), 10)]

    seeds = [0, 42]

    for seed in seeds:
        for (model_type, dyn, rep, latent_shape, data_path) in runs:

            if data_path == 10:
                data_path = "data/dqn_training_seed_0/experience_data_episode_10000.npz"
                exp_name = f"deterministic_10_dyn_{dyn}_rep_{rep}_seed_{seed}"
            else:
                data_path = "data/dqn_training_seed_0/experience_data_episode_30000.npz"
                exp_name = f"deterministic_30_dyn_{dyn}_rep_{rep}_seed_{seed}"

            print(f"Training {exp_name}...")

            try:
                learn_model.main(
                    dyn_loss_scale=dyn,
                    rep_loss_scale=rep,
                    kl_clip=0.75,
                    data_path=data_path,
                    exp_name=exp_name,
                    latent_state_type=model_type,
                    latent_state_shape=latent_shape
                )
            except:
                print(f"{exp_name} failed")



def test_all():
    pattern = "data/model_learning/discrete_small_dyn*_rep*"

    folder_list = glob.glob(pattern)
    names = [str(folder).split("\\")[-1] for folder in folder_list]

    best_mode = None

    for name in names:
        evaluate_model.main(
            output_folder=f"data/model_evaluation/{name}",
            model_folder=f"data/model_learning/",
            seed=42,
            num_episodes=5
        )


def test_some():
    dyn_weights = [0.2]
    rep_weights = [0.05, 0.07, 0.1, 0.15, 0.2]
    clips = [0.75]

    for dyn_weight in dyn_weights:
        for rep_weight in rep_weights:
            for clip in clips:
                exp_name = f"discrete_10_space_dyn{dyn_weight}_rep{rep_weight}_clip{clip}"

                try:
                    evaluate_model.main(
                        output_folder=f"data/model_evaluation/{exp_name}",
                        model_folder=f"data/model_learning/{exp_name}",
                        seed=42,
                        num_episodes=10
                    )
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    run_grid()

