import fire
import numpy as np

import matplotlib.pyplot as plt


def plot_training(exp_name: str, latent_state_type: str):
    training_data = np.load(f"data/model_learning/{exp_name}/losses.npz")

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    axs[0, 0].plot(training_data['total_losses'])
    axs[0, 0].set_title('Total Losses')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')

    axs[0, 1].plot(training_data['image_prediction_losses'])
    axs[0, 1].set_title('Image Reconstruction Losses')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')

    axs[0, 2].plot(training_data['dynamics_losses'])
    axs[0, 2].set_title('Dynamics/Representation Losses')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('Loss')

    axs[1, 0].plot(training_data['reward_prediction_losses'])
    axs[1, 0].set_title('Reward Prediction Losses')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')

    axs[1, 1].plot(training_data['continuity_prediction_losses'])
    axs[1, 1].set_title('Continuity Prediction Losses')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')

    if latent_state_type == 'gaussian':
        axs[1, 2].plot(training_data['kl_losses'])
        axs[1, 2].set_title('KL Losses')
        axs[1, 2].set_xlabel('Epoch')
        axs[1, 2].set_ylabel('Loss')
    else:
        axs[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"data/model_learning/{exp_name}/figures/losses.png")


if __name__ == '__main__':
    fire.Fire(plot_training)
