import os
import pickle
import fire
import torch

import numpy as np
import torch.nn as nn

from minatar import Environment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_actions = Environment("quick_breakout", sticky_action_prob=0.0, use_minimal_action_set=True).num_actions()


def load_data(data_path: str) -> list[tuple]:
    with open(data_path, 'rb') as transition_file:
        transitions = pickle.load(transition_file)

        return transitions


def preprocess_data(transitions: list[tuple]) -> [torch.tensor]:
    unpacked_tuples = list(zip(*transitions))
    initial_frames, actions, next_frames, dones, rewards = map(torch.tensor, map(np.array, unpacked_tuples))

    initial_frames, next_frames = initial_frames.float(), next_frames.float()

    actions = actions.unsqueeze(1) / num_actions

    return initial_frames.to(device), actions.to(device), next_frames.to(device), dones.to(device), rewards.to(device)



def evaluate_model(
        model_path: str,
        data_path: str,
        exp_name: str = None,
) -> None:
    transitions = load_data(data_path)

    # Load the models
    representation_net = torch.load(f'{model_path}/representation_net.pt', map_location=device)
    dynamics_net = torch.load(f'{model_path}/dynamics_net.pt', map_location=device)
    reconstruction_net = torch.load(f'{model_path}/reconstruction_net.pt', map_location=device)

    # Set the models to evaluation mode
    representation_net.eval()
    dynamics_net.eval()
    reconstruction_net.eval()

    # Define loss functions
    image_prediction_loss_fn = nn.MSELoss(reduction='none')
    reward_prediction_loss_fn = nn.MSELoss(reduction='none')
    continuity_prediction_loss_fn = nn.BCELoss(reduction='none')

    initial_frames, actions, next_frames, dones, rewards = preprocess_data(transitions)

    with torch.no_grad():
        z = representation_net.sample(initial_frames)

        net_input = torch.cat([z, actions], dim=1)

        predicted_next_z, predicted_rewards, predicted_dones = dynamics_net.sample(net_input)

        predicted_next_frame = reconstruction_net(predicted_next_z)

    image_prediction_loss = torch.clamp(image_prediction_loss_fn(predicted_next_frame, next_frames), min=0).mean().item()
    reward_prediction_loss = reward_prediction_loss_fn(predicted_rewards, rewards.float()).mean().item()
    continuity_prediction_loss = continuity_prediction_loss_fn(predicted_dones, dones.float()).mean().item()

    total_loss = image_prediction_loss + reward_prediction_loss + continuity_prediction_loss


    if exp_name is None:
        exp_name = os.path.basename(model_path)

    training_losses = np.load(f'{model_path}/losses.npz')
    training_image_prediction_losses = training_losses['image_prediction_losses'][-1]
    training_reward_prediction_losses = training_losses['reward_prediction_losses'][-1]
    training_continuity_prediction_losses = training_losses['continuity_prediction_losses'][-1]

    total_training_loss = training_image_prediction_losses + training_reward_prediction_losses + training_continuity_prediction_losses

    generalization_losses = np.array([total_loss, image_prediction_loss, reward_prediction_loss, continuity_prediction_loss])
    training_losses = np.array([total_training_loss, training_image_prediction_losses, training_reward_prediction_losses, training_continuity_prediction_losses])

    os.makedirs(f"data/model_generalization/{exp_name}", exist_ok=True)
    np.savez(f"data/model_generalization/{exp_name}/losses.npz",
             generalization_losses=generalization_losses,
             training_losses=training_losses,
             delta_losses=generalization_losses - training_losses)

    # print(image_prediction_loss)
    # print(reward_prediction_loss)
    # print(continuity_prediction_loss)


if __name__ == "__main__":
    fire.Fire(evaluate_model)
