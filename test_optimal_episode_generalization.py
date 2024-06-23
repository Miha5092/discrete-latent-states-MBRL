import torch
import numpy as np
import fire
import torch.nn as nn
import random
import os
import time
import wandb

from nets import RepresentationNetwork, DynamicsNetwork, ReconstructionNetwork
from mcts import Node, MCTS

from minatar import Environment


num_actions = Environment("quick_breakout", sticky_action_prob=0.0, use_minimal_action_set=True).num_actions()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class LearnedModel:

    def __init__(self, num_actions, model):
        self.num_actions = num_actions
        representation_net, dynamics_net = model
        self.dynamics_net = dynamics_net
        self.representation_net = representation_net

    def step(self, z, action):
        # encode the action
        action = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1) / self.num_actions
        net_input = torch.cat([z, action], dim=1)
        with torch.no_grad():
            next_z, reward, done = self.dynamics_net.sample(net_input)
        return next_z, reward.item(), done.item()

    def encode(self, env):
        z = self.representation_net.sample(
            torch.tensor(env.state(), dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2))
        return z


def test_episode_generalization(
        model_path: str,
        data_path: str = "data/dqn_optimal_seed_0/experience_data_full.npz",
        exp_name: str = None,
        trajectories: int = 2
):
    if exp_name is None:
        exp_name = os.path.basename(model_path)

    trajectory_data = np.load(data_path)

    states, actions, rewards, terminals = trajectory_data['states'], trajectory_data['actions'], trajectory_data[
        'rewards'], trajectory_data['terminals']

    states = torch.tensor(states, device=device).float()
    states = states.permute(0, 3, 1, 2)
    states = states.unsqueeze(1)

    actions = torch.tensor(actions, device=device).float()
    actions = actions.unsqueeze(1) / num_actions

    rewards = torch.tensor(rewards, device=device).float()
    terminals = torch.tensor(terminals, device=device)

    # states, actions, rewards, terminals = states[:states_to_check], actions[:states_to_check], rewards[:states_to_check], terminals[:states_to_check]
    transitions = list(zip(states, actions, rewards, terminals))

    # Load the models
    representation_net = torch.load(f'{model_path}/representation_net.pt', map_location=device)
    dynamics_net = torch.load(f'{model_path}/dynamics_net.pt', map_location=device)

    model = LearnedModel(num_actions, (representation_net, dynamics_net))

    # Set the models to evaluation mode
    representation_net.eval()
    dynamics_net.eval()

    actual_episode_returns = []
    predicted_episode_returns = []
    reward_difference = []
    episode_length_difference = []


    current_actual_return = 0
    current_predicted_reward = 0
    model_predicted_episode_termination = False
    # for state, action, reward, terminal in zip(states, actions, rewards, terminals):
    #
    #     with torch.no_grad():
    #         z = representation_net.sample(state)
    #
    #         net_input = torch.cat([z, action], dim=1)
    #
    #         _, predicted_rewards, predicted_dones = dynamics_net.sample(net_input)
    #
    #     current_actual_return += reward.item()
    #     if not model_predicted_episode_termination:
    #         current_predicted_reward += predicted_rewards.item()
    #
    #     if terminal:
    #         actual_episode_returns.append(current_actual_return)
    #         predicted_episode_returns.append(current_predicted_reward)
    #
    #         current_actual_return = 0
    #         current_predicted_reward = 0
    #         model_predicted_episode_termination = False

    episode_reward = 0
    predicted_episode_reward = 0

    no_data = False
    i = 0
    counter = 0
    while not no_data and counter < trajectories and i < len(transitions):
    # for i, (state, action, reward, terminal) in enumerate(transitions):
        (state, action, reward, terminal) = transitions[i]

        j = i + 1
        while not terminal and j < len(transitions):
            episode_reward += transitions[j][2].item()
            terminal = transitions[j][3].item()
            j += 1

        next_i = j + 1

        episode_length = j - i
        # print(terminal)

        z = representation_net.sample(state)
        done = False

        j = i
        while not done and j < len(transitions):
            with torch.no_grad():
                net_input = torch.cat([z, transitions[j][1]], dim=1)

                z, predicted_rewards, predicted_done = dynamics_net.sample(net_input)
                done = True if predicted_done > 0.5 else False
                predicted_episode_reward += predicted_rewards.item()
                j += 1

        predicted_episode_length = j - i

        actual_episode_returns.append(episode_reward)
        predicted_episode_returns.append(predicted_episode_reward)
        reward_difference.append(episode_reward - predicted_episode_reward)
        episode_length_difference.append(episode_length - predicted_episode_length)

        episode_reward = 0
        predicted_episode_reward = 0

        counter += 1
        i = next_i

    actual_episode_returns = np.array(actual_episode_returns)
    predicted_episode_returns = np.array(predicted_episode_returns)
    reward_difference = np.array(reward_difference)
    episode_length_difference = np.array(episode_length_difference)

    avg_episode_returns = np.mean(actual_episode_returns)
    avg_predicted_episode_returns = np.mean(predicted_episode_returns)
    avg_reward_difference = np.mean(reward_difference)
    avg_episode_length_difference = np.mean(episode_length_difference)

    os.makedirs(f"data/model_generalization/{exp_name}", exist_ok=True)
    np.savez(f"data/model_generalization/{exp_name}/episode_rewards.npz",
             actual_episode_returns=actual_episode_returns,
             predicted_episode_returns=predicted_episode_returns,
             reward_difference=reward_difference,
             episode_length_difference=episode_length_difference)

    print("Actual returns", avg_episode_returns)
    print("Predicted returns", avg_predicted_episode_returns)
    print("Reward Prediction difference", avg_reward_difference)
    print("Episode length difference", avg_episode_length_difference)


def all(trajectories: int = 2000):
    models = [
        "data/model_learning/optimal_deterministic_full_seed_0",
        "data/model_learning/optimal_deterministic_full_seed_42",
        "data/model_learning/optimal_deterministic_full_seed_420",
        "data/model_learning/optimal_discrete_full_seed_0",
        "data/model_learning/optimal_discrete_full_seed_42",
        "data/model_learning/optimal_discrete_full_seed_420",
    ]

    data = "data/dqn_optimal_seed_0/experience_data_full.npz"

    for model in models:
        test_episode_generalization(model_path=model, data_path=data, trajectories=trajectories)


if __name__ == '__main__':
    fire.Fire(test_episode_generalization)
