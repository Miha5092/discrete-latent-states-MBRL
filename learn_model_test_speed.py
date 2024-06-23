import torch
import numpy as np
import fire
import torch.nn as nn
import random
import os
from nets import RepresentationNetwork, DynamicsNetwork, ReconstructionNetwork
# import matplotlib.pyplot as plt

# batch size 128 = 500s, 512 = 200s, 1024 = 123s, 2048=118s
# seems like we should use 512 or 1024

import time

from minatar import Environment
num_actions = Environment("quick_breakout", sticky_action_prob=0.0, use_minimal_action_set=True).num_actions()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, states, actions, rewards, terminals, unroll_steps):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.unroll_steps = unroll_steps
    
    def __len__(self):
        return len(self.states)-self.unroll_steps
    
    def __getitem__(self, idx):
        states = self.states[idx:idx+self.unroll_steps+1]
        actions = self.actions[idx:idx+self.unroll_steps]
        rewards = self.rewards[idx:idx+self.unroll_steps]
        terminals = self.terminals[idx:idx+self.unroll_steps]
        return states, actions, rewards, terminals

def main(
    unroll_steps: int = 1,
    batch_size: int = 128,
    num_epochs: int = 10,
    learning_rate: float = 0.00025,
    out_channels: int = 128,
    seed: int = 42,
    exp_name: str = "reworked_deterministic_30",
    data_path: str = "data/dqn_training_seed_0/experience_data_full.npz",
    image_prediction_loss="MSE",
    image_prediction_loss_clip=0.0,
    weight_decay=0.0,
):
    
    # make the results folder
    os.makedirs(f"data/model_learning/{exp_name}", exist_ok=True)
    os.makedirs(f"data/model_learning/{exp_name}/figures", exist_ok=True)

    # save parameters from fire
    with open(f"data/model_learning/{exp_name}/params.txt", "w") as f:
        f.write(f"unroll_steps: {unroll_steps}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"exp_name: {exp_name}\n")
    
    random.seed(seed)
    torch_seed = random.randint(0, 2**32-1)
    numpy_seed = random.randint(0, 2**32-1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    np.random.seed(numpy_seed)

    # load data into device
    trajectory_data = np.load(data_path)
    states, actions, rewards, terminals = trajectory_data['states'], trajectory_data['actions'], trajectory_data['rewards'], trajectory_data['terminals']
    assert len(states) == len(actions) == len(rewards) == len(terminals)
    actions = torch.tensor(actions, device=device).float()
    if len(actions.shape) == 2:
        actions = actions.squeeze(1)
    if len(rewards.shape) == 2:
        rewards = rewards.squeeze(1)
    if len(terminals.shape) == 2:
        terminals = terminals.squeeze(1)
    # put states, rewards and temrinals into torch
    states = torch.tensor(states, device=device).float()
    # switch axis so channel is the second axis
    states = states.permute(0, 3, 1, 2)
    rewards = torch.tensor(rewards, device=device).float()
    terminals = torch.tensor(terminals, device=device).bool()
    # create dataset
    dataset = CustomDataset(states, actions, rewards, terminals, unroll_steps)
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        )

    # make the nets
    representation_net = RepresentationNetwork(in_channels=4, out_channels=out_channels).to(device)
    dynamics_net = DynamicsNetwork(num_channels=out_channels).to(device)
    reconstruction_net = ReconstructionNetwork(num_channels=64, in_channels=out_channels).to(device)

    # define the loss function
    image_prediction_loss_fn = nn.MSELoss(reduction='none')
    reward_prediction_loss_fn = nn.MSELoss(reduction='none')
    continuity_prediction_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    dynamics_loss_fn = nn.MSELoss(reduction='none')
    representation_loss_fn = nn.MSELoss(reduction='none')

    # define the optimizer
    optimizer = torch.optim.Adam(
        list(representation_net.parameters()) + list(dynamics_net.parameters()) + list(reconstruction_net.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    epoch_total_losses = []
    epoch_reward_prediction_losses = []
    epoch_image_prediction_losses = []
    epoch_continuity_prediction_losses = []
    epoch_dynamics_losses = []
    epoch_representation_losses = []
    
    # supervised-learning loop
    for epoch in range(num_epochs):

        epoch_start_time = time.time()

        epoch_total_loss = 0
        epoch_reward_prediction_loss = 0
        epoch_image_prediction_loss = 0
        epoch_continuity_prediction_loss = 0
        epoch_dynamics_loss = 0
        epoch_representation_loss = 0
       
        print("total number of minibatches:", len(data_loader))

        data_loader_iterator = iter(data_loader)
        for _ in range(len(data_loader)):

            batch_start_time = time.time()
            data_loading_start_time = time.time()

            batch_states, batch_actions, batch_rewards, batch_terminals = next(data_loader_iterator)
            data_loading_time = time.time() - data_loading_start_time

            forward_start_time = time.time()
            
            # preprocess the actions
            batch_actions = batch_actions[:, :, None].expand(-1, -1, 1) / num_actions

            # losses
            reward_prediction_losses = []
            image_prediction_losses = []
            continuity_prediction_losses = []
            dynamics_losses = []
            representation_losses = []

            # encode s_t to get z_t using the representation net
            latent_states = representation_net(batch_states[:, 0])
            
            # reconstruct s_t using z_t
            reconstructed_states = reconstruction_net(latent_states)
            image_prediction_loss = torch.clamp(image_prediction_loss_fn(reconstructed_states, batch_states[:, 0]), min=image_prediction_loss_clip).mean()
            image_prediction_losses.append(image_prediction_loss)
            
            for i in range(unroll_steps):
                
                # using latent_states (encoding of s_{t+i}) and a_{t+i}, we predict next latent states (encodings of s_{t+i+1})
                predicted_next_latent_states = dynamics_net.dynamics_net(torch.cat([latent_states, batch_actions[:, i]], dim=1))
                # next latent states encoded from the next states
                next_latent_states = representation_net(batch_states[:, i+1])

                # predict rewards and dones
                state_action = torch.cat([latent_states, batch_actions[:, i]], dim=1)
                state_action_embedding = dynamics_net.embedding_net(state_action)
                predicted_rewards = dynamics_net.reward_net(state_action_embedding)
                predicted_dones_logits = dynamics_net.done_net(state_action_embedding)

                # calculate losses
                reward_prediction_loss = reward_prediction_loss_fn(predicted_rewards, batch_rewards[:, i]).mean()
                reward_prediction_losses.append(reward_prediction_loss)
                continuity_prediction_loss = continuity_prediction_loss_fn(predicted_dones_logits, batch_terminals[:, i].float()).mean()
                continuity_prediction_losses.append(continuity_prediction_loss)

                # calculate reconstruction_loss
                reconstructed_states = reconstruction_net(next_latent_states)
                image_prediction_loss = torch.clamp(image_prediction_loss_fn(reconstructed_states, batch_states[:, i+1]), min=image_prediction_loss_clip).mean()
                image_prediction_losses.append(image_prediction_loss)

                # termination after time step i
                episode_continuing = (1 - batch_terminals[:, i].float())[:, None]

                # calculate dynamics loss and representation loss
                dynamics_loss = (dynamics_loss_fn(predicted_next_latent_states, next_latent_states.detach()) * episode_continuing).sum() / episode_continuing.sum()
                dynamics_losses.append(dynamics_loss)
                representation_loss = (representation_loss_fn(next_latent_states, predicted_next_latent_states.detach()) * episode_continuing).sum() / episode_continuing.sum()
                representation_losses.append(representation_loss)

                # update latent states
                latent_states = next_latent_states

            avg_image_prediction_loss = torch.stack(image_prediction_losses).mean()
            avg_reward_prediction_loss = torch.stack(reward_prediction_losses).mean()
            avg_continuity_prediction_loss = torch.stack(continuity_prediction_losses).mean()
            avg_dynamics_loss = torch.stack(dynamics_losses).mean()
            avg_representation_loss = torch.stack(representation_losses).mean()
            
            loss = avg_image_prediction_loss + avg_reward_prediction_loss + avg_continuity_prediction_loss + 0.5 * avg_dynamics_loss + 0.1 * avg_representation_loss

            forward_time = time.time() - forward_start_time
            backward_start_time = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            backward_time = time.time() - backward_start_time
            batch_time = time.time() - batch_start_time
            print(f"Batch Time: {batch_time:.4f}s, Data Loading Time: {data_loading_time:.4f}s, Forward Time: {forward_time:.4f}s, Backward Time: {backward_time:.4f}s")

            epoch_total_loss += loss.item()
            epoch_reward_prediction_loss += avg_reward_prediction_loss.item()
            epoch_image_prediction_loss += avg_image_prediction_loss.item()
            epoch_continuity_prediction_loss += avg_continuity_prediction_loss.item() 
            epoch_dynamics_loss += avg_dynamics_loss.item()
            epoch_representation_loss += avg_representation_loss.item()

        epoch_total_losses.append(epoch_total_loss/len(data_loader))
        epoch_reward_prediction_losses.append(epoch_reward_prediction_loss/len(data_loader))
        epoch_image_prediction_losses.append(epoch_image_prediction_loss/len(data_loader))
        epoch_continuity_prediction_losses.append(epoch_continuity_prediction_loss/len(data_loader))
        epoch_dynamics_losses.append(epoch_dynamics_loss/len(data_loader))
        epoch_representation_losses.append(epoch_representation_loss/len(data_loader))

        message = f"Epoch {epoch+1} losses: total: {epoch_total_losses[-1]:.4f}, reward: {epoch_reward_prediction_losses[-1]:.4f}, reconstruction: {epoch_image_prediction_losses[-1]:.4f}, done: {epoch_continuity_prediction_losses[-1]:.4f}, dynamics: {epoch_dynamics_losses[-1]:.4f}, representation: {epoch_representation_losses[-1]:.4f}"
        print(message)
        with open(f"data/model_learning/{exp_name}/log.txt", "a") as f:
            f.write(message + "\n")

        if epoch % 1 == 0:

            # save all models
            os.makedirs(f"data/model_learning/{exp_name}/epoch_{epoch}", exist_ok=True)
            torch.save(representation_net, f"data/model_learning/{exp_name}/epoch_{epoch}/representation_net.pt")
            torch.save(dynamics_net, f"data/model_learning/{exp_name}/epoch_{epoch}/dynamics_net.pt")
            torch.save(reconstruction_net, f"data/model_learning/{exp_name}/epoch_{epoch}/reconstruction_net.pt")
    
        epoch_time = time.time() - epoch_start_time  # Total time for the epoch
        print(f"Epoch {epoch+1} completed in {epoch_time:.4f}s")
        assert False

    
    # save model
    torch.save(representation_net, f"data/model_learning/{exp_name}/representation_net.pt")
    torch.save(dynamics_net, f"data/model_learning/{exp_name}/dynamics_net.pt")
    torch.save(reconstruction_net, f"data/model_learning/{exp_name}/reconstruction_net.pt")

    # save losses together
    np.savez_compressed(f"data/model_learning/{exp_name}/losses.npz",
                        total_losses=epoch_total_losses,
                        reward_prediction_losses=epoch_reward_prediction_losses,
                        image_prediction_losses=epoch_image_prediction_losses,
                        continuity_prediction_losses=epoch_continuity_prediction_losses,
                        dynamics_losses=epoch_dynamics_losses,
                        representation_losses=epoch_representation_losses)

if __name__ == '__main__':
    fire.Fire(main)