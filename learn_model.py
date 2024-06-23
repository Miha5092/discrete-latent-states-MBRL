import torch
import numpy as np
import fire
import torch.nn as nn
import random
import os
import time
import wandb

from nets import RepresentationNetwork, DynamicsNetwork, ReconstructionNetwork
from plotting.plot_training import plot_training

from minatar import Environment

from utils import rsample_categorical, compute_KL_divergence_between_two_independent_gaussians, \
    compute_KL_divergence_between_two_independent_categoricals, rsample_gaussian

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
        return len(self.states) - self.unroll_steps

    def __getitem__(self, idx):
        states = self.states[idx:idx + self.unroll_steps + 1]
        actions = self.actions[idx:idx + self.unroll_steps]
        rewards = self.rewards[idx:idx + self.unroll_steps]
        terminals = self.terminals[idx:idx + self.unroll_steps]
        return states, actions, rewards, terminals


def main(
        unroll_steps: int = 1,
        batch_size: int = 128,
        num_epochs: int = 50,
        learning_rate: float = 0.00025,
        hidden_channels: int = 128,
        seed: int = 420,
        exp_name: str = "optimal_deterministic_full_seed_420",
        data_path: str = "data/dqn_training_seed_0/experience_data_full.npz",
        image_prediction_loss_clip: float = 0.0,
        weight_decay: float = 0,
        latent_state_type: str = "deterministic",
        latent_state_shape: tuple = (256, ),
        kl_clip: float = 0.75,
        dyn_loss_scale: float = 0.3,
        rep_loss_scale: float = 0.05,
        kl_beta: float = 0,
        aggregate_method: str = "sum",
        monitor_model: bool = False,
        save_intermediary: bool = False
):
    assert (latent_state_type == "deterministic"
            or latent_state_type == "gaussian"
            or latent_state_type == "discrete"), f"Latent state type: {latent_state_type} can only be deterministic, gaussian, or discrete"

    if monitor_model:
        wandb.init(
            project="Bachelor's Thesis",
            name=exp_name,
            tags=[latent_state_type],

        )

        wandb.config.update({
            "unroll_steps": unroll_steps,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "hidden_channels": hidden_channels,
            "seed": seed,
            "exp_name": exp_name,
            "data_path": data_path,
            "image_prediction_loss_clip": image_prediction_loss_clip,
            "weight_decay": weight_decay,
            "latent_state_type": latent_state_type,
            "latent_state_shape": latent_state_shape,
            "KL_clip": kl_clip,
            "dyn_loss_scale": dyn_loss_scale,
            "rep_loss_scale": rep_loss_scale,
            "KL_beta": kl_beta,
            "aggregate_method": aggregate_method,
            "monitor_model": monitor_model,
            "save_intermediary": save_intermediary
        })

    start_time = time.time()

    # make the results folder
    os.makedirs(f"data/model_learning/{exp_name}", exist_ok=True)
    os.makedirs(f"data/model_learning/{exp_name}/figures", exist_ok=True)

    # save parameters from fire
    with open(f"data/model_learning/{exp_name}/params.txt", "w") as f:
        for arg_name, arg_value in locals().items():
            if arg_name != "self" and arg_name != "f":
                f.write(f"{arg_name}: {arg_value}\n")

    random.seed(seed)
    torch_seed = random.randint(0, 2 ** 32 - 1)
    numpy_seed = random.randint(0, 2 ** 32 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    np.random.seed(numpy_seed)

    # load data
    trajectory_data = np.load(data_path)
    states, actions, rewards, terminals = trajectory_data['states'], trajectory_data['actions'], trajectory_data[
        'rewards'], trajectory_data['terminals']
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
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # make the nets

    latent_state_channels = 1
    for x in latent_state_shape:
        latent_state_channels *= x

    representation_net = RepresentationNetwork(
        in_channels=4,
        hidden_channels=hidden_channels,
        latent_state_type=latent_state_type,
        latent_state_shape=latent_state_shape
    ).to(device)
    dynamics_net = DynamicsNetwork(
        hidden_channels=hidden_channels,
        latent_state_type=latent_state_type,
        latent_state_shape=latent_state_shape
    ).to(device)
    reconstruction_net = ReconstructionNetwork(
        in_channels=latent_state_channels,
        num_channels=64
    ).to(device)

    if monitor_model:
        wandb.watch(representation_net, log_freq=100, log="all")
        wandb.watch(dynamics_net, log_freq=100, log="all")

    # define the loss function
    image_prediction_loss_fn = nn.MSELoss(reduction='none')
    reward_prediction_loss_fn = nn.MSELoss(reduction='none')
    continuity_prediction_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    if latent_state_type == "deterministic":
        dynamics_loss_fn = nn.MSELoss(reduction='none')
        representation_loss_fn = nn.MSELoss(reduction='none')

        kl_loss_fn = None
    elif latent_state_type == "gaussian":
        dynamics_loss_fn = compute_KL_divergence_between_two_independent_gaussians
        representation_loss_fn = compute_KL_divergence_between_two_independent_gaussians

        kl_loss_fn = compute_KL_divergence_between_two_independent_gaussians
    elif latent_state_type == "discrete":
        dynamics_loss_fn = compute_KL_divergence_between_two_independent_categoricals
        representation_loss_fn = compute_KL_divergence_between_two_independent_categoricals

        kl_loss_fn = None
    else:
        raise ValueError(f"Latent state type: {latent_state_type} can only be deterministic, gaussian, or discrete")

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
    epoch_kl_losses = []

    # supervised-learning loop
    for epoch in range(num_epochs):

        epoch_total_loss = 0
        epoch_reward_prediction_loss = 0
        epoch_image_prediction_loss = 0
        epoch_continuity_prediction_loss = 0
        epoch_dynamics_loss = 0
        epoch_representation_loss = 0
        epoch_kl_loss = 0

        for batch_states, batch_actions, batch_rewards, batch_terminals in data_loader:

            # preprocess the actions
            batch_actions = batch_actions[:, :, None].expand(-1, -1, 1) / num_actions

            # losses
            reward_prediction_losses = []
            image_prediction_losses = []
            continuity_prediction_losses = []
            dynamics_losses = []
            representation_losses = []
            kl_losses = []

            # encode s_t to get z_t using the representation net
            if latent_state_type == "deterministic":
                latent_states = representation_net(batch_states[:, 0])
            else:
                latent_state_dist = representation_net(batch_states[:, 0])

                if latent_state_type == "gaussian":
                    latent_states = rsample_gaussian(latent_state_dist[0], latent_state_dist[1]).flatten(1)
                else:
                    latent_states = rsample_categorical(latent_state_dist).flatten(1)

            # reconstruct s_t using z_t
            reconstructed_states = reconstruction_net(latent_states)
            image_prediction_loss = torch.clamp(image_prediction_loss_fn(reconstructed_states, batch_states[:, 0]),
                                                min=image_prediction_loss_clip).mean()
            image_prediction_losses.append(image_prediction_loss)

            if latent_state_type == "gaussian":
                kl_loss = kl_loss_fn(
                    latent_state_dist[0], latent_state_dist[1],
                    torch.zeros_like(latent_state_dist[0]).detach(), torch.ones_like(latent_state_dist[1]).detach(),
                    aggregate_method
                )
                kl_losses.append(kl_loss)

            for i in range(unroll_steps):
                # termination after time step i
                episode_continuing = (1 - batch_terminals[:, i].float())

                state_action = torch.cat([latent_states, batch_actions[:, i]], dim=1)

                if latent_state_type == "deterministic":
                    next_latent_states = representation_net(batch_states[:, i + 1])
                    predicted_next_latent_states = dynamics_net.dynamics_net(
                        torch.cat([latent_states, batch_actions[:, i]], dim=1))

                    dynamics_loss = dynamics_loss_fn(
                        next_latent_states.detach(),
                        predicted_next_latent_states
                    ).sum(dim=1)
                    representation_loss = representation_loss_fn(
                        next_latent_states,
                        predicted_next_latent_states.detach()
                    ).sum(dim=1)

                else:
                    next_latent_states_dist = representation_net(batch_states[:, i + 1])
                    if latent_state_type == "gaussian":
                        next_latent_states = rsample_gaussian(mu=next_latent_states_dist[0], log_std=next_latent_states_dist[1]).flatten(1)
                    else:
                        next_latent_states = rsample_categorical(logits=next_latent_states_dist).flatten(1)

                    predicted_next_latent_states_dist = dynamics_net(state_action)

                    if latent_state_type == "gaussian":
                        dynamics_loss = dynamics_loss_fn(
                            mu1=next_latent_states_dist[0].detach(), log_std1=next_latent_states_dist[1].detach(),
                            mu2=predicted_next_latent_states_dist[0], log_std2=predicted_next_latent_states_dist[1],
                            aggregate_method=aggregate_method
                        )

                        representation_loss = representation_loss_fn(
                            mu1=next_latent_states_dist[0], log_std1=next_latent_states_dist[1],
                            mu2=predicted_next_latent_states_dist[0].detach(), log_std2=predicted_next_latent_states_dist[1].detach(),
                            aggregate_method=aggregate_method
                        )

                    elif latent_state_type == "discrete":

                        dynamics_loss = dynamics_loss_fn(
                            logits1=next_latent_states_dist.detach(),
                            logits2=predicted_next_latent_states_dist,
                            aggregate_method=aggregate_method
                        )

                        representation_loss = representation_loss_fn(
                            logits1=next_latent_states_dist,
                            logits2=predicted_next_latent_states_dist.detach(),
                            aggregate_method=aggregate_method
                        )
                    else:
                        raise ValueError(f"Latent state type: {latent_state_type} can only be deterministic, gaussian, or discrete")

                    if kl_clip is not None:
                        dynamics_loss = torch.clamp(dynamics_loss, min=kl_clip)
                        representation_loss = torch.clamp(representation_loss, min=kl_clip)

                assert dynamics_loss.shape == representation_loss.shape == episode_continuing.shape, "Shapes for dynamics and representation loss does not match episode continuing loss."
                dynamics_loss = (dynamics_loss * episode_continuing).sum() / episode_continuing.sum()
                representation_loss = (representation_loss * episode_continuing).sum() / episode_continuing.sum()

                # predict rewards and dones
                state_action_embedding = dynamics_net.embedding_net(state_action)
                predicted_rewards = dynamics_net.reward_net(state_action_embedding)
                predicted_dones_logits = dynamics_net.done_net(state_action_embedding)

                # calculate rewards and dones losses
                reward_prediction_loss = reward_prediction_loss_fn(predicted_rewards, batch_rewards[:, i]).mean()
                reward_prediction_losses.append(reward_prediction_loss)
                continuity_prediction_loss = continuity_prediction_loss_fn(predicted_dones_logits,
                                                                           batch_terminals[:, i].float()).mean()
                continuity_prediction_losses.append(continuity_prediction_loss)

                # calculate reconstruction_loss
                reconstructed_states = reconstruction_net(next_latent_states)
                image_prediction_loss = torch.clamp(
                    image_prediction_loss_fn(reconstructed_states, batch_states[:, i + 1]),
                    min=image_prediction_loss_clip).mean()
                image_prediction_losses.append(image_prediction_loss)

                if latent_state_type == "gaussian":
                    kl_loss = compute_KL_divergence_between_two_independent_gaussians(
                        next_latent_states_dist[0], next_latent_states_dist[1],
                        torch.zeros_like(next_latent_states_dist[0]).detach(), torch.ones_like(next_latent_states_dist[1]).detach(),
                        aggregate_method
                    )

                    kl_losses.append(kl_loss)

                # calculate dynamics loss and representation loss
                dynamics_losses.append(dynamics_loss)
                representation_losses.append(representation_loss)

                # update latent states
                latent_states = next_latent_states

            avg_image_prediction_loss = torch.stack(image_prediction_losses).mean()
            avg_reward_prediction_loss = torch.stack(reward_prediction_losses).mean()
            avg_continuity_prediction_loss = torch.stack(continuity_prediction_losses).mean()
            avg_dynamics_loss = torch.stack(dynamics_losses).mean()
            avg_representation_loss = torch.stack(representation_losses).mean()

            if latent_state_type == "gaussian":
                avg_kl_loss = torch.stack(kl_losses).mean()
                loss = avg_image_prediction_loss + avg_reward_prediction_loss + kl_beta * avg_kl_loss + avg_continuity_prediction_loss + dyn_loss_scale * avg_dynamics_loss + rep_loss_scale * avg_representation_loss
                epoch_kl_loss += avg_kl_loss.item()
            else:
                loss = avg_image_prediction_loss + avg_reward_prediction_loss + avg_continuity_prediction_loss + dyn_loss_scale * avg_dynamics_loss + rep_loss_scale * avg_representation_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norms = torch.nn.utils.clip_grad_norm_(
                list(representation_net.parameters()) + list(dynamics_net.parameters()) + list(reconstruction_net.parameters()),
                max_norm=500)

            if monitor_model:
                wandb.log({
                    "gradient_norms": grad_norms
                })

            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_reward_prediction_loss += avg_reward_prediction_loss.item()
            epoch_image_prediction_loss += avg_image_prediction_loss.item()
            epoch_continuity_prediction_loss += avg_continuity_prediction_loss.item()
            epoch_dynamics_loss += avg_dynamics_loss.item()
            epoch_representation_loss += avg_representation_loss.item()

        epoch_total_losses.append(epoch_total_loss / len(data_loader))
        epoch_reward_prediction_losses.append(epoch_reward_prediction_loss / len(data_loader))
        epoch_image_prediction_losses.append(epoch_image_prediction_loss / len(data_loader))
        epoch_continuity_prediction_losses.append(epoch_continuity_prediction_loss / len(data_loader))
        epoch_dynamics_losses.append(epoch_dynamics_loss / len(data_loader))
        epoch_representation_losses.append(epoch_representation_loss / len(data_loader))

        if latent_state_type == "gaussian":
            epoch_kl_losses.append(epoch_kl_loss / len(data_loader))
            message = (f"Epoch {epoch + 1} "
                       f"losses: total: {epoch_total_losses[-1]:.4f}, "
                       f"reconstruction: {epoch_image_prediction_losses[-1]:.4f}, "
                       f"kl: {epoch_kl_losses[-1]:.4f}, "
                       f"reward: {epoch_reward_prediction_losses[-1]:.4f}, "
                       f"done: {epoch_continuity_prediction_losses[-1]:.4f}, "
                       f"dynamics: {epoch_dynamics_losses[-1]:.4f}, "
                       f"representation: {epoch_representation_losses[-1]:.4f}")

            if monitor_model:
                wandb.log({
                    "loss": epoch_total_losses[-1],
                    "reconstruction": epoch_image_prediction_losses[-1],
                    "kl": epoch_kl_losses[-1],
                    "reward": epoch_reward_prediction_losses[-1],
                    "done": epoch_continuity_prediction_losses[-1],
                    "dynamics": epoch_dynamics_losses[-1],
                    "representation": epoch_representation_losses[-1]
                })
        else:
            message = (f"Epoch {epoch + 1} "
                       f"losses: total: {epoch_total_losses[-1]:.4f}, "
                       f"reconstruction: {epoch_image_prediction_losses[-1]:.4f}, "
                       f"reward: {epoch_reward_prediction_losses[-1]:.4f}, "
                       f"done: {epoch_continuity_prediction_losses[-1]:.4f}, "
                       f"dynamics: {epoch_dynamics_losses[-1]:.4f}, "
                       f"representation: {epoch_representation_losses[-1]:.4f}")

            if monitor_model:
                wandb.log({
                    "loss": epoch_total_losses[-1],
                    "reconstruction": epoch_image_prediction_losses[-1],
                    "reward": epoch_reward_prediction_losses[-1],
                    "done": epoch_continuity_prediction_losses[-1],
                    "dynamics": epoch_dynamics_losses[-1],
                    "representation": epoch_representation_losses[-1]
                })

        print(message, flush=True)
        with open(f"data/model_learning/{exp_name}/log.txt", "a") as f:
            f.write(message + "\n")

        if save_intermediary:
            # save all models
            os.makedirs(f"data/model_learning/{exp_name}/epoch_{epoch}", exist_ok=True)
            torch.save(representation_net, f"data/model_learning/{exp_name}/epoch_{epoch}/representation_net.pt")
            torch.save(dynamics_net, f"data/model_learning/{exp_name}/epoch_{epoch}/dynamics_net.pt")
            torch.save(reconstruction_net, f"data/model_learning/{exp_name}/epoch_{epoch}/reconstruction_net.pt")

    if monitor_model:
        wandb.unwatch()
        wandb.finish()

    # save model
    torch.save(representation_net, f"data/model_learning/{exp_name}/representation_net.pt")
    torch.save(dynamics_net, f"data/model_learning/{exp_name}/dynamics_net.pt")
    torch.save(reconstruction_net, f"data/model_learning/{exp_name}/reconstruction_net.pt")

    if latent_state_type == "gaussian":
        # save losses together
        np.savez_compressed(f"data/model_learning/{exp_name}/losses.npz",
                            total_losses=epoch_total_losses,
                            reward_prediction_losses=epoch_reward_prediction_losses,
                            image_prediction_losses=epoch_image_prediction_losses,
                            continuity_prediction_losses=epoch_continuity_prediction_losses,
                            dynamics_losses=epoch_dynamics_losses,
                            representation_losses=epoch_representation_losses,
                            kl_losses=epoch_kl_losses)
    else:
        np.savez_compressed(f"data/model_learning/{exp_name}/losses.npz",
                            total_losses=epoch_total_losses,
                            reward_prediction_losses=epoch_reward_prediction_losses,
                            image_prediction_losses=epoch_image_prediction_losses,
                            continuity_prediction_losses=epoch_continuity_prediction_losses,
                            dynamics_losses=epoch_dynamics_losses,
                            representation_losses=epoch_representation_losses)

    end_time = time.time()

    minutes, seconds = divmod(end_time - start_time, 60)
    hours, minutes = divmod(minutes, 60)

    print(f"Elapsed time: {int(hours)}:{int(minutes)}:{int(seconds)}", flush=True)
    with open(f"data/model_learning/{exp_name}/log.txt", "a") as f:
        f.write(f"Elapsed time: {int(hours)}:{int(minutes)}:{int(seconds)}\n")

    plot_training(exp_name=exp_name, latent_state_type=latent_state_type)


if __name__ == '__main__':
    fire.Fire(main)
