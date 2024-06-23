from mpi4py import MPI
import torch
from minatar import Environment
import random
import numpy as np
import copy
import fire
import os
import math
import time
from mcts import Node, MCTS
import multiprocessing as multiprocessing

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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


class TrueModel:

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def step(self, env, action):
        copied_env = copy.deepcopy(env)
        reward, terminal = copied_env.act(action)
        return copied_env, reward, terminal

    def encode(self, env):
        return copy.deepcopy(env)


def evaluate_episode(param):
    model_folder, eval_seed, mcts_params = param

    # load the models
    representation_net = torch.load(f'{model_folder}/representation_net.pt', map_location=device)
    dynamics_net = torch.load(f'{model_folder}/dynamics_net.pt', map_location=device)
    reconstruction_net = torch.load(f'{model_folder}/reconstruction_net.pt', map_location=device)

    # set the models to evaluation mode
    representation_net.eval()
    dynamics_net.eval()
    reconstruction_net.eval()

    # make the environment and model
    env = Environment('quick_breakout', sticky_action_prob=0.0, use_minimal_action_set=True)
    learned_model = LearnedModel(env.num_actions(), (representation_net, dynamics_net))
    model = learned_model

    # set the random seeds
    random.seed(eval_seed)
    env_seed = random.randint(0, 2 ** 31)
    torch_seed = random.randint(0, 2 ** 31)
    np_seed = random.randint(0, 2 ** 31)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    np.random.seed(np_seed)
    env.seed(env_seed)

    # reset the environment
    env.reset()

    # get mcts hyperparameters from mcts_params
    exploration_constant = mcts_params["exploration_constant"]
    discount_factor = mcts_params["discount_factor"]
    num_simulations = mcts_params["num_simulations"]
    planning_horizon = mcts_params["planning_horizon"] if "planning_horizon" in mcts_params else math.inf

    # the agent-environment loop
    with torch.inference_mode():
        undiscounted_return = 0
        done = False
        t = 0
        while not done:
            z = model.encode(env)
            root = Node(
                state=z,
                reward=0.0,
                continue_probability=1.0,
                parent=None,
                action=None,
                num_actions=env.num_actions()
            )
            best_action = MCTS(root, model, num_simulations, exploration_constant, discount_factor, planning_horizon)
            reward, done = env.act(best_action)
            undiscounted_return += reward
            t += 1
            if t >= 1000:
                break
            # print(t, reward, done)
    print(f"Finished episode after {t} steps with return {undiscounted_return}", flush=True)
    return undiscounted_return


def main(output_folder: str, model_folder: str, seed: int, num_episodes: int, exploration_constant: float = 1.0,
         discount_factor: float = 0.97, planning_horizon: int = 32, num_simulations: int = 128):
    os.makedirs(output_folder, exist_ok=True)

    start_time = time.time()

    local_params = None
    if rank == 0:

        # save params
        with open(f"{output_folder}/params.txt", 'w') as f:
            f.write(f"model_folder: {model_folder}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"num_episodes: {num_episodes}\n")
            f.write(f"exploration_constant: {exploration_constant}\n")
            f.write(f"discount_factor: {discount_factor}\n")

        # seeding
        random.seed(seed)
        print("master seed:", seed, flush=True)
        # generate random seeds for each episode
        eval_seeds = [random.randint(0, 2 ** 31) for _ in range(num_episodes)]

        mcts_params = {
            "exploration_constant": exploration_constant,
            "discount_factor": discount_factor,
            "num_simulations": num_simulations,
            "planning_horizon": planning_horizon if planning_horizon is not None else math.inf
        }

        job_params = [(model_folder, eval_seeds[i], mcts_params) for i in range(num_episodes)]
        print(f"jobs created with seeds {eval_seeds}.", flush=True)

        local_params = comm.scatter([job_params[i::size] for i in range(size)], root=0)
    else:
        local_params = comm.scatter(None, root=0)

    local_results = [evaluate_episode(param) for param in local_params]

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        episode_returns = [result for sublist in all_results for result in sublist]  # Flatten list of results

        np.save(f"{output_folder}/episode_returns.npy", np.array(episode_returns))
        print(f"Episode returns: {episode_returns}")
        print(f"Mean episode return: {np.mean(episode_returns)}")
        print(f"Median episode return: {np.median(episode_returns)}")
        print(f"Std episode return: {np.std(episode_returns)}")

        end_time = time.time()

        minutes, seconds = divmod(end_time - start_time, 60)
        hours, minutes = divmod(minutes, 60)

        print(f"Elapsed time: {int(hours)}:{int(minutes)}:{int(seconds)}", flush=True)

        with open(f"{output_folder}/log.txt", 'w') as f:
            f.write(f"Episode returns: {episode_returns}\n")
            f.write(f"Mean episode return: {np.mean(episode_returns)}\n")
            f.write(f"Median episode return: {np.median(episode_returns)}\n")
            f.write(f"Std episode return: {np.std(episode_returns)}\n")
            f.write(f"Elapsed time: {int(hours)}:{int(minutes)}:{int(seconds)}\n")



if __name__ == '__main__':
    fire.Fire(main)
