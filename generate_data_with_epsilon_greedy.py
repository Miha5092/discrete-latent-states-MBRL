import argparse
import torch
import numpy as np
from collections import namedtuple
from dqn import QNetwork
from minatar import Environment
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()

def main():
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--load_file', "-l", type=str)
    parser.add_argument('--output_folder', "-o", type=str, default="data/model_learning_data/reworked_deterministic_30")
    parser.add_argument('--game', "-g", type=str, default='quick_breakout')
    parser.add_argument('--num_episodes', "-n", type=int, default=100)
    parser.add_argument('--seed', "-s", type=int, default=42)
    parser.add_argument('--epsilon', "-e", type=float, default=0.0)
    args = parser.parse_args()
    print(args)

    env = Environment(args.game, sticky_action_prob=0.0, use_minimal_action_set=True)
    epsilon = args.epsilon

    # seeding
    seed = args.seed
    checkpoint = torch.load(args.load_file, map_location=device)
    random.seed(seed)
    np_seed = random.randint(0, 2**32 - 1)
    env_seed = random.randint(0, 2**32 - 1)
    print("seeds (np, env):", np_seed, env_seed)
    np.random.seed(np_seed)
    env.seed(env_seed)

    num_actions = env.num_actions()
    in_channels = env.state_shape()[2]
    policy_net = QNetwork(in_channels, num_actions).to(device)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    # S
    state_trajectories = []
    # A
    action_trajectories = []
    # R(S,A)
    reward_trajectories = []
    # Terminal after S and A
    terminal_trajectories = []

    for i in range(args.num_episodes):
        episode_rewards = 0
        env.seed(0)
        env.reset()
        done = False
        t = 0

        while not done:
            state = env.state()
            state_trajectories.append(state)
            if np.random.binomial(1, epsilon) == 1:
                action = torch.tensor([[random.randrange(num_actions)]], device=device)
            else:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            action = torch.tensor([[random.randint(0,2)]], device=device)
            action_trajectories.append(action.item())
            reward, done = env.act(action)
            reward_trajectories.append(reward)
            terminal_trajectories.append(done)
            episode_rewards += reward
            t += 1

        print(f"Episode {i+1} reward: {episode_rewards} timesteps: {t}")

    state_trajectories = np.array(state_trajectories)
    action_trajectories = np.array(action_trajectories)
    reward_trajectories = np.array(reward_trajectories)
    terminal_trajectories = np.array(terminal_trajectories)

    os.makedirs(args.output_folder, exist_ok=True)
    print(state_trajectories.shape)
    np.savez_compressed(f"{args.output_folder}/epsilon_{args.epsilon}_seed_{args.seed}_num_episodes_{args.num_episodes}.npz",
                        states=state_trajectories,
                        actions=action_trajectories,
                        rewards=reward_trajectories,
                        terminals=terminal_trajectories)

if __name__ == '__main__':
    main()