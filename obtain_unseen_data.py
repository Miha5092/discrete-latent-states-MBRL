import functools
import multiprocessing
import os
import pickle
from multiprocessing import Pool, cpu_count
import fire
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns


def save_transitions(data: list, output_folder: str, file_name: str) -> None:
    with open(output_folder + f"/{file_name}.pkl", 'wb') as f:
        pickle.dump(data, f)



def display_transition(transition):
    first_frame = transition[0]
    action = transition[1]
    second_frame = transition[2]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cmap = sns.color_palette("cubehelix", 4)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(4 + 2)]
    norm = colors.BoundaryNorm(bounds, 4 + 1)
    numerical_state = np.amax(first_frame.transpose(1, 2, 0) * np.reshape(np.arange(4) + 1, (1, 1, -1)), 2) + 0.5
    reconstructed_numerical_state = np.amax(second_frame.transpose(1, 2, 0) * np.reshape(np.arange(4) + 1, (1, 1, -1)),
                                            2) + 0.5
    ax[0].imshow(numerical_state, cmap=cmap, norm=norm, interpolation='none')
    ax[0].axis('off')
    ax[1].imshow(reconstructed_numerical_state, cmap=cmap, norm=norm, interpolation='none')
    ax[1].axis('off')

    fig.suptitle(f"Transition with action {action}")

    plt.show()


def equal_transitions(trans1, trans2):
    """Check if two transitions are equal."""
    return np.array_equal(trans1[0], trans2[0]) and trans1[1] == trans2[1] and np.array_equal(trans1[2], trans2[2])


def transition_not_in_training(test_transition, training_transitions):
    """Check if a test transition is not in training transitions."""
    for train_transition in training_transitions:
        if equal_transitions(test_transition, train_transition):
            return False, None
    return True, test_transition


def get_transitions(dataset: np.array) -> list:
    frames = dataset['states'].transpose(0, 3, 1, 2)
    actions = dataset['actions']
    dones = dataset['terminals']
    rewards = dataset['rewards']

    transitions = list(zip(frames[:-1], actions[:-1, 0], frames[1:], dones[:-1, 0], rewards[:-1, 0]))
    return transitions


def obtain_new(
        test_dataset: str = "data/dqn_optimal_seed_0/experience_data_episode_1000.npz",
        training_dataset: str = "data/dqn_training_seed_0/experience_data_episode_10000.npz",
        no_to_check: int = 100_000,
        required_no_transitions: int = 10_000,
        output_path: str = "data/unseen_transitions",
        save_interval: int = 1000
):
    # Load the datasets
    test_dataset = np.load(test_dataset)
    training_dataset = np.load(training_dataset)

    training_transitions = get_transitions(training_dataset)
    test_transitions = get_transitions(test_dataset)
    test_transitions = test_transitions[:no_to_check]

    print(f"Number of training transitions: {len(training_transitions)}")
    print(f"Number of test transitions: {len(test_transitions)}")

    # Remove seen transitions
    os.makedirs(output_path, exist_ok=True)

    unseen_transitions = []

    p = multiprocessing.Pool()
    for i, transition in enumerate(test_transitions):

        unseen = transition_not_in_training(transition, training_transitions)

        if unseen[0]:
            unseen_transitions.append(unseen[1])

        if len(unseen_transitions) != 0 and len(unseen_transitions) % save_interval == 0:
            print(f"Saving intermediary with size: {len(unseen_transitions)} after checking: {i}")
            save_transitions(unseen_transitions, output_path, f"intermediate_save_{len(unseen_transitions)}")

        if len(unseen_transitions) >= required_no_transitions:
            p.terminate()
            p.join()
            break

    save_transitions(unseen_transitions, output_path, "final")


if __name__ == "__main__":
    fire.Fire(obtain_new)
