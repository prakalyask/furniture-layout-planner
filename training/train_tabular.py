
import numpy as np
import json
import os
import time
import sys
import inspect
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Add parent directory to path to import from sibling modules
current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from agents.tabular_agent import TabularQAgent
from environment.furniture import Furniture
from environment.room import Room

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(dataset_path, "r") as f:
        return json.load(f)


def get_valid_actions(room: Room, furniture: Furniture) -> List[int]:
    """
    Get all valid actions for placing the given furniture in the room.

    Args:
        room: The room environment
        furniture: The furniture piece to place

    Returns:
        List of valid action indices (flat indices)
    """
    valid_actions = []

    # Try all possible positions and orientations
    for orientation in range(furniture.orientations):
        furniture_width, furniture_height = furniture.get_dimensions(
            orientation)
        orientation_offset = orientation * (room.width * room.height)

        for y in range(room.height - furniture_height + 1):
            for x in range(room.width - furniture_width + 1):
                # Check if placement is valid
                if room._is_valid_placement(furniture, x, y, orientation):
                    # Convert x,y,orientation to flat index
                    flat_idx = orientation_offset + y * room.width + x
                    valid_actions.append(flat_idx)

    return valid_actions


def train_agent(
    agent: TabularQAgent,
    dataset: List[Dict],
    num_episodes: int,
    save_dir: str = "models",
    checkpoint_freq: int = 100,
    log_freq: int = 10
):
    """
    Train the tabular Q-learning agent on the dataset.

    Args:
        agent: The Q-learning agent to train
        dataset: List of room layout examples
        num_episodes: Number of training episodes
        save_dir: Directory to save model checkpoints
        checkpoint_freq: Frequency to save model checkpoints
        log_freq: Frequency to log training progress
    """
    os.makedirs(save_dir, exist_ok=True)

    # Training metrics
    episode_rewards = []
    episode_lengths = []

    try:
        for episode in tqdm(range(num_episodes)):
            # Sample a random example from dataset
            example = dataset[np.random.randint(0, len(dataset))]

            # Setup room
            room = Room(width=example["room"]["width"],
                        height=example["room"]["height"])
            furniture_list = [
                Furniture(
                    name=f["name"],
                    width=f["width"],
                    height=f["height"],
                    orientations=f["orientations"]
                )
                for f in example["furniture"]
            ]
            room.add_furniture_list(furniture_list)

            # Reset environment
            state = room.reset()

            # Episode tracking
            episode_reward = 0
            episode_steps = 0

            # Run episode
            done = False
            while not done:
                # Get valid actions for the current state
                if room.current_furniture_idx >= len(furniture_list):
                    break

                valid_actions = get_valid_actions(
                    room, room.furniture_list[room.current_furniture_idx])

                if not valid_actions:
                    # No valid actions, terminate episode
                    break

                # Select action
                action = agent.select_action(state, valid_actions)

                # Take action
                next_state, reward, done, info = room.step(action)

                # Get valid actions for next state
                next_valid_actions = []
                if not done and room.current_furniture_idx < len(furniture_list):
                    next_valid_actions = get_valid_actions(
                        room, room.furniture_list[room.current_furniture_idx])

                # Update Q-table
                agent.update(state, action, reward, next_state,
                             done, next_valid_actions)

                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1

            # Decay epsilon
            agent.decay_epsilon()

            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            agent.episode_rewards.append(episode_reward)

            # Log progress
            if episode % log_freq == 0:
                avg_reward = np.mean(
                    episode_rewards[-log_freq:]) if episode_rewards else 0
                avg_length = np.mean(
                    episode_lengths[-log_freq:]) if episode_lengths else 0

                print(f"Episode {episode}/{num_episodes}")
                print(f"  Avg. Reward: {avg_reward:.2f}")
                print(f"  Avg. Length: {avg_length:.2f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  Q-table size: {len(agent.q_table)}")

            # Save checkpoint
            if episode % checkpoint_freq == 0:
                checkpoint_path = os.path.join(
                    save_dir, f"tabular_agent_episode_{episode}.pkl")
                agent.save(checkpoint_path)

        # Save final model
        agent.save(os.path.join(save_dir, "tabular_agent_final.pkl"))

        # Plot training curves
        plot_training_curves(episode_rewards, episode_lengths, save_dir)

    except Exception as e:
        print(f"Error during training: {e}")
        # Save model even if there was an error
        agent.save(os.path.join(save_dir, "tabular_agent_interrupted.pkl"))

    return episode_rewards, episode_lengths


def plot_training_curves(rewards, lengths, save_dir):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 5))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Plot episode lengths
    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves_tabular.png"))
    plt.close()


if __name__ == "__main__":
    # Load dataset
    dataset_path = os.path.join(parent_dir, "data", "synthetic_dataset.json")

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run generate_data.py first")
        sys.exit(1)

    dataset = load_dataset(dataset_path)

    # Get example room dimensions
    example = dataset[0]
    room_width, room_height = example["room"]["width"], example["room"]["height"]

    # Calculate action dimension (maximum possible)
    max_action_dim = room_width * room_height * 2  # 2 orientations

    # Initialize agent
    agent = TabularQAgent(max_action_dim)

    # Train agent
    num_episodes = 500  # Adjust as needed
    train_agent(agent, dataset, num_episodes)
