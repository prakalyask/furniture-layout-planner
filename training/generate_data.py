from environment.furniture import Furniture, create_common_furniture
import numpy as np
import json
from typing import List, Dict, Tuple
import os
import sys
import inspect

# Add parent directory to path to import from sibling modules
current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def generate_room_dimensions(min_size: int = 8, max_size: int = 20) -> Tuple[int, int]:
    """Generate random room dimensions."""
    width = np.random.randint(min_size, max_size + 1)
    height = np.random.randint(min_size, max_size + 1)
    return width, height


def generate_furniture_list(num_pieces: int = None) -> List[Furniture]:
    """Generate a list of furniture to be placed."""
    all_furniture = create_common_furniture()

    if num_pieces is None:
        # Random number of pieces
        num_pieces = np.random.randint(3, len(all_furniture) + 1)

    # Randomly select furniture pieces
    selected_indices = np.random.choice(
        len(all_furniture), num_pieces, replace=False)
    return [all_furniture[i] for i in selected_indices]


def generate_dataset(
    num_examples: int,
    output_dir: str,
    min_room_size: int = 8,
    max_room_size: int = 20
) -> None:
    """
    Generate a synthetic dataset of room layouts.

    Args:
        num_examples: Number of examples to generate
        output_dir: Directory to save dataset
        min_room_size: Minimum room dimension
        max_room_size: Maximum room dimension
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = []

    for i in range(num_examples):
        # Generate room dimensions
        width, height = generate_room_dimensions(min_room_size, max_room_size)

        # Generate furniture list
        furniture_list = generate_furniture_list()

        # Create example
        example = {
            "id": i,
            "room": {
                "width": width,
                "height": height
            },
            "furniture": [
                {
                    "name": f.name,
                    "width": f.width,
                    "height": f.height,
                    "orientations": f.orientations
                }
                for f in furniture_list
            ]
        }

        dataset.append(example)

    # Save dataset
    with open(os.path.join(output_dir, "synthetic_dataset.json"), "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {num_examples} synthetic room layouts.")


if __name__ == "__main__":
    # Generate a small dataset for testing
    generate_dataset(10, "data")
