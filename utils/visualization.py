from core.furniture import Furniture
from core.room import Room, Zone, Placement, RoomFeature
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import to_rgba
import sys
import os

# Add parent directory to path to import from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Define color scheme
COLORS = {
    'wall': '#343a40',
    'door': '#ced4da',
    'window': '#adb5bd',
    'zone_entertainment': to_rgba('#e9ecef', 0.3),
    'zone_dining': to_rgba('#dee2e6', 0.3),
    'zone_storage': to_rgba('#ced4da', 0.3),
    'zone_general': to_rgba('#adb5bd', 0.3),
    'furniture_tv': '#4361ee',
    'furniture_sofa': '#3a0ca3',
    'furniture_coffee_table': '#7209b7',
    'furniture_dining_table': '#f72585',
    'furniture_dining_chair': '#b5179e',
    'furniture_bookshelf': '#560bad',
    'furniture_side_table': '#480ca8',
    'furniture_lamp': '#3f37c9',
    'furniture_plant': '#4cc9f0',
    'clearance': to_rgba('#ff4d6d', 0.2),
    'relationship': to_rgba('#fb8500', 0.7)
}


def visualize_room(
    room: Room,
    show_grid: bool = False,
    show_zones: bool = True,
    show_relationships: bool = True,
    show_clearance: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    title: str = None,
    ax=None,
    save_path: str = None
):
    """
    Visualize the room layout.

    Args:
        room: Room object
        show_grid: Whether to show the grid
        show_zones: Whether to show zones
        show_relationships: Whether to show furniture relationships
        show_clearance: Whether to show furniture clearance
        figsize: Figure size (width, height) in inches
        title: Plot title
        ax: Matplotlib axis (if None, a new figure is created)
        save_path: Path to save the figure (if None, figure is not saved)

    Returns:
        Matplotlib axis
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Set title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(
            f"Room Layout ({room.width}' x {room.height}')", fontsize=14)

    # Set axis limits with margin
    margin = 2
    ax.set_xlim(-margin, room.width + margin)
    ax.set_ylim(-margin, room.height + margin)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Show grid
    if show_grid:
        step = room.grid_size
        x_ticks = np.arange(0, room.width + step, step)
        y_ticks = np.arange(0, room.height + step, step)

        ax.set_xticks(x_ticks, minor=True)
        ax.set_yticks(y_ticks, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', alpha=0.2)

    # Draw room outline
    room_rect = patches.Rectangle(
        (0, 0),
        room.width,
        room.height,
        linewidth=2,
        edgecolor=COLORS['wall'],
        facecolor='none',
        zorder=1
    )
    ax.add_patch(room_rect)

    # Draw zones if available
    if show_zones and room.zones:
        for zone in room.zones:
            zone_color = COLORS.get(
                f'zone_{zone.name}', COLORS['zone_general'])
            zone_rect = patches.Rectangle(
                (zone.x, zone.y),
                zone.width,
                zone.height,
                linewidth=1,
                edgecolor='gray',
                facecolor=zone_color,
                linestyle='--',
                zorder=2
            )
            ax.add_patch(zone_rect)

            # Add zone label
            ax.text(
                zone.x + zone.width / 2,
                zone.y + zone.height / 2,
                zone.name.capitalize(),
                fontsize=10,
                color='gray',
                ha='center',
                va='center',
                zorder=3
            )

    # Draw features (doors, windows, etc.)
    for feature in room.features:
        x, y, width, height = feature.get_coords()

        feature_color = COLORS.get(feature.feature_type, 'gray')
        feature_rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=1,
            edgecolor='black',
            facecolor=feature_color,
            zorder=4
        )
        ax.add_patch(feature_rect)

        # Add feature label
        ax.text(
            x + width / 2,
            y + height / 2,
            feature.feature_type.capitalize(),
            fontsize=8,
            color='black',
            ha='center',
            va='center',
            zorder=5
        )

    # Draw furniture clearance if requested
    if show_clearance:
        for placement in room.placements:
            x, y, width, height = placement.get_coords()
            clearance = placement.furniture.clearance_required

            # Draw clearance area
            if clearance > 0:
                clearance_rect = patches.Rectangle(
                    (x - clearance, y - clearance),
                    width + 2 * clearance,
                    height + 2 * clearance,
                    linewidth=1,
                    edgecolor=COLORS['clearance'],
                    facecolor=COLORS['clearance'],
                    zorder=6
                )
                ax.add_patch(clearance_rect)

    # Draw furniture
    for placement in room.placements:
        x, y, width, height = placement.get_coords()

        furniture_type = placement.furniture.furniture_type
        furniture_color = COLORS.get(f'furniture_{furniture_type}', 'blue')

        furniture_rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=1.5,
            edgecolor='black',
            facecolor=furniture_color,
            zorder=7
        )
        ax.add_patch(furniture_rect)

        # Add furniture label
        ax.text(
            x + width / 2,
            y + height / 2,
            placement.furniture.name,
            fontsize=8,
            color='white',
            ha='center',
            va='center',
            zorder=8
        )

        # Draw facing direction
        dir_x, dir_y = placement.get_facing_direction()
        center_x, center_y = placement.get_center()

        # Calculate arrow endpoints
        arrow_length = min(width, height) * 0.4
        end_x = center_x + dir_x * arrow_length
        end_y = center_y + dir_y * arrow_length

        ax.arrow(
            center_x, center_y,
            dir_x * arrow_length, dir_y * arrow_length,
            head_width=0.3,
            head_length=0.3,
            fc='white',
            ec='black',
            zorder=9
        )

    # Draw furniture relationships if requested
    if show_relationships:
        for placement in room.placements:
            # Skip if no relationships
            if not placement.furniture.related_furniture:
                continue

            # Draw relationships
            for related_type in placement.furniture.related_furniture:
                related_placements = room.get_placement_by_type(related_type)
                if not related_placements:
                    continue

                # Get relationship properties
                optimal_distance = placement.furniture.optimal_distances.get(
                    related_type, 0)

                # Draw relationship to each related furniture
                for related in related_placements:
                    # Calculate centers
                    c1_x, c1_y = placement.get_center()
                    c2_x, c2_y = related.get_center()

                    # Draw line
                    ax.plot(
                        [c1_x, c2_x],
                        [c1_y, c2_y],
                        linestyle=':',
                        color=COLORS['relationship'],
                        linewidth=1.5,
                        zorder=6
                    )

                    # Draw optimal distance (if applicable)
                    if optimal_distance > 0:
                        # Calculate midpoint
                        mid_x = (c1_x + c2_x) / 2
                        mid_y = (c1_y + c2_y) / 2

                        # Calculate actual distance
                        actual_distance = placement.distance_to(related)

                        # Add distance label
                        ax.text(
                            mid_x,
                            mid_y,
                            f"{actual_distance:.1f}' (opt: {optimal_distance:.1f}')",
                            fontsize=7,
                            color='black',
                            ha='center',
                            va='center',
                            bbox=dict(facecolor='white',
                                      alpha=0.7, edgecolor='none'),
                            zorder=10
                        )

    # Set labels
    ax.set_xlabel('Width (feet)', fontsize=12)
    ax.set_ylabel('Height (feet)', fontsize=12)

    # Remove ticks
    ax.set_xticks([0, room.width])
    ax.set_yticks([0, room.height])

    # Tight layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


def visualize_alternative_layouts(
    room: Room,
    layouts: List[Tuple[List[Placement], float]],
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize alternative layouts.

    Args:
        room: Room object
        layouts: List of layout tuples (placements, score)
        figsize: Figure size (width, height) in inches
    """
    # Calculate grid dimensions
    n_layouts = len(layouts)
    n_cols = min(3, n_layouts)
    n_rows = (n_layouts + n_cols - 1) // n_cols

    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single row or column case
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs.reshape(1, -1)
    elif n_cols == 1:
        axs = axs.reshape(-1, 1)

    # Draw each layout
    for i, (placements, score) in enumerate(layouts):
        row = i // n_cols
        col = i % n_cols

        # Make a fresh copy of the room
        room_copy = Room(room.width, room.height,
                         room.grid_size, room.wall_thickness)
        room_copy.features = room.features.copy()
        room_copy.zones = room.zones.copy()

        # Add placements
        for placement in placements:
            room_copy.place_furniture(
                placement.furniture,
                placement.x,
                placement.y,
                placement.orientation
            )

        # Visualize
        visualize_room(
            room_copy,
            show_grid=False,
            show_zones=True,
            show_relationships=True,
            show_clearance=False,
            title=f"Layout {i+1}: Score = {score:.2f}",
            ax=axs[row, col]
        )

    # Hide empty subplots
    for i in range(n_layouts, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axs[row, col].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    return fig
