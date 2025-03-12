"""
Ultra-fast demo script for furniture layout optimization.
Completely simplified to avoid any computational bottlenecks.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from core.room import Room, RoomFeature, Placement, Zone
from core.furniture import FurnitureFactory
import time

def create_sample_layout():
    """Create a sample layout directly without optimization."""
    # Create room
    room = Room(width=15, height=12)
    
    # Add zones manually
    entertainment = Zone("entertainment", 0, 0, 10, 12)
    dining = Zone("dining", 10, 0, 5, 6)
    storage = Zone("storage", 10, 6, 5, 6)
    room.add_zone(entertainment)
    room.add_zone(dining)
    room.add_zone(storage)
    
    # Add door and window
    door = RoomFeature('door', 7, 0, 3, 0.5, 0)
    window = RoomFeature('window', 14.5, 5, 0.5, 3, 0)
    room.add_feature(door)
    room.add_feature(window)
    
    # Create furniture
    tv = FurnitureFactory.create_tv()
    sofa = FurnitureFactory.create_sofa()
    coffee_table = FurnitureFactory.create_coffee_table()
    bookshelf = FurnitureFactory.create_bookshelf()
    
    # Manually place furniture (predefined good arrangement)
    room.place_furniture(tv, 1, 1, 0)
    room.place_furniture(sofa, 1, 6, 0)
    room.place_furniture(coffee_table, 3, 4, 0)
    room.place_furniture(bookshelf, 10, 7, 0)
    
    return room

def create_alternative_layout():
    """Create an alternative layout directly."""
    # Create room
    room = Room(width=15, height=12)
    
    # Add zones manually
    entertainment = Zone("entertainment", 0, 0, 10, 12)
    dining = Zone("dining", 10, 0, 5, 6)
    storage = Zone("storage", 10, 6, 5, 6)
    room.add_zone(entertainment)
    room.add_zone(dining)
    room.add_zone(storage)
    
    # Add door and window
    door = RoomFeature('door', 7, 0, 3, 0.5, 0)
    window = RoomFeature('window', 14.5, 5, 0.5, 3, 0)
    room.add_feature(door)
    room.add_feature(window)
    
    # Create furniture
    tv = FurnitureFactory.create_tv()
    sofa = FurnitureFactory.create_sofa()
    coffee_table = FurnitureFactory.create_coffee_table()
    bookshelf = FurnitureFactory.create_bookshelf()
    
    # Manually place furniture (different arrangement)
    room.place_furniture(tv, 8, 1, 0)
    room.place_furniture(sofa, 2, 4, 1)  # Different orientation
    room.place_furniture(coffee_table, 5, 4, 0)
    room.place_furniture(bookshelf, 11, 6, 1)  # Different position
    
    return room

def visualize_room(room, title="Room Layout", save_path=None):
    """Simplified visualization function."""
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Draw room outline
    ax.add_patch(plt.Rectangle((0, 0), room.width, room.height, fill=False, edgecolor='black'))
    
    # Draw zones
    zone_colors = {
        "entertainment": "lightblue",
        "dining": "lightyellow",
        "storage": "lightgreen"
    }
    
    for zone in room.zones:
        ax.add_patch(plt.Rectangle(
            (zone.x, zone.y), zone.width, zone.height,
            fill=True, alpha=0.3, edgecolor='gray', facecolor=zone_colors.get(zone.name, "lightgray")
        ))
        ax.text(
            zone.x + zone.width/2, zone.y + zone.height/2, zone.name,
            ha='center', va='center', fontsize=10
        )
    
    # Draw features
    for feature in room.features:
        if feature.feature_type == 'door':
            color = 'brown'
        elif feature.feature_type == 'window':
            color = 'lightblue'
        else:
            color = 'gray'
        
        ax.add_patch(plt.Rectangle(
            (feature.x, feature.y), feature.width, feature.height,
            fill=True, edgecolor='black', facecolor=color
        ))
        
    # Draw furniture
    furniture_colors = {
        "tv": "royalblue",
        "sofa": "navy",
        "coffee_table": "purple",
        "bookshelf": "darkgreen"
    }
    
    for placement in room.placements:
        furniture = placement.furniture
        color = furniture_colors.get(furniture.furniture_type, "blue")
        
        # Get coordinates
        width, height = furniture.get_dimensions(placement.orientation)
        
        ax.add_patch(plt.Rectangle(
            (placement.x, placement.y), width, height,
            fill=True, edgecolor='black', facecolor=color
        ))
        ax.text(
            placement.x + width/2, placement.y + height/2, furniture.name,
            ha='center', va='center', fontsize=8, color='white'
        )
    
    # Set limits and title
    ax.set_xlim(-1, room.width + 1)
    ax.set_ylim(-1, room.height + 1)
    ax.set_title(title)
    ax.set_aspect('equal')
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig, ax

def run_fastest_demo():
    """Run the fastest possible demo."""
    print("Starting ultra-fast furniture layout demo...")
    start_time = time.time()
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create and visualize first layout
    print("Creating first layout...")
    room1 = create_sample_layout()
    print("Visualizing first layout...")
    visualize_room(
        room1, 
        title="Layout 1 (TV along left wall)", 
        save_path='output/layout1.png'
    )
    
    # Create and visualize second layout
    print("Creating second layout...")
    room2 = create_alternative_layout()
    print("Visualizing second layout...")
    visualize_room(
        room2, 
        title="Layout 2 (TV along right wall)", 
        save_path='output/layout2.png'
    )
    
    # Compare layouts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # First layout
    visualize_room(room1, title="Layout 1", save_path=None)[1].figure = fig
    ax1.figure = fig
    
    # Second layout
    visualize_room(room2, title="Layout 2", save_path=None)[1].figure = fig
    ax2.figure = fig
    
    plt.tight_layout()
    plt.savefig('output/comparison.png', dpi=150, bbox_inches='tight')
    
    end_time = time.time()
    print(f"Demo completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to 'output/' directory")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    run_fastest_demo()