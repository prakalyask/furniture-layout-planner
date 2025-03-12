"""
Helper functions for the Streamlit application.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from core.room import Room, Zone, Placement, RoomFeature
from core.furniture import Furniture, FurnitureFactory
from utils.metrics import LayoutMetrics

def get_image_as_base64(fig):
    """
    Convert matplotlib figure to base64 encoded image.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded image
    """
    # Save figure to a PNG in memory
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def create_comparison_image(layouts, room):
    """
    Create a comparison image of all layouts.
    
    Args:
        layouts: List of layout tuples (placements, score)
        room: Room object
        
    Returns:
        Matplotlib figure with all layouts
    """
    n_layouts = len(layouts)
    if n_layouts == 0:
        return None
        
    # Calculate grid dimensions
    n_cols = min(3, n_layouts)
    n_rows = (n_layouts + n_cols - 1) // n_cols
    
    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    
    # Handle single axes case
    if n_layouts == 1:
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
        room_copy = Room(room.width, room.height, room.grid_size, room.wall_thickness)
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
        from utils.visualization import visualize_room
        visualize_room(
            room_copy,
            show_grid=False,
            show_zones=True,
            show_relationships=False,
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

def create_metrics_comparison(layouts, room):
    """
    Create a metrics comparison of all layouts.
    
    Args:
        layouts: List of layout tuples (placements, score)
        room: Room object
        
    Returns:
        Dictionary of metric comparisons
    """
    if not layouts:
        return {}
        
    metrics_comparison = {
        "scores": [],
        "space_efficiency": [],
        "relationship_score": [],
        "zone_coherence": [],
        "pathway_quality": [],
        "clearance_score": [],
        "balance_score": [],
        "aesthetics_score": []
    }
    
    for i, (placements, score) in enumerate(layouts):
        # Make a fresh copy of the room
        room_copy = Room(room.width, room.height, room.grid_size, room.wall_thickness)
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
        
        # Calculate metrics
        metrics = LayoutMetrics(room_copy)
        all_metrics = metrics.calculate_all_metrics()
        
        # Add to comparison
        metrics_comparison["scores"].append(score)
        for key in all_metrics:
            if key in metrics_comparison:
                metrics_comparison[key].append(all_metrics[key])
    
    return metrics_comparison

def get_furniture_details(placement):
    """
    Get detailed information about a furniture placement.
    
    Args:
        placement: Furniture placement
        
    Returns:
        Dictionary of furniture details
    """
    furniture = placement.furniture
    x, y, width, height = placement.get_coords()
    
    details = {
        "name": furniture.name,
        "type": furniture.furniture_type,
        "position": (x, y),
        "dimensions": (width, height),
        "orientation": placement.orientation,
        "zone": furniture.zone,
        "relationships": []
    }
    
    # Add relationship details
    for related_type in furniture.related_furniture:
        relationship = {
            "related_to": related_type,
            "optimal_distance": furniture.optimal_distances.get(related_type, 0.0)
        }
        details["relationships"].append(relationship)
        
    if furniture.facing_furniture:
        details["facing"] = furniture.facing_furniture
        
    return details

def generate_layout_report(room, layout_idx, placements, score, metrics):
    """
    Generate a detailed layout report.
    
    Args:
        room: Room object
        layout_idx: Layout index
        placements: List of furniture placements
        score: Layout score
        metrics: Layout metrics
        
    Returns:
        Report as markdown string
    """
    # Basic layout information
    report = f"""
    # Layout {layout_idx + 1} Report
    
    ## Overview
    
    - **Room Dimensions**: {room.width}' × {room.height}'
    - **Room Area**: {room.width * room.height} sq ft
    - **Furniture Pieces**: {len(placements)}
    - **Overall Score**: {score:.2f}
    
    ## Quality Metrics
    
    | Metric | Score |
    | ------ | ----- |
    | Space Efficiency | {metrics['space_efficiency']:.2f} |
    | Relationship Score | {metrics['relationship_score']:.2f} |
    | Zone Coherence | {metrics['zone_coherence']:.2f} |
    | Pathway Quality | {metrics['pathway_quality']:.2f} |
    | Clearance Score | {metrics['clearance_score']:.2f} |
    | Balance Score | {metrics['balance_score']:.2f} |
    | Aesthetics Score | {metrics['aesthetics_score']:.2f} |
    
    ## Furniture Placement
    """
    
    # Furniture details
    for i, placement in enumerate(placements):
        furniture = placement.furniture
        x, y, width, height = placement.get_coords()
        
        report += f"""
    ### {i+1}. {furniture.name}
    
    - **Position**: ({x:.1f}', {y:.1f}')
    - **Size**: {width:.1f}' × {height:.1f}'
    - **Zone**: {furniture.zone}
    - **Orientation**: {placement.orientation}
    """
        
        # Add relationship details if any
        if furniture.related_furniture:
            report += "\n    #### Relationships\n"
            for related_type in furniture.related_furniture:
                optimal_distance = furniture.optimal_distances.get(related_type, 0.0)
                report += f"\n    - {related_type.capitalize()}: Optimal distance {optimal_distance}'"
                
    return report

def modify_placement(room, placement, new_x, new_y, new_orientation):
    """
    Modify a furniture placement.
    
    Args:
        room: Room object
        placement: Original placement
        new_x: New X coordinate
        new_y: New Y coordinate
        new_orientation: New orientation
        
    Returns:
        New placement if successful, None otherwise
    """
    # Remove original placement
    furniture = placement.furniture
    room.placements.remove(placement)
    room.available_furniture.append(furniture)
    
    # Create new placement
    new_placement = room.place_furniture(furniture, new_x, new_y, new_orientation)
    
    return new_placement

def calculate_room_size