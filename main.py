import os
import argparse
import matplotlib.pyplot as plt
from core.room import Room, RoomFeature
from core.furniture import FurnitureFactory
from core.optimizer import RoomOptimizer
from utils.visualization import visualize_room, visualize_alternative_layouts
import time

def create_sample_room():
    room = Room(width=15, height=12)
    door = RoomFeature('door', 7, 0, 3, 0.5, 0)
    window = RoomFeature('window', 14.5, 5, 0.5, 3, 0)
    room.add_feature(door)
    room.add_feature(window)
    return room

def optimize_demo():
    start_time = time.time()
    print("Starting furniture layout demo...")
    
    room = create_sample_room()
    print(f"Room dimensions: {room.width}' x {room.height}'")
    
    optimizer = RoomOptimizer(room)
    optimizer.create_zones()
    print("Created zones: entertainment, dining, storage")
    
    furniture = [
        FurnitureFactory.create_tv(),
        FurnitureFactory.create_sofa(),
        FurnitureFactory.create_coffee_table(),
        FurnitureFactory.create_bookshelf()
    ]
    
    for piece in furniture:
        room.add_furniture(piece)
    
    print(f"Added {len(furniture)} furniture pieces")
    print("Generating layouts...")
    layouts = optimizer.create_alternative_layouts(2, furniture)
    print(f"Generated {len(layouts)} alternative layouts")
    
    best_placements, best_score = layouts[0]
    
    room.reset()
    for placement in best_placements:
        room.place_furniture(
            placement.furniture,
            placement.x,
            placement.y,
            placement.orientation
        )
    
    os.makedirs('output', exist_ok=True)
    
    print("Creating visualizations...")
    plt.figure(figsize=(10, 8))
    visualize_room(
        room,
        show_grid=False,
        show_zones=True,
        show_relationships=True,
        show_clearance=False,
        title=f"Optimized Room Layout (Score: {best_score:.2f})",
        save_path='output/optimized_layout.png'
    )
    
    fig = visualize_alternative_layouts(room, layouts)
    fig.savefig('output/alternative_layouts.png', dpi=150, bbox_inches='tight')
    
    end_time = time.time()
    print(f"Demo completed in {end_time - start_time:.2f} seconds")
    print(f"Best layout score: {best_score:.2f}")
    print(f"Output saved to 'output/' directory")
    
    plt.show()

def run_efficient_demo():
    print("Starting efficient furniture layout demo...")
    start_time = time.time()
    
    room = create_sample_room()
    print(f"Created room: {room.width}' x {room.height}'")
    
    optimizer = RoomOptimizer(room)
    optimizer.create_zones()
    print("Created room zones")
    
    furniture = [
        FurnitureFactory.create_tv(),
        FurnitureFactory.create_sofa(),
        FurnitureFactory.create_coffee_table(),
        FurnitureFactory.create_bookshelf()
    ]
    
    for piece in furniture:
        room.add_furniture(piece)
    
    print(f"Added {len(furniture)} furniture pieces")
    
    print("Optimizing layout...")
    placements, score = optimizer.optimize_layout(furniture, use_integrated_layout=False)
    
    print(f"Optimization completed with score: {score:.2f}")
    print(f"Successfully placed {len(placements)} of {len(furniture)} furniture pieces")
    
    os.makedirs('output', exist_ok=True)
    
    print("Generating visualization...")
    plt.figure(figsize=(10, 8))
    visualize_room(
        room,
        show_grid=False,
        show_zones=True,
        show_relationships=True,
        show_clearance=False,
        title=f"Optimized Layout (Score: {score:.2f})",
        save_path='output/optimized_layout.png'
    )
    
    print("Generating alternative layout...")
    room.reset()
    
    alternative_furniture = furniture.copy()
    alternative_furniture.reverse()
    
    alt_placements, alt_score = optimizer.optimize_layout(alternative_furniture, use_integrated_layout=False)
    
    plt.figure(figsize=(10, 8))
    visualize_room(
        room,
        show_grid=False,
        show_zones=True,
        show_relationships=True,
        show_clearance=False,
        title=f"Alternative Layout (Score: {alt_score:.2f})",
        save_path='output/alternative_layout.png'
    )
    
    end_time = time.time()
    print(f"Demo completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to 'output/' directory")
    
    plt.show()

def run_fastest_demo():
    print("Starting ultra-fast furniture layout demo...")
    start_time = time.time()
    
    os.makedirs('output', exist_ok=True)
    
    print("Creating first layout...")
    room1 = create_sample_layout()
    print("Visualizing first layout...")
    visualize_simple_room(
        room1, 
        title="Layout 1 (TV along left wall)", 
        save_path='output/layout1.png'
    )
    
    print("Creating second layout...")
    room2 = create_alternative_layout()
    print("Visualizing second layout...")
    visualize_simple_room(
        room2, 
        title="Layout 2 (TV along right wall)", 
        save_path='output/layout2.png'
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    visualize_simple_room(room1, title="Layout 1", save_path=None)[1].figure = fig
    ax1.figure = fig
    
    visualize_simple_room(room2, title="Layout 2", save_path=None)[1].figure = fig
    ax2.figure = fig
    
    plt.tight_layout()
    plt.savefig('output/comparison.png', dpi=150, bbox_inches='tight')
    
    end_time = time.time()
    print(f"Demo completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to 'output/' directory")
    
    plt.show()

def create_sample_layout():
    room = Room(width=15, height=12)
    
    entertainment = Zone("entertainment", 0, 0, 10, 12)
    dining = Zone("dining", 10, 0, 5, 6)
    storage = Zone("storage", 10, 6, 5, 6)
    room.add_zone(entertainment)
    room.add_zone(dining)
    room.add_zone(storage)
    
    door = RoomFeature('door', 7, 0, 3, 0.5, 0)
    window = RoomFeature('window', 14.5, 5, 0.5, 3, 0)
    room.add_feature(door)
    room.add_feature(window)
    
    tv = FurnitureFactory.create_tv()
    sofa = FurnitureFactory.create_sofa()
    coffee_table = FurnitureFactory.create_coffee_table()
    bookshelf = FurnitureFactory.create_bookshelf()
    
    room.place_furniture(tv, 1, 1, 0)
    room.place_furniture(sofa, 1, 6, 0)
    room.place_furniture(coffee_table, 3, 4, 0)
    room.place_furniture(bookshelf, 10, 7, 0)
    
    return room

def create_alternative_layout():
    room = Room(width=15, height=12)
    
    entertainment = Zone("entertainment", 0, 0, 10, 12)
    dining = Zone("dining", 10, 0, 5, 6)
    storage = Zone("storage", 10, 6, 5, 6)
    room.add_zone(entertainment)
    room.add_zone(dining)
    room.add_zone(storage)
    
    door = RoomFeature('door', 7, 0, 3, 0.5, 0)
    window = RoomFeature('window', 14.5, 5, 0.5, 3, 0)
    room.add_feature(door)
    room.add_feature(window)
    
    tv = FurnitureFactory.create_tv()
    sofa = FurnitureFactory.create_sofa()
    coffee_table = FurnitureFactory.create_coffee_table()
    bookshelf = FurnitureFactory.create_bookshelf()
    
    room.place_furniture(tv, 8, 1, 0)
    room.place_furniture(sofa, 2, 4, 1)
    room.place_furniture(coffee_table, 5, 4, 0)
    room.place_furniture(bookshelf, 11, 6, 1)
    
    return room

def visualize_simple_room(room, title="Room Layout", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.add_patch(plt.Rectangle((0, 0), room.width, room.height, fill=False, edgecolor='black'))
    
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
        
    furniture_colors = {
        "tv": "royalblue",
        "sofa": "navy",
        "coffee_table": "purple",
        "bookshelf": "darkgreen"
    }
    
    for placement in room.placements:
        furniture = placement.furniture
        color = furniture_colors.get(furniture.furniture_type, "blue")
        
        width, height = furniture.get_dimensions(placement.orientation)
        
        ax.add_patch(plt.Rectangle(
            (placement.x, placement.y), width, height,
            fill=True, edgecolor='black', facecolor=color
        ))
        ax.text(
            placement.x + width/2, placement.y + height/2, furniture.name,
            ha='center', va='center', fontsize=8, color='white'
        )
    
    ax.set_xlim(-1, room.width + 1)
    ax.set_ylim(-1, room.height + 1)
    ax.set_title(title)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig, ax

def run_optimized_demo():
    print("Starting optimized furniture layout demo...")
    start_time = time.time()
    
    room = create_sample_room()
    print(f"Created room: {room.width}' x {room.height}'")
    
    optimizer = RoomOptimizer(room)
    optimizer.create_zones()
    print("Created room zones")
    
    furniture = [
        FurnitureFactory.create_tv(),
        FurnitureFactory.create_sofa(),
        FurnitureFactory.create_coffee_table(),
        FurnitureFactory.create_bookshelf()
    ]
    
    for piece in furniture:
        room.add_furniture(piece)
    
    print(f"Added {len(furniture)} furniture pieces")
    
    print("Optimizing first layout...")
    placements, score = optimizer.optimize_layout(furniture, use_integrated_layout=False)
    
    print(f"Optimization completed with score: {score:.2f}")
    print(f"Successfully placed {len(placements)} of {len(furniture)} furniture pieces")
    
    os.makedirs('output', exist_ok=True)
    
    print("Generating visualization...")
    plt.figure(figsize=(10, 8))
    visualize_room(
        room,
        show_grid=False,
        show_zones=True,
        show_relationships=True,
        show_clearance=False,
        title=f"Optimized Layout (Score: {score:.2f})",
        save_path='output/optimized_layout.png'
    )
    
    print("Generating alternative layout...")
    print("Resetting room...")
    room.reset()
    
    alternative_furniture = furniture.copy()
    alternative_furniture.reverse()
    
    for piece in alternative_furniture:
        room.add_furniture(piece)
    
    print("Optimizing alternative layout...")
    alt_placements, alt_score = optimizer.optimize_layout(alternative_furniture, use_integrated_layout=True)
    
    print("Visualizing alternative layout...")
    plt.figure(figsize=(10, 8))
    visualize_room(
        room,
        show_grid=False,
        show_zones=True,
        show_relationships=True,
        show_clearance=False,
        title=f"Alternative Layout (Score: {alt_score:.2f})",
        save_path='output/alternative_layout.png'
    )
    
    print("Creating layout comparison...")
    plt.figure(figsize=(16, 6))
    plt.suptitle("Layout Comparison", fontsize=16)
    
    plt.subplot(1, 2, 1)
    img1 = plt.imread('output/optimized_layout.png')
    plt.imshow(img1)
    plt.axis('off')
    plt.title(f"Layout 1 (Score: {score:.2f})")
    
    plt.subplot(1, 2, 2)
    img2 = plt.imread('output/alternative_layout.png')
    plt.imshow(img2)
    plt.axis('off')
    plt.title(f"Layout 2 (Score: {alt_score:.2f})")
    
    plt.tight_layout()
    plt.savefig('output/comparison.png', dpi=150, bbox_inches='tight')
    
    end_time = time.time()
    print(f"Demo completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to 'output/' directory")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Furniture Layout Optimizer')
    parser.add_argument('--demo', action='store_true', help='Run the optimization demo')
    # parser.add_argument('--fast', action='store_true', help='Run a simplified fast demo')
    parser.add_argument('--fastest', action='store_true', help='Run the fastest possible demo')
    parser.add_argument('--optimized', action='store_true', help='Run optimized demo with actual optimization')
    args = parser.parse_args()
    
    if args.optimized:
        run_optimized_demo()
    elif args.fastest:
        run_fastest_demo()
    elif args.fast:
        run_efficient_demo()
    elif args.demo:
        optimize_demo()
    else:
        run_optimized_demo()

if __name__ == '__main__':
    main()