import streamlit as st
import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for streamlit

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import project modules
from utils.metrics import LayoutMetrics
from utils.visualization import visualize_room, visualize_alternative_layouts
from core.optimizer import RoomOptimizer
from core.furniture import Furniture, FurnitureFactory
from core.room import Room, Zone, Placement, RoomFeature
# Set page configuration
st.set_page_config(
    page_title="AI Furniture Layout Planner",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("AI Furniture Layout Planner")
st.markdown("""
This application uses AI to generate optimal furniture layouts for rooms based on design principles, 
ergonomics, and spatial relationships. Customize your room and furniture, and let the AI suggest 
the best arrangements.
""")

# Create session state variables if they don't exist
if 'room' not in st.session_state:
    st.session_state.room = None
if 'layouts' not in st.session_state:
    st.session_state.layouts = []
if 'current_layout_idx' not in st.session_state:
    st.session_state.current_layout_idx = 0
if 'furniture_list' not in st.session_state:
    st.session_state.furniture_list = []
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'room_features' not in st.session_state:
    st.session_state.room_features = []

# Function to reset the application


def reset_app():
    st.session_state.room = None
    st.session_state.layouts = []
    st.session_state.current_layout_idx = 0
    st.session_state.furniture_list = []
    st.session_state.optimization_complete = False
    st.session_state.metrics = {}
    st.session_state.room_features = []

# Initialize furniture options for selection


def get_furniture_options():
    return {
        "TV Stand": {"type": "tv", "width": (2.0, 6.0), "height": (1.5, 3.0), "orientations": 1},
        "Sofa": {"type": "sofa", "width": (5.0, 8.0), "height": (2.5, 4.0), "orientations": 4},
        "Coffee Table": {"type": "coffee_table", "width": (2.0, 5.0), "height": (1.5, 3.0), "orientations": 2},
        "Dining Table": {"type": "dining_table", "width": (3.0, 6.0), "height": (3.0, 6.0), "orientations": 2},
        "Dining Chair": {"type": "dining_chair", "width": (1.5, 2.0), "height": (1.5, 2.0), "orientations": 4},
        "Bookshelf": {"type": "bookshelf", "width": (2.0, 4.0), "height": (1.0, 2.0), "orientations": 2},
        "Side Table": {"type": "side_table", "width": (1.0, 2.0), "height": (1.0, 2.0), "orientations": 1},
        "Lamp": {"type": "lamp", "width": (1.0, 1.5), "height": (1.0, 1.5), "orientations": 1},
        "Plant": {"type": "plant", "width": (1.0, 2.0), "height": (1.0, 2.0), "orientations": 1}
    }


# Create sidebar for room configuration
st.sidebar.header("Room Configuration")

# Room dimensions
room_width = st.sidebar.slider("Room Width (feet)", 10, 30, 20, 1)
room_height = st.sidebar.slider("Room Height (feet)", 10, 30, 15, 1)

# Feature controls
st.sidebar.subheader("Room Features")

# Door controls
add_door = st.sidebar.checkbox("Add Door", value=True)
if add_door:
    door_position = st.sidebar.selectbox(
        "Door Position",
        ["Bottom Wall", "Top Wall", "Left Wall", "Right Wall"],
        index=0
    )
    door_width = st.sidebar.slider("Door Width (feet)", 2.0, 5.0, 3.0, 0.5)

# Window controls
add_window = st.sidebar.checkbox("Add Window", value=True)
if add_window:
    window_position = st.sidebar.selectbox(
        "Window Position",
        ["Bottom Wall", "Top Wall", "Left Wall", "Right Wall"],
        index=3
    )
    window_size = st.sidebar.slider("Window Size (feet)", 2.0, 8.0, 4.0, 0.5)

# Furniture selection
st.sidebar.header("Furniture Selection")

# Room size recommendation
room_area = room_width * room_height
room_size_category = "small" if room_area < 200 else "medium" if room_area < 300 else "large"
st.sidebar.info(f"Room area: {room_area} sq ft ({room_size_category} room)")

# Furniture presets based on room size
furniture_presets = {
    "small": ["TV Stand", "Sofa", "Coffee Table", "Bookshelf"],
    "medium": ["TV Stand", "Sofa", "Coffee Table", "Dining Table", "Dining Chair", "Dining Chair", "Bookshelf", "Side Table"],
    "large": ["TV Stand", "Sofa", "Sofa", "Coffee Table", "Dining Table", "Dining Chair", "Dining Chair", "Dining Chair", "Dining Chair", "Bookshelf", "Bookshelf", "Side Table", "Side Table", "Lamp", "Plant"]
}

preset_option = st.sidebar.radio(
    "Furniture Selection",
    ["Recommended for Room Size", "Custom Selection"],
    index=0
)

furniture_options = get_furniture_options()
selected_furniture = []

if preset_option == "Recommended for Room Size":
    preset_furniture = furniture_presets[room_size_category]

    st.sidebar.write(f"Recommended furniture for {room_size_category} room:")
    for furniture_name in preset_furniture:
        st.sidebar.write(f"- {furniture_name}")

    if st.sidebar.button("Use Recommended Furniture"):
        # Create furniture from preset
        for furniture_name in preset_furniture:
            options = furniture_options[furniture_name]

            # Use default values in the middle of the range
            width = (options["width"][0] + options["width"][1]) / 2
            height = (options["height"][0] + options["height"][1]) / 2

            # Create custom furniture for dining chairs to match the table
            if furniture_name == "Dining Chair":
                dining_chair = Furniture(
                    name=furniture_name,
                    width=width,
                    height=height,
                    furniture_type=options["type"],
                    orientations=options["orientations"],
                    zone="dining"
                )
                dining_chair.add_relationship("dining_table", 0.0)
                dining_chair.set_facing("dining_table")
                selected_furniture.append(dining_chair)
            else:
                # Use factory for other furniture
                factory_method = getattr(
                    FurnitureFactory, f"create_{options['type']}")
                furniture = factory_method(width=width, height=height)
                selected_furniture.append(furniture)
else:
    # Custom furniture selection
    furniture_to_add = st.sidebar.multiselect(
        "Select Furniture to Add",
        list(furniture_options.keys())
    )

    for furniture_name in furniture_to_add:
        options = furniture_options[furniture_name]
        width = st.sidebar.slider(
            f"{furniture_name} Width (feet)",
            options["width"][0],
            options["width"][1],
            (options["width"][0] + options["width"][1]) / 2,
            0.5
        )
        height = st.sidebar.slider(
            f"{furniture_name} Height (feet)",
            options["height"][0],
            options["height"][1],
            (options["height"][0] + options["height"][1]) / 2,
            0.5
        )

        if st.sidebar.button(f"Add {furniture_name}"):
            if furniture_name == "Dining Chair":
                dining_chair = Furniture(
                    name=furniture_name,
                    width=width,
                    height=height,
                    furniture_type=options["type"],
                    orientations=options["orientations"],
                    zone="dining"
                )
                dining_chair.add_relationship("dining_table", 0.0)
                dining_chair.set_facing("dining_table")
                selected_furniture.append(dining_chair)
            else:
                factory_method = getattr(
                    FurnitureFactory, f"create_{options['type']}")
                furniture = factory_method(width=width, height=height)
                selected_furniture.append(furniture)

            st.sidebar.success(f"Added {furniture_name}")

# Optimization controls
st.sidebar.header("Optimization Options")

num_layouts = st.sidebar.slider("Number of Alternative Layouts", 1, 5, 3, 1)
use_integrated_layout = st.sidebar.checkbox("Use Enhanced Layout Algorithm", value=True,
                                            help="More sophisticated but slower algorithm")

# Generate button
generate_btn = st.sidebar.button("Generate Layouts", type="primary")

# Main content area
main_col, details_col = st.columns([2, 1])

with main_col:
    st.header("Room Layout")

    # Room visualization
    room_viz_placeholder = st.empty()

    if generate_btn or (st.session_state.room is not None and st.session_state.optimization_complete):
        if generate_btn:
            # Create a new room
            room = Room(width=room_width, height=room_height)

            # Add door if selected
            if add_door:
                if door_position == "Bottom Wall":
                    door = RoomFeature('door', room_width /
                                       2 - door_width/2, 0, door_width, 0.5, 0)
                elif door_position == "Top Wall":
                    door = RoomFeature(
                        'door', room_width/2 - door_width/2, room_height - 0.5, door_width, 0.5, 0)
                elif door_position == "Left Wall":
                    door = RoomFeature(
                        'door', 0, room_height/2 - door_width/2, 0.5, door_width, 0)
                else:  # Right Wall
                    door = RoomFeature(
                        'door', room_width - 0.5, room_height/2 - door_width/2, 0.5, door_width, 0)
                room.add_feature(door)
                st.session_state.room_features.append(door)

            # Add window if selected
            if add_window:
                if window_position == "Bottom Wall":
                    window = RoomFeature(
                        'window', room_width/2 - window_size/2, 0, window_size, 0.5, 0)
                elif window_position == "Top Wall":
                    window = RoomFeature(
                        'window', room_width/2 - window_size/2, room_height - 0.5, window_size, 0.5, 0)
                elif window_position == "Left Wall":
                    window = RoomFeature(
                        'window', 0, room_height/2 - window_size/2, 0.5, window_size, 0)
                else:  # Right Wall
                    window = RoomFeature(
                        'window', room_width - 0.5, room_height/2 - window_size/2, 0.5, window_size, 0)
                room.add_feature(window)
                st.session_state.room_features.append(window)

            # Create optimizer
            optimizer = RoomOptimizer(room)

            # Create zones
            optimizer.create_zones()

            # Use either selected furniture or recommended furniture
            if selected_furniture:
                furniture_list = selected_furniture
            else:
                furniture_list = st.session_state.furniture_list

                if not furniture_list:
                    # Use recommended furniture by default
                    furniture_list = optimizer.recommend_furniture(
                        room_size_category)

            # Add furniture to room
            for furniture in furniture_list:
                room.add_furniture(furniture)

            # Store in session state
            st.session_state.room = room
            st.session_state.furniture_list = furniture_list

            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Creating zones...")
            time.sleep(0.5)  # For UI responsiveness
            progress_bar.progress(10)

            status_text.text("Analyzing furniture relationships...")
            time.sleep(0.5)
            progress_bar.progress(20)

            status_text.text("Generating alternative layouts...")

            # Generate layouts
            layouts = optimizer.create_alternative_layouts(
                num_layouts, furniture_list)

            # Store layouts
            st.session_state.layouts = layouts
            st.session_state.current_layout_idx = 0
            st.session_state.optimization_complete = True

            # Calculate metrics for best layout
            best_layout, best_score = layouts[0]
            metrics = LayoutMetrics(room)
            all_metrics = metrics.calculate_all_metrics()
            st.session_state.metrics = all_metrics

            # Place furniture according to best layout
            room.reset()
            for placement in best_layout:
                room.place_furniture(
                    placement.furniture,
                    placement.x,
                    placement.y,
                    placement.orientation
                )

            status_text.text("Layouts generated successfully!")
            progress_bar.progress(100)

            # Clear progress elements after completion
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

        # Get current layout
        current_idx = st.session_state.current_layout_idx
        current_layout, current_score = st.session_state.layouts[current_idx]

        # Visualize layout
        fig, ax = plt.subplots(figsize=(10, 8))
        visualize_room(
            st.session_state.room,
            show_grid=False,
            show_zones=True,
            show_relationships=True,
            show_clearance=False,
            title=f"Layout {current_idx + 1} (Score: {current_score:.2f})",
            ax=ax
        )
        room_viz_placeholder.pyplot(fig)

        # Layout navigation
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if st.button("Previous Layout", disabled=current_idx == 0):
                # Show previous layout
                st.session_state.current_layout_idx = (
                    current_idx - 1) % len(st.session_state.layouts)

                # Update room with previous layout
                previous_layout, _ = st.session_state.layouts[st.session_state.current_layout_idx]
                st.session_state.room.reset()
                for placement in previous_layout:
                    st.session_state.room.place_furniture(
                        placement.furniture,
                        placement.x,
                        placement.y,
                        placement.orientation
                    )

                # Recalculate metrics
                metrics = LayoutMetrics(st.session_state.room)
                st.session_state.metrics = metrics.calculate_all_metrics()

                st.rerun()

        with col2:
            st.write(
                f"Layout {current_idx + 1} of {len(st.session_state.layouts)}")

        with col3:
            if st.button("Next Layout", disabled=current_idx == len(st.session_state.layouts) - 1):
                # Show next layout
                st.session_state.current_layout_idx = (
                    current_idx + 1) % len(st.session_state.layouts)

                # Update room with next layout
                next_layout, _ = st.session_state.layouts[st.session_state.current_layout_idx]
                st.session_state.room.reset()
                for placement in next_layout:
                    st.session_state.room.place_furniture(
                        placement.furniture,
                        placement.x,
                        placement.y,
                        placement.orientation
                    )

                # Recalculate metrics
                metrics = LayoutMetrics(st.session_state.room)
                st.session_state.metrics = metrics.calculate_all_metrics()

                st.rerun()

    else:
        # Show empty room
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.add_patch(plt.Rectangle((0, 0), room_width,
                     room_height, fill=False, edgecolor='black'))
        ax.set_xlim(-1, room_width + 1)
        ax.set_ylim(-1, room_height + 1)
        ax.set_title(f"Empty Room ({room_width}' x {room_height}')")
        ax.set_aspect('equal')
        room_viz_placeholder.pyplot(fig)

        st.info(
            "Configure your room and furniture in the sidebar, then click 'Generate Layouts'.")

with details_col:
    if st.session_state.optimization_complete:
        # Layout details
        st.header("Layout Details")

        # Show furniture placements
        st.subheader("Furniture Placement")

        current_layout, _ = st.session_state.layouts[st.session_state.current_layout_idx]

        for i, placement in enumerate(current_layout):
            furniture = placement.furniture
            x, y, width, height = placement.get_coords()

            st.markdown(f"**{i+1}. {furniture.name}**")
            st.markdown(f"Position: ({x:.1f}', {y:.1f}')")
            st.markdown(f"Size: {width:.1f}' × {height:.1f}'")
            st.markdown(f"Zone: {furniture.zone}")

            if i < len(current_layout) - 1:
                st.markdown("---")

        # Show quality metrics
        st.subheader("Quality Metrics")

        if st.session_state.metrics:
            metrics = st.session_state.metrics

            # Overall score
            st.metric("Overall Score", f"{metrics['overall_score']:.2f}")

            # Individual metrics
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Space Efficiency",
                          f"{metrics['space_efficiency']:.2f}")
                st.metric("Zone Coherence", f"{metrics['zone_coherence']:.2f}")
                st.metric("Clearance Score",
                          f"{metrics['clearance_score']:.2f}")

            with col2:
                st.metric("Relationship Score",
                          f"{metrics['relationship_score']:.2f}")
                st.metric("Pathway Quality",
                          f"{metrics['pathway_quality']:.2f}")
                st.metric("Aesthetics Score",
                          f"{metrics['aesthetics_score']:.2f}")

        # Download button
        if os.path.exists("output/optimized_layout.png"):
            with open("output/optimized_layout.png", "rb") as f:
                btn = st.download_button(
                    label="Download Layout as Image",
                    data=f.read(),
                    file_name=f"furniture_layout_{st.session_state.current_layout_idx + 1}.png",
                    mime="image/png"
                )
        else:
            st.download_button(
                label="Download Layout as Image",
                data=None,
                file_name="furniture_layout.png",
                mime="image/png",
                disabled=True
            )

        # Reset button
        if st.button("Start Over", type="secondary"):
            reset_app()
            st.rerun()
    else:
        # Help information
        st.header("How It Works")

        st.markdown("""
        **1. Configure Room**
        - Set room dimensions
        - Add doors and windows
        
        **2. Select Furniture**
        - Choose recommended furniture or customize
        - Adjust furniture dimensions
        
        **3. Generate Layouts**
        - AI creates optimal arrangements
        - Browse alternative layouts
        
        **How the AI Works**
        
        The AI uses fuzzy logic and spatial optimization to create furniture layouts based on:
        
        - **Furniture Relationships**: Sofas face TVs, coffee tables near sofas, etc.
        - **Zone Planning**: Entertainment, dining, and storage zones
        - **Ergonomics**: Ensures comfortable walking paths and clearances
        - **Design Principles**: Balance, symmetry, and aesthetic considerations
        
        Each layout is scored on multiple metrics to ensure an optimal arrangement.
        """)

# Footer
st.markdown("---")
st.caption(
    "AI Furniture Layout Planner | Created with Streamlit and Reinforcement Learning")
