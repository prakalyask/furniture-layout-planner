from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from .room import Room, Zone, Placement, RoomFeature
from .furniture import Furniture, FurnitureFactory
from .fuzzy_logic import FuzzyEvaluator
from layout import IntegratedLayout
from utils.metrics import LayoutMetrics


class RoomOptimizer:
    """Class for optimizing room layouts."""

    def __init__(self, room: Room):
        """
        Initialize a room optimizer.

        Args:
            room: Room object
        """
        self.room = room
        self.evaluator = FuzzyEvaluator()
        self.integrated_layout = None

    def create_zones(self) -> None:
        """Create default zones in the room."""
        # Simple zone creation based on room size
        width = self.room.width
        height = self.room.height

        # Default zone allocation (percentages)
        entertainment_pct = 0.5
        dining_pct = 0.3
        storage_pct = 0.2

        # Check room shape
        if width >= height * 1.5:  # Wide room
            # Entertainment zone on the left
            entertainment_width = width * entertainment_pct
            entertainment = Zone(
                name="entertainment",
                x=0,
                y=0,
                width=entertainment_width,
                height=height
            )

            # Dining zone in the middle
            dining_width = width * dining_pct
            dining = Zone(
                name="dining",
                x=entertainment_width,
                y=0,
                width=dining_width,
                height=height
            )

            # Storage zone on the right
            storage_width = width * storage_pct
            storage = Zone(
                name="storage",
                x=entertainment_width + dining_width,
                y=0,
                width=storage_width,
                height=height
            )
        elif height >= width * 1.5:  # Tall room
            # Entertainment zone on the top
            entertainment_height = height * entertainment_pct
            entertainment = Zone(
                name="entertainment",
                x=0,
                y=0,
                width=width,
                height=entertainment_height
            )

            # Dining zone in the middle
            dining_height = height * dining_pct
            dining = Zone(
                name="dining",
                x=0,
                y=entertainment_height,
                width=width,
                height=dining_height
            )

            # Storage zone on the bottom
            storage_height = height * storage_pct
            storage = Zone(
                name="storage",
                x=0,
                y=entertainment_height + dining_height,
                width=width,
                height=storage_height
            )
        else:  # Square room
            # Entertainment zone on the left
            entertainment_width = width * 0.7
            entertainment_height = height * 0.7
            entertainment = Zone(
                name="entertainment",
                x=0,
                y=0,
                width=entertainment_width,
                height=entertainment_height
            )

            # Dining zone on the right
            dining = Zone(
                name="dining",
                x=entertainment_width,
                y=0,
                width=width - entertainment_width,
                height=entertainment_height
            )

            # Storage zone on the bottom
            storage = Zone(
                name="storage",
                x=0,
                y=entertainment_height,
                width=width,
                height=height - entertainment_height
            )

        # Add zones to room
        self.room.add_zone(entertainment)
        self.room.add_zone(dining)
        self.room.add_zone(storage)

    def optimize_layout(
        self,
        furniture_list: List[Furniture] = None,
        use_integrated_layout: bool = True
    ) -> Tuple[List[Placement], float]:
        """
        Optimize the entire room layout.

        Args:
            furniture_list: List of furniture to place (if None, use room's available furniture)
            use_integrated_layout: Whether to use the enhanced integrated layout

        Returns:
            Tuple of (placements, score)
        """
        # Reset room if necessary
        if self.room.placements:
            self.room.reset()

        # Use provided furniture or room's available furniture
        if furniture_list is None:
            furniture_list = self.room.available_furniture.copy()
        else:
            # Add furniture to room
            self.room.available_furniture = furniture_list.copy()

        if use_integrated_layout:
            # Use enhanced integrated layout with zone-specific algorithms
            self.integrated_layout = IntegratedLayout(self.room)
            placements, score = self.integrated_layout.optimize()
        else:
            # Use basic optimization (from original RoomOptimizer)
            placements = []

            # Sort furniture by importance
            furniture_list.sort(
                key=lambda f: self._get_furniture_importance(f))

            # Place furniture one by one
            for furniture in furniture_list:
                placement = self.optimize_placement(furniture)
                if placement:
                    placements.append(placement)

            # Evaluate final layout using metrics
            metrics = LayoutMetrics(self.room)
            all_metrics = metrics.calculate_all_metrics()
            score = all_metrics["overall_score"]

        return placements, score

    def optimize_placement(
        self,
        furniture: Furniture,
        num_candidates: int = 100
    ) -> Optional[Placement]:
        """
        Find optimal placement for a furniture piece.

        Args:
            furniture: Furniture to place
            num_candidates: Number of candidate positions to evaluate

        Returns:
            Optimal placement if found, None otherwise
        """
        best_placement, best_score = self.evaluator.find_best_placement(
            self.room,
            furniture,
            num_positions=num_candidates,
            num_orientations=furniture.orientations
        )

        if best_placement and best_score > 0.0:
            # Place furniture in the room
            self.room.place_furniture(
                furniture,
                best_placement.x,
                best_placement.y,
                best_placement.orientation
            )
            return best_placement

        return None

    def create_alternative_layouts(
        self,
        num_layouts: int = 3,
        furniture_list: List[Furniture] = None
    ) -> List[Tuple[List[Placement], float]]:
        """
        Create multiple alternative layouts.

        Args:
            num_layouts: Number of layouts to create
            furniture_list: List of furniture to place

        Returns:
            List of layout tuples (placements, score)
        """
        # Use provided furniture or room's available furniture
        if furniture_list is None:
            furniture_list = self.room.available_furniture.copy()

        # Create layouts with different strategies
        layouts = []

        # Strategy 1: Use integrated layout (zone-based)
        self.room.reset()
        placements, score = self.optimize_layout(
            furniture_list, use_integrated_layout=True)
        layouts.append((placements.copy(), score))

        # Strategy 2: Use basic optimization (furniture-based)
        self.room.reset()
        placements, score = self.optimize_layout(
            furniture_list, use_integrated_layout=False)
        layouts.append((placements.copy(), score))

        # Strategy 3: Prioritize entertainment zone
        self.room.reset()
        # Adjust zone sizes to prioritize entertainment
        for zone in self.room.zones:
            if zone.name == "entertainment":
                # Expand entertainment zone
                zone.width *= 1.2
                zone.height *= 1.2
                # Ensure within room bounds
                zone.width = min(zone.width, self.room.width)
                zone.height = min(zone.height, self.room.height)
        placements, score = self.optimize_layout(
            furniture_list, use_integrated_layout=True)
        layouts.append((placements.copy(), score))

        # Additional layouts if requested
        if num_layouts > 3:
            # Strategy 4: Prioritize dining zone
            self.room.reset()
            # Reset and adjust zone sizes to prioritize dining
            self.create_zones()  # Reset zones
            for zone in self.room.zones:
                if zone.name == "dining":
                    # Expand dining zone
                    zone.width *= 1.2
                    zone.height *= 1.2
                    # Ensure within room bounds
                    zone.width = min(zone.width, self.room.width)
                    zone.height = min(zone.height, self.room.height)
            placements, score = self.optimize_layout(
                furniture_list, use_integrated_layout=True)
            layouts.append((placements.copy(), score))

        if num_layouts > 4:
            # Strategy 5: Symmetric layout
            self.room.reset()
            self.create_zones()  # Reset zones
            # Create symmetrical zones
            width = self.room.width
            height = self.room.height

            # Entertainment zone in center
            entertainment = Zone(
                name="entertainment",
                x=width * 0.1,
                y=height * 0.1,
                width=width * 0.8,
                height=height * 0.5
            )

            # Dining zone on left
            dining = Zone(
                name="dining",
                x=width * 0.1,
                y=height * 0.6,
                width=width * 0.4,
                height=height * 0.3
            )

            # Storage zone on right
            storage = Zone(
                name="storage",
                x=width * 0.5,
                y=height * 0.6,
                width=width * 0.4,
                height=height * 0.3
            )

            # Clear and add new zones
            self.room.zones = []
            self.room.add_zone(entertainment)
            self.room.add_zone(dining)
            self.room.add_zone(storage)

            placements, score = self.optimize_layout(
                furniture_list, use_integrated_layout=True)
            layouts.append((placements.copy(), score))

        # Sort layouts by score (descending)
        layouts.sort(key=lambda x: x[1], reverse=True)

        return layouts[:num_layouts]

    def recommend_furniture(self, room_size: str = 'medium') -> List[Furniture]:
        """
        Recommend furniture based on room size.

        Args:
            room_size: Room size ('small', 'medium', 'large')

        Returns:
            List of recommended furniture
        """
        if room_size == 'small':
            # Small room (up to 200 sq ft)
            return [
                FurnitureFactory.create_tv(width=2.5, height=1.5),
                FurnitureFactory.create_sofa(width=5.0, height=2.5),
                FurnitureFactory.create_coffee_table(width=3.0, height=1.5),
                FurnitureFactory.create_bookshelf(),
                FurnitureFactory.create_side_table(),
                FurnitureFactory.create_lamp()
            ]
        elif room_size == 'medium':
            # Medium room (200-300 sq ft)
            return [
                FurnitureFactory.create_tv(width=3.0, height=2.0),
                FurnitureFactory.create_sofa(width=6.0, height=3.0),
                FurnitureFactory.create_coffee_table(width=4.0, height=2.0),
                FurnitureFactory.create_dining_table(width=4.0, height=3.0),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_bookshelf(),
                FurnitureFactory.create_side_table(),
                FurnitureFactory.create_lamp(),
                FurnitureFactory.create_plant()
            ]
        else:  # large
            # Large room (300+ sq ft)
            return [
                FurnitureFactory.create_tv(width=4.0, height=2.0),
                FurnitureFactory.create_sofa(width=7.0, height=3.0),
                FurnitureFactory.create_sofa(
                    width=5.0, height=3.0),  # Second sofa
                FurnitureFactory.create_coffee_table(width=5.0, height=2.5),
                FurnitureFactory.create_dining_table(width=6.0, height=3.5),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_dining_chair(),
                FurnitureFactory.create_bookshelf(),
                FurnitureFactory.create_bookshelf(),  # Second bookshelf
                FurnitureFactory.create_side_table(),
                FurnitureFactory.create_side_table(),  # Second side table
                FurnitureFactory.create_lamp(),
                FurnitureFactory.create_lamp(),  # Second lamp
                FurnitureFactory.create_plant(),
                FurnitureFactory.create_plant()  # Second plant
            ]

    def _get_furniture_importance(self, furniture: Furniture) -> int:
        """
        Get importance ranking of furniture (lower is more important).

        Args:
            furniture: Furniture object

        Returns:
            Importance ranking (0 is most important)
        """
        importance_order = {
            'tv': 0,
            'sofa': 1,
            'dining_table': 2,
            'coffee_table': 3,
            'bookshelf': 4,
            'dining_chair': 5,
            'side_table': 6,
            'lamp': 7,
            'plant': 8
        }

        return importance_order.get(furniture.furniture_type, 100)

    def __repr__(self) -> str:
        return f"RoomOptimizer(room={self.room})"
