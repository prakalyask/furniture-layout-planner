"""
Dining zone layout algorithm.
Handles optimal placement of dining tables, chairs, etc.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from core.room import Room, Zone, Placement, RoomFeature
from core.furniture import Furniture


class DiningLayout:
    """Class for optimizing dining zone layout."""

    def __init__(self, room: Room, zone: Zone):
        """
        Initialize a dining zone layout optimizer.

        Args:
            room: Room object
            zone: Dining zone
        """
        self.room = room
        self.zone = zone
        # Store furniture by type for quick access
        self.furniture_by_type = {}

    def get_furniture_of_type(self, furniture_type: str) -> List[Furniture]:
        """Get all furniture of a specific type in the room's available furniture."""
        if furniture_type in self.furniture_by_type:
            return self.furniture_by_type[furniture_type]

        furniture_list = [f for f in self.room.available_furniture
                          if f.furniture_type == furniture_type]
        self.furniture_by_type[furniture_type] = furniture_list
        return furniture_list

    def optimize(self) -> List[Placement]:
        """
        Optimize the dining zone layout.

        Returns:
            List of furniture placements
        """
        placements = []

        # First, place the dining table (the focal point)
        table_placement = self._place_dining_table()
        if table_placement:
            placements.append(table_placement)

        # Next, place chairs around the table
        chair_placements = self._place_dining_chairs(table_placement)
        placements.extend(chair_placements)

        return placements

    def _find_center_position(self) -> Tuple[float, float]:
        """
        Find the center position for the dining table.

        Returns:
            Tuple of (center_x, center_y)
        """
        # Default to zone center
        center_x = self.zone.x + self.zone.width / 2
        center_y = self.zone.y + self.zone.height / 2

        # Check if any walls nearby
        wall_distance = 3.0  # Minimum distance from walls

        # Adjust for walls
        if center_x - self.zone.x < wall_distance:
            center_x = self.zone.x + wall_distance
        elif self.zone.x + self.zone.width - center_x < wall_distance:
            center_x = self.zone.x + self.zone.width - wall_distance

        if center_y - self.zone.y < wall_distance:
            center_y = self.zone.y + wall_distance
        elif self.zone.y + self.zone.height - center_y < wall_distance:
            center_y = self.zone.y + self.zone.height - wall_distance

        return center_x, center_y

    def _place_dining_table(self) -> Optional[Placement]:
        """
        Place the dining table in the optimal position.

        Returns:
            Dining table placement if successful, None otherwise
        """
        # Get dining table furniture
        tables = self.get_furniture_of_type('dining_table')
        if not tables:
            return None

        table = tables[0]  # Use the first dining table

        # Find center position
        center_x, center_y = self._find_center_position()

        # Try different orientations
        best_placement = None
        best_score = -float('inf')

        for orientation in range(table.orientations):
            table_width, table_height = table.get_dimensions(orientation)

            # Calculate position (centered)
            x = center_x - table_width / 2
            y = center_y - table_height / 2

            # Ensure position is within zone bounds
            x = max(self.zone.x, min(x, self.zone.x +
                    self.zone.width - table_width))
            y = max(self.zone.y, min(y, self.zone.y +
                    self.zone.height - table_height))

            # Create placement
            placement = Placement(table, x, y, orientation)

            # Check if valid
            if not self.room._is_valid_placement(placement):
                continue

            # Calculate distance to center
            distance_to_center = np.sqrt(
                (x + table_width / 2 - center_x) ** 2 +
                (y + table_height / 2 - center_y) ** 2
            )

            # Calculate distance to walls
            min_wall_distance = min(
                x - self.zone.x,  # Left wall
                self.zone.x + self.zone.width -
                (x + table_width),  # Right wall
                y - self.zone.y,  # Bottom wall
                self.zone.y + self.zone.height - (y + table_height)  # Top wall
            )

            # Calculate score (prefer centered position with adequate wall clearance)
            center_score = 1.0 - min(distance_to_center / 5.0, 1.0)
            wall_score = min(min_wall_distance / 3.0, 1.0)

            # Calculate total score
            score = center_score * 0.7 + wall_score * 0.3

            if score > best_score:
                best_score = score
                best_placement = placement

        # Add best placement
        if best_placement:
            self.room.place_furniture(
                table,
                best_placement.x,
                best_placement.y,
                best_placement.orientation
            )
            return best_placement

        return None

    def _place_dining_chairs(self, table_placement: Optional[Placement]) -> List[Placement]:
        """
        Place dining chairs around the table.

        Args:
            table_placement: Dining table placement

        Returns:
            List of dining chair placements
        """
        placements = []

        # Get dining chair furniture
        chairs = self.get_furniture_of_type('dining_chair')
        if not chairs or not table_placement:
            return placements

        # Get table position
        table_x, table_y, table_width, table_height = table_placement.get_coords()

        # Calculate number of chairs per side
        chairs_per_long_side = max(1, int(max(table_width, table_height) / 2))
        chairs_per_short_side = max(1, int(min(table_width, table_height) / 2))

        # Calculate chair positions around table
        positions = []

        if table_placement.orientation % 2 == 0:  # Table in horizontal orientation
            # Chairs on top side
            for i in range(chairs_per_long_side):
                x = table_x + (i + 0.5) * table_width / \
                    chairs_per_long_side - chairs[0].width / 2
                y = table_y - chairs[0].height - 0.5
                positions.append((x, y, 2))  # Facing down

            # Chairs on bottom side
            for i in range(chairs_per_long_side):
                x = table_x + (i + 0.5) * table_width / \
                    chairs_per_long_side - chairs[0].width / 2
                y = table_y + table_height + 0.5
                positions.append((x, y, 0))  # Facing up

            # Chairs on left side
            for i in range(chairs_per_short_side):
                x = table_x - chairs[0].width - 0.5
                y = table_y + (i + 0.5) * table_height / \
                    chairs_per_short_side - chairs[0].height / 2
                positions.append((x, y, 1))  # Facing right

            # Chairs on right side
            for i in range(chairs_per_short_side):
                x = table_x + table_width + 0.5
                y = table_y + (i + 0.5) * table_height / \
                    chairs_per_short_side - chairs[0].height / 2
                positions.append((x, y, 3))  # Facing left
        else:  # Table in vertical orientation
            # Chairs on left side
            for i in range(chairs_per_long_side):
                x = table_x - chairs[0].width - 0.5
                y = table_y + (i + 0.5) * table_height / \
                    chairs_per_long_side - chairs[0].height / 2
                positions.append((x, y, 1))  # Facing right

            # Chairs on right side
            for i in range(chairs_per_long_side):
                x = table_x + table_width + 0.5
                y = table_y + (i + 0.5) * table_height / \
                    chairs_per_long_side - chairs[0].height / 2
                positions.append((x, y, 3))  # Facing left

            # Chairs on top side
            for i in range(chairs_per_short_side):
                x = table_x + (i + 0.5) * table_width / \
                    chairs_per_short_side - chairs[0].width / 2
                y = table_y - chairs[0].height - 0.5
                positions.append((x, y, 2))  # Facing down

            # Chairs on bottom side
            for i in range(chairs_per_short_side):
                x = table_x + (i + 0.5) * table_width / \
                    chairs_per_short_side - chairs[0].width / 2
                y = table_y + table_height + 0.5
                positions.append((x, y, 0))  # Facing up

        # Try each position for each chair
        for i, position in enumerate(positions):
            if i >= len(chairs):
                break

            chair = chairs[i]
            x, y, orientation = position

            # Ensure position is within zone bounds
            x = max(self.zone.x, min(x, self.zone.x +
                    self.zone.width - chair.width))
            y = max(self.zone.y, min(y, self.zone.y +
                    self.zone.height - chair.height))

            # Create placement
            placement = Placement(chair, x, y, orientation)

            # Check if valid
            if self.room._is_valid_placement(placement):
                # Add to room
                self.room.place_furniture(chair, x, y, orientation)
                placements.append(placement)

        return placements
