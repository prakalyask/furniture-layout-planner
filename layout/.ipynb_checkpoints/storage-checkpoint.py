"""
Storage zone layout algorithm.
Handles optimal placement of bookshelves, cabinets, etc.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from core.room import Room, Zone, Placement, RoomFeature
from core.furniture import Furniture


class StorageLayout:
    """Class for optimizing storage zone layout."""

    def __init__(self, room: Room, zone: Zone):
        """
        Initialize a storage zone layout optimizer.

        Args:
            room: Room object
            zone: Storage zone
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
        Optimize the storage zone layout.

        Returns:
            List of furniture placements
        """
        placements = []

        # Place bookshelves
        bookshelf_placements = self._place_bookshelves()
        placements.extend(bookshelf_placements)

        # Place plants
        plant_placements = self._place_plants()
        placements.extend(plant_placements)

        return placements

    def _find_corner_positions(self) -> List[Tuple[float, float, int]]:
        """
        Find corner positions in the zone.

        Returns:
            List of (x, y, orientation) tuples
        """
        corners = [
            # (x, y, orientation)
            (self.zone.x, self.zone.y, 0),  # Bottom-left
            (self.zone.x + self.zone.width, self.zone.y, 3),  # Bottom-right
            (self.zone.x, self.zone.y + self.zone.height, 1),  # Top-left
            (self.zone.x + self.zone.width, self.zone.y +
             self.zone.height, 2)  # Top-right
        ]

        return corners

    def _find_wall_positions(self) -> List[Tuple[float, float, int]]:
        """
        Find wall positions in the zone.

        Returns:
            List of (x, y, orientation) tuples
        """
        positions = []

        # Bottom wall
        x = self.zone.x + self.zone.width / 4
        y = self.zone.y
        positions.append((x, y, 0))

        x = self.zone.x + 3 * self.zone.width / 4
        positions.append((x, y, 0))

        # Top wall
        x = self.zone.x + self.zone.width / 4
        y = self.zone.y + self.zone.height
        positions.append((x, y, 2))

        x = self.zone.x + 3 * self.zone.width / 4
        positions.append((x, y, 2))

        # Left wall
        x = self.zone.x
        y = self.zone.y + self.zone.height / 4
        positions.append((x, y, 3))

        y = self.zone.y + 3 * self.zone.height / 4
        positions.append((x, y, 3))

        # Right wall
        x = self.zone.x + self.zone.width
        y = self.zone.y + self.zone.height / 4
        positions.append((x, y, 1))

        y = self.zone.y + 3 * self.zone.height / 4
        positions.append((x, y, 1))

        return positions

    def _place_bookshelves(self) -> List[Placement]:
        """
        Place bookshelves along walls or in corners.

        Returns:
            List of bookshelf placements
        """
        placements = []

        # Get bookshelf furniture
        bookshelves = self.get_furniture_of_type('bookshelf')
        if not bookshelves:
            return placements

        # Try corner positions first
        corner_positions = self._find_corner_positions()
        wall_positions = self._find_wall_positions()

        # Combined positions, corners first (they're preferred)
        positions = corner_positions + wall_positions

        # Place bookshelves
        for i, bookshelf in enumerate(bookshelves):
            best_placement = None
            best_score = -float('inf')

            for x, y, orientation in positions:
                # Adjust position based on orientation
                adj_x, adj_y = x, y
                shelf_width, shelf_height = bookshelf.get_dimensions(
                    orientation)

                if orientation == 0:  # Bottom wall
                    adj_y += 0.5  # Offset from wall
                elif orientation == 1:  # Left wall
                    adj_x += 0.5  # Offset from wall
                elif orientation == 2:  # Top wall
                    adj_y -= shelf_height + 0.5  # Offset from wall
                elif orientation == 3:  # Right wall
                    adj_x -= shelf_width + 0.5  # Offset from wall

                # Ensure position is within zone bounds
                adj_x = max(self.zone.x, min(
                    adj_x, self.zone.x + self.zone.width - shelf_width))
                adj_y = max(self.zone.y, min(adj_y, self.zone.y +
                            self.zone.height - shelf_height))

                # Create placement
                placement = Placement(
                    bookshelf, adj_x, adj_y, orientation % bookshelf.orientations)

                # Check if valid
                if not self.room._is_valid_placement(placement):
                    continue

                # Calculate score
                # Prefer corners, then walls
                position_idx = positions.index((x, y, orientation))
                position_score = 1.0 - min(position_idx / len(positions), 1.0)

                # Check if near window (beneficial for bookshelves)
                window_score = 0.0
                for feature in self.room.features:
                    if feature.feature_type == 'window':
                        feature_x, feature_y, feature_width, feature_height = feature.get_coords()
                        feature_center_x = feature_x + feature_width / 2
                        feature_center_y = feature_y + feature_height / 2

                        placement_center_x = adj_x + shelf_width / 2
                        placement_center_y = adj_y + shelf_height / 2

                        distance = np.sqrt(
                            (placement_center_x - feature_center_x) ** 2 +
                            (placement_center_y - feature_center_y) ** 2
                        )

                        # Closer to window is better (up to 5 feet)
                        window_score = max(
                            window_score, 1.0 - min(distance / 5.0, 1.0))

                # Calculate total score
                score = position_score * 0.7 + window_score * 0.3

                if score > best_score:
                    best_score = score
                    best_placement = placement

            # Add best placement for this bookshelf
            if best_placement:
                self.room.place_furniture(
                    bookshelf,
                    best_placement.x,
                    best_placement.y,
                    best_placement.orientation
                )
                placements.append(best_placement)

                # Remove position from available positions
                positions = [
                    (x, y, o) for x, y, o in positions
                    if not Placement(bookshelf, x, y, o % bookshelf.orientations).overlaps(best_placement)
                ]

        return placements

    def _place_plants(self) -> List[Placement]:
        """
        Place plants near windows or in corners.

        Returns:
            List of plant placements
        """
        placements = []

        # Get plant furniture
        plants = self.get_furniture_of_type('plant')
        if not plants:
            return placements

        # Find window positions
        window_positions = []
        for feature in self.room.features:
            if feature.feature_type == 'window':
                feature_x, feature_y, feature_width, feature_height = feature.get_coords()

                # Left of window
                window_positions.append(
                    (feature_x - plants[0].width - 0.5, feature_y, 0))

                # Right of window
                window_positions.append(
                    (feature_x + feature_width + 0.5, feature_y, 0))

                # Bottom of window
                window_positions.append(
                    (feature_x, feature_y - plants[0].height - 0.5, 0))

                # Top of window
                window_positions.append(
                    (feature_x, feature_y + feature_height + 0.5, 0))

        # Combine with corner positions
        corner_positions = self._find_corner_positions()
        positions = window_positions + corner_positions

        # Place plants
        for i, plant in enumerate(plants):
            if not positions:
                break

            best_placement = None
            best_score = -float('inf')

            for x, y, orientation in positions:
                # Adjust position based on plant size
                plant_width, plant_height = plant.get_dimensions(orientation)

                # Ensure position is within zone bounds
                adj_x = max(self.zone.x, min(
                    x, self.zone.x + self.zone.width - plant_width))
                adj_y = max(self.zone.y, min(y, self.zone.y +
                            self.zone.height - plant_height))

                # Create placement
                placement = Placement(
                    plant, adj_x, adj_y, orientation % plant.orientations)

                # Check if valid
                if not self.room._is_valid_placement(placement):
                    continue

                # Calculate score
                # Prefer window positions
                position_idx = positions.index((x, y, orientation))
                position_score = 1.0 - min(position_idx / len(positions), 1.0)

                # Check if actually near window
                window_score = 0.0
                for feature in self.room.features:
                    if feature.feature_type == 'window':
                        feature_x, feature_y, feature_width, feature_height = feature.get_coords()
                        feature_center_x = feature_x + feature_width / 2
                        feature_center_y = feature_y + feature_height / 2

                        placement_center_x = adj_x + plant_width / 2
                        placement_center_y = adj_y + plant_height / 2

                        distance = np.sqrt(
                            (placement_center_x - feature_center_x) ** 2 +
                            (placement_center_y - feature_center_y) ** 2
                        )

                        # Closer to window is better (up to 5 feet)
                        window_score = max(
                            window_score, 1.0 - min(distance / 5.0, 1.0))

                # Calculate total score
                score = position_score * 0.3 + window_score * 0.7

                if score > best_score:
                    best_score = score
                    best_placement = placement

            # Add best placement for this plant
            if best_placement:
                self.room.place_furniture(
                    plant,
                    best_placement.x,
                    best_placement.y,
                    best_placement.orientation
                )
                placements.append(best_placement)

                # Remove position from available positions
                positions = [
                    (x, y, o) for x, y, o in positions
                    if not Placement(plant, x, y, o % plant.orientations).overlaps(best_placement)
                ]

        return placements
