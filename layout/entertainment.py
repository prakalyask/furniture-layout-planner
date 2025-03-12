"""
Entertainment zone layout algorithm.
Handles optimal placement of TV, sofas, coffee tables, etc.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from core.room import Room, Zone, Placement, RoomFeature
from core.furniture import Furniture


class EntertainmentLayout:
    """Class for optimizing entertainment zone layout."""

    def __init__(self, room: Room, zone: Zone):
        """
        Initialize an entertainment zone layout optimizer.

        Args:
            room: Room object
            zone: Entertainment zone
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
        Optimize the entertainment zone layout.

        Returns:
            List of furniture placements
        """
        placements = []

        # First, place the TV (the focal point)
        tv_placement = self._place_tv()
        if tv_placement:
            placements.append(tv_placement)

        # Next, place sofas facing the TV
        sofa_placements = self._place_sofas(tv_placement)
        placements.extend(sofa_placements)

        # Place coffee table between sofas and TV
        coffee_table_placement = self._place_coffee_table(
            tv_placement, sofa_placements)
        if coffee_table_placement:
            placements.append(coffee_table_placement)

        # Place side tables next to sofas
        side_table_placements = self._place_side_tables(sofa_placements)
        placements.extend(side_table_placements)

        # Place lamps near side tables or sofas
        lamp_placements = self._place_lamps(
            sofa_placements, side_table_placements)
        placements.extend(lamp_placements)

        return placements

    def _find_best_wall_position(self) -> Tuple[float, float, int]:
        """
        Find the best wall position for the TV.

        Returns:
            Tuple of (x, y, orientation)
        """
        # Check all walls of the zone
        wall_candidates = []

        # Define zone walls
        zone_walls = [
            # (x, y, width, height, orientation)
            (self.zone.x, self.zone.y, self.zone.width, 0, 0),  # Bottom wall
            (self.zone.x, self.zone.y + self.zone.height,
             self.zone.width, 0, 0),  # Top wall
            (self.zone.x, self.zone.y, 0, self.zone.height, 1),  # Left wall
            (self.zone.x + self.zone.width, self.zone.y,
             0, self.zone.height, 1)  # Right wall
        ]

        # Get windows and doors
        windows = [f for f in self.room.features if f.feature_type == 'window']
        doors = [f for f in self.room.features if f.feature_type == 'door']

        # Check each wall
        for x, y, width, height, orientation in zone_walls:
            # Skip if wall is too short
            if (orientation == 0 and width < 3) or (orientation == 1 and height < 3):
                continue

            # Check if wall overlaps with window or door
            overlaps = False
            for feature in windows + doors:
                fx, fy, fw, fh = feature.get_coords()
                if orientation == 0:  # Horizontal wall
                    if (fy == y or fy + fh == y) and fx < x + width and fx + fw > x:
                        overlaps = True
                        break
                else:  # Vertical wall
                    if (fx == x or fx + fw == x) and fy < y + height and fy + fh > y:
                        overlaps = True
                        break

            if not overlaps:
                # Calculate position score (prefer centered position on longer walls)
                if orientation == 0:  # Horizontal wall
                    score = width
                    position = (x + width / 2, y)
                else:  # Vertical wall
                    score = height
                    position = (x, y + height / 2)

                wall_candidates.append(
                    (position[0], position[1], orientation, score))

        # Sort by score (descending)
        wall_candidates.sort(key=lambda x: x[3], reverse=True)

        if not wall_candidates:
            # No suitable wall found, use center of zone
            x = self.zone.x + self.zone.width / 2
            y = self.zone.y + self.zone.height / 2
            return (x, y, 0)

        # Return best wall position
        x, y, orientation, _ = wall_candidates[0]
        return (x, y, orientation)

    def _place_tv(self) -> Optional[Placement]:
        """
        Place the TV in the optimal position.

        Returns:
            TV placement if successful, None otherwise
        """
        # Get TV furniture
        tvs = self.get_furniture_of_type('tv')
        if not tvs:
            return None

        tv = tvs[0]  # Use the first TV

        # Find best wall position
        x, y, orientation = self._find_best_wall_position()

        # Adjust position based on TV size
        tv_width, tv_height = tv.get_dimensions(orientation)
        if orientation == 0:  # Horizontal wall
            x -= tv_width / 2
            if y == self.zone.y:  # Bottom wall
                y = self.zone.y + 0.5  # Add offset from wall
            else:  # Top wall
                y = self.zone.y + self.zone.height - tv_height - 0.5  # Subtract height and offset
        else:  # Vertical wall
            y -= tv_height / 2
            if x == self.zone.x:  # Left wall
                x = self.zone.x + 0.5  # Add offset from wall
            else:  # Right wall
                x = self.zone.x + self.zone.width - tv_width - 0.5  # Subtract width and offset

        # Ensure position is within zone bounds
        x = max(self.zone.x, min(x, self.zone.x + self.zone.width - tv_width))
        y = max(self.zone.y, min(y, self.zone.y + self.zone.height - tv_height))

        # Create placement
        placement = Placement(tv, x, y, orientation)

        # Check if valid
        if self.room._is_valid_placement(placement):
            # Add to room
            self.room.place_furniture(tv, x, y, orientation)
            return placement

        return None

    def _place_sofas(self, tv_placement: Optional[Placement]) -> List[Placement]:
        """
        Place sofas facing the TV.

        Args:
            tv_placement: TV placement

        Returns:
            List of sofa placements
        """
        placements = []

        # Get sofa furniture
        sofas = self.get_furniture_of_type('sofa')
        if not sofas or not tv_placement:
            return placements

        # Get TV position
        tv_x, tv_y, tv_width, tv_height = tv_placement.get_coords()
        tv_center_x = tv_x + tv_width / 2
        tv_center_y = tv_y + tv_height / 2

        # Calculate optimal viewing distance (8 feet from TV)
        optimal_distance = 8.0

        # Calculate candidate positions
        sofa_positions = []

        for sofa in sofas:
            # Try different orientations
            for orientation in range(sofa.orientations):
                sofa_width, sofa_height = sofa.get_dimensions(orientation)

                # Calculate position to face TV
                # Position 1: In front of TV
                x1 = tv_center_x - sofa_width / 2
                y1 = tv_center_y + optimal_distance - sofa_height  # Subtract sofa depth
                sofa_positions.append((x1, y1, orientation))

                # Position 2: To the left of TV
                x2 = tv_center_x - optimal_distance
                y2 = tv_center_y - sofa_height / 2
                sofa_positions.append(
                    (x2, y2, (orientation + 1) % sofa.orientations))

                # Position 3: To the right of TV
                x3 = tv_center_x + optimal_distance - sofa_width
                y3 = tv_center_y - sofa_height / 2
                sofa_positions.append(
                    (x3, y3, (orientation + 1) % sofa.orientations))

                # Position 4: Multiple sofas in L-shape (if more than one sofa)
                if len(sofas) > 1:
                    # L-shape left corner
                    x4 = tv_center_x - sofa_width / 2 - optimal_distance / 2
                    y4 = tv_center_y + optimal_distance - sofa_height  # Front sofa
                    sofa_positions.append((x4, y4, orientation))

                    # L-shape right corner
                    x5 = tv_center_x + sofa_width / 2 + optimal_distance / 2 - sofa_width
                    y5 = tv_center_y + optimal_distance - sofa_height  # Front sofa
                    sofa_positions.append((x5, y5, orientation))

        # Try each sofa at each position
        for sofa in sofas:
            best_placement = None
            best_score = -float('inf')

            for x, y, orientation in sofa_positions:
                # Create placement
                placement = Placement(sofa, x, y, orientation)

                # Check if valid and within zone
                if not self.room._is_valid_placement(placement):
                    continue

                # Check if within zone
                if not (self.zone.x <= x <= self.zone.x + self.zone.width and
                        self.zone.y <= y <= self.zone.y + self.zone.height):
                    continue

                # Check if facing TV
                if not placement.is_facing(tv_placement):
                    continue

                # Calculate distance to TV
                distance = placement.distance_to(tv_placement)

                # Calculate score (closer to optimal distance is better)
                distance_score = 1.0 - \
                    min(abs(distance - optimal_distance) / optimal_distance, 1.0)

                # Prefer positions directly in front of TV
                front_score = 0.0
                sofa_center_x, sofa_center_y = placement.get_center()
                if abs(sofa_center_x - tv_center_x) < tv_width:
                    front_score = 1.0

                # Calculate total score
                score = distance_score * 0.7 + front_score * 0.3

                if score > best_score:
                    best_score = score
                    best_placement = placement

            # Add best placement for this sofa
            if best_placement:
                self.room.place_furniture(
                    sofa,
                    best_placement.x,
                    best_placement.y,
                    best_placement.orientation
                )
                placements.append(best_placement)

                # Remove from sofa positions that would overlap
                sofa_positions = [
                    (x, y, o) for x, y, o in sofa_positions
                    if not Placement(sofa, x, y, o).overlaps(best_placement)
                ]

        return placements

    def _place_coffee_table(
        self,
        tv_placement: Optional[Placement],
        sofa_placements: List[Placement]
    ) -> Optional[Placement]:
        """
        Place a coffee table between sofas and TV.

        Args:
            tv_placement: TV placement
            sofa_placements: Sofa placements

        Returns:
            Coffee table placement if successful, None otherwise
        """
        # Get coffee table furniture
        coffee_tables = self.get_furniture_of_type('coffee_table')
        if not coffee_tables or not tv_placement or not sofa_placements:
            return None

        coffee_table = coffee_tables[0]  # Use the first coffee table

        # Get positions
        tv_center_x, tv_center_y = tv_placement.get_center()

        # Calculate average sofa position
        sofa_center_x = sum(p.get_center()[0]
                            for p in sofa_placements) / len(sofa_placements)
        sofa_center_y = sum(p.get_center()[1]
                            for p in sofa_placements) / len(sofa_placements)

        # Calculate coffee table position (between sofas and TV)
        table_center_x = (tv_center_x + sofa_center_x) / 2
        table_center_y = (tv_center_y + sofa_center_y) / 2

        # Try different orientations
        best_placement = None
        best_score = -float('inf')

        for orientation in range(coffee_table.orientations):
            table_width, table_height = coffee_table.get_dimensions(
                orientation)

            # Calculate position (centered)
            x = table_center_x - table_width / 2
            y = table_center_y - table_height / 2

            # Create placement
            placement = Placement(coffee_table, x, y, orientation)

            # Check if valid
            if not self.room._is_valid_placement(placement):
                continue

            # Calculate distance to sofas
            sofa_distances = [placement.distance_to(
                sofa) for sofa in sofa_placements]
            avg_sofa_distance = sum(sofa_distances) / len(sofa_distances)

            # Calculate distance to TV
            tv_distance = placement.distance_to(tv_placement)

            # Calculate score (prefer placement roughly 2 feet from sofas)
            optimal_sofa_distance = 2.0
            sofa_score = 1.0 - \
                min(abs(avg_sofa_distance - optimal_sofa_distance) / 3.0, 1.0)

            # Prefer placement not too close to TV
            optimal_tv_distance = 5.0
            tv_score = 1.0 - \
                min(abs(tv_distance - optimal_tv_distance) / 5.0, 1.0)

            # Calculate total score
            score = sofa_score * 0.7 + tv_score * 0.3

            if score > best_score:
                best_score = score
                best_placement = placement

        # Add best placement
        if best_placement:
            self.room.place_furniture(
                coffee_table,
                best_placement.x,
                best_placement.y,
                best_placement.orientation
            )
            return best_placement

        return None

    def _place_side_tables(self, sofa_placements: List[Placement]) -> List[Placement]:
        """
        Place side tables next to sofas.

        Args:
            sofa_placements: Sofa placements

        Returns:
            List of side table placements
        """
        placements = []

        # Get side table furniture
        side_tables = self.get_furniture_of_type('side_table')
        if not side_tables or not sofa_placements:
            return placements

        # Place a side table next to each sofa (if available)
        for i, sofa in enumerate(sofa_placements):
            if i >= len(side_tables):
                break

            side_table = side_tables[i]

            # Get sofa position
            sofa_x, sofa_y, sofa_width, sofa_height = sofa.get_coords()

            # Calculate side table positions (on both sides of sofa)
            positions = []

            # Position depends on sofa orientation
            if sofa.orientation % 2 == 0:  # Horizontal orientation
                # Left side
                positions.append((sofa_x - side_table.width - 0.5, sofa_y, 0))
                # Right side
                positions.append((sofa_x + sofa_width + 0.5, sofa_y, 0))
            else:  # Vertical orientation
                # Top side
                positions.append((sofa_x, sofa_y - side_table.height - 0.5, 0))
                # Bottom side
                positions.append((sofa_x, sofa_y + sofa_height + 0.5, 0))

            # Try each position
            for x, y, orientation in positions:
                placement = Placement(side_table, x, y, orientation)

                # Check if valid
                if self.room._is_valid_placement(placement):
                    # Add to room
                    self.room.place_furniture(side_table, x, y, orientation)
                    placements.append(placement)
                    break

        return placements

    def _place_lamps(
        self,
        sofa_placements: List[Placement],
        side_table_placements: List[Placement]
    ) -> List[Placement]:
        """
        Place lamps near side tables or sofas.

        Args:
            sofa_placements: Sofa placements
            side_table_placements: Side table placements

        Returns:
            List of lamp placements
        """
        placements = []

        # Get lamp furniture
        lamps = self.get_furniture_of_type('lamp')
        if not lamps:
            return placements

        # Place lamps on side tables first
        for i, side_table in enumerate(side_table_placements):
            if i >= len(lamps):
                break

            lamp = lamps[i]

            # Get side table position
            table_x, table_y, table_width, table_height = side_table.get_coords()

            # Place lamp in center of side table
            x = table_x + (table_width - lamp.width) / 2
            y = table_y + (table_height - lamp.height) / 2

            placement = Placement(lamp, x, y, 0)

            # Check if valid
            if self.room._is_valid_placement(placement):
                # Add to room
                self.room.place_furniture(lamp, x, y, 0)
                placements.append(placement)

                # Remove from lamps
                lamps.remove(lamp)

        # Place remaining lamps near sofas
        for i, sofa in enumerate(sofa_placements):
            if not lamps:
                break

            lamp = lamps[0]

            # Get sofa position
            sofa_x, sofa_y, sofa_width, sofa_height = sofa.get_coords()

            # Calculate lamp positions (near corners of sofa)
            positions = [
                (sofa_x - lamp.width - 0.5, sofa_y -
                 lamp.height - 0.5, 0),  # Top-left
                (sofa_x + sofa_width + 0.5, sofa_y -
                 lamp.height - 0.5, 0),  # Top-right
                (sofa_x - lamp.width - 0.5, sofa_y +
                 sofa_height + 0.5, 0),  # Bottom-left
                (sofa_x + sofa_width + 0.5, sofa_y +
                 sofa_height + 0.5, 0)  # Bottom-right
            ]

            # Try each position
            for x, y, orientation in positions:
                placement = Placement(lamp, x, y, orientation)

                # Check if valid
                if self.room._is_valid_placement(placement):
                    # Add to room
                    self.room.place_furniture(lamp, x, y, orientation)
                    placements.append(placement)

                    # Remove from lamps
                    lamps.remove(lamp)
                    break

        return placements
