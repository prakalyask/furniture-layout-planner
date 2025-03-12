"""
Layout package for furniture arrangement optimization.
Integrates zone-specific layout algorithms and pathway generation.
"""

from typing import Dict, List, Tuple, Optional
from core.room import Room, Zone, Placement
from .entertainment import EntertainmentLayout
from .dining import DiningLayout
from .storage import StorageLayout
from .pathway import PathwayGenerator, Path


class IntegratedLayout:
    """Class that integrates all layout algorithms for room optimization."""

    def __init__(self, room: Room):
        """
        Initialize an integrated layout optimizer.

        Args:
            room: Room object
        """
        self.room = room
        self.pathway_generator = None

    def optimize(self) -> Tuple[List[Placement], float]:
        """
        Optimize the entire room layout.

        Returns:
            Tuple of (placements, score)
        """
        placements = []

        # Optimize each zone
        for zone in self.room.zones:
            zone_placements = self._optimize_zone(zone)
            placements.extend(zone_placements)

        # Generate and evaluate pathways
        pathway_score = self._evaluate_pathways()

        # Calculate overall score
        layout_score = self._evaluate_layout(placements, pathway_score)

        return placements, layout_score

    def _optimize_zone(self, zone: Zone) -> List[Placement]:
        """
        Optimize layout for a specific zone.

        Args:
            zone: Zone to optimize

        Returns:
            List of furniture placements
        """
        if zone.name == "entertainment":
            optimizer = EntertainmentLayout(self.room, zone)
        elif zone.name == "dining":
            optimizer = DiningLayout(self.room, zone)
        elif zone.name == "storage":
            optimizer = StorageLayout(self.room, zone)
        else:
            # Unknown zone type, skip
            return []

        return optimizer.optimize()

    def _evaluate_pathways(self) -> float:
        """
        Generate and evaluate pathways.

        Returns:
            Pathway quality score
        """
        # Create pathway generator
        self.pathway_generator = PathwayGenerator(self.room)

        # Generate pathways
        self.pathway_generator.generate_pathways()

        # Evaluate pathways
        return self.pathway_generator.evaluate_pathway_quality()

    def _evaluate_layout(self, placements: List[Placement], pathway_score: float) -> float:
        """
        Evaluate the overall layout quality.

        Args:
            placements: List of furniture placements
            pathway_score: Pathway quality score

        Returns:
            Overall layout quality score
        """
        if not placements:
            return 0.0

        # Calculate furniture placement score
        from core.fuzzy_logic import FuzzyEvaluator
        evaluator = FuzzyEvaluator()
        furniture_score = evaluator.evaluate_layout(self.room)

        # Calculate balance score (distribution of furniture)
        balance_score = self._evaluate_balance()

        # Calculate final score (weighted average)
        score = 0.5 * furniture_score + 0.3 * pathway_score + 0.2 * balance_score

        return score

    def _evaluate_balance(self) -> float:
        """
        Evaluate the balance of furniture distribution.

        Returns:
            Balance score (0-1, higher is better)
        """
        if not self.room.placements:
            return 0.0

        # Calculate furniture density in different regions
        width = self.room.width
        height = self.room.height

        # Divide room into quadrants
        quadrants = [
            (0, 0, width/2, height/2),  # Bottom-left
            (width/2, 0, width/2, height/2),  # Bottom-right
            (0, height/2, width/2, height/2),  # Top-left
            (width/2, height/2, width/2, height/2)  # Top-right
        ]

        # Count furniture in each quadrant
        quadrant_counts = [0, 0, 0, 0]

        for placement in self.room.placements:
            p_x, p_y, p_width, p_height = placement.get_coords()
            center_x = p_x + p_width / 2
            center_y = p_y + p_height / 2

            # Determine quadrant
            quadrant_idx = 0
            if center_x >= width / 2:
                quadrant_idx += 1
            if center_y >= height / 2:
                quadrant_idx += 2

            quadrant_counts[quadrant_idx] += 1

        # Calculate balance score (variation in counts)
        total_furniture = len(self.room.placements)
        expected_per_quadrant = total_furniture / 4

        # Sum of squared differences from expected
        sum_squared_diff = sum(
            (count - expected_per_quadrant) ** 2 for count in quadrant_counts)

        # Normalize to [0, 1] (lower is better)
        max_variance = total_furniture ** 2  # Maximum possible squared difference
        variance = sum_squared_diff / max_variance

        # Convert to score (higher is better)
        balance_score = 1.0 - variance

        return balance_score
