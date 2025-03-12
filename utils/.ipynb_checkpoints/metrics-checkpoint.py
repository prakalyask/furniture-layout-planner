"""
Metrics for evaluating layout quality.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from core.room import Room, Zone, Placement
from layout.pathway import PathwayGenerator


class LayoutMetrics:
    """Class for calculating various layout quality metrics."""

    def __init__(self, room: Room):
        """
        Initialize layout metrics calculator.

        Args:
            room: Room object
        """
        self.room = room

    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all layout metrics.

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        # Space efficiency
        metrics["space_efficiency"] = self.calculate_space_efficiency()

        # Furniture relationship score
        metrics["relationship_score"] = self.calculate_relationship_score()

        # Zone coherence
        metrics["zone_coherence"] = self.calculate_zone_coherence()

        # Pathway quality
        metrics["pathway_quality"] = self.calculate_pathway_quality()

        # Clearance score
        metrics["clearance_score"] = self.calculate_clearance_score()

        # Balance score
        metrics["balance_score"] = self.calculate_balance_score()

        # Aesthetics score
        metrics["aesthetics_score"] = self.calculate_aesthetics_score()

        # Calculate overall score (weighted average)
        weights = {
            "space_efficiency": 0.15,
            "relationship_score": 0.25,
            "zone_coherence": 0.15,
            "pathway_quality": 0.20,
            "clearance_score": 0.10,
            "balance_score": 0.05,
            "aesthetics_score": 0.10
        }

        metrics["overall_score"] = sum(
            metrics[key] * weights[key] for key in weights)

        return metrics

    def calculate_space_efficiency(self) -> float:
        """
        Calculate space efficiency (how well the room space is utilized).

        Returns:
            Space efficiency score (0-1, higher is better)
        """
        if not self.room.placements:
            return 0.0

        # Calculate total room area
        room_area = self.room.width * self.room.height

        # Calculate total furniture area
        furniture_area = 0.0
        for placement in self.room.placements:
            _, _, width, height = placement.get_coords()
            furniture_area += width * height

        # Calculate efficiency
        efficiency = furniture_area / room_area

        # Ideal efficiency is around 30-50%
        if efficiency < 0.3:
            # Too sparse
            return efficiency / 0.3
        elif efficiency <= 0.5:
            # Optimal range
            return 1.0
        else:
            # Too crowded
            return max(0.0, 1.0 - (efficiency - 0.5) / 0.5)

    def calculate_relationship_score(self) -> float:
        """
        Calculate furniture relationship score (how well related furniture is positioned).

        Returns:
            Relationship score (0-1, higher is better)
        """
        if not self.room.placements:
            return 0.0

        relationship_scores = []

        for placement in self.room.placements:
            furniture = placement.furniture

            # Skip if no relationships
            if not furniture.related_furniture:
                continue

            # Check each relationship
            for related_type in furniture.related_furniture:
                # Find related furniture
                related_placements = self.room.get_placement_by_type(
                    related_type)

                if not related_placements:
                    continue

                # Calculate optimal distance
                optimal_distance = furniture.optimal_distances.get(
                    related_type, 0.0)

                # Calculate score for each related furniture
                for related in related_placements:
                    # Calculate actual distance
                    actual_distance = placement.distance_to(related)

                    # Calculate score based on distance
                    if optimal_distance > 0:
                        # Distance-based score
                        distance_score = 1.0 - \
                            min(abs(actual_distance - optimal_distance) /
                                optimal_distance, 1.0)
                    else:
                        # Default score
                        distance_score = 1.0

                    # Check if facing is required
                    if furniture.facing_furniture == related_type:
                        # Check if actually facing
                        facing_score = 1.0 if placement.is_facing(
                            related) else 0.0

                        # Combined score (weighted average)
                        score = 0.6 * distance_score + 0.4 * facing_score
                    else:
                        score = distance_score

                    relationship_scores.append(score)

        # Return average score
        return sum(relationship_scores) / len(relationship_scores) if relationship_scores else 0.0

    def calculate_zone_coherence(self) -> float:
        """
        Calculate zone coherence (how well furniture is placed in appropriate zones).

        Returns:
            Zone coherence score (0-1, higher is better)
        """
        if not self.room.placements or not self.room.zones:
            return 0.0

        correct_zone_count = 0

        for placement in self.room.placements:
            # Get placement coordinates
            p_x, p_y, _, _ = placement.get_coords()

            # Find containing zone
            containing_zone = None
            for zone in self.room.zones:
                if zone.contains_point(p_x, p_y):
                    containing_zone = zone
                    break

            # Check if zone matches furniture zone
            if containing_zone and containing_zone.name == placement.furniture.zone:
                correct_zone_count += 1

        # Calculate percentage
        return correct_zone_count / len(self.room.placements)

    def calculate_pathway_quality(self) -> float:
        """
        Calculate pathway quality (how well the layout supports movement).

        Returns:
            Pathway quality score (0-1, higher is better)
        """
        # Create pathway generator
        pathway_generator = PathwayGenerator(self.room)

        # Generate pathways
        pathway_generator.generate_pathways()

        # Evaluate pathways
        return pathway_generator.evaluate_pathway_quality()

    def calculate_clearance_score(self) -> float:
        """
        Calculate clearance score (how well furniture respects clearance requirements).

        Returns:
            Clearance score (0-1, higher is better)
        """
        if not self.room.placements:
            return 0.0

        clearance_scores = []

        for i, p1 in enumerate(self.room.placements):
            # Get clearance requirement
            clearance = p1.furniture.clearance_required

            if clearance <= 0:
                continue

            # Check distance to other furniture
            for j, p2 in enumerate(self.room.placements):
                if i == j:
                    continue

                # Calculate distance
                p1_x, p1_y, p1_width, p1_height = p1.get_coords()
                p2_x, p2_y, p2_width, p2_height = p2.get_coords()

                # Calculate closest distance between rectangles
                dx = max(0, max(p1_x - (p2_x + p2_width),
                         p2_x - (p1_x + p1_width)))
                dy = max(0, max(p1_y - (p2_y + p2_height),
                         p2_y - (p1_y + p1_height)))
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # Calculate score
                score = min(distance / clearance, 1.0)
                clearance_scores.append(score)

            # Check distance to walls
            wall_scores = []

            # Left wall
            if p1_x < clearance:
                wall_scores.append(p1_x / clearance)

            # Right wall
            if self.room.width - (p1_x + p1_width) < clearance:
                wall_scores.append(
                    (self.room.width - (p1_x + p1_width)) / clearance)

            # Bottom wall
            if p1_y < clearance:
                wall_scores.append(p1_y / clearance)

            # Top wall
            if self.room.height - (p1_y + p1_height) < clearance:
                wall_scores.append(
                    (self.room.height - (p1_y + p1_height)) / clearance)

            # Add wall scores
            clearance_scores.extend(wall_scores)

        # Return average score
        return sum(clearance_scores) / len(clearance_scores) if clearance_scores else 1.0

    def calculate_balance_score(self) -> float:
        """
        Calculate balance score (how well furniture is distributed in the room).

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

        # Calculate furniture area in each quadrant
        quadrant_areas = [0.0, 0.0, 0.0, 0.0]

        for placement in self.room.placements:
            p_x, p_y, p_width, p_height = placement.get_coords()

            # Calculate overlap with each quadrant
            for i, (q_x, q_y, q_width, q_height) in enumerate(quadrants):
                # Calculate overlap rectangle
                overlap_x = max(p_x, q_x)
                overlap_y = max(p_y, q_y)
                overlap_width = min(p_x + p_width, q_x + q_width) - overlap_x
                overlap_height = min(
                    p_y + p_height, q_y + q_height) - overlap_y

                # Calculate overlap area
                if overlap_width > 0 and overlap_height > 0:
                    overlap_area = overlap_width * overlap_height
                    quadrant_areas[i] += overlap_area

        # Calculate total furniture area
        total_area = sum(quadrant_areas)

        if total_area <= 0:
            return 0.0

        # Calculate expected area per quadrant
        expected_area = total_area / 4

        # Calculate variance
        variance = sum((area - expected_area) **
                       2 for area in quadrant_areas) / total_area

        # Convert to score (higher is better)
        balance_score = 1.0 - min(variance, 1.0)

        return balance_score

    def calculate_aesthetics_score(self) -> float:
        """
        Calculate aesthetics score (how visually pleasing the layout is).

        Returns:
            Aesthetics score (0-1, higher is better)
        """
        if not self.room.placements:
            return 0.0

        # Calculate symmetry score
        symmetry_score = self._calculate_symmetry()

        # Calculate alignment score
        alignment_score = self._calculate_alignment()

        # Calculate proportional spacing
        spacing_score = self._calculate_spacing()

        # Calculate final score (weighted average)
        aesthetics_score = 0.4 * symmetry_score + \
            0.4 * alignment_score + 0.2 * spacing_score

        return aesthetics_score

    def _calculate_symmetry(self) -> float:
        """Calculate symmetry score."""
        # Get room center
        center_x = self.room.width / 2
        center_y = self.room.height / 2

        # Calculate symmetry along x-axis and y-axis
        x_symmetry = 0.0
        y_symmetry = 0.0

        # Group furniture by type
        furniture_by_type = {}
        for placement in self.room.placements:
            furniture_type = placement.furniture.furniture_type
            if furniture_type not in furniture_by_type:
                furniture_by_type[furniture_type] = []
            furniture_by_type[furniture_type].append(placement)

        # Calculate symmetry for each furniture type
        symmetry_scores = []

        for furniture_type, placements in furniture_by_type.items():
            if len(placements) <= 1:
                continue

            # Calculate center of mass
            com_x = sum(p.get_center()[0]
                        for p in placements) / len(placements)
            com_y = sum(p.get_center()[1]
                        for p in placements) / len(placements)

            # Calculate symmetry score
            x_offset = abs(com_x - center_x) / center_x
            y_offset = abs(com_y - center_y) / center_y

            symmetry_score = 1.0 - (x_offset + y_offset) / 2
            symmetry_scores.append(symmetry_score)

        # Return average symmetry score
        return sum(symmetry_scores) / len(symmetry_scores) if symmetry_scores else 0.0

    def _calculate_alignment(self) -> float:
        """Calculate alignment score."""
        if len(self.room.placements) <= 1:
            return 0.0

        # Calculate alignment between furniture pieces
        alignment_scores = []

        for i, p1 in enumerate(self.room.placements):
            p1_x, p1_y, p1_width, p1_height = p1.get_coords()

            for j, p2 in enumerate(self.room.placements):
                if i >= j:
                    continue

                p2_x, p2_y, p2_width, p2_height = p2.get_coords()

                # Check horizontal alignment
                h_aligned = False
                h_score = 0.0

                # Top edges aligned
                if abs(p1_y - p2_y) < 0.5:
                    h_aligned = True
                    h_score = 1.0
                # Bottom edges aligned
                elif abs((p1_y + p1_height) - (p2_y + p2_height)) < 0.5:
                    h_aligned = True
                    h_score = 1.0
                # Centers aligned
                elif abs((p1_y + p1_height/2) - (p2_y + p2_height/2)) < 0.5:
                    h_aligned = True
                    h_score = 1.0

                # Check vertical alignment
                v_aligned = False
                v_score = 0.0

                # Left edges aligned
                if abs(p1_x - p2_x) < 0.5:
                    v_aligned = True
                    v_score = 1.0
                # Right edges aligned
                elif abs((p1_x + p1_width) - (p2_x + p2_width)) < 0.5:
                    v_aligned = True
                    v_score = 1.0
                # Centers aligned
                elif abs((p1_x + p1_width/2) - (p2_x + p2_width/2)) < 0.5:
                    v_aligned = True
                    v_score = 1.0

                # Add alignment score if either horizontally or vertically aligned
                if h_aligned or v_aligned:
                    alignment_scores.append(max(h_score, v_score))

        # Return average alignment score
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    def _calculate_spacing(self) -> float:
        """Calculate spacing score."""
        if len(self.room.placements) <= 1:
            return 0.0

        # Calculate spacing between furniture pieces
        spacing_scores = []

        for i, p1 in enumerate(self.room.placements):
            p1_x, p1_y, p1_width, p1_height = p1.get_coords()

            for j, p2 in enumerate(self.room.placements):
                if i >= j:
                    continue

                p2_x, p2_y, p2_width, p2_height = p2.get_coords()

                # Calculate closest distance between rectangles
                dx = max(0, max(p1_x - (p2_x + p2_width),
                         p2_x - (p1_x + p1_width)))
                dy = max(0, max(p1_y - (p2_y + p2_height),
                         p2_y - (p1_y + p1_height)))
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # Ideal spacing is around 1-3 feet
                if distance < 1.0:
                    # Too close
                    score = distance
                elif distance <= 3.0:
                    # Optimal range
                    score = 1.0
                else:
                    # Too far
                    score = max(0.0, 1.0 - (distance - 3.0) / 5.0)

                spacing_scores.append(score)

        # Return average spacing score
        return sum(spacing_scores) / len(spacing_scores) if spacing_scores else 0.0
