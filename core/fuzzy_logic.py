from typing import Dict, List, Tuple, Callable, Optional
import numpy as np


from core.furniture import Furniture
from .room import Room, Placement, Zone


class FuzzyMembershipFunction:
    """Base class for fuzzy membership functions."""

    def __call__(self, x: float) -> float:
        """
        Evaluate the membership function at x.

        Args:
            x: Input value

        Returns:
            Membership value in [0, 1]
        """
        raise NotImplementedError("Subclasses must implement this method")


class TriangularMF(FuzzyMembershipFunction):
    """Triangular membership function."""

    def __init__(self, a: float, b: float, c: float):
        """
        Initialize a triangular membership function.

        Args:
            a: Left boundary (0 membership)
            b: Peak (1 membership)
            c: Right boundary (0 membership)
        """
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a)
        else:  # self.b < x < self.c
            return (self.c - x) / (self.c - self.b)


class TrapezoidalMF(FuzzyMembershipFunction):
    """Trapezoidal membership function."""

    def __init__(self, a: float, b: float, c: float, d: float):
        """
        Initialize a trapezoidal membership function.

        Args:
            a: Left boundary (0 membership)
            b: Left peak (1 membership)
            c: Right peak (1 membership)
            d: Right boundary (0 membership)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b < x <= self.c:
            return 1.0
        else:  # self.c < x < self.d
            return (self.d - x) / (self.d - self.c)


class GaussianMF(FuzzyMembershipFunction):
    """Gaussian membership function."""

    def __init__(self, mu: float, sigma: float):
        """
        Initialize a Gaussian membership function.

        Args:
            mu: Mean (peak)
            sigma: Standard deviation
        """
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: float) -> float:
        return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)


class FuzzyRule:
    """Base class for fuzzy rules."""

    def evaluate(self, room: Room, placement: Placement) -> float:
        """
        Evaluate the rule for a placement.

        Args:
            room: Room object
            placement: Placement to evaluate

        Returns:
            Rule satisfaction value in [0, 1]
        """
        raise NotImplementedError("Subclasses must implement this method")


class DistanceRule(FuzzyRule):
    """Rule evaluating distance between furniture types."""

    def __init__(
        self,
        target_type: str,
        membership_function: FuzzyMembershipFunction,
        weight: float = 1.0
    ):
        """
        Initialize a distance rule.

        Args:
            target_type: Target furniture type
            membership_function: Membership function for distance evaluation
            weight: Rule weight
        """
        self.target_type = target_type
        self.membership_function = membership_function
        self.weight = weight

    def evaluate(self, room: Room, placement: Placement) -> float:
        # Find target furniture placements
        targets = room.get_placement_by_type(self.target_type)

        if not targets:
            return 0.0  # No targets to evaluate against

        # Calculate distances to all targets
        distances = [placement.distance_to(target) for target in targets]

        # Evaluate membership for each distance
        memberships = [self.membership_function(d) for d in distances]

        # Return maximum membership (best match)
        return max(memberships) if memberships else 0.0


class FacingRule(FuzzyRule):
    """Rule evaluating if furniture is facing a target."""

    def __init__(
        self,
        target_type: str,
        weight: float = 1.0
    ):
        """
        Initialize a facing rule.

        Args:
            target_type: Target furniture type
            weight: Rule weight
        """
        self.target_type = target_type
        self.weight = weight

    def evaluate(self, room: Room, placement: Placement) -> float:
        # Find target furniture placements
        targets = room.get_placement_by_type(self.target_type)

        if not targets:
            return 0.0  # No targets to evaluate against

        # Check if placement is facing any target
        for target in targets:
            if placement.is_facing(target):
                return 1.0  # Facing at least one target

        return 0.0  # Not facing any target


class WallDistanceRule(FuzzyRule):
    """Rule evaluating distance to walls."""

    def __init__(
        self,
        prefer_wall: bool,
        membership_function: FuzzyMembershipFunction,
        weight: float = 1.0
    ):
        """
        Initialize a wall distance rule.

        Args:
            prefer_wall: Whether furniture prefers to be near walls
            membership_function: Membership function for distance evaluation
            weight: Rule weight
        """
        self.prefer_wall = prefer_wall
        self.membership_function = membership_function
        self.weight = weight

    def evaluate(self, room: Room, placement: Placement) -> float:
        x, y, width, height = placement.get_coords()

        # Calculate distances to walls
        dist_left = x
        dist_right = room.width - (x + width)
        dist_top = y
        dist_bottom = room.height - (y + height)

        # Minimum distance to any wall
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        # Evaluate membership
        membership = self.membership_function(min_dist)

        # If prefer_wall is False, invert the membership
        if not self.prefer_wall:
            return 1.0 - membership

        return membership


class WindowDistanceRule(FuzzyRule):
    """Rule evaluating distance to windows."""

    def __init__(
        self,
        prefer_window: bool,
        membership_function: FuzzyMembershipFunction,
        weight: float = 1.0
    ):
        """
        Initialize a window distance rule.

        Args:
            prefer_window: Whether furniture prefers to be near windows
            membership_function: Membership function for distance evaluation
            weight: Rule weight
        """
        self.prefer_window = prefer_window
        self.membership_function = membership_function
        self.weight = weight

    def evaluate(self, room: Room, placement: Placement) -> float:
        # Find windows
        windows = [f for f in room.features if f.feature_type == 'window']

        if not windows:
            return 0.0 if self.prefer_window else 1.0  # No windows to evaluate against

        # Calculate distances to all windows
        p_x, p_y, p_w, p_h = placement.get_coords()
        p_center_x = p_x + p_w / 2
        p_center_y = p_y + p_h / 2

        min_dist = float('inf')
        for window in windows:
            w_x, w_y, w_w, w_h = window.get_coords()
            w_center_x = w_x + w_w / 2
            w_center_y = w_y + w_h / 2

            dist = np.sqrt((p_center_x - w_center_x) ** 2 +
                           (p_center_y - w_center_y) ** 2)
            min_dist = min(min_dist, dist)

        # Evaluate membership
        membership = self.membership_function(min_dist)

        # If prefer_window is False, invert the membership
        if not self.prefer_window:
            return 1.0 - membership

        return membership


class ZoneRule(FuzzyRule):
    """Rule evaluating if furniture is in the right zone."""

    def __init__(
        self,
        weight: float = 1.0
    ):
        """
        Initialize a zone rule.

        Args:
            weight: Rule weight
        """
        self.weight = weight

    def evaluate(self, room: Room, placement: Placement) -> float:
        # Find the zone containing the placement
        x, y, _, _ = placement.get_coords()

        containing_zone = None
        for zone in room.zones:
            if zone.contains_point(x, y):
                containing_zone = zone
                break

        if not containing_zone:
            return 0.0  # Not in any zone

        # Check if the zone matches the furniture's zone
        return 1.0 if containing_zone.name == placement.furniture.zone else 0.0


class CornerRule(FuzzyRule):
    """Rule evaluating if furniture is in a corner."""

    def __init__(
        self,
        prefer_corner: bool,
        corner_distance: float = 2.0,
        weight: float = 1.0
    ):
        """
        Initialize a corner rule.

        Args:
            prefer_corner: Whether furniture prefers to be in corners
            corner_distance: Maximum distance to be considered in a corner
            weight: Rule weight
        """
        self.prefer_corner = prefer_corner
        self.corner_distance = corner_distance
        self.weight = weight

    def evaluate(self, room: Room, placement: Placement) -> float:
        x, y, width, height = placement.get_coords()

        # Calculate distances to walls
        dist_left = x
        dist_right = room.width - (x + width)
        dist_top = y
        dist_bottom = room.height - (y + height)

        # Check if in any corner
        in_top_left = dist_left <= self.corner_distance and dist_top <= self.corner_distance
        in_top_right = dist_right <= self.corner_distance and dist_top <= self.corner_distance
        in_bottom_left = dist_left <= self.corner_distance and dist_bottom <= self.corner_distance
        in_bottom_right = dist_right <= self.corner_distance and dist_bottom <= self.corner_distance

        in_corner = in_top_left or in_top_right or in_bottom_left or in_bottom_right

        # If prefer_corner is False, invert the result
        if not self.prefer_corner:
            return 0.0 if in_corner else 1.0

        return 1.0 if in_corner else 0.0


class RelationshipRule(FuzzyRule):
    """Rule evaluating relationships between furniture pieces."""

    def __init__(
        self,
        rules: List[FuzzyRule],
        aggregation_method: str = 'min',
        weight: float = 1.0
    ):
        """
        Initialize a relationship rule.

        Args:
            rules: List of rules to evaluate
            aggregation_method: Method to aggregate rules ('min', 'max', 'mean')
            weight: Rule weight
        """
        self.rules = rules
        self.aggregation_method = aggregation_method
        self.weight = weight

    def evaluate(self, room: Room, placement: Placement) -> float:
        # Evaluate all rules
        rule_values = [rule.evaluate(room, placement) for rule in self.rules]

        # Aggregate results
        if self.aggregation_method == 'min':
            return min(rule_values) if rule_values else 0.0
        elif self.aggregation_method == 'max':
            return max(rule_values) if rule_values else 0.0
        elif self.aggregation_method == 'mean':
            return sum(rule_values) / len(rule_values) if rule_values else 0.0
        else:
            raise ValueError(
                f"Unknown aggregation method: {self.aggregation_method}")


class FuzzyEvaluator:
    """Class for evaluating furniture placements using fuzzy logic."""

    def __init__(self):
        """Initialize a fuzzy evaluator."""
        self.rules_by_type = {}
        self._init_rules()

    def _init_rules(self) -> None:
        """Initialize rules for different furniture types."""
        # TV rules
        tv_rules = [
            WallDistanceRule(
                prefer_wall=True,
                membership_function=TriangularMF(0.0, 0.0, 3.0),
                weight=2.0
            ),
            WindowDistanceRule(
                prefer_window=False,
                membership_function=TriangularMF(0.0, 0.0, 5.0),
                weight=1.0
            ),
            ZoneRule(weight=2.0)
        ]
        self.rules_by_type['tv'] = RelationshipRule(tv_rules, 'mean', 1.0)

        # Sofa rules
        sofa_rules = [
            DistanceRule(
                target_type='tv',
                membership_function=TriangularMF(6.0, 8.0, 12.0),
                weight=3.0
            ),
            FacingRule(
                target_type='tv',
                weight=4.0
            ),
            WallDistanceRule(
                prefer_wall=True,
                membership_function=TriangularMF(0.0, 0.0, 3.0),
                weight=2.0
            ),
            ZoneRule(weight=2.0)
        ]
        self.rules_by_type['sofa'] = RelationshipRule(sofa_rules, 'mean', 1.0)

        # Coffee table rules
        coffee_table_rules = [
            DistanceRule(
                target_type='sofa',
                membership_function=TriangularMF(1.0, 2.0, 3.0),
                weight=3.0
            ),
            DistanceRule(
                target_type='tv',
                membership_function=TriangularMF(3.0, 5.0, 7.0),
                weight=2.0
            ),
            ZoneRule(weight=2.0)
        ]
        self.rules_by_type['coffee_table'] = RelationshipRule(
            coffee_table_rules, 'mean', 1.0)

        # Dining table rules
        dining_table_rules = [
            WallDistanceRule(
                prefer_wall=False,
                membership_function=TriangularMF(0.0, 0.0, 3.0),
                weight=1.0
            ),
            ZoneRule(weight=3.0)
        ]
        self.rules_by_type['dining_table'] = RelationshipRule(
            dining_table_rules, 'mean', 1.0)

        # Dining chair rules
        dining_chair_rules = [
            DistanceRule(
                target_type='dining_table',
                membership_function=TriangularMF(0.0, 0.0, 1.0),
                weight=4.0
            ),
            FacingRule(
                target_type='dining_table',
                weight=3.0
            ),
            ZoneRule(weight=2.0)
        ]
        self.rules_by_type['dining_chair'] = RelationshipRule(
            dining_chair_rules, 'mean', 1.0)

        # Bookshelf rules
        bookshelf_rules = [
            WallDistanceRule(
                prefer_wall=True,
                membership_function=TriangularMF(0.0, 0.0, 1.0),
                weight=3.0
            ),
            CornerRule(
                prefer_corner=True,
                corner_distance=2.0,
                weight=2.0
            ),
            ZoneRule(weight=1.0)
        ]
        self.rules_by_type['bookshelf'] = RelationshipRule(
            bookshelf_rules, 'mean', 1.0)

        # Side table rules
        side_table_rules = [
            DistanceRule(
                target_type='sofa',
                membership_function=TriangularMF(0.0, 0.5, 1.5),
                weight=3.0
            ),
            ZoneRule(weight=1.0)
        ]
        self.rules_by_type['side_table'] = RelationshipRule(
            side_table_rules, 'mean', 1.0)

        # Lamp rules
        lamp_rules = [
            DistanceRule(
                target_type='sofa',
                membership_function=TriangularMF(0.0, 1.0, 3.0),
                weight=2.0
            ),
            DistanceRule(
                target_type='side_table',
                membership_function=TriangularMF(0.0, 0.0, 1.0),
                weight=3.0
            )
        ]
        self.rules_by_type['lamp'] = RelationshipRule(lamp_rules, 'mean', 1.0)

        # Plant rules
        plant_rules = [
            WindowDistanceRule(
                prefer_window=True,
                membership_function=TriangularMF(0.0, 0.0, 5.0),
                weight=2.0
            ),
            CornerRule(
                prefer_corner=True,
                corner_distance=2.0,
                weight=1.0
            )
        ]
        self.rules_by_type['plant'] = RelationshipRule(
            plant_rules, 'mean', 1.0)

    def evaluate_placement(self, room: Room, placement: Placement) -> float:
        """
        Evaluate a placement using fuzzy rules.

        Args:
            room: Room object
            placement: Placement to evaluate

        Returns:
            Evaluation score in [0, 1]
        """
        furniture_type = placement.furniture.furniture_type

        if furniture_type not in self.rules_by_type:
            return 0.5  # Default score for unknown types

        # Evaluate rules for this furniture type
        rule = self.rules_by_type[furniture_type]
        score = rule.evaluate(room, placement)

        return score

    def evaluate_layout(self, room: Room) -> float:
        """
        Evaluate the entire room layout.

        Args:
            room: Room object

        Returns:
            Overall evaluation score in [0, 1]
        """
        if not room.placements:
            return 0.0  # Empty room

        # Evaluate each placement
        scores = []
        for placement in room.placements:
            score = self.evaluate_placement(room, placement)
            scores.append(score)

        # Return average score
        return sum(scores) / len(scores)

    def find_best_placement(
        self,
        room: Room,
        furniture: 'Furniture',
        num_positions: int = 20,
        num_orientations: int = 4
    ) -> Tuple[Optional[Placement], float]:
        """
        Find the best placement for a furniture piece.

        Args:
            room: Room object
            furniture: Furniture to place
            num_positions: Number of positions to evaluate
            num_orientations: Number of orientations to evaluate

        Returns:
            Tuple of (best placement, score)
        """
        # Make a temporary copy of the room for evaluation
        room_copy = Room(room.width, room.height,
                         room.grid_size, room.wall_thickness)
        room_copy.grid = room.grid.copy()
        room_copy.features = room.features.copy()
        room_copy.zones = room.zones.copy()
        room_copy.placements = room.placements.copy()

        # Generate candidate positions
        step_x = room.width / (num_positions // 2)
        step_y = room.height / (num_positions // 2)

        positions = []
        for i in range(num_positions // 2):
            for j in range(num_positions // 2):
                positions.append((i * step_x, j * step_y))

        # Generate candidate orientations
        orientations = list(
            range(min(furniture.orientations, num_orientations)))

        # Evaluate all candidates
        best_placement = None
        best_score = -float('inf')

        for x, y in positions:
            for orientation in orientations:
                # Try to place furniture
                placement = Placement(furniture, x, y, orientation)

                # Check if valid placement
                if not room_copy._is_valid_placement(placement):
                    continue

                # Temporarily add placement for evaluation
                room_copy.placements.append(placement)

                # Evaluate placement
                score = self.evaluate_placement(room_copy, placement)

                # Remove temporary placement
                room_copy.placements.pop()

                # Update best placement
                if score > best_score:
                    best_score = score
                    best_placement = placement

        return best_placement, best_score if best_placement else 0.0

    def __repr__(self) -> str:
        return f"FuzzyEvaluator(rules={len(self.rules_by_type)})"
