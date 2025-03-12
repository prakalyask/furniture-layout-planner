"""
Pathway generation algorithm.
Finds optimal pathways between functional areas in a room.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import heapq
from core.room import Room, Zone, Placement, RoomFeature

# Define a grid cell (for pathfinding)


class Cell:
    """Class representing a cell in the pathfinding grid."""

    def __init__(self, x: int, y: int, walkable: bool = True, cost: float = 1.0):
        """
        Initialize a cell.

        Args:
            x: X-coordinate
            y: Y-coordinate
            walkable: Whether the cell is walkable
            cost: Movement cost for this cell
        """
        self.x = x
        self.y = y
        self.walkable = walkable
        self.cost = cost
        self.g_cost = float('inf')  # Cost from start
        self.h_cost = 0  # Heuristic cost to goal
        self.f_cost = float('inf')  # Total cost (g + h)
        self.parent = None

    def __lt__(self, other):
        """Compare cells by f_cost (for priority queue)."""
        return self.f_cost < other.f_cost

# Define a path (sequence of cells)


class Path:
    """Class representing a path between two points."""

    def __init__(self, cells: List[Cell] = None):
        """
        Initialize a path.

        Args:
            cells: List of cells in the path
        """
        self.cells = cells or []

    def get_length(self) -> float:
        """Get the length of the path."""
        return len(self.cells)

    def get_cost(self) -> float:
        """Get the total cost of the path."""
        return sum(cell.cost for cell in self.cells)

    def get_points(self) -> List[Tuple[float, float]]:
        """Get the points along the path."""
        return [(cell.x, cell.y) for cell in self.cells]


class PathwayGenerator:
    """Class for generating optimal pathways in a room."""

    def __init__(self, room: Room, grid_size: float = 0.5):
        """
        Initialize a pathway generator.

        Args:
            room: Room object
            grid_size: Size of each grid cell (in feet)
        """
        self.room = room
        self.grid_size = grid_size

        # Calculate grid dimensions
        self.grid_width = int(room.width / grid_size) + 1
        self.grid_height = int(room.height / grid_size) + 1

        # Create grid
        self.grid = self._create_grid()

        # Important points (for pathfinding)
        self.important_points = []
        self._find_important_points()

        # Generated pathways
        self.pathways = []

    def _create_grid(self) -> List[List[Cell]]:
        """
        Create a grid for pathfinding.

        Returns:
            2D grid of cells
        """
        grid = []

        for y in range(self.grid_height):
            row = []
            for x in range(self.grid_width):
                # Convert to room coordinates
                room_x = x * self.grid_size
                room_y = y * self.grid_size

                # Check if walkable
                walkable = True
                cost = 1.0

                # Check if outside room bounds
                if room_x < 0 or room_x > self.room.width or room_y < 0 or room_y > self.room.height:
                    walkable = False
                    cost = float('inf')

                # Check if overlaps with furniture
                if walkable:
                    for placement in self.room.placements:
                        p_x, p_y, p_width, p_height = placement.get_coords()

                        # Add clearance
                        clearance = placement.furniture.clearance_required
                        p_x -= clearance
                        p_y -= clearance
                        p_width += 2 * clearance
                        p_height += 2 * clearance

                        if (p_x <= room_x <= p_x + p_width and
                                p_y <= room_y <= p_y + p_height):
                            walkable = False
                            cost = float('inf')
                            break

                # Check if overlaps with walls
                if walkable:
                    # Walls at room edges
                    if (room_x < self.room.wall_thickness or
                        room_x > self.room.width - self.room.wall_thickness or
                        room_y < self.room.wall_thickness or
                            room_y > self.room.height - self.room.wall_thickness):
                        walkable = False
                        cost = float('inf')

                # Check if near furniture (higher cost)
                if walkable:
                    for placement in self.room.placements:
                        p_x, p_y, p_width, p_height = placement.get_coords()

                        # Calculate distance to furniture
                        dist_x = max(
                            0, max(p_x - room_x, room_x - (p_x + p_width)))
                        dist_y = max(
                            0, max(p_y - room_y, room_y - (p_y + p_height)))
                        distance = np.sqrt(dist_x ** 2 + dist_y ** 2)

                        # If near furniture, increase cost
                        near_distance = 1.0  # feet
                        if distance < near_distance:
                            # Cost increases as distance decreases
                            cost = max(
                                cost, 1.0 + (near_distance - distance) * 2)

                row.append(Cell(x, y, walkable, cost))
            grid.append(row)

        return grid

    def _find_important_points(self) -> None:
        """Find important points for pathfinding (doors, zone centers, etc.)."""
        points = []

        # Add doors
        for feature in self.room.features:
            if feature.feature_type == 'door':
                feature_x, feature_y, feature_width, feature_height = feature.get_coords()
                center_x = feature_x + feature_width / 2
                center_y = feature_y + feature_height / 2

                # Convert to grid coordinates
                grid_x = int(center_x / self.grid_size)
                grid_y = int(center_y / self.grid_size)

                # Add point
                points.append((grid_x, grid_y))

        # Add zone centers
        for zone in self.room.zones:
            center_x = zone.x + zone.width / 2
            center_y = zone.y + zone.height / 2

            # Convert to grid coordinates
            grid_x = int(center_x / self.grid_size)
            grid_y = int(center_y / self.grid_size)

            # Add point
            points.append((grid_x, grid_y))

        # Add furniture centers (for major pieces)
        for placement in self.room.placements:
            if placement.furniture.furniture_type in ['tv', 'sofa', 'dining_table']:
                center_x, center_y = placement.get_center()

                # Convert to grid coordinates
                grid_x = int(center_x / self.grid_size)
                grid_y = int(center_y / self.grid_size)

                # Add point
                points.append((grid_x, grid_y))

        # Store points
        self.important_points = points

    def _heuristic(self, a: Cell, b: Cell) -> float:
        """
        Calculate heuristic cost between two cells.

        Args:
            a: First cell
            b: Second cell

        Returns:
            Heuristic cost
        """
        # Manhattan distance
        return abs(a.x - b.x) + abs(a.y - b.y)

    def _get_neighbors(self, cell: Cell) -> List[Cell]:
        """
        Get neighbors of a cell.

        Args:
            cell: Cell to get neighbors for

        Returns:
            List of neighbor cells
        """
        neighbors = []

        # Check orthogonal neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = cell.x + dx, cell.y + dy

            # Check if within bounds
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                neighbors.append(self.grid[ny][nx])

        # Check diagonal neighbors
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = cell.x + dx, cell.y + dy

            # Check if within bounds
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                # Check if both orthogonal neighbors are walkable
                if self.grid[cell.y][nx].walkable and self.grid[ny][cell.x].walkable:
                    neighbors.append(self.grid[ny][nx])

        return neighbors

    def _find_path(self, start_x: int, start_y: int, goal_x: int, goal_y: int) -> Optional[Path]:
        """
        Find a path between two points using A* algorithm.

        Args:
            start_x: Start X-coordinate
            start_y: Start Y-coordinate
            goal_x: Goal X-coordinate
            goal_y: Goal Y-coordinate

        Returns:
            Path if found, None otherwise
        """
        # Get start and goal cells
        start_cell = self.grid[start_y][start_x]
        goal_cell = self.grid[goal_y][goal_x]

        # Check if start or goal is not walkable
        if not start_cell.walkable or not goal_cell.walkable:
            return None

        # Initialize open and closed sets
        open_set = []
        closed_set = set()

        # Initialize start cell
        start_cell.g_cost = 0
        start_cell.h_cost = self._heuristic(start_cell, goal_cell)
        start_cell.f_cost = start_cell.g_cost + start_cell.h_cost

        # Add start cell to open set
        heapq.heappush(open_set, (start_cell.f_cost, start_cell))

        while open_set:
            # Get cell with lowest f_cost
            _, current = heapq.heappop(open_set)

            # Check if goal reached
            if current.x == goal_cell.x and current.y == goal_cell.y:
                # Reconstruct path
                path = []
                while current:
                    path.append(current)
                    current = current.parent
                path.reverse()

                return Path(path)

            # Add to closed set
            closed_set.add((current.x, current.y))

            # Check neighbors
            for neighbor in self._get_neighbors(current):
                # Skip if in closed set
                if (neighbor.x, neighbor.y) in closed_set:
                    continue

                # Skip if not walkable
                if not neighbor.walkable:
                    continue

                # Calculate new g_cost
                new_g_cost = current.g_cost + neighbor.cost

                # Check if new path is better
                if new_g_cost < neighbor.g_cost:
                    # Update costs
                    neighbor.g_cost = new_g_cost
                    neighbor.h_cost = self._heuristic(neighbor, goal_cell)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    neighbor.parent = current

                    # Add to open set (or update position)
                    heapq.heappush(open_set, (neighbor.f_cost, neighbor))

        # No path found
        return None

    def generate_pathways(self) -> List[Path]:
        """
        Generate pathways between important points.

        Returns:
            List of paths
        """
        # Reset pathways
        self.pathways = []

        # Update grid (in case furniture changed)
        self.grid = self._create_grid()

        # Update important points
        self._find_important_points()

        # Generate paths between all pairs of important points
        for i in range(len(self.important_points)):
            for j in range(i + 1, len(self.important_points)):
                start_x, start_y = self.important_points[i]
                goal_x, goal_y = self.important_points[j]

                # Find path
                path = self._find_path(start_x, start_y, goal_x, goal_y)

                if path:
                    self.pathways.append(path)

        return self.pathways

    def get_pathway_grid(self) -> np.ndarray:
        """
        Get a grid representation of pathways.

        Returns:
            Grid where 1 indicates a pathway cell
        """
        # Create grid
        grid = np.zeros((self.grid_height, self.grid_width), dtype=int)

        # Mark pathways
        for path in self.pathways:
            for cell in path.cells:
                grid[cell.y, cell.x] = 1

        return grid

    def evaluate_pathway_quality(self) -> float:
        """
        Evaluate the quality of generated pathways.

        Returns:
            Quality score (0-1, higher is better)
        """
        if not self.pathways:
            return 0.0

        # Calculate average path length
        avg_length = sum(path.get_length()
                         for path in self.pathways) / len(self.pathways)

        # Calculate average path cost
        avg_cost = sum(path.get_cost()
                       for path in self.pathways) / len(self.pathways)

        # Calculate coverage (percentage of important points connected)
        connected_points = set()
        for path in self.pathways:
            for point in path.get_points():
                connected_points.add(point)

        coverage = len(connected_points) / len(self.important_points)

        # Calculate pathway clarity (avoid complex paths)
        clarity = 1.0 - min(avg_cost / (10 * avg_length), 1.0)

        # Calculate final score
        score = 0.4 * coverage + 0.3 * clarity + \
            0.3 * (1.0 / min(avg_length / 10, 1.0))

        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
