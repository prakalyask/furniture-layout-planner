from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from .furniture import Furniture

class RoomFeature:
    def __init__(
        self,
        feature_type: str,
        x: float,
        y: float,
        width: float,
        height: float,
        orientation: int = 0
    ):
        self.feature_type = feature_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.orientation = orientation
        
    def get_coords(self) -> Tuple[float, float, float, float]:
        if self.orientation == 0 or self.orientation == 2:
            return (self.x, self.y, self.width, self.height)
        else:
            return (self.x, self.y, self.height, self.width)
            
    def __repr__(self) -> str:
        return f"RoomFeature(type='{self.feature_type}', pos=({self.x}, {self.y}), size={self.width}x{self.height})"


class Zone:
    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        width: float,
        height: float
    ):
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.furniture = []
        
    def contains_point(self, x: float, y: float) -> bool:
        return (
            self.x <= x <= self.x + self.width and
            self.y <= y <= self.y + self.height
        )
        
    def add_furniture(self, furniture: Furniture) -> None:
        self.furniture.append(furniture)
        
    def __repr__(self) -> str:
        return f"Zone(name='{self.name}', pos=({self.x}, {self.y}), size={self.width}x{self.height})"


class Placement:
    def __init__(
        self,
        furniture: Furniture,
        x: float,
        y: float,
        orientation: int = 0
    ):
        self.furniture = furniture
        self.x = x
        self.y = y
        self.orientation = orientation
        
    def get_coords(self) -> Tuple[float, float, float, float]:
        width, height = self.furniture.get_dimensions(self.orientation)
        return (self.x, self.y, width, height)
    
    def get_center(self) -> Tuple[float, float]:
        width, height = self.furniture.get_dimensions(self.orientation)
        return (self.x + width / 2, self.y + height / 2)
    
    def get_facing_direction(self) -> Tuple[float, float]:
        if self.orientation == 0:
            return (0, 1)
        elif self.orientation == 1:
            return (1, 0)
        elif self.orientation == 2:
            return (0, -1)
        else:
            return (-1, 0)
            
    def overlaps(self, other) -> bool:
        x1, y1, w1, h1 = self.get_coords()
        x2, y2, w2, h2 = other.get_coords()
        
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
            
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
            
        return True
        
    def distance_to(self, other) -> float:
        c1_x, c1_y = self.get_center()
        c2_x, c2_y = other.get_center()
        
        return np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)
        
    def is_facing(self, other) -> bool:
        c1_x, c1_y = self.get_center()
        c2_x, c2_y = other.get_center()
        
        dir_x = c2_x - c1_x
        dir_y = c2_y - c1_y
        
        mag = np.sqrt(dir_x ** 2 + dir_y ** 2)
        if mag > 0:
            dir_x /= mag
            dir_y /= mag
            
        f_x, f_y = self.get_facing_direction()
        
        dot = dir_x * f_x + dir_y * f_y
        
        return dot > 0.7
    
    def __repr__(self) -> str:
        return f"Placement({self.furniture.name}, pos=({self.x}, {self.y}), orientation={self.orientation})"


class Room:
    def __init__(
        self,
        width: float = 20.0,
        height: float = 15.0,
        grid_size: float = 0.5,
        wall_thickness: float = 0.5
    ):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.wall_thickness = wall_thickness
        
        self.grid_width = int(width / grid_size)
        self.grid_height = int(height / grid_size)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        
        self._init_walls()
        
        self.features = []
        self.zones = []
        self.placements = []
        self.available_furniture = []
        
    def _init_walls(self) -> None:
        wall_cells = int(self.wall_thickness / self.grid_size)
        if wall_cells < 1:
            wall_cells = 1
            
        self.grid[:wall_cells, :] = -1
        self.grid[-wall_cells:, :] = -1
        self.grid[:, :wall_cells] = -1
        self.grid[:, -wall_cells:] = -1
        
    def add_feature(self, feature: RoomFeature) -> None:
        self.features.append(feature)
        
        x, y, width, height = feature.get_coords()
        
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_width = int(width / self.grid_size)
        grid_height = int(height / self.grid_size)
        
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        grid_width = max(1, min(grid_width, self.grid_width - grid_x))
        grid_height = max(1, min(grid_height, self.grid_height - grid_y))
        
        if feature.feature_type == 'door':
            self.grid[grid_y:grid_y+grid_height, grid_x:grid_x+grid_width] = -2
        elif feature.feature_type == 'window':
            self.grid[grid_y:grid_y+grid_height, grid_x:grid_x+grid_width] = -3
        else:
            self.grid[grid_y:grid_y+grid_height, grid_x:grid_x+grid_width] = -4
            
    def add_zone(self, zone: Zone) -> None:
        self.zones.append(zone)
        
    def add_furniture(self, furniture: Furniture) -> None:
        self.available_furniture.append(furniture)
        
    def place_furniture(
        self,
        furniture: Furniture,
        x: float,
        y: float,
        orientation: int = 0
    ) -> Optional[Placement]:
        placement = Placement(furniture, x, y, orientation)
        
        if not self._is_valid_placement(placement):
            return None
            
        self.placements.append(placement)
        self._update_grid_for_placement(placement)
        
        if furniture in self.available_furniture:
            self.available_furniture.remove(furniture)
            
        for zone in self.zones:
            if zone.contains_point(x, y):
                zone.add_furniture(furniture)
                break
                
        return placement
        
    def _is_valid_placement(self, placement: Placement) -> bool:
        x, y, width, height = placement.get_coords()
        
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            return False
            
        for other in self.placements:
            if placement.overlaps(other):
                return False
                
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_width = int(width / self.grid_size)
        grid_height = int(height / self.grid_size)
        
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        grid_width = max(1, min(grid_width, self.grid_width - grid_x))
        grid_height = max(1, min(grid_height, self.grid_height - grid_y))
        
        for i in range(grid_height):
            for j in range(grid_width):
                cell_x = grid_x + j
                cell_y = grid_y + i
                
                if cell_x >= self.grid_width or cell_y >= self.grid_height:
                    return False
                    
                if self.grid[cell_y, cell_x] == -1 or self.grid[cell_y, cell_x] == -2:
                    return False
                    
                if self.grid[cell_y, cell_x] > 0:
                    return False
                    
        return True
        
    def _update_grid_for_placement(self, placement: Placement) -> None:
        x, y, width, height = placement.get_coords()
        
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_width = int(width / self.grid_size)
        grid_height = int(height / self.grid_size)
        
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        grid_width = max(1, min(grid_width, self.grid_width - grid_x))
        grid_height = max(1, min(grid_height, self.grid_height - grid_y))
        
        furniture_id = len(self.placements)
        
        self.grid[grid_y:grid_y+grid_height, grid_x:grid_x+grid_width] = furniture_id
        
    def get_placement_by_type(self, furniture_type: str) -> List[Placement]:
        return [p for p in self.placements if p.furniture.furniture_type == furniture_type]
        
    def get_zone_by_name(self, name: str) -> Optional[Zone]:
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None
        
    def reset(self) -> None:
        # Optimized reset method
        # Store original grid dimensions
        grid_height, grid_width = self.grid.shape
        
        # Create a new empty grid (faster than zeroing out each cell)
        self.grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Initialize walls (fast operation)
        self._init_walls()
        
        # Reset placements
        self.available_furniture.extend([p.furniture for p in self.placements])
        self.placements = []
        
        # Reset zones
        for zone in self.zones:
            zone.furniture = []
        
        # Re-add features to grid efficiently
        for feature in self.features:
            # Get feature coordinates
            x, y, width, height = feature.get_coords()
            
            # Convert to grid coordinates (this is where the slowdown was happening)
            grid_x = max(0, min(int(x / self.grid_size), self.grid_width - 1))
            grid_y = max(0, min(int(y / self.grid_size), self.grid_height - 1))
            
            # Calculate grid dimensions with bounds checking
            grid_width = max(1, min(int(width / self.grid_size), self.grid_width - grid_x))
            grid_height = max(1, min(int(height / self.grid_size), self.grid_height - grid_y))
            
            # Set feature type in grid (faster than calling add_feature)
            feature_value = -1  # default for walls
            if feature.feature_type == 'door':
                feature_value = -2
            elif feature.feature_type == 'window':
                feature_value = -3
            else:
                feature_value = -4
                
            # Update grid directly
            self.grid[grid_y:grid_y+grid_height, grid_x:grid_x+grid_width] = feature_value
        
        return self.grid
            
    def __repr__(self) -> str:
        return f"Room(width={self.width}, height={self.height}, furniture={len(self.placements)})"