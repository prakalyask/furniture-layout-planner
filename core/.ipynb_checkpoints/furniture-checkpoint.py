from typing import Dict, List, Tuple, Optional, Set
import numpy as np

class Furniture:
    """
    Enhanced furniture class with relationship awareness and constraints.
    
    Attributes:
        name: Name of the furniture piece
        width: Width in feet
        height: Depth in feet
        furniture_type: Type of furniture (sofa, tv, table, etc.)
        orientations: Number of possible orientations (1-4)
        related_furniture: List of furniture types this piece relates to
        optimal_distances: Dictionary of optimal distances to related furniture
        min_wall_distance: Minimum distance from walls (in feet)
        prefer_wall: Whether this furniture prefers to be against a wall
        prefer_corner: Whether this furniture prefers to be in a corner
        prefer_window: Whether this furniture prefers to be near a window
        avoid_window: Whether this furniture should avoid windows
        clearance_required: Clearance required around the furniture (in feet)
        zone: Zone this furniture belongs to
    """
    
    def __init__(
        self,
        name: str,
        width: float,
        height: float,
        furniture_type: str,
        orientations: int = 2,
        zone: str = "general",
        min_wall_distance: float = 0.0,
        prefer_wall: bool = False,
        prefer_corner: bool = False,
        prefer_window: bool = False,
        avoid_window: bool = False,
        clearance_required: float = 1.0,
    ):
        """Initialize a furniture piece with enhanced properties."""
        self.name = name
        self.width = width
        self.height = height
        self.furniture_type = furniture_type
        self.orientations = orientations
        self.zone = zone
        self.min_wall_distance = min_wall_distance
        self.prefer_wall = prefer_wall
        self.prefer_corner = prefer_corner
        self.prefer_window = prefer_window
        self.avoid_window = avoid_window
        self.clearance_required = clearance_required
        
        # Relationships with other furniture
        self.related_furniture = []
        self.optimal_distances = {}
        self.facing_furniture = None  # Furniture this piece should face
        self.group_with = []  # Furniture to group together with
        
    def get_dimensions(self, orientation: int) -> Tuple[float, float]:
        """
        Get dimensions of furniture for given orientation.
        
        Args:
            orientation: 0 for normal, 1 for rotated 90 degrees, etc.
            
        Returns:
            Tuple of (width, height) for the given orientation
        """
        if orientation == 0 or self.orientations == 1:
            return (self.width, self.height)
        elif orientation == 1:
            return (self.height, self.width)
        elif orientation == 2:
            return (self.width, self.height)  # 180 degrees, same as 0
        elif orientation == 3:
            return (self.height, self.width)  # 270 degrees, same as 90
        else:
            raise ValueError(f"Invalid orientation: {orientation}")
    
    def add_relationship(self, furniture_type: str, optimal_distance: float) -> None:
        """
        Add a relationship with another furniture type.
        
        Args:
            furniture_type: Type of related furniture
            optimal_distance: Optimal distance in feet
        """
        if furniture_type not in self.related_furniture:
            self.related_furniture.append(furniture_type)
        self.optimal_distances[furniture_type] = optimal_distance
    
    def set_facing(self, furniture_type: str) -> None:
        """
        Set which furniture type this piece should face.
        
        Args:
            furniture_type: Type of furniture to face
        """
        self.facing_furniture = furniture_type
    
    def add_to_group(self, furniture_type: str) -> None:
        """
        Add a furniture type to group with.
        
        Args:
            furniture_type: Type of furniture to group with
        """
        if furniture_type not in self.group_with:
            self.group_with.append(furniture_type)
    
    def __repr__(self) -> str:
        return f"Furniture(name='{self.name}', type='{self.furniture_type}', size={self.width}x{self.height})"


class FurnitureFactory:
    """Factory for creating furniture with predefined relationships."""
    
    @staticmethod
    def create_tv(width: float = 3.0, height: float = 2.0) -> Furniture:
        """Create a TV stand."""
        tv = Furniture(
            name="TV Stand",
            width=width,
            height=height,
            furniture_type="tv",
            orientations=1,
            zone="entertainment",
            prefer_wall=True,
            min_wall_distance=0.0
        )
        return tv
    
    @staticmethod
    def create_sofa(width: float = 6.0, height: float = 3.0) -> Furniture:
        """Create a sofa."""
        sofa = Furniture(
            name="Sofa",
            width=width,
            height=height,
            furniture_type="sofa",
            orientations=4,
            zone="entertainment",
            min_wall_distance=0.0,
            prefer_wall=True,
            clearance_required=2.0
        )
        sofa.add_relationship("tv", 8.0)  # 8 feet optimal distance to TV
        sofa.set_facing("tv")  # Sofa should face the TV
        sofa.add_relationship("coffee_table", 2.0)  # 2 feet to coffee table
        return sofa
    
    @staticmethod
    def create_coffee_table(width: float = 4.0, height: float = 2.0) -> Furniture:
        """Create a coffee table."""
        table = Furniture(
            name="Coffee Table",
            width=width,
            height=height,
            furniture_type="coffee_table",
            orientations=2,
            zone="entertainment",
            clearance_required=1.5
        )
        table.add_relationship("sofa", 2.0)  # 2 feet to sofa
        table.add_relationship("tv", 5.0)  # 5 feet to TV
        return table
    
    @staticmethod
    def create_dining_table(width: float = 5.0, height: float = 3.0) -> Furniture:
        """Create a dining table."""
        table = Furniture(
            name="Dining Table",
            width=width,
            height=height,
            furniture_type="dining_table",
            orientations=2,
            zone="dining",
            clearance_required=3.0
        )
        return table
    
    @staticmethod
    def create_dining_chair() -> Furniture:
        """Create a dining chair."""
        chair = Furniture(
            name="Dining Chair",
            width=1.5,
            height=1.5,
            furniture_type="dining_chair",
            orientations=4,
            zone="dining",
            clearance_required=1.0
        )
        chair.add_relationship("dining_table", 0.0)  # Right next to table
        chair.set_facing("dining_table")  # Chair should face the table
        chair.add_to_group("dining_chair")  # Group with other chairs
        return chair
    
    @staticmethod
    def create_bookshelf(width: float = 3.0, height: float = 1.0) -> Furniture:
        """Create a bookshelf."""
        bookshelf = Furniture(
            name="Bookshelf",
            width=width,
            height=height,
            furniture_type="bookshelf",
            orientations=2,
            zone="storage",
            prefer_wall=True,
            prefer_corner=True,
            min_wall_distance=0.0,
            clearance_required=2.0
        )
        bookshelf.add_to_group("bookshelf")  # Group with other bookshelves
        return bookshelf
    
    @staticmethod
    def create_side_table() -> Furniture:
        """Create a side table."""
        table = Furniture(
            name="Side Table",
            width=1.5,
            height=1.5,
            furniture_type="side_table",
            orientations=1,
            zone="entertainment",
            clearance_required=1.0
        )
        table.add_relationship("sofa", 0.5)  # 0.5 feet to sofa
        return table
    
    @staticmethod
    def create_lamp() -> Furniture:
        """Create a lamp."""
        lamp = Furniture(
            name="Lamp",
            width=1.0,
            height=1.0,
            furniture_type="lamp",
            orientations=1,
            zone="general",
            clearance_required=0.5
        )
        lamp.add_relationship("sofa", 1.0)  # 1 foot to sofa
        lamp.add_relationship("side_table", 0.0)  # On a side table
        return lamp
    
    @staticmethod
    def create_plant() -> Furniture:
        """Create a plant."""
        plant = Furniture(
            name="Plant",
            width=1.0,
            height=1.0,
            furniture_type="plant",
            orientations=1,
            zone="general",
            prefer_window=True,
            prefer_corner=True,
            clearance_required=0.5
        )
        return plant
    
    @staticmethod
    def create_furniture_set() -> List[Furniture]:
        """Create a default set of furniture."""
        return [
            FurnitureFactory.create_tv(),
            FurnitureFactory.create_sofa(),
            FurnitureFactory.create_coffee_table(),
            FurnitureFactory.create_dining_table(),
            FurnitureFactory.create_dining_chair(),
            FurnitureFactory.create_dining_chair(),
            FurnitureFactory.create_dining_chair(),
            FurnitureFactory.create_dining_chair(),
            FurnitureFactory.create_bookshelf(),
            FurnitureFactory.create_side_table(),
            FurnitureFactory.create_lamp(),
            FurnitureFactory.create_plant()
        ]