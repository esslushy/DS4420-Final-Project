from World.Entity import Entity
from World.Vector2D import Vector2D

class Robot(Entity):
    """
    Represents the robot in the world
    """
    def __init__(self, radius, starting_position):
        super().__init__(radius, starting_position, Vector2D(0, 0))
        self.num_collisions = 0