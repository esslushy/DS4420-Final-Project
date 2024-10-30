from World.Entity import Entity
from World.Vector2D import Vector2D
from typing import List
import math
class CircleEntity(Entity):
    """
    Represents an entity that goes in a circle
    """

    def __init__(self, radius, position, velocity, angle):
        super().__init__(radius, position, velocity)
        self.angle = angle

    def rotate(self, vector: Vector2D):
        return Vector2D(
            (vector.x * math.cos(self.angle)) - (vector.y * math.sin(self.angle)),
            (vector.x * math.sin(self.angle)) + (vector.y * math.cos(self.angle))
        )

    def update_velocity(self, world_width, world_height):
        self.velocity = self.rotate(self.velocity)