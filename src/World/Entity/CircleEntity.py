from World.Entity import Entity
from World.Vector2D import Vector2D
import math
class CircleEntity(Entity):
    """
    Represents an entity that goes in a circle
    """

    def __init__(self, radius: float, position: Vector2D, velocity: Vector2D, angle: float):
        super().__init__(radius, position, velocity)
        """
        Initializes a new entity at the given position with the given velocity 
        and radius that goes in a circle.

        Args:
            radius: The radius of the entity.
            position: The initial position of the entity.
            velocity: The velocity of the entity.
            angle: The angle of rotation for the velocity.
        """
        self.angle = angle

    def rotate(self, vector: Vector2D):
        """
        Rotates a 2D vector by the given angle

        Args:
            vector: The vector to rotate
        """
        return Vector2D(
            (vector.x * math.cos(self.angle)) - (vector.y * math.sin(self.angle)),
            (vector.x * math.sin(self.angle)) + (vector.y * math.cos(self.angle))
        )

    def update_velocity(self, world_width, world_height):
        self.velocity = self.rotate(self.velocity)