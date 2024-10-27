from abc import ABC
from World.Vector2D import Vector2D
from ProjectParameters import STEP_SIZE

class Entity(ABC):
    """
    Represents an entity in the world.
    """
    def __init__(self, position: Vector2D, velocity: Vector2D):
        self.position = position
        self.velocity = velocity

    def update_position(self):
        """
        Updates position based on velocity
        """
        self.position += self.velocity * STEP_SIZE

    def update_velocity(self):
        """
        Updates velocity
        """
        raise NotImplementedError("Velocity updates must be implemented in base class")